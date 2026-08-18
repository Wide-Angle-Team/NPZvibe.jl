# The `.npy` container: a 64-byte-aligned header holding a Python dict literal,
# followed by the raw element data.

const NPY_MAGIC = UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]  # \x93NUMPY
const ARRAY_ALIGN = 64
const WRITE_CHUNK_BYTES = 8 * 1024 * 1024

struct NPYHeader
    dt::NPYDType
    fortran_order::Bool
    shape::Tuple{Vararg{Int}}
    dataoffset::Int
end

"""
    read_npy_header(io) -> NPYHeader

Read and validate an `.npy` header from the current position of `io`.
"""
function read_npy_header(io::IO)
    start = position(io)
    magic = read(io, 6)
    (length(magic) == 6 && magic == NPY_MAGIC) &&
        return read_npy_header_body(io, start)
    throw(NPZError("not a .npy file: bad magic $(repr(String(copy(magic))))"))
end

function read_npy_header_body(io::IO, start::Int)
    major = read(io, UInt8)
    minor = read(io, UInt8)
    minor == 0 || throw(NPZError("unsupported .npy version $major.$minor"))
    hlen = if major == 0x01
        Int(ltoh(read(io, UInt16)))
    elseif major == 0x02 || major == 0x03
        Int(ltoh(read(io, UInt32)))
    else
        throw(NPZError("unsupported .npy version $major.$minor"))
    end
    hbytes = read(io, hlen)
    length(hbytes) == hlen || throw(NPZError("truncated .npy header"))
    # v1.0 and v2.0 headers are latin-1, v3.0 headers are utf-8.
    hstr = major == 0x03 ? String(hbytes) : join(Char(b) for b in hbytes)
    d = pyparse(hstr)
    d isa AbstractDict || throw(NPZError(".npy header is not a dict"))
    for k in ("descr", "fortran_order", "shape")
        haskey(d, k) || throw(NPZError(".npy header is missing the '$k' key"))
    end
    fortran = d["fortran_order"]
    fortran isa Bool || throw(NPZError(".npy header has a non-boolean 'fortran_order'"))
    shapev = d["shape"]
    (shapev isa Tuple || shapev isa AbstractVector) ||
        throw(NPZError(".npy header has a malformed 'shape'"))
    shape = Tuple(Int(x) for x in shapev)
    any(<(0), shape) && throw(NPZError(".npy header has a negative dimension"))
    NPYHeader(parse_dtype(d["descr"]), fortran, shape, position(io) - start)
end

# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

"""
    npyread(path; mmap=false) -> Array
    npyread(io) -> Array

Read a single `.npy` array.  With `mmap=true` the data is memory-mapped instead
of copied whenever its dtype and layout allow it, which keeps arrays larger than
RAM usable.  A mapped array is a read-only view of the file (it is opened for
reading), so copy it before modifying it.
"""
function npyread(path::AbstractString; mmap::Bool=false)
    open(path, "r") do io
        npyread(io; mmapfile=(mmap ? io : nothing))
    end
end

function npyread(io::IO; mmapfile::Union{Nothing,IOStream}=nothing)
    h = read_npy_header(io)
    read_npy_data(io, h; mmapfile)
end

function read_npy_data(io::IO, h::NPYHeader; mmapfile::Union{Nothing,IOStream}=nothing)
    dt, shape = h.dt, h.shape
    N = length(shape)
    # A 1-d or 0-d array is both C- and Fortran-contiguous, and Julia arrays are
    # column-major, i.e. Fortran order, so only C-ordered N>1 data needs a
    # transpose after reading.
    contiguous = h.fortran_order || N <= 1
    dims = contiguous ? shape : reverse(shape)
    try
        A = if isbitskind(dt)
            read_bits_array(io, dt.T, dims, dt.swap, contiguous ? mmapfile : nothing)
        elseif dt.kind === :bool
            raw = read_bits_array(io, UInt8, dims, false, nothing)
            map(!=(0x00), raw)
        elseif dt.kind === :datetime || dt.kind === :timedelta
            raw = read_bits_array(io, Int64, dims, dt.swap, nothing)
            time_array(dt, raw)
        else
            read_generic_array(io, dt, dims)
        end
        contiguous ? A : permutedims(A, ntuple(i -> N - i + 1, N))
    catch e
        e isa EOFError && throw(NPZError("truncated .npy data: the file ends before the array does"))
        rethrow()
    end
end

function read_bits_array(io::IO, ::Type{T}, dims::Tuple{Vararg{Int}}, swap::Bool,
                         mmapfile::Union{Nothing,IOStream}) where {T}
    if mmapfile !== nothing && !swap && !isempty(dims) && position(io) % sizeof(T) == 0 &&
       position(io) + sizeof(T) * prod(dims) <= filesize(mmapfile)
        # `grow=false` so that a damaged file is never enlarged; a short file
        # falls through to the copying path, which reports it as truncated.
        return Mmap.mmap(mmapfile, Array{T,length(dims)}, dims, position(io); grow=false)
    end
    A = Array{T}(undef, dims)
    read!(io, A)
    swap && map!(_bswap, A, A)
    A
end

"Convert raw numpy time counts, widening the element type only if NaT occurs."
function time_array(dt::NPYDType, raw::Array{Int64})
    value = dt.kind === :datetime ?
            v -> datetime64_value(dt.mult, dt.unit, v) :
            v -> timedelta64_value(dt.mult, dt.unit, v)
    T = any(==(NPY_NAT), raw) ? Union{Missing,dt.T} : dt.T
    A = Array{T}(undef, size(raw))
    @inbounds for i in eachindex(raw)
        A[i] = value(raw[i])
    end
    A
end

function read_generic_array(io::IO, dt::NPYDType, dims::Tuple{Vararg{Int}})
    n = prod(dims; init=1)
    A = Array{array_eltype(dt)}(undef, dims)
    dt.itemsize == 0 && return fill!(A, readvalue(dt, UInt8[], 0))
    nchunk = max(1, div(WRITE_CHUNK_BYTES, dt.itemsize))
    buf = Vector{UInt8}(undef, min(n, nchunk) * dt.itemsize)
    i = 0
    while i < n
        m = min(nchunk, n - i)
        nbytes = m * dt.itemsize
        read!(io, view(buf, 1:nbytes))
        @inbounds for k in 1:m
            A[i+k] = readvalue(dt, buf, (k - 1) * dt.itemsize)
        end
        i += m
    end
    A
end

# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

"""
    npywrite(path, A; stringdtype=:U)
    npywrite(io, A; stringdtype=:U)

Write `A` as a single `.npy` array.  Julia `String`s are stored as numpy's
UCS-4 `U` dtype by default; `stringdtype=:S` stores them as raw byte strings.
"""
function npywrite(path::AbstractString, A; stringdtype::Symbol=:U)
    open(path, "w") do io
        npywrite(io, A; stringdtype)
    end
    nothing
end

npywrite(io::IO, x; stringdtype::Symbol=:U) = npywrite(io, as_array(x); stringdtype)

as_array(A::AbstractArray) = A
as_array(x) = fill(x)

function npywrite(io::IO, A::AbstractArray; stringdtype::Symbol=:U)
    dt = wdtype(A; stringdtype)
    check_writable(A, dt)
    fortran_order = ndims(A) >= 2
    write_npy_header(io, dt, fortran_order, size(A))
    write_npy_data(io, A, dt)
    nothing
end

function check_writable(A::AbstractArray, dt::NPYDType)
    if Missing <: eltype(A) && !(dt.kind === :datetime || dt.kind === :timedelta)
        throw(NPZError("`missing` is only representable in numpy for datetime64/timedelta64 " *
                       "(NaT); got an array of $(eltype(A))"))
    end
end

function npy_header_string(dt::NPYDType, fortran_order::Bool, shape::Tuple{Vararg{Int}})
    string("{'descr': ", pystr(descr_of(dt)),
           ", 'fortran_order': ", pystr(fortran_order),
           ", 'shape': ", pystr(shape), ", }")
end

function write_npy_header(io::IO, dt::NPYDType, fortran_order::Bool,
                          shape::Tuple{Vararg{Int}})
    body = Vector{UInt8}(codeunits(npy_header_string(dt, fortran_order, shape)))
    ascii = all(<(0x80), body)
    # v1.0 stores the header length in a UInt16 and is limited to latin-1; the
    # bigger UInt32 field of v2.0 takes over above that, and non-ASCII field
    # names need the utf-8 header of v3.0.
    version, lensize = if !ascii
        (0x03, 4)
    elseif padded_length(length(body), 2) <= typemax(UInt16)
        (0x01, 2)
    else
        (0x02, 4)
    end
    padlen = ARRAY_ALIGN - ((6 + 2 + lensize + length(body) + 1) % ARRAY_ALIGN)
    hlen = length(body) + padlen + 1
    write(io, NPY_MAGIC)
    write(io, version, 0x00)
    lensize == 2 ? write(io, htol(UInt16(hlen))) : write(io, htol(UInt32(hlen)))
    write(io, body)
    write(io, fill(UInt8(' '), padlen))
    write(io, UInt8('\n'))
    nothing
end

function padded_length(bodylen::Int, lensize::Int)
    padlen = ARRAY_ALIGN - ((6 + 2 + lensize + bodylen + 1) % ARRAY_ALIGN)
    bodylen + padlen + 1
end

function write_npy_data(io::IO, A::AbstractArray, dt::NPYDType)
    if isbitskind(dt) || dt.kind === :bool
        # Julia stores `Bool` as one 0/1 byte per element, which is exactly
        # numpy's `|b1`, so both go out as a straight memory dump.
        B = A isa Array{dt.T} ? A : convert(Array{dt.T}, A)
        write(io, B)
    elseif dt.kind === :datetime || dt.kind === :timedelta
        raw = Vector{Int64}(undef, length(A))
        i = 0
        for v in A
            raw[i+=1] = npy_count(v)
        end
        write(io, raw)
    else
        write_generic_data(io, A, dt)
    end
    nothing
end

function write_generic_data(io::IO, A::AbstractArray, dt::NPYDType)
    dt.itemsize == 0 && return nothing
    n = length(A)
    nchunk = max(1, div(WRITE_CHUNK_BYTES, dt.itemsize))
    buf = Vector{UInt8}(undef, min(n, nchunk) * dt.itemsize)
    i = 0
    for v in A
        writevalue!(buf, i * dt.itemsize, dt, v)
        i += 1
        if i == nchunk
            write(io, buf)
            i = 0
        end
    end
    i > 0 && write(io, view(buf, 1:i*dt.itemsize))
    nothing
end
