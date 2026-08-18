# numpy dtype descriptors <-> Julia types, plus the element codecs used by the
# generic (non-memcpy) read/write paths.

const HOST_LITTLE = (ENDIAN_BOM == 0x04030201)
const NATIVE_ORDER = HOST_LITTLE ? '<' : '>'

struct NPYField{D}
    name::Symbol
    offset::Int
    dt::D
    shape::Tuple{Vararg{Int}}
end

"""
    NPYDType

A parsed numpy dtype: its Julia element type, byte width, whether its byte order
differs from the host's, and — for structured dtypes — its fields.
"""
struct NPYDType
    kind::Symbol      # :bool :int :uint :float :complex :bytes :unicode :void :datetime :timedelta :struct
    T::Type           # Julia element type
    itemsize::Int
    swap::Bool
    n::Int            # character/byte count for S, U and V dtypes
    mult::Int         # time unit multiple
    unit::Symbol      # time unit
    fields::Vector{NPYField{NPYDType}}
end

NPYDType(kind, T, itemsize; swap=false, n=0, mult=1, unit=:none,
         fields=NPYField{NPYDType}[]) =
    NPYDType(kind, T, itemsize, swap, n, mult, unit, fields)

const BITS_KINDS = (:int, :uint, :float, :complex)

isbitskind(dt::NPYDType) = dt.kind in BITS_KINDS

const INT_TYPES = Dict(1 => Int8, 2 => Int16, 4 => Int32, 8 => Int64)
const UINT_TYPES = Dict(1 => UInt8, 2 => UInt16, 4 => UInt32, 8 => UInt64)
const FLOAT_TYPES = Dict(2 => Float16, 4 => Float32, 8 => Float64)
const COMPLEX_TYPES = Dict(4 => Complex{Float16}, 8 => ComplexF32, 16 => ComplexF64)

longdouble_error(descr) = throw(NPZError(
    "dtype $(repr(descr)) is numpy's platform-specific `longdouble` (x86 80-bit extended " *
    "padded to 12 or 16 bytes); no Julia type represents it, so it is not supported"))

# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

parse_dtype(descr) = throw(NPZError("unsupported dtype descriptor $(repr(descr))"))

function parse_dtype(descr::AbstractString)
    isempty(descr) && throw(NPZError("empty dtype descriptor"))
    bo = descr[1]
    rest = bo in ('<', '>', '|', '=') ? descr[nextind(descr, 1):end] : descr
    bo in ('<', '>', '|', '=') || (bo = '|')
    swap = (bo == '>' && HOST_LITTLE) || (bo == '<' && !HOST_LITTLE)
    isempty(rest) && throw(NPZError("malformed dtype descriptor $(repr(descr))"))

    kindchar = rest[1]
    tail = rest[nextind(rest, 1):end]

    if kindchar == 'M' || kindchar == 'm'
        startswith(tail, "8") || throw(NPZError("malformed dtype descriptor $(repr(descr))"))
        mult, unit = parse_timeunit(tail[nextind(tail, 1):end], descr)
        if kindchar == 'M'
            return NPYDType(:datetime, datetime64_type(unit), 8; swap, mult, unit)
        else
            return NPYDType(:timedelta, timedelta64_type(unit), 8; swap, mult, unit)
        end
    elseif kindchar == 'b' || kindchar == '?'
        (kindchar == '?' || tail == "1") ||
            throw(NPZError("unsupported boolean dtype $(repr(descr))"))
        return NPYDType(:bool, Bool, 1)
    elseif kindchar == 'S' || kindchar == 'a'
        n = parse_size(tail, descr)
        return NPYDType(:bytes, String, n; n)
    elseif kindchar == 'U'
        n = parse_size(tail, descr)
        return NPYDType(:unicode, String, 4n; swap, n)
    elseif kindchar == 'V'
        n = parse_size(tail, descr)
        return NPYDType(:void, NTuple{n,UInt8}, n; n)
    elseif kindchar == 'O'
        throw(NPZError(
            "dtype $(repr(descr)) is a numpy object array; its contents are Python pickles " *
            "(numpy itself needs `allow_pickle=True` to load them), which NPZvibe does not read"))
    elseif kindchar == 'T'
        throw(NPZError(
            "dtype $(repr(descr)) is numpy's variable-width StringDType, which is not supported; " *
            "convert the array to a fixed-width `U`/`S` dtype before saving"))
    elseif kindchar == 'g' || kindchar == 'G'
        longdouble_error(descr)
    end

    size = parse_size(tail, descr)
    if kindchar == 'i'
        haskey(INT_TYPES, size) || throw(NPZError("unsupported integer dtype $(repr(descr))"))
        NPYDType(:int, INT_TYPES[size], size; swap)
    elseif kindchar == 'u'
        haskey(UINT_TYPES, size) || throw(NPZError("unsupported integer dtype $(repr(descr))"))
        NPYDType(:uint, UINT_TYPES[size], size; swap)
    elseif kindchar == 'f'
        (size == 12 || size == 16) && longdouble_error(descr)
        haskey(FLOAT_TYPES, size) || throw(NPZError("unsupported float dtype $(repr(descr))"))
        NPYDType(:float, FLOAT_TYPES[size], size; swap)
    elseif kindchar == 'c'
        (size == 24 || size == 32) && longdouble_error(descr)
        haskey(COMPLEX_TYPES, size) || throw(NPZError("unsupported complex dtype $(repr(descr))"))
        NPYDType(:complex, COMPLEX_TYPES[size], size; swap)
    else
        throw(NPZError("unknown dtype descriptor $(repr(descr))"))
    end
end

function parse_size(s::AbstractString, descr)
    n = tryparse(Int, s)
    (n === nothing || n < 0) && throw(NPZError("malformed dtype descriptor $(repr(descr))"))
    n
end

"Structured dtype in numpy's list form: `[('a', '<i4'), ('b', '<f8', (3,))]`."
function parse_dtype(descr::AbstractVector)
    fields = NPYField{NPYDType}[]
    offset = 0
    for item in descr
        item isa Tuple && length(item) >= 2 ||
            throw(NPZError("malformed structured dtype entry $(repr(item))"))
        name = fieldname_of(item[1])
        fdt = parse_dtype(item[2])
        shape = length(item) >= 3 ? subshape(item[3]) : ()
        if !isempty(name)
            push!(fields, NPYField{NPYDType}(Symbol(name), offset, fdt, shape))
        end
        offset += fdt.itemsize * prod(shape; init=1)
    end
    struct_dtype(fields, offset)
end

"Structured dtype in numpy's dict form: `{'names': …, 'formats': …, 'offsets': …}`."
function parse_dtype(descr::AbstractDict)
    haskey(descr, "names") && haskey(descr, "formats") ||
        throw(NPZError("structured dtype dict needs 'names' and 'formats'"))
    names = descr["names"]
    formats = descr["formats"]
    length(names) == length(formats) ||
        throw(NPZError("structured dtype has $(length(names)) names but $(length(formats)) formats"))
    offsets = get(descr, "offsets", nothing)
    fields = NPYField{NPYDType}[]
    offset = 0
    for (k, (name, fmt)) in enumerate(zip(names, formats))
        base, shape = fmt isa Tuple ? (fmt[1], subshape(fmt[2])) : (fmt, ())
        fdt = parse_dtype(base)
        off = offsets === nothing ? offset : Int(offsets[k])
        nm = fieldname_of(name)
        isempty(nm) || push!(fields, NPYField{NPYDType}(Symbol(nm), off, fdt, shape))
        offset = max(offset, off + fdt.itemsize * prod(shape; init=1))
    end
    itemsize = haskey(descr, "itemsize") ? Int(descr["itemsize"]) : offset
    struct_dtype(fields, itemsize)
end

fieldname_of(name::AbstractString) = String(name)
# numpy writes `(('title', 'name'), '<i4')` for fields that carry a title.
fieldname_of(name::Tuple) = String(name[end])
fieldname_of(name) = throw(NPZError("unsupported structured field name $(repr(name))"))

subshape(s::Tuple) = Tuple(Int(x) for x in s)
subshape(s::AbstractVector) = Tuple(Int(x) for x in s)
subshape(s::Integer) = (Int(s),)
subshape(s) = throw(NPZError("unsupported sub-array shape $(repr(s))"))

function struct_dtype(fields::Vector{NPYField{NPYDType}}, itemsize::Int)
    names = Tuple(f.name for f in fields)
    length(unique(names)) == length(names) ||
        throw(NPZError("structured dtype has duplicate field names"))
    types = Tuple(sub_type(struct_field_type(f.dt), f.shape) for f in fields)
    T = NamedTuple{names,Tuple{types...}}
    NPYDType(:struct, T, itemsize; fields)
end

sub_type(T::Type, shape::Tuple{Vararg{Int}}) =
    isempty(shape) ? T : NTuple{shape[1],sub_type(T, Base.tail(shape))}

# NaT can appear anywhere inside a structured array, and unlike the top-level
# case there is no cheap way to specialise per array, so struct fields always
# admit `missing`.
struct_field_type(dt::NPYDType) =
    dt.kind === :datetime || dt.kind === :timedelta ? Union{Missing,dt.T} : dt.T

"Julia element type of an array with this dtype (before NaT widening)."
array_eltype(dt::NPYDType) = dt.T

# ---------------------------------------------------------------------------
# element codecs
# ---------------------------------------------------------------------------

_bswap(x::Union{Int8,UInt8,Bool}) = x
_bswap(x::Union{Int16,UInt16,Int32,UInt32,Int64,UInt64}) = bswap(x)
_bswap(x::Float16) = reinterpret(Float16, bswap(reinterpret(UInt16, x)))
_bswap(x::Float32) = reinterpret(Float32, bswap(reinterpret(UInt32, x)))
_bswap(x::Float64) = reinterpret(Float64, bswap(reinterpret(UInt64, x)))
_bswap(z::Complex) = Complex(_bswap(real(z)), _bswap(imag(z)))

"Load a `T` from `buf` at byte offset `off`, tolerating unaligned offsets."
function loadbits(::Type{T}, buf::AbstractVector{UInt8}, off::Int) where {T}
    r = Ref{T}()
    GC.@preserve r begin
        p = Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, r))
        for k in 1:sizeof(T)
            unsafe_store!(p, buf[off+k], k)
        end
    end
    r[]
end

function storebits!(buf::AbstractVector{UInt8}, off::Int, x::T) where {T}
    r = Ref{T}(x)
    GC.@preserve r begin
        p = Ptr{UInt8}(Base.unsafe_convert(Ptr{T}, r))
        for k in 1:sizeof(T)
            buf[off+k] = unsafe_load(p, k)
        end
    end
    buf
end

function decode_bytes(buf::AbstractVector{UInt8}, off::Int, n::Int)
    last = n
    while last > 0 && buf[off+last] == 0x00
        last -= 1
    end
    String(@view buf[off+1:off+last])
end

function decode_unicode(buf::AbstractVector{UInt8}, off::Int, n::Int, swap::Bool)
    cps = Vector{UInt32}(undef, n)
    for k in 1:n
        u = loadbits(UInt32, buf, off + 4(k - 1))
        cps[k] = swap ? bswap(u) : u
    end
    last = n
    while last > 0 && cps[last] == 0
        last -= 1
    end
    io = IOBuffer()
    for k in 1:last
        u = cps[k]
        (u <= 0x10ffff && !(0xd800 <= u <= 0xdfff)) ||
            throw(NPZError("invalid unicode code point 0x$(string(u; base=16)) in a U dtype"))
        print(io, Char(u))
    end
    String(take!(io))
end

function readvalue(dt::NPYDType, buf::AbstractVector{UInt8}, off::Int)
    k = dt.kind
    if k === :bool
        return buf[off+1] != 0x00
    elseif isbitskind(dt)
        x = loadbits(dt.T, buf, off)
        return dt.swap ? _bswap(x) : x
    elseif k === :bytes
        return decode_bytes(buf, off, dt.n)
    elseif k === :unicode
        return decode_unicode(buf, off, dt.n, dt.swap)
    elseif k === :void
        return ntuple(i -> buf[off+i], dt.n)
    elseif k === :datetime
        v = loadbits(Int64, buf, off)
        return datetime64_value(dt.mult, dt.unit, dt.swap ? bswap(v) : v)
    elseif k === :timedelta
        v = loadbits(Int64, buf, off)
        return timedelta64_value(dt.mult, dt.unit, dt.swap ? bswap(v) : v)
    elseif k === :struct
        return dt.T(Tuple(readfield(f, buf, off + f.offset) for f in dt.fields))
    end
    throw(NPZError("cannot read dtype of kind $(dt.kind)"))
end

readfield(f::NPYField, buf, off) =
    isempty(f.shape) ? readvalue(f.dt, buf, off) : readsub(f.dt, f.shape, buf, off)

function readsub(dt::NPYDType, shape::Tuple{Vararg{Int}}, buf, off::Int)
    isempty(shape) && return readvalue(dt, buf, off)
    rest = Base.tail(shape)
    stride = prod(rest; init=1) * dt.itemsize
    ntuple(i -> readsub(dt, rest, buf, off + (i - 1) * stride), shape[1])
end

function writevalue!(buf::AbstractVector{UInt8}, off::Int, dt::NPYDType, v)
    k = dt.kind
    if k === :bool
        buf[off+1] = v ? 0x01 : 0x00
    elseif isbitskind(dt)
        storebits!(buf, off, convert(dt.T, v))
    elseif k === :bytes
        cu = codeunits(v)
        length(cu) <= dt.n ||
            throw(NPZError("string of $(length(cu)) bytes does not fit dtype |S$(dt.n)"))
        for k2 in 1:dt.n
            buf[off+k2] = k2 <= length(cu) ? cu[k2] : 0x00
        end
    elseif k === :unicode
        i = 0
        for c in v
            i += 1
            i <= dt.n ||
                throw(NPZError("string of more than $(dt.n) characters does not fit dtype U$(dt.n)"))
            storebits!(buf, off + 4(i - 1), UInt32(c))
        end
        for k2 in i+1:dt.n
            storebits!(buf, off + 4(k2 - 1), UInt32(0))
        end
    elseif k === :void
        length(v) == dt.n || throw(NPZError("expected $(dt.n) bytes for dtype |V$(dt.n)"))
        for k2 in 1:dt.n
            buf[off+k2] = v[k2]
        end
    elseif k === :datetime || k === :timedelta
        storebits!(buf, off, npy_count(v))
    elseif k === :struct
        for (i, f) in enumerate(dt.fields)
            writefield!(buf, off + f.offset, f, getfield(v, i))
        end
    else
        throw(NPZError("cannot write dtype of kind $(dt.kind)"))
    end
    buf
end

writefield!(buf, off, f::NPYField, v) =
    isempty(f.shape) ? writevalue!(buf, off, f.dt, v) : writesub!(buf, off, f.dt, f.shape, v)

function writesub!(buf, off::Int, dt::NPYDType, shape::Tuple{Vararg{Int}}, v)
    isempty(shape) && return writevalue!(buf, off, dt, v)
    rest = Base.tail(shape)
    stride = prod(rest; init=1) * dt.itemsize
    length(v) == shape[1] ||
        throw(NPZError("expected $(shape[1]) elements in a sub-array field, got $(length(v))"))
    for i in 1:shape[1]
        writesub!(buf, off + (i - 1) * stride, dt, rest, v[i])
    end
    buf
end

# ---------------------------------------------------------------------------
# Julia array -> dtype
# ---------------------------------------------------------------------------

const StringPath = Vector{Int}

"Does this element type contain strings whose widths have to be scanned for?"
function has_strings(::Type{T}) where {T}
    S = Base.nonmissingtype(T)
    S <: AbstractString && return true
    # `Any` matches `Union{Missing,S}` with `S === Any`, so recurse on the
    # stripped type only, and only when its fields are known.
    (S <: Tuple || S <: NamedTuple) && isconcretetype(S) &&
        return any(has_strings, fieldtypes(S))
    false
end

strwidth(s::AbstractString, stringdtype::Symbol) =
    stringdtype === :S ? ncodeunits(s) : length(s)

function scan_strings!(d::Dict{StringPath,Int}, path::StringPath, v, sk::Symbol)
    if v isa AbstractString
        d[copy(path)] = max(get(d, path, 0), strwidth(v, sk))
    elseif v isa NamedTuple
        for i in 1:nfields(v)
            push!(path, i)
            scan_strings!(d, path, getfield(v, i), sk)
            pop!(path)
        end
    elseif v isa Tuple
        for x in v
            scan_strings!(d, path, x, sk)
        end
    end
    d
end

"Split a struct field type into its base type and numpy sub-array shape."
function unwrap_subarray(::Type{T}) where {T}
    if T <: Tuple && !(T <: NamedTuple)
        ps = fieldtypes(T)
        isempty(ps) && throw(NPZError("cannot write a zero-length tuple field"))
        S = ps[1]
        all(p -> p === S, ps) ||
            throw(NPZError("heterogeneous tuple field of type $T is not supported"))
        base, shape = unwrap_subarray(S)
        return (base, (length(ps), shape...))
    end
    (T, ())
end

"""
    wdtype(A; stringdtype=:U) -> NPYDType

Build the dtype NPZvibe writes for `A`. String widths (which numpy needs up
front) come from a scan over the data, including strings nested in structs.
"""
function wdtype(A::AbstractArray; stringdtype::Symbol=:U)
    stringdtype in (:U, :S) ||
        throw(NPZError("stringdtype must be :U or :S, got $(repr(stringdtype))"))
    sizes = Dict{StringPath,Int}()
    if has_strings(eltype(A))
        path = Int[]
        for v in A
            scan_strings!(sizes, path, v, stringdtype)
        end
    end
    wdtype_for(eltype(A), Int[], sizes, stringdtype, true)
end

function wdtype_for(::Type{T0}, path::StringPath, sizes::Dict{StringPath,Int},
                    sk::Symbol, toplevel::Bool) where {T0}
    T = Base.nonmissingtype(T0)
    if T === Bool
        NPYDType(:bool, Bool, 1)
    elseif T === Int8 || T === UInt8 || T === Int16 || T === UInt16 ||
           T === Int32 || T === UInt32 || T === Int64 || T === UInt64
        NPYDType(T <: Signed ? :int : :uint, T, sizeof(T))
    elseif T === Float16 || T === Float32 || T === Float64
        NPYDType(:float, T, sizeof(T))
    elseif T === ComplexF32 || T === ComplexF64
        NPYDType(:complex, T, sizeof(T))
    elseif T === Complex{Float16}
        # `<c4` parses on the way in, but numpy has no complex32 dtype, so a
        # file containing one would be unreadable there.
        throw(NPZError("numpy has no complex32 dtype; convert $(T) data to " *
                       "ComplexF32 before writing"))
    elseif T <: AbstractString
        n = max(get(sizes, path, 0), 1)
        sk === :S ? NPYDType(:bytes, String, n; n) : NPYDType(:unicode, String, 4n; n)
    elseif T === Date || T === DateTime || T === NanoDate
        NPYDType(:datetime, T, 8; unit=npy_unit_of(T))
    elseif T <: Period && haskey(PERIOD_OF_UNIT, npy_unit_of_safe(T))
        NPYDType(:timedelta, T, 8; unit=npy_unit_of(T))
    elseif toplevel && T <: Tuple && !(T <: NamedTuple) && all(==(UInt8), fieldtypes(T))
        n = fieldcount(T)
        NPYDType(:void, NTuple{n,UInt8}, n; n)
    elseif T <: NamedTuple
        wstruct_dtype(T, path, sizes, sk)
    else
        throw(NPZError("cannot write Julia type $T to a numpy array"))
    end
end

npy_unit_of_safe(::Type{T}) where {T<:Period} =
    hasmethod(npy_unit_of, Tuple{Type{T}}) ? npy_unit_of(T) : :none

function wstruct_dtype(::Type{T}, path::StringPath, sizes::Dict{StringPath,Int},
                       sk::Symbol) where {T<:NamedTuple}
    fields = NPYField{NPYDType}[]
    offset = 0
    for (i, (nm, ft)) in enumerate(zip(fieldnames(T), fieldtypes(T)))
        base, shape = unwrap_subarray(ft)
        push!(path, i)
        fdt = wdtype_for(base, path, sizes, sk, false)
        pop!(path)
        push!(fields, NPYField{NPYDType}(nm, offset, fdt, shape))
        offset += fdt.itemsize * prod(shape; init=1)
    end
    names = Tuple(f.name for f in fields)
    types = Tuple(sub_type(struct_field_type(f.dt), f.shape) for f in fields)
    NPYDType(:struct, NamedTuple{names,Tuple{types...}}, offset; fields)
end

# ---------------------------------------------------------------------------
# dtype -> descr
# ---------------------------------------------------------------------------

ordchar(itemsize::Int) = itemsize == 1 ? '|' : NATIVE_ORDER

function descr_of(dt::NPYDType)
    k = dt.kind
    if k === :bool
        "|b1"
    elseif k === :int
        string(ordchar(dt.itemsize), 'i', dt.itemsize)
    elseif k === :uint
        string(ordchar(dt.itemsize), 'u', dt.itemsize)
    elseif k === :float
        string(NATIVE_ORDER, 'f', dt.itemsize)
    elseif k === :complex
        string(NATIVE_ORDER, 'c', dt.itemsize)
    elseif k === :bytes
        string('|', 'S', dt.n)
    elseif k === :unicode
        string(NATIVE_ORDER, 'U', dt.n)
    elseif k === :void
        string('|', 'V', dt.n)
    elseif k === :datetime
        string(NATIVE_ORDER, "M8[", dt.unit, "]")
    elseif k === :timedelta
        string(NATIVE_ORDER, "m8[", dt.unit, "]")
    elseif k === :struct
        Any[field_descr(f) for f in dt.fields]
    else
        throw(NPZError("cannot describe dtype of kind $(dt.kind)"))
    end
end

field_descr(f::NPYField) = isempty(f.shape) ?
    (String(f.name), descr_of(f.dt)) :
    (String(f.name), descr_of(f.dt), f.shape)
