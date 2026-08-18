# The `.npz` container: a ZIP archive whose members are `<key>.npy` files.

const ZIP_STORE = UInt16(0)

"""
    npzread(path; mmap=false) -> Dict{String,Any} or Array
    npzread(path, names; mmap=false) -> Dict{String,Any}

Read a `.npz` archive into a dictionary, or a bare `.npy` file into an array —
the file's magic bytes decide, so the extension does not matter.  Pass `names` to
read only some of the archive's members.  `mmap=true` memory-maps array data
instead of copying it wherever the dtype, layout and (for `.npz`) an
uncompressed entry allow it; mapped arrays are read-only views of the file.
"""
function npzread(path::AbstractString; mmap::Bool=false)
    kind = file_kind(path)
    kind === :npy ? npyread(path; mmap) : npzread_entries(path, nothing; mmap)
end

function npzread(path::AbstractString, names; mmap::Bool=false)
    kind = file_kind(path)
    kind === :npy && throw(NPZError("$path is a .npy file, which has no named entries"))
    npzread_entries(path, names; mmap)
end

function file_kind(path::AbstractString)
    magic = open(path, "r") do io
        read(io, 4)
    end
    length(magic) >= 4 || throw(NPZError("$path is too short to be a .npy or .npz file"))
    if magic[1:4] == NPY_MAGIC[1:4]
        :npy
    elseif magic[1:2] == UInt8['P', 'K']
        :npz
    else
        throw(NPZError("$path is neither a .npy file nor a .npz (zip) archive"))
    end
end

function npzread_entries(path::AbstractString, names; mmap::Bool=false)
    io = open(path, "r")
    try
        r = ZipReader(Mmap.mmap(io))
        wanted = names === nothing ? nothing : Set(String(k) for k in names)
        out = Dict{String,Any}()
        for i in 1:zip_nentries(r)
            name = zip_name(r, i)
            key = endswith(name, ".npy") ? name[1:prevind(name, end, 4)] : name
            (wanted === nothing || key in wanted) || continue
            out[key] = read_zip_entry(r, i, io, mmap)
        end
        if wanted !== nothing
            absent = setdiff(wanted, keys(out))
            isempty(absent) && return out
            throw(NPZError("$path has no entries named $(join(sort!(collect(absent)), ", "))"))
        end
        out
    finally
        close(io)
    end
end

function read_zip_entry(r::ZipReader, i::Int, io::IOStream, mmap::Bool)
    if mmap && zip_compression_method(r, i) == ZIP_STORE
        # Stored entries live verbatim in the file, so they can be read (and
        # memory-mapped) straight through the file handle.
        seek(io, zip_entry_data_offset(r, i))
        return npyread(io; mmapfile=io)
    end
    zip_openentry(r, i) do entry
        npyread(entry)
    end
end

"""
    npzwrite(path, data::AbstractDict; compress=false, stringdtype=:U)
    npzwrite(path; compress=false, stringdtype=:U, key=value, …)
    npzwrite(path, A::AbstractArray; stringdtype=:U)

Write a `.npz` archive from a dictionary (or keyword arguments), one `.npy`
member per key.  `compress=true` deflates the members, matching numpy's
`savez_compressed`; the default matches `savez`.  Given a single array instead of
a dictionary, a bare `.npy` file is written.
"""
function npzwrite(path::AbstractString, data::AbstractDict;
                  compress::Bool=false, stringdtype::Symbol=:U)
    ZipWriter(path) do w
        for (k, v) in data
            name = String(k)
            zip_newfile(w, name * ".npy"; compress)
            npywrite(w, v; stringdtype)
        end
    end
    nothing
end

function npzwrite(path::AbstractString; compress::Bool=false, stringdtype::Symbol=:U, kwargs...)
    npzwrite(path, Dict{String,Any}(String(k) => v for (k, v) in kwargs); compress, stringdtype)
end

npzwrite(path::AbstractString, A; stringdtype::Symbol=:U) =
    npywrite(path, A; stringdtype)
