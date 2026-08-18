"""
    NPZvibe

Read and write numpy's `.npy` and `.npz` files, covering every dtype numpy can
store in them, arrays and strings of arbitrary size (ZIP64), and both C and
Fortran memory order.

    npzwrite("data.npz", Dict("a" => rand(3, 4), "s" => ["hé", "world"]))
    d = npzread("data.npz")

Structured dtypes become `NamedTuple` arrays, `datetime64`/`timedelta64` become
`Dates`/`NanoDates` values (with `NaT` as `missing`), and fixed-width `S`/`U`
strings become `String`s.  Object (pickled) arrays and numpy's platform-specific
`longdouble` are reported as errors rather than read incorrectly.
"""
module NPZvibe

using Dates
using Mmap
using NanoDates
using ZipArchives

export npzread, npzwrite, npyread, npywrite

"""
    NPZError(msg)

Raised for malformed `.npy`/`.npz` files and for dtypes that cannot be mapped to
Julia (or Julia types that cannot be mapped to numpy).
"""
struct NPZError <: Exception
    msg::String
end

Base.showerror(io::IO, e::NPZError) = print(io, "NPZError: ", e.msg)

include("pyliteral.jl")
include("times.jl")
include("dtypes.jl")
include("npy.jl")
include("npz.jl")

end # module
