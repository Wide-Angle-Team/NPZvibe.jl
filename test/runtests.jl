using NPZvibe
using Test
using Dates
using NanoDates
using Random
using Mmap
using ZipArchives

include("pyutils.jl")

@testset "NPZvibe.jl" begin
    include("pyliteral.jl")
    include("dtypes.jl")
    include("times.jl")
    include("roundtrip.jl")
    include("errors.jl")
    include("fixtures.jl")
    include("julia_reads_python.jl")
    include("python_reads_julia.jl")
    include("large.jl")
end
