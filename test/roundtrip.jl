# Pure-Julia write/read round trips: no numpy involved, so these run everywhere.

@testset "round trip" begin
    nd(ms, ns) = NanoDate(DateTime(2020, 1, 2, 3, 4, 5, ms), Nanosecond(ns))

    cases = Pair{String,Any}[
        "bool" => [true, false, true],
        "int8" => Int8[typemin(Int8), 0, typemax(Int8)],
        "int16" => Int16[typemin(Int16), 0, typemax(Int16)],
        "int32" => Int32[typemin(Int32), 0, typemax(Int32)],
        "int64" => Int64[typemin(Int64), 0, typemax(Int64)],
        "uint8" => UInt8[0, 1, typemax(UInt8)],
        "uint16" => UInt16[0, 1, typemax(UInt16)],
        "uint32" => UInt32[0, 1, typemax(UInt32)],
        "uint64" => UInt64[0, 1, typemax(UInt64)],
        "float16" => Float16[1.5, -0.0, Inf, -Inf],
        "float32" => Float32[1.5, -0.0, Inf, -Inf, floatmin(Float32), floatmax(Float32)],
        "float64" => Float64[1.5, -0.0, Inf, -Inf, floatmin(Float64), floatmax(Float64)],
        "complexf32" => ComplexF32[1.5 + 2.5im, -3 - 4im],
        "complexf64" => ComplexF64[1.5 + 2.5im, -3 - 4im],
        "strings" => ["", "a", "héllo", "\U0001d418nicode", "with\0nul"],
        "strings empty" => String[],
        "void" => [(0x01, 0x02, 0x03), (0xff, 0x00, 0x80)],
        "date" => [Date(1900, 1, 1), Date(1970, 1, 1), Date(2222, 12, 31)],
        "datetime" => [DateTime(1900, 1, 1), DateTime(2020, 1, 2, 3, 4, 5, 6)],
        "nanodate" => [nd(0, 0), nd(999, 999999), nd(6, 7)],
        "day" => [Day(-3), Day(0), Day(5)],
        "second" => [Second(-1), Second(90)],
        "nanosecond" => [Nanosecond(-1), Nanosecond(0), Nanosecond(1)],
        "year" => [Year(1), Year(-2)],
        "datetime with NaT" => [DateTime(2020, 1, 2), missing, DateTime(1970, 1, 1)],
        "period with NaT" => [missing, Microsecond(12)],
        "0-d" => fill(3.5),
        "0-d string" => fill("hé"),
        "empty" => Float64[],
        "empty matrix" => zeros(Float32, 0, 3),
        "matrix" => reshape(collect(1.0:6.0), 2, 3),
        "cube" => reshape(Int32.(1:24), 2, 3, 4),
        "5-d" => reshape(collect(1.0:120.0), 2, 3, 4, 5, 1),
        "string matrix" => ["a" "bb"; "ccc" ""],
        "struct" => [(a=Int32(1), b=(1.0, 2.0, 3.0), c="xy"),
                     (a=Int32(-2), b=(4.0, 5.0, 6.0), c="")],
        "struct nested" => [(x=(u=Int32(1), v=0.5f0), y=Int64(-10))],
        "struct 2-d field" => [(m=((Int16(1), Int16(2), Int16(3)),
                                   (Int16(4), Int16(5), Int16(6))),)],
        "struct of bytes" => [(v=(0x01, 0x02),), (v=(0x03, 0x04),)],
    ]

    # element types that survive a round trip exactly
    exact_eltype = Set(["bool", "int8", "int16", "int32", "int64", "uint8", "uint16",
                        "uint32", "uint64", "float16", "float32", "float64",
                        "complexf32", "complexf64", "strings",
                        "strings empty", "void", "date", "datetime", "nanodate", "day",
                        "second", "nanosecond", "year", "0-d", "0-d string", "empty",
                        "empty matrix", "matrix", "cube", "5-d", "string matrix",
                        "struct", "struct nested", "struct 2-d field", "struct of bytes",
                        "datetime with NaT", "period with NaT"])

    mktempdir() do dir
        @testset ".npy: $name" for (name, A) in cases
            path = joinpath(dir, "case.npy")
            npywrite(path, A)
            B = npyread(path)
            @test size(B) == size(A)
            @test ndims(B) == ndims(A)
            @test isequal(B, A)
            name in exact_eltype && @test eltype(B) === eltype(A)
        end

        @testset "non-contiguous input" begin
            A = reshape(collect(1.0:12.0), 3, 4)
            for B in (view(A, 2:3, 2:4), permutedims(A), A', reshape(A, 4, 3),
                      view(A, :, 1), reverse(A; dims=1))
                path = joinpath(dir, "nc.npy")
                npywrite(path, B)
                @test npyread(path) == B
                @test size(npyread(path)) == size(B)
            end
        end

        @testset "other array types" begin
            path = joinpath(dir, "other.npy")
            npywrite(path, BitArray([true, false, true]))
            @test npyread(path) == [true, false, true]
            @test eltype(npyread(path)) === Bool
            npywrite(path, 1:5)
            @test npyread(path) == 1:5
            npywrite(path, reshape(1:6, 2, 3))
            @test npyread(path) == reshape(1:6, 2, 3)
        end

        @testset "scalars become 0-d arrays" begin
            path = joinpath(dir, "scalar.npy")
            npywrite(path, 42)
            @test npyread(path)[] === 42
            npywrite(path, "hé")
            @test npyread(path)[] == "hé"
            npywrite(path, DateTime(2020))
            @test npyread(path)[] == DateTime(2020)
        end

        @testset "byte-string dtype" begin
            path = joinpath(dir, "bytes.npy")
            npywrite(path, ["abc", "d"]; stringdtype=:S)
            @test npyread(path) == ["abc", "d"]
            # `S` keeps utf-8 code units, so a non-ascii string round trips too
            npywrite(path, ["héllo"]; stringdtype=:S)
            @test npyread(path) == ["héllo"]
        end

        @testset ".npz archives" begin
            data = Dict("a" => reshape(collect(1.0:6.0), 2, 3),
                        "s" => ["x", "yy"],
                        "t" => [Date(2020, 2, 29), missing],
                        "n" => fill(Int32(7)))
            for compress in (false, true)
                path = joinpath(dir, "archive.npz")
                npzwrite(path, data; compress)
                back = npzread(path)
                @test sort(collect(keys(back))) == ["a", "n", "s", "t"]
                for (k, v) in data
                    @test isequal(back[k], v)
                end
                @test isequal(npzread(path, ["a", "s"]), Dict("a" => data["a"], "s" => data["s"]))
                @test isequal(npzread(path; mmap=true)["a"], data["a"])
            end

            kwpath = joinpath(dir, "kwargs.npz")
            npzwrite(kwpath; alpha=[1, 2, 3], beta=reshape(1:4, 2, 2))
            back = npzread(kwpath)
            @test back["alpha"] == [1, 2, 3]
            @test back["beta"] == reshape(1:4, 2, 2)

            # a single array written through npzwrite is a bare .npy, and
            # npzread dispatches on the file's magic rather than its name
            arrpath = joinpath(dir, "single.npz")
            npzwrite(arrpath, [1.0, 2.0])
            @test npzread(arrpath) == [1.0, 2.0]
        end

        @testset "memory-mapped reads" begin
            A = reshape(collect(1.0:1000.0), 100, 10)
            path = joinpath(dir, "mm.npy")
            npywrite(path, A)
            @test npyread(path; mmap=true) == A
            npzwrite(joinpath(dir, "mm.npz"), Dict("a" => A))
            @test npzread(joinpath(dir, "mm.npz"); mmap=true)["a"] == A
            # compressed entries cannot be mapped, but must still read correctly
            npzwrite(joinpath(dir, "mmz.npz"), Dict("a" => A); compress=true)
            @test npzread(joinpath(dir, "mmz.npz"); mmap=true)["a"] == A
        end
    end
end
