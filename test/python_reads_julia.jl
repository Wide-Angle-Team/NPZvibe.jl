# NPZvibe writes, numpy reads.  `check_written.py` holds the expected values as
# numpy literals and exits nonzero (with a report) on any mismatch.

@testset "python reads julia" begin
    if !HAVE_NUMPY
        @test_skip "no numpy available"
    else
        mktempdir() do dir
            w(name, A; kw...) = npywrite(joinpath(dir, name * ".npy"), A; kw...)

            w("bool", [true, false, true])
            for T in (Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64)
                w(lowercase(string(T <: Signed ? "i" : "u", sizeof(T))),
                  T[typemin(T), 0, 7, typemax(T)])
            end
            for (name, T) in (("f2", Float16), ("f4", Float32), ("f8", Float64))
                w(name, T[1.5, -0.0, NaN, Inf, -Inf])
            end
            for (name, T) in (("c8", ComplexF32), ("c16", ComplexF64))
                w(name, T[1.5+2.5im, T(-0.0, Inf)])
            end

            mat = reshape(collect(0.0:5.0), 2, 3)
            w("mat", mat)
            w("cube", reshape(Int32.(0:23), 2, 3, 4))
            w("zerod", fill(3.5))
            w("empty", zeros(Float32, 0, 3))
            w("transposed", permutedims(mat))
            w("subview", view(reshape(Int16.(0:11), 3, 4), 2:3, 3:4))

            strings = ["", "héllo", "\U0001d418nicode"]
            w("str_U", strings)
            w("str_S", strings; stringdtype=:S)
            w("void", [(0x01, 0x02, 0x03), (0xff, 0x00, 0x80)])

            w("date", [Date(2020, 2, 29), Date(1900, 1, 1)])
            w("datetime", [DateTime(2020, 1, 2, 3, 4, 5, 6)])
            w("nanodate", [NanoDate(DateTime(2020, 1, 2, 3, 4, 5), Nanosecond(8)),
                           NanoDate(DateTime(1969, 12, 31, 23, 59, 59, 999), Nanosecond(999999))])
            w("nat", [DateTime(2020, 1, 2), missing])
            w("period_s", [Second(0), Second(90), Second(-1)])
            w("period_ns", [Nanosecond(0), Nanosecond(1), Nanosecond(-1)])
            w("period_year", [Year(1), Year(-2)])
            w("period_nat", [Microsecond(12), missing])

            w("struct", [(a=Int32(7), b=(1.0, 2.0, 3.0), c="xy"),
                         (a=Int32(8), b=(4.0, 5.0, 6.0), c="z")])
            w("struct_nested", [(x=(u=Int32(1), v=0.5f0), y=Int64(-10)),
                                (x=(u=Int32(2), v=1.5f0), y=Int64(20))])

            archive = Dict("a" => mat, "s" => ["one", "two"],
                           "t" => [Date(2020, 2, 29), Date(1900, 1, 1)])
            npzwrite(joinpath(dir, "all.npz"), archive)
            npzwrite(joinpath(dir, "all_compressed.npz"), archive; compress=true)

            @test occursin("ok", pyrun(joinpath(PYDIR, "check_written.py"), dir))
        end
    end
end

@testset "round trip through numpy" begin
    if !HAVE_NUMPY
        @test_skip "no numpy available"
    else
        mktempdir() do dir
            cases = Dict{String,Any}(
                "bool" => rand(Bool, 7),
                "i16" => Int16[-5, 0, 5],
                "u64" => UInt64[0, typemax(UInt64)],
                "f32" => Float32[1.0f0, -2.5f0, Inf32],
                "f64_matrix" => reshape(collect(1.0:12.0), 3, 4),
                "f64_cube" => reshape(collect(1.0:24.0), 2, 3, 4),
                "c64" => ComplexF64[1+2im, -3-4im],
                "f16" => Float16[1.5, 2.5],
                "strings" => ["a", "bb", "héllo", "\U0001d418"],
                "string_grid" => ["a" "bb"; "ccc" ""],
                "void" => [(0x01, 0x02), (0x03, 0x04)],
                "dates" => [Date(1900, 1, 1), Date(2222, 12, 31)],
                "datetimes" => [DateTime(2020, 1, 2, 3, 4, 5, 6), missing],
                "nanodates" => [NanoDate(DateTime(2020, 1, 1), Nanosecond(123456789 % 1000000))],
                "periods" => [Millisecond(1), Millisecond(-2)],
                "zerod" => fill(Int32(7)),
                "empty" => Float64[],
                "struct" => [(a=Int32(1), b="xy", c=(1.0, 2.0)),
                             (a=Int32(2), b="z", c=(3.0, 4.0))],
                "struct_time" => [(t=DateTime(2020, 1, 1), d=Day(3))],
            )
            inpath = joinpath(dir, "rt_in.npz")
            outpath = joinpath(dir, "rt_out.npz")
            npzwrite(inpath, cases)
            pyexec("""
import sys, numpy as np
d = np.load(sys.argv[1])
np.savez_compressed(sys.argv[2], **{k: d[k] for k in d.files})
""", inpath, outpath)
            back = npzread(outpath)
            @test sort(collect(keys(back))) == sort(collect(keys(cases)))
            for (k, v) in cases
                @test isequal(collect(back[k]), collect(v)) || (@show k; false)
            end
            # element types survive the trip, except that NaT-capable struct
            # fields always widen to Union{Missing,...} on the way back
            for k in ("bool", "i16", "u64", "f32", "f64_matrix", "c64", "f16",
                      "strings", "void", "dates", "periods", "zerod", "empty")
                @test eltype(back[k]) === eltype(cases[k])
            end
            @test eltype(back["datetimes"]) === Union{Missing,DateTime}
            @test size(back["f64_cube"]) == (2, 3, 4)
        end
    end
end
