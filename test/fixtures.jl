# Hand-built `.npy` bytes for cases the local numpy cannot produce (this machine
# has no float128 at all) or that a well-behaved writer never emits.

using NPZvibe: NPZError, NPY_MAGIC, parse_dtype, descr_of, write_npy_header, read_npy_header

"Write a .npy file with a literal header, so the reader can be fed anything."
function fixture(path, descr_literal::String, shape::Tuple, payload::Vector{UInt8};
                 fortran=false, version=(0x01, 0x00), magic=NPY_MAGIC)
    shapestr = length(shape) == 1 ? "($(shape[1]),)" : "(" * join(shape, ", ") * ")"
    header = "{'descr': $descr_literal, 'fortran_order': $(fortran ? "True" : "False"), " *
             "'shape': $shapestr, }"
    body = Vector{UInt8}(codeunits(header))
    lensize = version[1] == 0x01 ? 2 : 4
    padlen = 64 - ((6 + 2 + lensize + length(body) + 1) % 64)
    open(path, "w") do io
        write(io, magic)
        write(io, version[1], version[2])
        hlen = length(body) + padlen + 1
        lensize == 2 ? write(io, htol(UInt16(hlen))) : write(io, htol(UInt32(hlen)))
        write(io, body)
        write(io, fill(UInt8(' '), padlen))
        write(io, UInt8('\n'))
        write(io, payload)
    end
    path
end

@testset "fixtures" begin
    mktempdir() do dir
        p = joinpath(dir, "f.npy")

        @testset "dtypes with no Julia counterpart" begin
            # x86 long double: 80-bit extended padded to 16 bytes
            fixture(p, "'<f16'", (1,), zeros(UInt8, 16))
            e = try
                npyread(p)
            catch err
                err
            end
            @test e isa NPZError
            @test occursin("longdouble", sprint(showerror, e))

            fixture(p, "'<c32'", (1,), zeros(UInt8, 32))
            @test_throws NPZError npyread(p)
            fixture(p, "'|O'", (1,), zeros(UInt8, 8))
            @test_throws NPZError npyread(p)
            fixture(p, "'|T'", (1,), zeros(UInt8, 8))
            @test_throws NPZError npyread(p)
            fixture(p, "'<M8[ps]'", (1,), zeros(UInt8, 8))
            @test_throws NPZError npyread(p)
        end

        @testset "header versions" begin
            payload = reinterpret(UInt8, Int32[7, 8])
            # v2.0 differs from v1.0 only in the width of the length field
            fixture(p, "'<i4'", (2,), collect(payload); version=(0x02, 0x00))
            @test npyread(p) == Int32[7, 8]
            # v3.0 headers are utf-8, which is how non-latin-1 field names travel
            fixture(p, "[('ä', '<i4')]", (2,), collect(payload); version=(0x03, 0x00))
            A = npyread(p)
            @test eltype(A) === NamedTuple{(Symbol("ä"),),Tuple{Int32}}
            @test A[1].ä == 7 && A[2].ä == 8
            fixture(p, "'<i4'", (2,), collect(payload); version=(0x04, 0x00))
            @test_throws NPZError npyread(p)
        end

        @testset "a header too big for v1.0" begin
            # 400 long field names push the header past the UInt16 length field
            names = [("field_$(i)_" * repeat("x", 300), "|i1") for i in 1:400]
            dt = parse_dtype(Any[(n, d) for (n, d) in names])
            io = IOBuffer()
            write_npy_header(io, dt, false, (0,))
            bytes = take!(io)
            @test bytes[7] == 0x02          # version 2.0
            @test length(bytes) % 64 == 0
            h = read_npy_header(IOBuffer(bytes))
            @test descr_of(h.dt) == descr_of(dt)
            @test h.shape == (0,)
        end

        @testset "damaged files" begin
            fixture(p, "'<i4'", (10,), zeros(UInt8, 12))          # data ends early
            e = try
                npyread(p)
            catch err
                err
            end
            @test e isa NPZError
            @test occursin("truncated", sprint(showerror, e))

            fixture(p, "'<i4'", (2,), zeros(UInt8, 8);
                    magic=UInt8[0x93, 0x4e, 0x4f, 0x50, 0x45, 0x21])
            @test_throws NPZError npyread(p)

            fixture(p, "'<i4'", (-2,), zeros(UInt8, 8))
            @test_throws NPZError npyread(p)

            fixture(p, "'<i4'", (2,), zeros(UInt8, 8))
            open(p, "r+") do io                                   # break the header dict
                seek(io, 10)
                write(io, "[[[")
            end
            @test_throws NPZError npyread(p)

            # a header that parses but is not a dict, and one missing a key
            open(joinpath(dir, "nodict.npy"), "w") do io
                write(io, NPY_MAGIC, 0x01, 0x00, htol(UInt16(54)))
                write(io, rpad("[1, 2, 3]", 53), '\n')
            end
            @test_throws NPZError npyread(joinpath(dir, "nodict.npy"))
        end

        @testset "duplicate archive entries" begin
            # a zip may legally repeat a name; the last entry is the live one
            dup = joinpath(dir, "dup.npz")
            open(dup, "w") do io
                ZipArchives.ZipWriter(io; check_names=false) do w
                    for v in 1:2
                        ZipArchives.zip_newfile(w, "k.npy")
                        npywrite(w, [v])
                    end
                end
            end
            @test npzread(dup)["k"] == [2]
        end

        if TEST_NUMPY
            @testset "numpy reads our v2.0 header" begin
                names = ntuple(i -> Symbol("field_", i, "_", repeat("x", 300)), 400)
                v = NamedTuple{names,NTuple{400,Int8}}(ntuple(i -> Int8(i % 128), 400))
                big = joinpath(dir, "bigheader.npy")
                npywrite(big, [v])
                @test open(io -> read(io, 8), big)[7] == 0x02
                out = pyexec("""
import sys, numpy as np
a = np.load(sys.argv[1], max_header_size=10**7)
assert len(a.dtype.names) == 400, a.dtype.names
assert a[0][6] == 7, a[0][6]
print("ok")
""", big)
                @test occursin("ok", out)
                @test npyread(big)[1][7] == Int8(7)
            end
        end
    end
end
