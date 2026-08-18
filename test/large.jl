# Large arrays and large strings.  The multi-gigabyte cases need real disk space
# and time, so they are gated behind NPZVIBE_TEST_LARGE=1.

const TEST_LARGE = get(ENV, "NPZVIBE_TEST_LARGE", "") == "1"

@testset "large data" begin
    mktempdir() do dir
        @testset "64 MiB array" begin
            n = 8_000_000
            A = collect(1.0:n)
            path = joinpath(dir, "big.npy")
            npywrite(path, A)
            @test filesize(path) == 8n + 128
            B = npyread(path; mmap=true)
            @test length(B) == n
            @test B[1] == 1.0 && B[end] == n && B[n÷2] == n ÷ 2
            @test sum(B) == n * (n + 1) / 2
            if HAVE_NUMPY
                out = pyexec("""
import sys, numpy as np
a = np.load(sys.argv[1], mmap_mode='r')
n = int(sys.argv[2])
assert a.dtype.str == '<f8' and a.shape == (n,), (a.dtype, a.shape)
assert a[0] == 1 and a[-1] == n
assert float(np.sum(a, dtype='f8')) == n * (n + 1) / 2
print("ok")
""", path, n)
                @test occursin("ok", out)
            end
        end

        @testset "multi-megabyte strings" begin
            big = [repeat("a", 1_000_000), repeat("é", 1_000_000), ""]
            path = joinpath(dir, "bigstr.npz")
            npzwrite(path, Dict("s" => big))
            back = npzread(path)["s"]
            @test back == big
            @test length(back[2]) == 1_000_000
            if HAVE_NUMPY
                out = pyexec("""
import sys, numpy as np
a = np.load(sys.argv[1])['s']
assert a.dtype.str == '<U1000000', a.dtype
assert len(a[0]) == 1000000 and a[0][0] == 'a'
assert len(a[1]) == 1000000 and a[1][-1] == 'é'
assert a[2] == ''
print("ok")
""", path)
                @test occursin("ok", out)
            end
        end

        if !TEST_LARGE
            @info "NPZvibe: skipping the >4 GiB ZIP64 tests; set NPZVIBE_TEST_LARGE=1 to run them"
        else
            @testset "entry larger than 4 GiB (ZIP64)" begin
                n = 4_500_000_000
                scratch = joinpath(dir, "scratch.bin")
                A = open(scratch, "w+") do io
                    Mmap.mmap(io, Vector{UInt8}, n)
                end
                for i in 1:1_000_000:n            # a sparse, checkable pattern
                    A[i] = UInt8(i % 251)
                end
                A[end] = 0xab
                Mmap.sync!(A)
                path = joinpath(dir, "huge.npz")
                npzwrite(path, Dict("h" => A))
                @test filesize(path) > n

                B = npzread(path; mmap=true)["h"]
                @test length(B) == n
                @test B[end] == 0xab
                @test all(B[i] == UInt8(i % 251) for i in 1:100_000_000:n)

                if HAVE_NUMPY
                    out = pyexec("""
import sys, zipfile, numpy as np
path = sys.argv[1]
n = int(sys.argv[2])
with zipfile.ZipFile(path) as z:
    info = z.infolist()[0]
    assert info.file_size == n + 128, info.file_size
a = np.load(path)['h']
assert a.shape == (n,) and a.dtype.str == '|u1', (a.shape, a.dtype)
assert a[-1] == 0xab
assert a[0] == 1        # julia index 1 was set to 1 % 251
print("ok")
""", path, n)
                    @test occursin("ok", out)
                end
                finalize(A)
                rm(scratch; force=true)
            end

            @testset "more than 2^32 elements" begin
                n = 4_300_000_000
                scratch = joinpath(dir, "scratch2.bin")
                A = open(scratch, "w+") do io
                    Mmap.mmap(io, Vector{UInt8}, n)
                end
                A[end] = 0x7f
                Mmap.sync!(A)
                path = joinpath(dir, "huge.npy")
                npywrite(path, A)
                B = npyread(path; mmap=true)
                @test length(B) == n
                @test B[end] == 0x7f
                finalize(A)
                rm(scratch; force=true)
            end
        end
    end
end
