# What NPZvibe refuses to do, and how it says so.

using NPZvibe: NPZError

@testset "errors" begin
    mktempdir() do dir
        path = joinpath(dir, "x.npy")

        @testset "unwritable Julia values" begin
            @test_throws NPZError npywrite(path, Any[1, "two"])
            @test_throws NPZError npywrite(path, [1 // 2, 3 // 4])
            @test_throws NPZError npywrite(path, [BigInt(1)])
            @test_throws NPZError npywrite(path, [Union{Missing,Float64}[1.0, missing]])
            e = try
                npywrite(path, Union{Missing,Float64}[1.0, missing])
            catch err
                err
            end
            @test e isa NPZError
            @test occursin("missing", sprint(showerror, e))
            # heterogeneous tuple fields have no numpy equivalent
            @test_throws NPZError npywrite(path, [(a=(1, 2.0),)])
            @test_throws NPZError npywrite(path, ["a"]; stringdtype=:utf8)
            # numpy has no complex32, so `<c4` is readable but not writable
            e = try
                npywrite(path, Complex{Float16}[1 + 2im])
            catch err
                err
            end
            @test e isa NPZError
            @test occursin("complex32", sprint(showerror, e))
        end

        @testset "unreadable files" begin
            write(joinpath(dir, "garbage.bin"), "not a numpy file at all")
            @test_throws NPZError npzread(joinpath(dir, "garbage.bin"))
            @test_throws NPZError npyread(joinpath(dir, "garbage.bin"))
            write(joinpath(dir, "tiny.bin"), "ab")
            @test_throws NPZError npzread(joinpath(dir, "tiny.bin"))
            @test_throws SystemError npzread(joinpath(dir, "does_not_exist.npz"))

            # a truncated file must report truncation, and mmap must never grow it
            npywrite(joinpath(dir, "short.npy"), collect(1.0:100.0))
            full = filesize(joinpath(dir, "short.npy"))
            open(joinpath(dir, "short.npy"), "r+") do io
                truncate(io, full - 8)
            end
            @test_throws NPZError npyread(joinpath(dir, "short.npy"))
            @test_throws NPZError npyread(joinpath(dir, "short.npy"); mmap=true)
            @test filesize(joinpath(dir, "short.npy")) == full - 8
        end

        @testset "missing archive entries" begin
            npzwrite(joinpath(dir, "a.npz"), Dict("a" => [1, 2]))
            e = try
                npzread(joinpath(dir, "a.npz"), ["a", "nope"])
            catch err
                err
            end
            @test e isa NPZError
            @test occursin("nope", sprint(showerror, e))
            npywrite(path, [1, 2])
            @test_throws NPZError npzread(path, ["a"])
        end

        @testset "a zip that is not an npz" begin
            zippath = joinpath(dir, "notnpz.zip")
            ZipArchives.ZipWriter(zippath) do w
                ZipArchives.zip_newfile(w, "readme.txt")
                write(w, "hello")
            end
            @test_throws NPZError npzread(zippath)
        end
    end
end
