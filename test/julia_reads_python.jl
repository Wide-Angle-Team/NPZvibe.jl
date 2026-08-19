# numpy writes, NPZvibe reads.  The expected values are built here independently
# of NPZvibe's own writer, so a symmetric bug cannot hide.

@testset "julia reads python" begin
    if !TEST_NUMPY
        @info "NPZvibe: skipping julia-reads-python tests; set NPZVIBE_TEST_NUMPY=1 to run them"
    else
        mktempdir() do dir
            pyrun(joinpath(PYDIR, "gen_reference.py"), dir)
            rd(name; kw...) = npzread(joinpath(dir, name); kw...)

            @testset "basic dtypes" begin
                @test rd("bool.npy") == [true, false, true]
                @test eltype(rd("bool.npy")) === Bool
                for (name, T) in (("i1", Int8), ("i2", Int16), ("i4", Int32), ("i8", Int64),
                                  ("u1", UInt8), ("u2", UInt16), ("u4", UInt32), ("u8", UInt64))
                    A = rd(name * ".npy")
                    @test eltype(A) === T
                    @test A == T[typemin(T), 0, 7, typemax(T)]
                end
                for (name, T) in (("f2", Float16), ("f4", Float32), ("f8", Float64))
                    A = rd(name * ".npy")
                    @test eltype(A) === T
                    @test A[1] === T(1.5)
                    @test A[2] === T(-0.0)
                    @test signbit(A[2])
                    @test isnan(A[3])
                    @test A[4] === T(Inf)
                    @test A[5] === T(-Inf)
                end
                @test rd("c8.npy") == ComplexF32[1.5+2.5im, ComplexF32(-0.0, Inf)]
                @test eltype(rd("c8.npy")) === ComplexF32
                c16 = rd("c16.npy")
                @test c16[1] === ComplexF64(1.5, 2.5)
                @test isnan(real(c16[2])) && imag(c16[2]) == -3
                @test rd("void3.npy") == [(0x01, 0x02, 0x03), (0xff, 0x00, 0x80)]
            end

            @testset "shape and memory order" begin
                expected = reshape(Int32.(0:23), 4, 3, 2)             # numpy C order
                expected = permutedims(expected, (3, 2, 1))
                @test rd("c_order.npy") == expected
                @test rd("f_order.npy") == expected
                @test rd("c_order.npy") == rd("f_order.npy")
                z = rd("zerod.npy")
                @test z isa Array{Float64,0}
                @test z[] === 3.5
                e = rd("empty.npy")
                @test size(e) == (0, 3)
                @test eltype(e) === Float32
                d5 = rd("dim5.npy")
                @test size(d5) == (2, 3, 4, 5, 6)
                @test d5[1, 1, 1, 1, 1] == 0.0
                @test d5[2, 3, 4, 5, 6] == 2 * 3 * 4 * 5 * 6 - 1
                @test d5[1, 1, 1, 1, 2] == 1.0     # last axis is fastest in C order
            end

            @testset "byte order" begin
                # booleans are single-byte, so endianness is irrelevant, but
                # the '>' prefix still needs to parse and read correctly
                @test rd("be_bool.npy") == [true, false, true]
                @test eltype(rd("be_bool.npy")) === Bool

                # signed integers
                @test rd("be_i2.npy") == Int16[typemin(Int16), 0, typemax(Int16)]
                @test eltype(rd("be_i2.npy")) === Int16
                @test rd("be_i4.npy") == reshape(Int32.(0:5), 3, 2)'
                @test eltype(rd("be_i4.npy")) === Int32
                @test rd("be_i8.npy") == Int64[0, -1, Int64(2)^53]
                @test eltype(rd("be_i8.npy")) === Int64

                # unsigned integers
                @test rd("be_u2.npy") == UInt16[0, 1, 65535]
                @test rd("be_u4.npy") == UInt32[0, 1, typemax(UInt32)]
                @test eltype(rd("be_u4.npy")) === UInt32
                @test rd("be_u8.npy") == UInt64[0, 1, UInt64(2)^53]
                @test eltype(rd("be_u8.npy")) === UInt64

                # floats
                bf2 = rd("be_f2.npy")
                @test bf2[1] === Float16(1.5)
                @test bf2[2] === Float16(-0.0) && signbit(bf2[2])
                @test isnan(bf2[3])
                @test bf2[4] === Float16(Inf)
                @test eltype(bf2) === Float16
                bf4 = rd("be_f4.npy")
                @test bf4[1] === Float32(1.5)
                @test bf4[2] === Float32(-0.0) && signbit(bf4[2])
                @test isnan(bf4[3])
                @test bf4[4] === Float32(Inf) && bf4[5] === Float32(-Inf)
                @test eltype(bf4) === Float32
                bf = rd("be_f8.npy")
                @test bf[1] === 1.5 && bf[2] === -2.25 && isnan(bf[3])
                @test eltype(bf) === Float64

                # complex
                bc8 = rd("be_c8.npy")
                @test bc8[1] === ComplexF32(1.5, 2.5)
                @test bc8[2] === ComplexF32(-3, -4)
                @test eltype(bc8) === ComplexF32
                @test rd("be_c16.npy") == ComplexF64[1.5+2.5im, -3-4im]

                # timedelta
                @test rd("be_m8_s.npy") == [Second(0), Second(90), Second(-1)]
                @test eltype(rd("be_m8_s.npy")) === Second
                @test rd("be_m8_D.npy") == [Day(0), Day(5), Day(-3)]
                @test eltype(rd("be_m8_D.npy")) === Day

                # strings and datetime
                @test rd("be_str_U3.npy") == ["ab", "cde"]
                @test rd("be_dt_s.npy") == [DateTime(2020, 1, 2, 3, 4, 5), DateTime(1960, 1, 1)]
            end

            @testset "strings" begin
                @test rd("str_S5.npy") == ["ab", "cde", ""]
                @test rd("str_S4_nul.npy") == ["a\0b"]                # only trailing NULs go
                @test rd("str_S2_raw.npy")[1] == String(UInt8[0xff, 0xfe])
                @test rd("str_U8.npy") == ["", "héllo", "\U0001d418nicode"]
                @test rd("str_U1_grid.npy") == ["a" "b"; "c" "d"]
            end

            @testset "datetime64 and timedelta64" begin
                @test rd("dt_Y.npy") == [Date(1970), Date(2020), Date(1900)]
                @test rd("dt_mon.npy") == [Date(1970, 1), Date(2020, 7), Date(1900, 12)]
                # 2020-01-02 is a whole number of weeks after the epoch, so W truncates to itself
                @test rd("dt_W.npy") == [Date(1970, 1, 1), Date(2020, 1, 2)]
                @test rd("dt_D.npy") == [Date(1970, 1, 1), Date(2020, 2, 29), Date(1900, 1, 1)]
                @test rd("dt_h.npy") == [DateTime(2020, 1, 2, 3), DateTime(1969, 12, 31, 23)]
                @test rd("dt_min.npy") == [DateTime(2020, 1, 2, 3, 4), DateTime(1969, 12, 31, 23, 59)]
                @test rd("dt_s.npy") == [DateTime(2020, 1, 2, 3, 4, 5), DateTime(1969, 12, 31, 23, 59, 59)]
                @test rd("dt_ms.npy") == [DateTime(2020, 1, 2, 3, 4, 5, 6)]
                @test eltype(rd("dt_D.npy")) === Date
                @test eltype(rd("dt_s.npy")) === DateTime

                us = rd("dt_us.npy")
                @test eltype(us) === NanoDate
                @test us == [NanoDate(DateTime(2020, 1, 2, 3, 4, 5), Nanosecond(7000))]
                ns = rd("dt_ns.npy")
                @test ns == [NanoDate(DateTime(2020, 1, 2, 3, 4, 5), Nanosecond(8)),
                             NanoDate(DateTime(1969, 12, 31, 23, 59, 59, 999), Nanosecond(999999))]
                @test rd("dt_10s.npy") == [DateTime(1970), DateTime(1970, 1, 1, 0, 0, 10),
                                           DateTime(1969, 12, 31, 23, 59, 40)]

                nat = rd("dt_ns_nat.npy")
                @test eltype(nat) === Union{Missing,NanoDate}
                @test ismissing(nat[1])
                @test nat[2] == NanoDate(DateTime(2020, 1, 2, 3, 4, 5), Nanosecond(8))
                natd = rd("dt_D_nat.npy")
                @test eltype(natd) === Union{Missing,Date}
                @test isequal(natd, [missing, Date(2020, 2, 29)])

                @test rd("td_D.npy") == [Day(0), Day(5), Day(-3)]
                @test rd("td_s.npy") == [Second(0), Second(90), Second(-1)]
                @test rd("td_ns.npy") == [Nanosecond(0), Nanosecond(1), Nanosecond(-1)]
                @test rd("td_ms.npy") == [Millisecond(1500), Millisecond(-250)]
                @test rd("td_year.npy") == [Year(1), Year(-2)]
                @test isequal(rd("td_us_nat.npy"), [missing, Microsecond(12)])
            end

            @testset "structured dtypes" begin
                s = rd("struct_simple.npy")
                @test eltype(s) === @NamedTuple{a::Int32, b::Float64}
                @test s[1] === (a=Int32(1), b=1.5)
                @test s[2].a == -2 && isnan(s[2].b)
                @test s[3] === (a=Int32(3), b=-0.25)

                sub = rd("struct_sub.npy")
                @test eltype(sub) === @NamedTuple{a::Int32, b::NTuple{3,Float64}, c::String}
                @test sub[1] === (a=Int32(7), b=(1.0, 2.0, 3.0), c="xy")
                @test sub[2] === (a=Int32(8), b=(4.0, 5.0, 6.0), c="zzzz")

                s2 = rd("struct_sub2d.npy")
                @test eltype(s2) === @NamedTuple{m::NTuple{2,NTuple{3,Int16}}}
                @test s2[1].m == ((0, 1, 2), (3, 4, 5))       # C order: last axis fastest
                @test s2[2].m == ((6, 7, 8), (9, 10, 11))

                n = rd("struct_nested.npy")
                @test eltype(n) === @NamedTuple{x::@NamedTuple{u::Int32, v::Float32}, y::Int64}
                @test n[1] === (x=(u=Int32(1), v=0.5f0), y=Int64(-10))
                @test n[2] === (x=(u=Int32(2), v=1.5f0), y=Int64(20))

                p = rd("struct_padded.npy")
                @test eltype(p) === @NamedTuple{a::Int16, b::Float64}
                @test p == [(a=Int16(1), b=3.5), (a=Int16(2), b=4.5)]

                bt = rd("struct_be_time.npy")
                @test bt[1].a == Int32(1) && bt[1].b == 1.25
                @test bt[1].t == DateTime(2020, 1, 2, 3, 4, 5)
                @test ismissing(bt[2].t)

                u = rd("struct_U.npy")
                @test u == [(i=Int32(1), s="héllo"), (i=Int32(2), s="")]
            end

            @testset "structured byte order" begin
                s = rd("struct_all_be.npy")
                @test eltype(s) === @NamedTuple{a::Int32, b::Float64}
                @test s[1] === (a=Int32(1), b=1.5)
                @test s[2].a == -2 && isnan(s[2].b)
                @test s[3] === (a=Int32(3), b=-0.25)
            end

            @testset "npz byte order" begin
                be = rd("stored_be.npz")
                @test sort(collect(keys(be))) == ["big_i8", "little_f4"]
                @test be["big_i8"] == Int64[0, -1, Int64(2)^53]
                @test eltype(be["big_i8"]) === Int64
                @test be["little_f4"] == Float32[1.5, -2.25]
                @test eltype(be["little_f4"]) === Float32
            end

            @testset "npz archives" begin
                stored = rd("stored.npz")
                @test sort(collect(keys(stored))) == ["a", "b", "c"]
                @test stored["a"] == permutedims(reshape(0.0:5.0, 3, 2), (2, 1))
                @test stored["b"] == ["x", "yy"]
                @test stored["c"] == Int16[0 1; 2 3]

                defl = rd("deflated.npz")
                @test defl["big"] == Int32.(0:9999)
                @test defl["t"] == [NanoDate(DateTime(2020, 1, 2, 3, 4, 5), Nanosecond(8))]

                @test rd("noext.npz")["plain"] == Int64.(0:2)
                @test rd("zip64.npz")["small"] == UInt8.(0:4)

                # reading a subset, and memory-mapped reads of stored entries
                @test collect(keys(npzread(joinpath(dir, "stored.npz"), ["a"]))) == ["a"]
                mapped = npzread(joinpath(dir, "stored.npz"); mmap=true)
                @test mapped["a"] == stored["a"]
                @test mapped["c"] == stored["c"]
                @test npyread(joinpath(dir, "dim5.npy"); mmap=true) == rd("dim5.npy")
            end
        end
    end
end
