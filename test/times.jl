# datetime64 / timedelta64 conversions.

using NPZvibe: parse_dtype, parse_timeunit, datetime64_value, timedelta64_value,
               npy_count, NPY_NAT, NPZError

@testset "times" begin
    @testset "unit parsing" begin
        @test parse_timeunit("[ns]", "<M8[ns]") == (1, :ns)
        @test parse_timeunit("[10s]", "<M8[10s]") == (10, :s)
        @test parse_timeunit("[us]", "<m8[us]") == (1, :us)
        @test_throws NPZError parse_timeunit("", "<M8")          # numpy's generic unit
        @test_throws NPZError parse_timeunit("[zz]", "<M8[zz]")
        @test_throws NPZError parse_timeunit("ns", "<M8ns")
    end

    @testset "dtype element types" begin
        for (u, T) in (("Y", Date), ("M", Date), ("W", Date), ("D", Date),
                       ("h", DateTime), ("m", DateTime), ("s", DateTime), ("ms", DateTime),
                       ("us", NanoDate), ("ns", NanoDate))
            @test parse_dtype("<M8[$u]").T === T
        end
        for (u, T) in (("Y", Year), ("M", Month), ("W", Week), ("D", Day), ("h", Hour),
                       ("m", Minute), ("s", Second), ("ms", Millisecond),
                       ("us", Microsecond), ("ns", Nanosecond))
            @test parse_dtype("<m8[$u]").T === T
        end
    end

    @testset "values" begin
        @test datetime64_value(1, :Y, Int64(50)) == Date(2020)
        @test datetime64_value(1, :M, Int64(-1)) == Date(1969, 12)
        @test datetime64_value(1, :D, Int64(0)) == Date(1970, 1, 1)
        @test datetime64_value(1, :s, Int64(-1)) == DateTime(1969, 12, 31, 23, 59, 59)
        @test datetime64_value(1, :ms, Int64(1)) == DateTime(1970, 1, 1, 0, 0, 0, 1)
        @test datetime64_value(1, :ns, Int64(-1)) ==
              NanoDate(DateTime(1969, 12, 31, 23, 59, 59, 999), Nanosecond(999999))
        @test datetime64_value(10, :s, Int64(3)) == DateTime(1970, 1, 1, 0, 0, 30)
        @test ismissing(datetime64_value(1, :ns, NPY_NAT))

        @test timedelta64_value(1, :D, Int64(-3)) === Day(-3)
        @test timedelta64_value(1, :ns, Int64(7)) === Nanosecond(7)
        @test timedelta64_value(1000, :ms, Int64(2)) === Millisecond(2000)
        @test ismissing(timedelta64_value(1, :s, NPY_NAT))

        # and back again
        @test npy_count(Date(2020, 2, 29)) == 18321
        @test npy_count(DateTime(1970, 1, 1, 0, 0, 0, 1)) == 1
        @test npy_count(NanoDate(DateTime(1970), Nanosecond(8))) == 8
        @test npy_count(Second(-1)) == -1
        @test npy_count(missing) == NPY_NAT
        for x in (Date(1900, 3, 4), DateTime(2100, 5, 6, 7, 8, 9, 10),
                  NanoDate(DateTime(2020, 1, 2, 3, 4, 5), Nanosecond(999999)))
            u = NPZvibe.npy_unit_of(typeof(x))
            @test datetime64_value(1, u, npy_count(x)) == x
        end
    end

    @testset "resolutions we refuse" begin
        for u in ("ps", "fs", "as")
            e = try
                parse_dtype("<M8[$u]")
            catch err
                err
            end
            @test e isa NPZError
            @test occursin("sub-nanosecond", sprint(showerror, e))
            @test_throws NPZError parse_dtype("<m8[$u]")
        end
        @test_throws NPZError parse_dtype("<M8")
    end
end
