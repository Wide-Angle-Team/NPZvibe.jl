# dtype descriptor <-> Julia type mapping.

using NPZvibe: parse_dtype, descr_of, NPZError, NATIVE_ORDER

@testset "dtypes" begin
    @testset "scalar dtypes" begin
        for (descr, T, itemsize) in (
            ("|b1", Bool, 1), ("?", Bool, 1),
            ("|i1", Int8, 1), ("<i2", Int16, 2), ("<i4", Int32, 4), ("<i8", Int64, 8),
            ("|u1", UInt8, 1), ("<u2", UInt16, 2), ("<u4", UInt32, 4), ("<u8", UInt64, 8),
            ("<f2", Float16, 2), ("<f4", Float32, 4), ("<f8", Float64, 8),
            ("<c4", Complex{Float16}, 4), ("<c8", ComplexF32, 8), ("<c16", ComplexF64, 16),
            ("|S5", String, 5), ("|a5", String, 5), ("<U3", String, 12), ("|V7", NTuple{7,UInt8}, 7),
            ("|S0", String, 0),
        )
            dt = parse_dtype(descr)
            @test dt.T === T
            @test dt.itemsize == itemsize
            @test !dt.swap
        end

        # byte order: '>' needs swapping on a little-endian host and vice versa
        @test parse_dtype(">i4").swap == (ENDIAN_BOM == 0x04030201)
        @test parse_dtype("<i4").swap == (ENDIAN_BOM != 0x04030201)
        @test !parse_dtype("=i4").swap
        @test !parse_dtype("|i1").swap
        @test parse_dtype(">i4").T === Int32
        @test parse_dtype(">U3").itemsize == 12
    end

    @testset "structured dtypes" begin
        dt = parse_dtype(Any[("a", "<i4"), ("b", "<f8", (3,)), ("c", "|S4")])
        @test dt.kind === :struct
        @test dt.itemsize == 32
        @test dt.T === @NamedTuple{a::Int32, b::NTuple{3,Float64}, c::String}
        @test [f.offset for f in dt.fields] == [0, 4, 28]

        # padding entries carry no name and only advance the offset
        padded = parse_dtype(Any[("a", "<i2"), ("", "|V6"), ("b", "<f8")])
        @test padded.T === @NamedTuple{a::Int16, b::Float64}
        @test [f.offset for f in padded.fields] == [0, 8]
        @test padded.itemsize == 16

        # numpy's dict form, with explicit offsets and itemsize
        dict = parse_dtype(Dict{String,Any}("names" => Any["a", "b"],
                                            "formats" => Any["<i2", "<f8"],
                                            "offsets" => Any[0, 8],
                                            "itemsize" => 16))
        @test dict.T === padded.T
        @test dict.itemsize == 16
        @test [f.offset for f in dict.fields] == [0, 8]

        nested = parse_dtype(Any[("x", Any[("u", "<i4"), ("v", "<f4")]), ("y", "<i8")])
        @test nested.T === @NamedTuple{x::@NamedTuple{u::Int32, v::Float32}, y::Int64}

        # 2-d sub-array fields nest in C order
        sub2d = parse_dtype(Any[("m", "<i2", (2, 3))])
        @test sub2d.T === @NamedTuple{m::NTuple{2,NTuple{3,Int16}}}
        @test sub2d.itemsize == 12

        # a field with a title is described as (('title', 'name'), descr)
        titled = parse_dtype(Any[(("t", "a"), "<i4")])
        @test titled.T === @NamedTuple{a::Int32}

        # NaT can hide anywhere in a struct, so time fields always admit missing
        timed = parse_dtype(Any[("t", "<M8[s]")])
        @test timed.T === @NamedTuple{t::Union{Missing,DateTime}}
    end

    @testset "descriptors we refuse" begin
        for descr in ("<f16", ">f16", "<f12", "|g16", "<c32", "<c24")
            @test_throws NPZError parse_dtype(descr)
        end
        @test occursin("longdouble", sprint(showerror, try
            parse_dtype("<f16")
        catch e
            e
        end))
        @test occursin("pickle", sprint(showerror, try
            parse_dtype("|O")
        catch e
            e
        end))
        @test occursin("StringDType", sprint(showerror, try
            parse_dtype("|T")
        catch e
            e
        end))
        @test_throws NPZError parse_dtype("")
        @test_throws NPZError parse_dtype("<i3")
        @test_throws NPZError parse_dtype("<q8")
        @test_throws NPZError parse_dtype("<Ux")
        @test_throws NPZError parse_dtype(Any[("a",)])
        @test_throws NPZError parse_dtype(Dict{String,Any}("names" => Any["a"]))
        @test_throws NPZError parse_dtype(Any[("a", "<i4"), ("a", "<i4")])
    end

    @testset "descriptors we emit" begin
        n = NATIVE_ORDER
        @test descr_of(parse_dtype("|b1")) == "|b1"
        @test descr_of(parse_dtype("<i4")) == "$(n)i4"
        @test descr_of(parse_dtype("|u1")) == "|u1"
        @test descr_of(parse_dtype("<f8")) == "$(n)f8"
        @test descr_of(parse_dtype("<c16")) == "$(n)c16"
        @test descr_of(parse_dtype("|S5")) == "|S5"
        @test descr_of(parse_dtype("<U3")) == "$(n)U3"
        @test descr_of(parse_dtype("|V7")) == "|V7"
        @test descr_of(parse_dtype("<M8[ns]")) == "$(n)M8[ns]"
        @test descr_of(parse_dtype("<m8[us]")) == "$(n)m8[us]"
        @test descr_of(parse_dtype(Any[("a", "<i4"), ("b", "<f8", (3,))])) ==
              Any[("a", "$(n)i4"), ("b", "$(n)f8", (3,))]
        # big-endian input is re-described in native order, since that is what
        # NPZvibe writes
        @test descr_of(parse_dtype(">i4")) == "$(n)i4"
    end
end
