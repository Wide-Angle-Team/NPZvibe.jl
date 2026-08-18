# The header parser, exercised on real numpy headers and on the malformed input
# it has to reject.

using NPZvibe: pyparse, pystr, NPZError

@testset "python literal parser" begin
    @testset "numpy headers" begin
        h = pyparse("{'descr': '<i4', 'fortran_order': False, 'shape': (2, 3), }")
        @test h["descr"] == "<i4"
        @test h["fortran_order"] === false
        @test h["shape"] === (2, 3)

        @test pyparse("{'descr': '<f8', 'fortran_order': True, 'shape': (), }")["shape"] === ()
        @test pyparse("{'descr': '|u1', 'fortran_order': False, 'shape': (0,), }")["shape"] === (0,)

        s = pyparse("{'descr': [('a', '<i4'), ('b', '<f8', (3,))], 'fortran_order': False, 'shape': (2,), }")
        @test s["descr"] == [("a", "<i4"), ("b", "<f8", (3,))]

        d = pyparse("{'descr': {'names': ['a', 'b'], 'formats': ['<i2', '<f8'], " *
                    "'offsets': [0, 8], 'itemsize': 16}, 'fortran_order': False, 'shape': (2,), }")
        @test d["descr"]["names"] == ["a", "b"]
        @test d["descr"]["offsets"] == [0, 8]
        @test d["descr"]["itemsize"] == 16

        n = pyparse("{'descr': [('x', [('u', '<i4'), ('v', '<f4')]), ('y', '<i8')], " *
                    "'fortran_order': False, 'shape': (2,), }")
        @test n["descr"][1][2] == [("u", "<i4"), ("v", "<f4")]
    end

    @testset "syntax tolerance" begin
        # key order, quoting, whitespace and trailing commas are all free
        a = pyparse("""{"shape" : (1,2,), 'fortran_order':True ,
                        "descr":'<i4' ,}""")
        @test a["shape"] === (1, 2)
        @test a["descr"] == "<i4"
        @test a["fortran_order"] === true
        @test pyparse("[]") == []
        @test pyparse("()") === ()
        @test pyparse("(5,)") === (5,)
        @test pyparse("-17") === -17
        @test pyparse("None") === nothing
        @test pyparse("'a\\'b'") == "a'b"
        @test pyparse("'\\x41\\u00e9\\U0001d418'") == "Aé\U0001d418"
        @test pyparse("'tab\\there\\nand\\\\'") == "tab\there\nand\\"
    end

    @testset "malformed input" begin
        @test_throws NPZError pyparse("{'a': 1")
        @test_throws NPZError pyparse("(1, 2")
        @test_throws NPZError pyparse("{'a' 1}")
        @test_throws NPZError pyparse("{1: 2}")
        @test_throws NPZError pyparse("'unterminated")
        @test_throws NPZError pyparse("Nope")
        @test_throws NPZError pyparse("{'a': 1} trailing")
        @test_throws NPZError pyparse("'\\q'")
    end

    @testset "serializer" begin
        @test pystr(true) == "True"
        @test pystr(false) == "False"
        @test pystr(-3) == "-3"
        @test pystr(()) == "()"
        @test pystr((3,)) == "(3,)"
        @test pystr((2, 3)) == "(2, 3)"
        @test pystr("<i4") == "'<i4'"
        @test pystr("it's") == "'it\\'s'"
        @test pystr(Any[("a", "<i4"), ("b", "<f8", (3,))]) ==
              "[('a', '<i4'), ('b', '<f8', (3,))]"
        # everything we emit must parse back to itself
        for v in (Any[("a", "<i4")], (2, 3), (), "hé", true, -7)
            @test pyparse(pystr(v)) == v
        end
    end
end
