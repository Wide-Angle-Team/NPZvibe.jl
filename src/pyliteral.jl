# Parser and serializer for the subset of Python literals that appears in `.npy`
# headers: dicts, lists, tuples, strings, integers, `True`/`False`/`None`.
#
# A regex-based reader cannot handle structured dtypes (whose `descr` is a
# possibly-nested list of tuples), so the header is parsed properly instead.

mutable struct PyReader
    c::Vector{Char}
    i::Int
end
PyReader(s::AbstractString) = PyReader(collect(s), 1)

atend(p::PyReader) = p.i > length(p.c)
cur(p::PyReader) = p.c[p.i]

function skipws!(p::PyReader)
    while !atend(p) && isspace(cur(p))
        p.i += 1
    end
end

function matchword(p::PyReader, word::String)
    n = length(word)
    p.i + n - 1 <= length(p.c) || return false
    for (k, ch) in enumerate(word)
        p.c[p.i+k-1] == ch || return false
    end
    true
end

"""
    pyparse(s) -> Any

Parse a Python literal into Julia values: `dict` becomes `Dict{String,Any}`,
`list` becomes `Vector{Any}`, `tuple` becomes `Tuple`, `str` becomes `String`,
`int` becomes `Int`, `True`/`False` become `Bool` and `None` becomes `nothing`.
"""
function pyparse(s::AbstractString)
    p = PyReader(s)
    v = parsevalue!(p)
    skipws!(p)
    atend(p) || throw(NPZError("trailing data after Python literal at character $(p.i)"))
    v
end

function parsevalue!(p::PyReader)
    skipws!(p)
    atend(p) && throw(NPZError("unexpected end of Python literal"))
    c = cur(p)
    if c == '{'
        parsedict!(p)
    elseif c == '['
        parseseq!(p, ']')
    elseif c == '('
        Tuple(parseseq!(p, ')'))
    elseif c == '\'' || c == '"'
        parsestring!(p)
    elseif isdigit(c) || c == '-' || c == '+'
        parseint!(p)
    elseif matchword(p, "True")
        p.i += 4
        true
    elseif matchword(p, "False")
        p.i += 5
        false
    elseif matchword(p, "None")
        p.i += 4
        nothing
    else
        throw(NPZError("unexpected character $(repr(c)) at $(p.i) in Python literal"))
    end
end

function parsedict!(p::PyReader)
    p.i += 1  # '{'
    d = Dict{String,Any}()
    while true
        skipws!(p)
        atend(p) && throw(NPZError("unterminated dict in Python literal"))
        if cur(p) == '}'
            p.i += 1
            return d
        end
        key = parsevalue!(p)
        key isa String || throw(NPZError("dict keys must be strings, got $(repr(key))"))
        skipws!(p)
        (!atend(p) && cur(p) == ':') || throw(NPZError("expected ':' after dict key $(repr(key))"))
        p.i += 1
        d[key] = parsevalue!(p)
        skipws!(p)
        atend(p) && throw(NPZError("unterminated dict in Python literal"))
        if cur(p) == ','
            p.i += 1
        elseif cur(p) != '}'
            throw(NPZError("expected ',' or '}' in dict at character $(p.i)"))
        end
    end
end

function parseseq!(p::PyReader, close::Char)
    p.i += 1  # '[' or '('
    v = Any[]
    while true
        skipws!(p)
        atend(p) && throw(NPZError("unterminated sequence in Python literal"))
        if cur(p) == close
            p.i += 1
            return v
        end
        push!(v, parsevalue!(p))
        skipws!(p)
        atend(p) && throw(NPZError("unterminated sequence in Python literal"))
        if cur(p) == ','
            p.i += 1
        elseif cur(p) != close
            throw(NPZError("expected ',' or '$close' in sequence at character $(p.i)"))
        end
    end
end

function parsestring!(p::PyReader)
    q = cur(p)
    p.i += 1
    io = IOBuffer()
    while true
        atend(p) && throw(NPZError("unterminated string in Python literal"))
        c = cur(p)
        p.i += 1
        if c == q
            return String(take!(io))
        elseif c == '\\'
            atend(p) && throw(NPZError("unterminated escape in Python literal"))
            e = cur(p)
            p.i += 1
            if e == 'n'; print(io, '\n')
            elseif e == 't'; print(io, '\t')
            elseif e == 'r'; print(io, '\r')
            elseif e == '0'; print(io, '\0')
            elseif e == 'a'; print(io, '\a')
            elseif e == 'b'; print(io, '\b')
            elseif e == 'f'; print(io, '\f')
            elseif e == 'v'; print(io, '\v')
            elseif e == '\\' || e == '\'' || e == '"'; print(io, e)
            elseif e == 'x'; print(io, Char(readhex!(p, 2)))
            elseif e == 'u'; print(io, Char(readhex!(p, 4)))
            elseif e == 'U'; print(io, Char(readhex!(p, 8)))
            else
                throw(NPZError("unsupported string escape \\$e in Python literal"))
            end
        else
            print(io, c)
        end
    end
end

function readhex!(p::PyReader, n::Int)
    p.i + n - 1 <= length(p.c) || throw(NPZError("truncated hex escape in Python literal"))
    v = UInt32(0)
    for _ in 1:n
        d = cur(p)
        p.i += 1
        digit = tryparse(UInt32, string(d); base=16)
        digit === nothing && throw(NPZError("bad hex digit $(repr(d)) in Python literal"))
        v = v * 16 + digit
    end
    v
end

function parseint!(p::PyReader)
    start = p.i
    (cur(p) == '-' || cur(p) == '+') && (p.i += 1)
    d0 = p.i
    while !atend(p) && isdigit(cur(p))
        p.i += 1
    end
    p.i > d0 || throw(NPZError("expected a number at character $start in Python literal"))
    s = String(p.c[start:p.i-1])
    v = tryparse(Int, s)
    v === nothing && throw(NPZError("integer $s out of range in Python literal"))
    v
end

"""
    pystr(x) -> String

Serialize a Julia value back to the Python literal syntax numpy uses in headers.
"""
pystr(x::Bool) = x ? "True" : "False"
pystr(x::Integer) = string(x)
pystr(::Nothing) = "None"
pystr(x::AbstractString) = pystr_string(x)
pystr(x::Tuple) = length(x) == 1 ? "(" * pystr(x[1]) * ",)" :
                  "(" * join(map(pystr, x), ", ") * ")"
pystr(x::AbstractVector) = "[" * join(map(pystr, x), ", ") * "]"
function pystr(d::AbstractDict)
    "{" * join(("$(pystr(k)): $(pystr(v))" for (k, v) in d), ", ") * "}"
end

function pystr_string(s::AbstractString)
    io = IOBuffer()
    print(io, '\'')
    for c in s
        if c == '\\'
            print(io, "\\\\")
        elseif c == '\''
            print(io, "\\'")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif c < ' ' || c == '\x7f'
            print(io, "\\x", string(UInt32(c); base=16, pad=2))
        else
            print(io, c)
        end
    end
    print(io, '\'')
    String(take!(io))
end
