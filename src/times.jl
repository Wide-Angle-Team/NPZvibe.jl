# numpy `datetime64` / `timedelta64` <-> `Dates` and `NanoDates`.
#
# numpy stores both as an `Int64` count of `unit`s (times an optional multiple)
# since the unix epoch, with `typemin(Int64)` meaning NaT.  No new Julia types
# are introduced here: `datetime64` becomes `Date`, `DateTime` or `NanoDate`
# depending on the resolution it needs, `timedelta64` becomes the matching
# `Dates.Period`, and NaT becomes `missing`.

const NPY_NAT = typemin(Int64)

const EPOCH_DATE = Date(1970, 1, 1)
const EPOCH_DATETIME = DateTime(1970, 1, 1)

"Units numpy accepts, longest name first so that parsing is unambiguous."
const NPY_TIME_UNITS = (:as, :fs, :ps, :ns, :us, :ms, :Y, :M, :W, :D, :h, :m, :s)

"Units too fine for any Julia time type."
const SUBNANO_UNITS = (:ps, :fs, :as)

const DATE_UNITS = (:Y, :M, :W, :D)
const DATETIME_UNITS = (:h, :m, :s, :ms)
const NANODATE_UNITS = (:us, :ns)

const PERIOD_OF_UNIT = Dict{Symbol,Type}(
    :Y => Year, :M => Month, :W => Week, :D => Day, :h => Hour, :m => Minute,
    :s => Second, :ms => Millisecond, :us => Microsecond, :ns => Nanosecond,
)

"""
    parse_timeunit(s) -> (multiple, unit)

Parse the unit part of an `M8`/`m8` dtype: `"[ns]"`, `"[10s]"` or `""` (numpy's
unit-less "generic" form, which carries no resolution and is rejected).
"""
function parse_timeunit(s::AbstractString, descr::AbstractString)
    isempty(s) && throw(NPZError(
        "dtype $(repr(descr)) has no time unit (numpy's \"generic\" datetime64/timedelta64); " *
        "it carries no resolution and cannot be converted to a Julia time type"))
    (startswith(s, '[') && endswith(s, ']')) ||
        throw(NPZError("malformed time unit in dtype $(repr(descr))"))
    body = s[nextind(s, 1):prevind(s, lastindex(s))]
    j = 1
    while j <= lastindex(body) && isdigit(body[j])
        j = nextind(body, j)
    end
    mult = j == 1 ? 1 : parse(Int, body[1:prevind(body, j)])
    name = body[j:end]
    unit = Symbol(name)
    unit in NPY_TIME_UNITS ||
        throw(NPZError("unknown time unit $(repr(name)) in dtype $(repr(descr))"))
    mult >= 1 || throw(NPZError("non-positive time unit multiple in dtype $(repr(descr))"))
    (mult, unit)
end

function check_time_unit(unit::Symbol, kind::Symbol)
    if unit in SUBNANO_UNITS
        throw(NPZError(
            "$(kind === :datetime ? "datetime64" : "timedelta64") unit $(repr(String(unit))) " *
            "has sub-nanosecond resolution; no Julia time type can represent it"))
    end
    unit
end

"""
    datetime64_type(unit)

Julia type used for a numpy `datetime64` with the given unit.
"""
function datetime64_type(unit::Symbol)
    check_time_unit(unit, :datetime)
    unit in DATE_UNITS && return Date
    unit in DATETIME_UNITS && return DateTime
    unit in NANODATE_UNITS && return NanoDate
    throw(NPZError("unsupported datetime64 unit $(repr(String(unit)))"))
end

"""
    timedelta64_type(unit)

`Dates.Period` subtype used for a numpy `timedelta64` with the given unit.
"""
function timedelta64_type(unit::Symbol)
    check_time_unit(unit, :timedelta)
    get(PERIOD_OF_UNIT, unit) do
        throw(NPZError("unsupported timedelta64 unit $(repr(String(unit)))"))
    end
end

"Scale a raw count by the dtype's unit multiple, checking for overflow."
function scale_count(v::Int64, mult::Int)
    mult == 1 && return v
    r = widemul(v, mult)
    typemin(Int64) < r <= typemax(Int64) ||
        throw(NPZError("time value $v * $mult overflows Int64"))
    Int64(r)
end

"Convert a raw numpy datetime64 count to a Julia date/time value (NaT -> missing)."
function datetime64_value(mult::Int, unit::Symbol, v::Int64)
    v == NPY_NAT && return missing
    n = scale_count(v, mult)
    if unit === :Y
        EPOCH_DATE + Year(n)
    elseif unit === :M
        EPOCH_DATE + Month(n)
    elseif unit === :W
        EPOCH_DATE + Week(n)
    elseif unit === :D
        EPOCH_DATE + Day(n)
    elseif unit === :h
        EPOCH_DATETIME + Hour(n)
    elseif unit === :m
        EPOCH_DATETIME + Minute(n)
    elseif unit === :s
        EPOCH_DATETIME + Second(n)
    elseif unit === :ms
        EPOCH_DATETIME + Millisecond(n)
    elseif unit === :us
        unixmicros2nanodate(n)
    elseif unit === :ns
        unixnanos2nanodate(n)
    else
        throw(NPZError("unsupported datetime64 unit $(repr(String(unit)))"))
    end
end

"Convert a raw numpy timedelta64 count to a `Dates.Period` (NaT -> missing)."
function timedelta64_value(mult::Int, unit::Symbol, v::Int64)
    v == NPY_NAT && return missing
    timedelta64_type(unit)(scale_count(v, mult))
end

"The unit NPZvibe writes a given Julia time type with."
npy_unit_of(::Type{Date}) = :D
npy_unit_of(::Type{DateTime}) = :ms
npy_unit_of(::Type{NanoDate}) = :ns
npy_unit_of(::Type{Year}) = :Y
npy_unit_of(::Type{Month}) = :M
npy_unit_of(::Type{Week}) = :W
npy_unit_of(::Type{Day}) = :D
npy_unit_of(::Type{Hour}) = :h
npy_unit_of(::Type{Minute}) = :m
npy_unit_of(::Type{Second}) = :s
npy_unit_of(::Type{Millisecond}) = :ms
npy_unit_of(::Type{Microsecond}) = :us
npy_unit_of(::Type{Nanosecond}) = :ns

"Raw numpy count for a Julia date/time value, in the unit `npy_unit_of` picks."
npy_count(::Missing) = NPY_NAT
npy_count(x::Date) = Int64(Dates.value(x - EPOCH_DATE))
npy_count(x::DateTime) = Int64(Dates.value(x - EPOCH_DATETIME))
function npy_count(x::NanoDate)
    n = nanodate2unixnanos(x)
    typemin(Int64) < n <= typemax(Int64) ||
        throw(NPZError("NanoDate $x is outside the range of datetime64[ns]"))
    Int64(n)
end
npy_count(x::Period) = Int64(Dates.value(x))
