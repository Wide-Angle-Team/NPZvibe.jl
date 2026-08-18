# NPZvibe.jl

Read and write numpy's `.npy` and `.npz` files from Julia — every dtype numpy can
store in them, arrays and strings of any size, and both C and Fortran memory order.

The test suite checks both directions against a real numpy: files NPZvibe writes are
loaded and verified by numpy, and files numpy writes are read back and compared to
independently constructed Julia values.

You have three guesses how this package was created.

## Usage

```julia
using NPZvibe

npzwrite("data.npz", Dict("a" => rand(3, 4), "s" => ["hé", "world"]))
npzwrite("data.npz"; a = rand(3, 4), s = ["hé", "world"])      # same thing
npzwrite("data.npz", Dict("a" => rand(3, 4)); compress = true) # like savez_compressed

d = npzread("data.npz")          # Dict{String,Any}
d = npzread("data.npz", ["a"])   # only the entries you need
a = npzread("array.npy")         # a bare .npy gives the array itself

npywrite("array.npy", rand(10))
a = npyread("array.npy")
a = npyread("huge.npy"; mmap = true)   # map the data instead of copying it
```

`npzread` dispatches on the file's magic bytes, not its extension, so either function
name works for either kind of file.

## Type mapping

| numpy dtype | Julia |
| --- | --- |
| `bool` | `Bool` |
| `int8` … `int64`, `uint8` … `uint64` | `Int8` … `Int64`, `UInt8` … `UInt64` |
| `float16`, `float32`, `float64` | `Float16`, `Float32`, `Float64` |
| `complex64`, `complex128` | `ComplexF32`, `ComplexF64` |
| `S<n>` (bytes), `U<n>` (UCS-4) | `String` |
| `V<n>` (raw void) | `NTuple{n,UInt8}` |
| `datetime64[Y\|M\|W\|D]` | `Dates.Date` |
| `datetime64[h\|m\|s\|ms]` | `Dates.DateTime` |
| `datetime64[us\|ns]` | `NanoDates.NanoDate` |
| `timedelta64[unit]` | the matching `Dates.Period` (`Year` … `Nanosecond`) |
| `NaT` | `missing` |
| structured dtypes | `NamedTuple`, including nesting and sub-array fields |

Both byte orders are read; NPZvibe writes native order. Writing goes the other way:
`String` arrays become `U` (pass `stringdtype = :S` for `S`), `Date`/`DateTime`/`NanoDate`
become `datetime64[D]`/`[ms]`/`[ns]`, `Period`s become the matching `timedelta64`, and
`NamedTuple` arrays become packed structured dtypes.

A few details worth knowing:

* Julia arrays are column-major, so arrays with two or more dimensions are written with
  `fortran_order = True` and no transpose. C-ordered files are transposed on read.
* Trailing NULs are stripped from `S`/`U` strings, matching numpy; embedded NULs are kept.
  `S` data that is not valid UTF-8 is preserved verbatim in the resulting `String`.
* Sub-array fields keep numpy's C ordering, so `('m', '<i2', (2, 3))` reads as
  `NTuple{2,NTuple{3,Int16}}`.
* Structured fields holding `datetime64`/`timedelta64` always read as `Union{Missing,T}`,
  since `NaT` can appear in any row.
* An `NTuple{n,UInt8}` is written as a `V<n>` void dtype at the top level, but as a
  sub-array of `u1` inside a struct — numpy stores both as the same `n` bytes.
* numpy has no `complex32`, so `Complex{Float16}` arrays are refused on write (a `<c4`
  file from some other writer still reads).

## Large files

`.npz` members larger than 4 GiB are written and read through ZIP64, and array data is
streamed rather than buffered twice. `mmap = true` maps the data instead of copying it
whenever the dtype, layout and (for `.npz`) an uncompressed entry allow it, falling back
to a normal read otherwise.  A mapped array is a read-only view of the file — copy it
before modifying it.

The multi-gigabyte tests are gated: run them with

```
NPZVIBE_TEST_LARGE=1 julia --project=. -e 'using Pkg; Pkg.test()'
```

## Not supported

These raise an informative `NPZError` rather than reading something wrong:

* **Object arrays** (`|O`): their contents are Python pickles, which numpy itself only
  loads with `allow_pickle=True`.
* **`longdouble`** (`float128`/`complex256`, i.e. `f16`/`g16`/`c32`): x86's 80-bit extended
  format padded to 12 or 16 bytes, which has no Julia counterpart.
* **`datetime64`/`timedelta64` below nanosecond resolution** (`ps`, `fs`, `as`) and numpy's
  unit-less "generic" datetimes.
* **numpy 2's variable-width `StringDType`** (`|T`).

## Contributing

Every contribution to this project must be 100% LLM-generated. Human-written code
will not be accepted.
