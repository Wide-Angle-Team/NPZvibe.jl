"""Assert that numpy reads exactly what NPZvibe wrote.

Called as `check_written.py <dir>`; the Julia side wrote one `<name>.npy` per
case plus `all.npz` / `all_compressed.npz`.  Expected values are built here from
numpy literals, independently of NPZvibe.
"""
import os
import sys

import numpy as np

d = sys.argv[1]
failures = []


def check(name, dtype_str, expected, *, order=None, exact_bytes=True):
    path = os.path.join(d, name + ".npy")
    a = np.load(path)
    e = np.asarray(expected)
    if a.dtype.str != dtype_str:
        failures.append(f"{name}: dtype {a.dtype.str!r} != {dtype_str!r}")
        return
    if a.shape != e.shape:
        failures.append(f"{name}: shape {a.shape} != {e.shape}")
        return
    if order is not None and not a.flags[order]:
        failures.append(f"{name}: expected {order}")
    if exact_bytes:
        if a.tobytes(order="C") != e.tobytes(order="C"):
            failures.append(f"{name}: bytes differ\n  got {a!r}\n  want {e!r}")
    elif not np.array_equal(a, e, equal_nan=True):
        failures.append(f"{name}: values differ\n  got {a!r}\n  want {e!r}")


# --- numbers ------------------------------------------------------------------
check("bool", "|b1", np.array([True, False, True]))
for k in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"):
    info = np.iinfo(k)
    check(k, np.dtype(k).str, np.array([info.min, 0, 7, info.max], dtype=k))
for k in ("f2", "f4", "f8"):
    check(k, np.dtype(k).str, np.array([1.5, -0.0, np.nan, np.inf, -np.inf], dtype=k))
for k in ("c8", "c16"):
    check(k, np.dtype(k).str, np.array([1.5 + 2.5j, complex(-0.0, np.inf)], dtype=k))

# --- shapes and order ---------------------------------------------------------
check("mat", "<f8", np.arange(6, dtype="<f8").reshape(2, 3, order="F"), order="F_CONTIGUOUS")
check("cube", "<i4", np.arange(24, dtype="<i4").reshape(2, 3, 4, order="F"))
check("zerod", "<f8", np.array(3.5))
check("empty", "<f4", np.zeros((0, 3), dtype="f4"))
check("transposed", "<f8", np.arange(6, dtype="<f8").reshape(2, 3, order="F").T)
check("subview", "<i2", np.arange(12, dtype="<i2").reshape(3, 4, order="F")[1:3, 2:4])

# --- strings and raw bytes ----------------------------------------------------
check("str_U", "<U7", np.array(["", "héllo", "\U0001d418nicode"], dtype="U7"))
# `stringdtype=:S` stores the strings' utf-8 code units verbatim
check("str_S", "|S10", np.array([b"", "héllo".encode(), "\U0001d418nicode".encode()], dtype="S10"))
check("void", "|V3", np.frombuffer(bytes([1, 2, 3, 255, 0, 128]), dtype="V3"))

# --- times --------------------------------------------------------------------
check("date", "<M8[D]", np.array(["2020-02-29", "1900-01-01"], dtype="M8[D]"))
check("datetime", "<M8[ms]", np.array(["2020-01-02T03:04:05.006"], dtype="M8[ms]"))
check("nanodate", "<M8[ns]",
      np.array(["2020-01-02T03:04:05.000000008", "1969-12-31T23:59:59.999999999"],
               dtype="M8[ns]"))
check("nat", "<M8[ms]", np.array(["2020-01-02T00:00:00", "NaT"], dtype="M8[ms]"))
check("period_s", "<m8[s]", np.array([0, 90, -1], dtype="m8[s]"))
check("period_ns", "<m8[ns]", np.array([0, 1, -1], dtype="m8[ns]"))
check("period_year", "<m8[Y]", np.array([1, -2], dtype="m8[Y]"))
check("period_nat", "<m8[us]", np.array([12, np.iinfo("i8").min], dtype="i8").view("m8[us]"))

# --- structured ---------------------------------------------------------------
struct = np.zeros(2, dtype=[("a", "<i4"), ("b", "<f8", (3,)), ("c", "<U2")])
struct["a"] = [7, 8]
struct["b"] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
struct["c"] = ["xy", "z"]
check("struct", "|V36", struct)

nested = np.zeros(2, dtype=[("x", [("u", "<i4"), ("v", "<f4")]), ("y", "<i8")])
nested["x"]["u"] = [1, 2]
nested["x"]["v"] = [0.5, 1.5]
nested["y"] = [-10, 20]
check("struct_nested", "|V16", nested)

# --- npz archives -------------------------------------------------------------
for archive, compressed in (("all.npz", False), ("all_compressed.npz", True)):
    z = np.load(os.path.join(d, archive))
    if sorted(z.files) != ["a", "s", "t"]:
        failures.append(f"{archive}: keys {sorted(z.files)}")
    else:
        if not np.array_equal(z["a"], np.arange(6, dtype="<f8").reshape(2, 3, order="F")):
            failures.append(f"{archive}: 'a' differs")
        if not np.array_equal(z["s"], np.array(["one", "two"], dtype="U3")):
            failures.append(f"{archive}: 's' differs")
        if z["t"].dtype.str != "<M8[D]" or z["t"][0] != np.datetime64("2020-02-29"):
            failures.append(f"{archive}: 't' differs")
    import zipfile
    with zipfile.ZipFile(os.path.join(d, archive)) as zf:
        methods = {i.compress_type for i in zf.infolist()}
        want = {8} if compressed else {0}
        if methods != want:
            failures.append(f"{archive}: compression methods {methods} != {want}")
        if sorted(i.filename for i in zf.infolist()) != ["a.npy", "s.npy", "t.npy"]:
            failures.append(f"{archive}: member names {[i.filename for i in zf.infolist()]}")

# --- header shape numpy itself accepts ---------------------------------------
with open(os.path.join(d, "mat.npy"), "rb") as f:
    head = f.read(128)
if not head.startswith(b"\x93NUMPY\x01\x00"):
    failures.append(f"mat.npy: unexpected magic/version {head[:8]!r}")
if len(head) % 64 != 0 or head[:128].find(b"\n") + 1 != 128:
    failures.append("mat.npy: header is not padded to a 64-byte boundary")

if failures:
    print("\n".join(failures), file=sys.stderr)
    sys.exit(1)
print(f"ok ({np.__version__})")
