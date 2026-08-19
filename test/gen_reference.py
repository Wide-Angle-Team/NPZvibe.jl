"""Write reference .npy/.npz files for the Julia side to read back.

Everything here is deterministic so that `julia_reads_python.jl` can build the
same values independently instead of trusting NPZvibe's own writer.
"""
import io
import os
import sys
import zipfile

import numpy as np

out = sys.argv[1]


def save(name, arr, **kw):
    np.save(os.path.join(out, name + ".npy"), arr, **kw)


# --- one array per basic dtype ------------------------------------------------
save("bool", np.array([True, False, True]))
for k in ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"):
    info = np.iinfo(k)
    save(k, np.array([info.min, 0, 7, info.max], dtype=k))
save("f2", np.array([1.5, -0.0, np.nan, np.inf, -np.inf], dtype="f2"))
save("f4", np.array([1.5, -0.0, np.nan, np.inf, -np.inf], dtype="f4"))
save("f8", np.array([1.5, -0.0, np.nan, np.inf, -np.inf], dtype="f8"))
save("c8", np.array([1.5 + 2.5j, complex(-0.0, np.inf)], dtype="c8"))
save("c16", np.array([1.5 + 2.5j, np.nan - 3j], dtype="c16"))
save("void3", np.frombuffer(bytes([1, 2, 3, 255, 0, 128]), dtype="V3"))

# --- shapes and memory order --------------------------------------------------
a = np.arange(24, dtype="<i4").reshape(2, 3, 4)
save("c_order", a)
save("f_order", np.asfortranarray(a))
save("zerod", np.array(3.5))
save("empty", np.zeros((0, 3), dtype="f4"))
save("dim5", np.arange(2 * 3 * 4 * 5 * 6, dtype="<f8").reshape(2, 3, 4, 5, 6))

# --- byte order ---------------------------------------------------------------
save("be_bool", np.array([True, False, True], dtype=">b1"))
save("be_i2", np.array([np.iinfo("i2").min, 0, np.iinfo("i2").max], dtype=">i2"))
save("be_i4", np.arange(6, dtype=">i4").reshape(2, 3))
save("be_i8", np.array([0, -1, 2**53], dtype=">i8"))
save("be_u2", np.array([0, 1, 65535], dtype=">u2"))
save("be_u4", np.array([0, 1, np.iinfo("u4").max], dtype=">u4"))
save("be_u8", np.array([0, 1, 2**53], dtype=">u8"))
save("be_f2", np.array([1.5, -0.0, np.nan, np.inf], dtype=">f2"))
save("be_f4", np.array([1.5, -0.0, np.nan, np.inf, -np.inf], dtype=">f4"))
save("be_f8", np.array([1.5, -2.25, np.nan], dtype=">f8"))
save("be_c8", np.array([1.5 + 2.5j, -3 - 4j], dtype=">c8"))
save("be_c16", np.array([1.5 + 2.5j, -3 - 4j], dtype=">c16"))
save("be_m8_s", np.array([0, 90, -1], dtype=">m8[s]"))
save("be_m8_D", np.array([0, 5, -3], dtype=">m8[D]"))
save("be_str_U3", np.array(["ab", "cde"], dtype=">U3"))
save("be_dt_s", np.array(["2020-01-02T03:04:05", "1960-01-01T00:00:00"], dtype=">M8[s]"))

struct_all_be = np.zeros(3, dtype=[("a", ">i4"), ("b", ">f8")])
struct_all_be["a"] = [1, -2, 3]
struct_all_be["b"] = [1.5, np.nan, -0.25]
save("struct_all_be", struct_all_be)

np.savez(os.path.join(out, "stored_be.npz"),
         big_i8=np.array([0, -1, 2**53], dtype=">i8"),
         little_f4=np.array([1.5, -2.25], dtype="<f4"))

# --- strings ------------------------------------------------------------------
save("str_S5", np.array([b"ab", b"cde", b""], dtype="S5"))
save("str_S4_nul", np.array([b"a\x00b"], dtype="S4"))
save("str_S2_raw", np.array([b"\xff\xfe"], dtype="S2"))
save("str_U8", np.array(["", "héllo", "\U0001d418nicode"], dtype="U8"))
save("str_U1_grid", np.array([["a", "b"], ["c", "d"]], dtype="U1"))

# --- datetime64 / timedelta64 -------------------------------------------------
save("dt_Y", np.array(["1970", "2020", "1900"], dtype="M8[Y]"))
save("dt_mon", np.array(["1970-01", "2020-07", "1900-12"], dtype="M8[M]"))
save("dt_W", np.array(["1970-01-01", "2020-01-02"], dtype="M8[W]"))
save("dt_D", np.array(["1970-01-01", "2020-02-29", "1900-01-01"], dtype="M8[D]"))
save("dt_h", np.array(["2020-01-02T03", "1969-12-31T23"], dtype="M8[h]"))
save("dt_min", np.array(["2020-01-02T03:04", "1969-12-31T23:59"], dtype="M8[m]"))
save("dt_s", np.array(["2020-01-02T03:04:05", "1969-12-31T23:59:59"], dtype="M8[s]"))
save("dt_ms", np.array(["2020-01-02T03:04:05.006"], dtype="M8[ms]"))
save("dt_us", np.array(["2020-01-02T03:04:05.000007"], dtype="M8[us]"))
save("dt_ns", np.array(["2020-01-02T03:04:05.000000008", "1969-12-31T23:59:59.999999999"],
                      dtype="M8[ns]"))
save("dt_ns_nat", np.array(["NaT", "2020-01-02T03:04:05.000000008"], dtype="M8[ns]"))
save("dt_D_nat", np.array(["NaT", "2020-02-29"], dtype="M8[D]"))
save("dt_10s", np.array([0, 1, -2], dtype="i8").astype("M8[10s]"))
save("td_D", np.array([0, 5, -3], dtype="m8[D]"))
save("td_s", np.array([0, 90, -1], dtype="m8[s]"))
save("td_ns", np.array([0, 1, -1], dtype="m8[ns]"))
save("td_us_nat", np.array([np.iinfo("i8").min, 12], dtype="i8").view("m8[us]"))
save("td_year", np.array([1, -2], dtype="m8[Y]"))
save("td_ms", np.array([1500, -250], dtype="m8[ms]"))

# --- structured dtypes --------------------------------------------------------
simple = np.zeros(3, dtype=[("a", "<i4"), ("b", "<f8")])
simple["a"] = [1, -2, 3]
simple["b"] = [1.5, np.nan, -0.25]
save("struct_simple", simple)

sub = np.zeros(2, dtype=[("a", "<i4"), ("b", "<f8", (3,)), ("c", "S4")])
sub["a"] = [7, 8]
sub["b"] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
sub["c"] = [b"xy", b"zzzz"]
save("struct_sub", sub)

sub2d = np.zeros(2, dtype=[("m", "<i2", (2, 3))])
sub2d["m"] = np.arange(12, dtype="<i2").reshape(2, 2, 3)
save("struct_sub2d", sub2d)

nested = np.zeros(2, dtype=[("x", [("u", "<i4"), ("v", "<f4")]), ("y", "<i8")])
nested["x"]["u"] = [1, 2]
nested["x"]["v"] = [0.5, 1.5]
nested["y"] = [-10, 20]
save("struct_nested", nested)

padded = np.zeros(2, dtype=np.dtype({"names": ["a", "b"], "formats": ["<i2", "<f8"],
                                     "offsets": [0, 8], "itemsize": 16}))
padded["a"] = [1, 2]
padded["b"] = [3.5, 4.5]
save("struct_padded", padded)

bemixed = np.zeros(2, dtype=[("a", ">i4"), ("b", "<f8"), ("t", "<M8[s]")])
bemixed["a"] = [1, 2]
bemixed["b"] = [1.25, 2.5]
bemixed["t"] = np.array(["2020-01-02T03:04:05", "NaT"], dtype="M8[s]")
save("struct_be_time", bemixed)

save("struct_U", np.array([(1, "héllo"), (2, "")], dtype=[("i", "<i4"), ("s", "<U6")]))

# --- npz archives -------------------------------------------------------------
np.savez(os.path.join(out, "stored.npz"),
         a=np.arange(6, dtype="<f8").reshape(2, 3),
         b=np.array(["x", "yy"], dtype="U2"),
         c=np.asfortranarray(np.arange(4, dtype="<i2").reshape(2, 2)))
np.savez_compressed(os.path.join(out, "deflated.npz"),
                    big=np.arange(10000, dtype="<i4"),
                    t=np.array(["2020-01-02T03:04:05.000000008"], dtype="M8[ns]"))

# An entry whose name lacks the .npy suffix: numpy keys it by the raw name.
buf = io.BytesIO()
np.lib.format.write_array(buf, np.arange(3, dtype="<i8"))
with zipfile.ZipFile(os.path.join(out, "noext.npz"), "w") as z:
    z.writestr("plain", buf.getvalue())

# A ZIP64 archive that is nonetheless small, so the ZIP64 read path is always
# exercised without writing 4 GiB.
buf = io.BytesIO()
np.lib.format.write_array(buf, np.arange(5, dtype="<u1"))
with zipfile.ZipFile(os.path.join(out, "zip64.npz"), "w", allowZip64=True) as z:
    with z.open(zipfile.ZipInfo("small.npy"), "w", force_zip64=True) as f:
        f.write(buf.getvalue())
