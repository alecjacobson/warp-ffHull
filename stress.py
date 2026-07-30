"""Robustness + performance stress harness for ffHull.

Robustness: a large battery of general-position and adversarial/degenerate
inputs, each validated (vs scipy for general position; via a validity check
otherwise).  Performance: size sweeps with GPU timing vs qhull.
"""
import time
import numpy as np
import warp as wp
from scipy.spatial import ConvexHull

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffhull.hull import convex_hull

wp.init()
DEV = "cuda:0"


def valid_hull(pts, faces, tol_rel=1e-7):
    from collections import Counter
    ec = Counter()
    for f in faces:
        for i in range(3):
            ec[frozenset((int(f[(i + 1) % 3]), int(f[(i + 2) % 3])))] += 1
    if any(c != 2 for c in ec.values()):
        return False, "non-manifold"
    # compare extreme-vertex sets by COORDINATE so exact duplicates (which alias
    # to different indices) are not miscounted as missing.
    def coords(idxs):
        return {tuple(np.round(pts[i], 9)) for i in idxs}
    try:
        exp = coords(ConvexHull(pts).vertices.tolist())
    except Exception:
        exp = set()
    vidx = [int(x) for x in np.unique(faces)]
    got = coords(vidx)
    if not exp <= got:
        return False, f"missing {len(exp - got)} extreme verts"
    c = pts[vidx].mean(0)
    scale = np.abs(pts).max() + 1.0
    tol = tol_rel * scale ** 3
    for f in faces:
        a, b, cc = pts[f[0]], pts[f[1]], pts[f[2]]
        n = np.cross(b - a, cc - a)
        if np.dot(n, c - a) > 0:
            n = -n
        if ((pts - a) @ n).max() > tol:
            return False, "point outside"
    return True, "ok"


def exact_match(pts, faces):
    def coords(idxs):
        return {tuple(np.round(pts[i], 9)) for i in idxs}
    return coords(int(x) for x in np.unique(faces)) == coords(ConvexHull(pts).vertices.tolist())


# ----------------------------------------------------------------------------
# Robustness battery
# ----------------------------------------------------------------------------

def gen_cases():
    r = np.random.default_rng
    yield "gaussian-2k", r(0).standard_normal((2000, 3)), True
    yield "gaussian-scaled", r(1).standard_normal((2000, 3)) * [1e3, 1.0, 1e-3], True
    yield "uniform-cube-rand", r(2).uniform(-1, 1, (3000, 3)), True
    x = r(3).standard_normal((3000, 3)); yield "on-sphere", x / np.linalg.norm(x, axis=1, keepdims=True), True
    yield "huge-offset", r(4).standard_normal((2000, 3)) + 1e8, True
    yield "tiny-magnitude", r(5).standard_normal((2000, 3)) * 1e-6, True
    yield "clustered-blobs", np.concatenate([r(6).standard_normal((500, 3)) * 0.2 + c
                                             for c in r(6).standard_normal((6, 3)) * 4]), True
    # thin slabs (near coplanar)
    for k in (1e-2, 1e-4, 1e-6):
        p = r(7).standard_normal((2000, 3)); p[:, 2] *= k
        yield f"thin-slab-{k:.0e}", p, True
    # anisotropic pancake in random orientation
    p = r(8).standard_normal((2000, 3)) * [1.0, 1.0, 1e-5]
    Q, _ = np.linalg.qr(r(8).standard_normal((3, 3)))
    yield "rotated-pancake", p @ Q, True
    # --- degenerate / structured (validity only, extra coplanar verts allowed) ---
    yield "cube-8-corners", np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float), False
    yield "int-grid-5", np.array([[x, y, z] for x in range(5) for y in range(5) for z in range(5)], float), False
    g = []
    rng = r(9)
    for s in (-1., 1.):
        for _ in range(60):
            g += [[s, rng.uniform(-1, 1), rng.uniform(-1, 1)],
                  [rng.uniform(-1, 1), s, rng.uniform(-1, 1)],
                  [rng.uniform(-1, 1), rng.uniform(-1, 1), s]]
    yield "cube-face-samples", np.array(g), False
    yield "sphere+dups", np.concatenate([(lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(
        r(10).standard_normal((1500, 3)))] * 1)[np.r_[np.arange(1500), np.arange(300)]], False
    yield "two-coincident-clusters", np.concatenate([np.zeros((50, 3)) + [1, 0, 0],
                                                     np.zeros((50, 3)) + [-1, 0, 0],
                                                     r(11).standard_normal((200, 3))]), False


def run_robustness():
    print("=== ROBUSTNESS ===")
    npass = 0; total = 0
    for name, pts, general in gen_cases():
        total += 1
        pts = np.ascontiguousarray(pts, float)
        try:
            faces = convex_hull(pts, device=DEV)
        except Exception as e:
            print(f"  {name:22s} n={len(pts):>6d}  EXCEPTION {type(e).__name__}: {e}")
            continue
        ok, why = valid_hull(pts, faces)
        exact = exact_match(pts, faces) if general and ok else None
        tag = "VALID" if ok else f"INVALID({why})"
        extra = "" if exact is None else (" exact" if exact else " valid+extra-verts")
        print(f"  {name:22s} n={len(pts):>6d} F={len(faces):>6d}  {tag}{extra}")
        npass += int(ok)
    print(f"  -> {npass}/{total} valid")


def run_determinism():
    print("=== DETERMINISM (valid every run; topology may vary via atomics) ===")
    rng = np.random.default_rng(0)
    for name, pts in [("gaussian", rng.standard_normal((3000, 3))),
                      ("cube-face", None)]:
        if pts is None:
            g = []
            for s in (-1., 1.):
                for _ in range(40):
                    g += [[s, rng.uniform(-1, 1), rng.uniform(-1, 1)]]
            pts = np.array(g + list(rng.standard_normal((50, 3)) * 2))
        vsets = []
        for _ in range(3):
            f = convex_hull(pts, device=DEV)
            ok, _ = valid_hull(pts, f)
            vsets.append((ok, frozenset(int(x) for x in np.unique(f))))
        allvalid = all(v[0] for v in vsets)
        same = len(set(v[1] for v in vsets)) == 1
        print(f"  {name:12s}: all_valid={allvalid} identical_vertex_set={same}")


# ----------------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------------

def run_perf():
    print("=== PERFORMANCE (ffHull GPU vs qhull CPU) ===")
    rng = np.random.default_rng(0)
    for kind in ("gaussian", "sphere"):
        for n in (10_000, 100_000, 1_000_000, 5_000_000):
            if kind == "sphere":
                x = rng.standard_normal((n, 3)); pts = x / np.linalg.norm(x, axis=1, keepdims=True)
            else:
                pts = rng.standard_normal((n, 3))
            convex_hull(pts, device=DEV); wp.synchronize()
            t = []
            for _ in range(3):
                wp.synchronize(); t0 = time.perf_counter()
                f = convex_hull(pts, device=DEV)
                wp.synchronize(); t.append(time.perf_counter() - t0)
            g = min(t)
            t0 = time.perf_counter(); h = ConvexHull(pts); cpu = time.perf_counter() - t0
            ok = set(map(int, np.unique(f))) == set(h.vertices.tolist())
            print(f"  {kind:8s} n={n:>9d}  ffHull={g*1e3:9.1f}ms  qhull={cpu*1e3:9.1f}ms  "
                  f"speedup={cpu/g:5.2f}x  hullV={len(h.vertices):>7d} ok={ok}", flush=True)


if __name__ == "__main__":
    run_robustness()
    run_determinism()
    run_perf()
