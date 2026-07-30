"""Adversarial-input regression tests (subset of stress.py) — assert validity."""
import numpy as np
import warp as wp
from scipy.spatial import ConvexHull

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.hull import convex_hull

wp.init()
DEV = "cuda:0"


def assert_valid(pts, tol_rel=1e-7, exact=False):
    faces = convex_hull(pts, device=DEV)
    from collections import Counter
    ec = Counter()
    for f in faces:
        for i in range(3):
            ec[frozenset((int(f[(i + 1) % 3]), int(f[(i + 2) % 3])))] += 1
    assert all(c == 2 for c in ec.values()), "non-manifold"

    def coords(idxs):
        return {tuple(np.round(pts[i], 9)) for i in idxs}
    vidx = [int(x) for x in np.unique(faces)]
    exp = coords(ConvexHull(pts).vertices.tolist())
    got = coords(vidx)
    assert exp <= got, f"missing extreme verts: {len(exp - got)}"
    if exact:
        assert got == exp, f"not vertex-minimal: extra {len(got - exp)}"
    c = pts[vidx].mean(0)
    scale = np.abs(pts).max() + 1.0
    tol = tol_rel * scale ** 3
    for f in faces:
        a, b, cc = pts[f[0]], pts[f[1]], pts[f[2]]
        n = np.cross(b - a, cc - a)
        if np.dot(n, c - a) > 0:
            n = -n
        assert ((pts - a) @ n).max() <= tol, "point outside hull"


def test_anisotropic_scales():
    r = np.random.default_rng(1)
    assert_valid(r.standard_normal((2000, 3)) * [1e3, 1.0, 1e-3], exact=True)


def test_huge_offset():
    r = np.random.default_rng(4)
    assert_valid(r.standard_normal((2000, 3)) + 1e8, exact=True)


def test_tiny_magnitude():
    r = np.random.default_rng(5)
    assert_valid(r.standard_normal((2000, 3)) * 1e-6, exact=True)


def test_thin_slabs():
    for k in (1e-2, 1e-4, 1e-6):
        p = np.random.default_rng(7).standard_normal((2000, 3)); p[:, 2] *= k
        assert_valid(p, exact=True)


def test_rotated_pancake():
    r = np.random.default_rng(8)
    p = r.standard_normal((2000, 3)) * [1.0, 1.0, 1e-5]
    Q, _ = np.linalg.qr(r.standard_normal((3, 3)))
    assert_valid(p @ Q, exact=True)


def test_duplicates_on_sphere():
    r = np.random.default_rng(10)
    x = r.standard_normal((1500, 3)); s = x / np.linalg.norm(x, axis=1, keepdims=True)
    assert_valid(s[np.r_[np.arange(1500), np.arange(300)]])  # 300 exact dups


def test_coincident_clusters():
    r = np.random.default_rng(11)
    pts = np.concatenate([np.zeros((50, 3)) + [1, 0, 0], np.zeros((50, 3)) + [-1, 0, 0],
                          r.standard_normal((200, 3))])
    assert_valid(pts)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("robustness OK")
