"""The Warp-native entry points accept device wp.arrays and return device
wp.arrays, matching the numpy shims and scipy."""
import numpy as np
import warp as wp
from scipy.spatial import ConvexHull, Delaunay

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.hull import convex_hull, convex_hull_wp
from ffhull.delaunay import delaunay_2d, delaunay_2d_wp

wp.init()
DEV = "cuda:0"


def _triset(t):
    return set(frozenset(int(i) for i in tt) for tt in t)


def test_hull_wp_device_in_device_out():
    P = np.random.default_rng(0).standard_normal((20000, 3))
    ref = set(map(int, ConvexHull(P).vertices))

    pv = wp.array(np.ascontiguousarray(P), dtype=wp.vec3d, device=DEV)
    faces = convex_hull_wp(pv, device=DEV)
    assert isinstance(faces, wp.array) and faces.dtype == wp.vec3i
    assert str(faces.device) == DEV
    assert set(map(int, np.unique(faces.numpy()))) == ref


def test_hull_wp_accepts_float32():
    P = np.random.default_rng(1).standard_normal((15000, 3))
    ref = set(map(int, ConvexHull(P).vertices))
    pf = wp.array(np.ascontiguousarray(P).astype(np.float32), dtype=wp.vec3f, device=DEV)
    faces = convex_hull_wp(pf, device=DEV)
    assert set(map(int, np.unique(faces.numpy()))) == ref


def test_hull_wp_return_vertices_sorted_device():
    P = np.random.default_rng(2).standard_normal((12000, 3))
    ref = set(map(int, ConvexHull(P).vertices))
    pv = wp.array(np.ascontiguousarray(P), dtype=wp.vec3d, device=DEV)
    faces, verts = convex_hull_wp(pv, device=DEV, return_vertices=True)
    assert isinstance(verts, wp.array) and verts.dtype == wp.int32
    v = verts.numpy()
    assert np.all(np.diff(v) > 0)                 # ascending, unique
    assert set(map(int, v)) == ref


def test_shim_matches_core():
    P = np.random.default_rng(3).standard_normal((8000, 3))
    pv = wp.array(np.ascontiguousarray(P), dtype=wp.vec3d, device=DEV)
    a = set(map(int, np.unique(convex_hull(P, device=DEV))))
    b = set(map(int, np.unique(convex_hull_wp(pv, device=DEV).numpy())))
    assert a == b == set(map(int, ConvexHull(P).vertices))


def test_delaunay_wp_device_in_device_out():
    P = np.random.default_rng(4).standard_normal((600, 2))
    exp = _triset(Delaunay(P).simplices)
    pv = wp.array(np.ascontiguousarray(P), dtype=wp.vec2d, device=DEV)
    tris = delaunay_2d_wp(pv, device=DEV)
    assert isinstance(tris, wp.array) and tris.dtype == wp.vec3i
    T = tris.numpy()
    assert _triset(T) == exp
    a = P[T[:, 0]]; b = P[T[:, 1]]; c = P[T[:, 2]]
    area2 = (b[:, 0]-a[:, 0])*(c[:, 1]-a[:, 1]) - (b[:, 1]-a[:, 1])*(c[:, 0]-a[:, 0])
    assert np.all(area2 > 0)                       # CCW


def test_delaunay_wp_accepts_float32():
    P = np.random.default_rng(5).standard_normal((500, 2))
    exp = _triset(Delaunay(P).simplices)
    pf = wp.array(np.ascontiguousarray(P).astype(np.float32), dtype=wp.vec2f, device=DEV)
    assert _triset(delaunay_2d_wp(pf, device=DEV).numpy()) == exp


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(name, "OK")
