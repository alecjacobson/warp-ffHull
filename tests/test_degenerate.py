"""Degenerate / lower-dimensional inputs.

The float64 predicate path is exact enough for full-dimensional inputs down to
aspect ratios of ~1e6.  Lower-dimensional inputs (coincident/collinear/coplanar)
are dispatched to host handlers.  Exactly-coplanar *facets* (grids, cube face
samples) and extreme aspect ratios need the exact Shewchuk predicate + SoS
(see warp_orient3d_plan.md) and are marked xfail until that lands.
"""
import numpy as np
import warp as wp
import pytest
from scipy.spatial import ConvexHull

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.hull import convex_hull

wp.init()
DEV = "cuda:0"


def test_coincident():
    pts = np.ones((10, 3)) * 3.0
    f, v = convex_hull(pts, device=DEV, return_vertices=True)
    assert len(v) == 1 and len(f) == 0


def test_collinear():
    t = np.linspace(0, 1, 50)
    pts = np.outer(t, [1.0, 2.0, 3.0]) + [5, 5, 5]
    f, v = convex_hull(pts, device=DEV, return_vertices=True)
    assert sorted(v.tolist()) == [0, 49]


def test_coplanar_2d():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((200, 2))
    pts = np.c_[q, np.full(200, 2.0)]
    f, v = convex_hull(pts, device=DEV, return_vertices=True)
    exp = set(ConvexHull(q).vertices.tolist())
    assert set(v.tolist()) == exp


def test_duplicates():
    rng = np.random.default_rng(2)
    p = rng.standard_normal((300, 3))
    pts = np.concatenate([p, p[:80]])  # 80 exact duplicates
    f, v = convex_hull(pts, device=DEV, return_vertices=True)
    h = ConvexHull(pts)
    # every true hull vertex is found (a duplicate may stand in for its twin)
    hv = set(h.vertices.tolist())
    got = set(int(x) for x in np.unique(f))
    coords_got = {tuple(pts[i]) for i in got}
    coords_exp = {tuple(pts[i]) for i in hv}
    assert coords_got == coords_exp


def test_thin_slab_ok_to_1e6():
    rng = np.random.default_rng(1)
    for thick in (1e-1, 1e-3, 1e-6):
        pts = rng.standard_normal((2000, 3)); pts[:, 2] *= thick
        f = convex_hull(pts, device=DEV)
        got = set(int(x) for x in np.unique(f))
        exp = set(ConvexHull(pts).vertices.tolist())
        assert got == exp, f"thick={thick}"


def _assert_valid_hull(pts, faces, tol_rel=1e-9):
    """A valid convex hull: closed manifold, every true extreme vertex present,
    and no input point outside any (outward-oriented) face.  Coplanar-facet
    points may appear as extra simplicial vertices — that is still valid."""
    from collections import Counter
    ec = Counter()
    for f in faces:
        for i in range(3):
            ec[frozenset((int(f[(i + 1) % 3]), int(f[(i + 2) % 3])))] += 1
    assert all(c == 2 for c in ec.values()), "non-manifold"
    # no missing extreme vertices
    exp = set(ConvexHull(pts).vertices.tolist())
    got = set(int(x) for x in np.unique(faces))
    assert exp <= got, f"missing extreme vertices: {exp - got}"
    # containment: orient each face outward via the vertex centroid, check all in
    c = pts[list(got)].mean(0)
    scale = np.abs(pts).max() + 1.0
    tol = tol_rel * scale ** 3
    for f in faces:
        a, b, cc = pts[f[0]], pts[f[1]], pts[f[2]]
        n = np.cross(b - a, cc - a)
        if np.dot(n, c - a) > 0:
            n = -n
        assert ((pts - a) @ n).max() <= tol, "point outside hull"


def test_cube_face_samples():
    # points sampled on the 6 faces of a cube: heavy exact-coplanar degeneracy.
    for seed in range(3):
        rng = np.random.default_rng(seed)
        g = []
        for s in (-1.0, 1.0):
            for _ in range(40):
                g.append([s, rng.uniform(-1, 1), rng.uniform(-1, 1)])
                g.append([rng.uniform(-1, 1), s, rng.uniform(-1, 1)])
                g.append([rng.uniform(-1, 1), rng.uniform(-1, 1), s])
        pts = np.array(g)
        f = convex_hull(pts, device=DEV)
        _assert_valid_hull(pts, f)


def test_integer_grid_lattice():
    # dense integer lattice inside a box: maximal coplanar degeneracy
    g = np.array([[x, y, z] for x in range(5) for y in range(5) for z in range(5)],
                 dtype=float)
    f = convex_hull(g, device=DEV)
    _assert_valid_hull(g, f)


if __name__ == "__main__":
    test_coincident(); test_collinear(); test_coplanar_2d()
    test_duplicates(); test_thin_slab_ok_to_1e6()
    test_cube_face_samples(); test_integer_grid_lattice()
    print("degenerate OK")
