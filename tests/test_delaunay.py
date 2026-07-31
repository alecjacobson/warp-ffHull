"""2D Delaunay (via lifting) must match scipy.spatial.Delaunay in general position."""
import numpy as np
import warp as wp
from scipy.spatial import Delaunay

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.delaunay import delaunay_2d

wp.init()
DEV = "cuda:0"


def _triset(tris):
    return set(frozenset(int(i) for i in t) for t in tris)


def _empty_circumcircle(P, tris):
    """No point lies strictly inside any triangle's circumcircle (the Delaunay
    property), via the in-circle determinant."""
    for t in tris:
        a, b, c = P[t[0]], P[t[1]], P[t[2]]
        M = np.array([[a[0]-P[:,0], a[1]-P[:,1], (a[0]-P[:,0])**2+(a[1]-P[:,1])**2],
                      [b[0]-P[:,0], b[1]-P[:,1], (b[0]-P[:,0])**2+(b[1]-P[:,1])**2],
                      [c[0]-P[:,0], c[1]-P[:,1], (c[0]-P[:,0])**2+(c[1]-P[:,1])**2]])
        # det per point
        d = (M[0,0]*(M[1,1]*M[2,2]-M[1,2]*M[2,1])
             - M[0,1]*(M[1,0]*M[2,2]-M[1,2]*M[2,0])
             + M[0,2]*(M[1,0]*M[2,1]-M[1,1]*M[2,0]))
        # CCW triangle: interior-of-circumcircle => det>0; allow the 3 verts (~0)
        assert d.max() <= 1e-6 * (np.abs(P).max()**4 + 1), "non-Delaunay triangle"


def test_matches_scipy_random():
    for seed in range(4):
        P = np.random.default_rng(seed).standard_normal((500, 2))
        tris = delaunay_2d(P, device=DEV)
        exp = _triset(Delaunay(P).simplices)
        assert _triset(tris) == exp, f"seed {seed}: {len(_triset(tris))} vs {len(exp)}"


def test_delaunay_property_large():
    P = np.random.default_rng(7).uniform(-1, 1, (3000, 2))
    tris = delaunay_2d(P, device=DEV)
    # every triangle CCW and empty-circumcircle
    a = P[tris[:, 0]]; b = P[tris[:, 1]]; c = P[tris[:, 2]]
    area2 = (b[:,0]-a[:,0])*(c[:,1]-a[:,1]) - (b[:,1]-a[:,1])*(c[:,0]-a[:,0])
    assert np.all(area2 > 0), "not all CCW"
    _empty_circumcircle(P, tris)


if __name__ == "__main__":
    test_matches_scipy_random()
    test_delaunay_property_large()
    print("delaunay OK")
