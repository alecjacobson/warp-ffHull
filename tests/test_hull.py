"""Compare ffHull output against scipy.spatial.ConvexHull."""
import numpy as np
import warp as wp
from scipy.spatial import ConvexHull

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.hull import convex_hull, _orient3d_np

wp.init()
DEV = "cuda:0"


def face_set(faces):
    return set(frozenset(int(x) for x in f) for f in faces)


def check_against_scipy(pts, verbose=False):
    faces = convex_hull(pts, device=DEV, verbose=verbose)
    hull = ConvexHull(pts)

    got = face_set(faces)
    exp = face_set(hull.simplices)

    # outward orientation: interior centroid must be beneath every face
    c = pts.mean(axis=0)
    for f in faces:
        assert _orient3d_np(pts[f[0]], pts[f[1]], pts[f[2]], c) < 0, "face not outward"

    # vertex sets must match exactly
    got_v = set(int(x) for f in faces for x in f)
    exp_v = set(hull.vertices.tolist())
    assert got_v == exp_v, f"vertex mismatch: extra={got_v-exp_v} missing={exp_v-got_v}"

    # triangle sets must match (unique triangulation when in general position)
    assert got == exp, (f"face mismatch: n_got={len(got)} n_exp={len(exp)} "
                        f"extra={len(got-exp)} missing={len(exp-got)}")
    return len(faces)


def test_sphere():
    rng = np.random.default_rng(7)
    for n in (200, 800, 2000):
        x = rng.standard_normal((n, 3))
        pts = x / np.linalg.norm(x, axis=1, keepdims=True)
        m = check_against_scipy(pts)
        print(f"sphere n={n}: {m} faces == scipy")


def test_gaussian_ball():
    for seed in range(6):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((1000, 3))
        m = check_against_scipy(pts)
        print(f"gaussian n=1000 seed={seed}: {m} faces == scipy")


def test_uniform_cube():
    for seed in range(4):
        rng = np.random.default_rng(100 + seed)
        pts = rng.uniform(-1, 1, size=(1500, 3))
        m = check_against_scipy(pts)
        print(f"uniform-cube n=1500 seed={seed}: {m} faces == scipy")


def test_clustered():
    # several gaussian blobs: many interior (non-extreme) points to remove
    rng = np.random.default_rng(3)
    blobs = [rng.standard_normal((400, 3)) * 0.3 + c
             for c in rng.standard_normal((5, 3)) * 3]
    pts = np.concatenate(blobs)
    m = check_against_scipy(pts)
    print(f"clustered n={len(pts)}: {m} faces == scipy")


def test_large():
    rng = np.random.default_rng(11)
    pts = rng.standard_normal((50000, 3))
    m = check_against_scipy(pts)
    print(f"gaussian n=50000: {m} faces == scipy")


if __name__ == "__main__":
    test_sphere()
    test_gaussian_ball()
    test_uniform_cube()
    test_clustered()
    test_large()
    print("hull OK")
