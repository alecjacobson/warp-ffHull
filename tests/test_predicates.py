"""Verify orient3d sign convention and cone test against numpy ground truth."""
import numpy as np
import warp as wp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.predicates import orient3d, in_cone

wp.init()
DEV = "cuda:0"


@wp.kernel
def _orient_k(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d,
              pts: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = orient3d(a, b, c, pts[i])


@wp.kernel
def _cone_k(s: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d,
            pts: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    out[i] = wp.int32(in_cone(s, a, b, c, pts[i]))


def np_orient3d(a, b, c, d):
    return np.linalg.det(np.array([a - d, b - d, c - d]))


def test_orient3d_matches_numpy():
    rng = np.random.default_rng(0)
    a, b, c = rng.standard_normal(3), rng.standard_normal(3), rng.standard_normal(3)
    pts = rng.standard_normal((200, 3))
    out = wp.zeros(200, dtype=wp.float64, device=DEV)
    wp.launch(_orient_k, dim=200,
              inputs=[wp.vec3d(*a), wp.vec3d(*b), wp.vec3d(*c),
                      wp.array(pts, dtype=wp.vec3d, device=DEV), out], device=DEV)
    wp.synchronize()
    got = out.numpy()
    exp = np.array([np_orient3d(a, b, c, p) for p in pts])
    assert np.allclose(got, exp, rtol=1e-9, atol=1e-9), np.abs(got - exp).max()


def test_face_orientation_convention():
    # Tetra with s inside. Orient face so orient3d(face, s) < 0, then a point
    # pushed outward along the face normal must give orient3d(face, p) > 0.
    a = np.array([0.0, 0.0, 0.0]); b = np.array([1.0, 0.0, 0.0]); c = np.array([0.0, 1.0, 0.0])
    s = np.array([0.3, 0.3, -1.0])  # below the z=0 plane
    ds = np_orient3d(a, b, c, s)
    if ds > 0:  # enforce s beneath
        b, c = c, b
        ds = np_orient3d(a, b, c, s)
    assert ds < 0
    outside = np.array([0.3, 0.3, 1.0])  # opposite side from s
    assert np_orient3d(a, b, c, outside) > 0


def test_in_cone_matches_bruteforce():
    rng = np.random.default_rng(1)
    s = np.array([0.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.5]); b = np.array([0.0, 1.0, 0.5]); c = np.array([0.0, 0.0, 1.0])
    pts = rng.standard_normal((500, 3)) * 0.8 + np.array([0.3, 0.3, 0.6])
    out = wp.zeros(len(pts), dtype=wp.int32, device=DEV)
    wp.launch(_cone_k, dim=len(pts),
              inputs=[wp.vec3d(*s), wp.vec3d(*a), wp.vec3d(*b), wp.vec3d(*c),
                      wp.array(pts, dtype=wp.vec3d, device=DEV), out], device=DEV)
    wp.synchronize()
    got = out.numpy().astype(bool)

    # Ground truth: p in cone iff barycentric solve of ray s->p through plane abc
    # yields nonneg coords. Solve p - s = t*(u*(a-s)+v*(b-s)+w*(c-s)) form:
    # express (p - s) in basis (a-s, b-s, c-s); inside iff all coords >= 0.
    M = np.stack([a - s, b - s, c - s], axis=1)  # columns
    Minv = np.linalg.inv(M)
    exp = np.array([np.all(Minv @ (p - s) >= -1e-12) for p in pts])
    mism = np.sum(got != exp)
    assert mism == 0, f"{mism} mismatches"


if __name__ == "__main__":
    test_orient3d_matches_numpy()
    test_face_orientation_convention()
    test_in_cone_matches_bruteforce()
    print("predicates OK")
