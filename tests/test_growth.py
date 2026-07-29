"""Validate Phase A: the grown polyhedron is a closed, star-shaped manifold
that contains all input points."""
import numpy as np
import warp as wp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.hull import build_star, live_faces, _orient3d_np

wp.init()
DEV = "cuda:0"


def _validate_star(pts, mesh, s):
    faces = live_faces(mesh)
    sv = np.array([float(s[0]), float(s[1]), float(s[2])])
    m = len(faces)

    # 1. closed 2-manifold: every undirected edge shared by exactly 2 faces
    from collections import Counter
    ec = Counter()
    for f in faces:
        for i in range(3):
            ec[frozenset((int(f[(i + 1) % 3]), int(f[(i + 2) % 3])))] += 1
    bad = [e for e, c in ec.items() if c != 2]
    assert not bad, f"{len(bad)} non-manifold edges"

    # Euler characteristic V - E + F = 2 (sphere)
    V = len(set(int(x) for f in faces for x in f))
    E = len(ec)
    assert V - E + m == 2, f"Euler V-E+F = {V-E+m} (V={V} E={E} F={m})"

    # 2. star-shaped: kernel s strictly beneath every face
    for f in faces:
        a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
        assert _orient3d_np(a, b, c, sv) < 0, "face not oriented outward wrt s"

    # 3. contains all points: each point must be beneath the face whose cone
    # (from s) contains it. (A star-shaped polyhedron is generally non-convex,
    # so a point may sit in the outer half-space of unrelated faces.)
    scale = np.abs(pts).max() + 1.0
    tol = 1e-6 * scale ** 3
    A = pts[faces[:, 0]]; B = pts[faces[:, 1]]; C = pts[faces[:, 2]]  # (F,3)

    def det3(P, Q, R, X):  # orient3d(P,Q,R,X) batched over faces for one point X
        M = np.stack([P - X, Q - X, R - X], axis=1)  # (F,3,3)
        return np.linalg.det(M)

    worst = -1e18
    for p in pts:
        # cone membership: same sign as third vertex on all three s-walls
        def wall(P, Q, ref):
            M = np.stack([np.broadcast_to(sv, P.shape) - p, P - p, Q - p], axis=1)
            return np.linalg.det(M)
        w_ab = wall(A, B, C)
        w_bc = wall(B, C, A)
        w_ca = wall(C, A, B)
        # reference = orient3d(s, a, b, c) etc (independent of p)
        def ref(P, Q, R):
            M = np.stack([np.broadcast_to(sv, P.shape) - R, P - R, Q - R], axis=1)
            return np.linalg.det(M)
        r_ab = ref(A, B, C); r_bc = ref(B, C, A); r_ca = ref(C, A, B)
        same = (np.sign(w_ab) * np.sign(r_ab) >= 0) & \
               (np.sign(w_bc) * np.sign(r_bc) >= 0) & \
               (np.sign(w_ca) * np.sign(r_ca) >= 0)
        idx = np.where(same)[0]
        if len(idx) == 0:
            continue  # on a boundary; skip
        # orient3d(face, p) for candidate faces; point must be beneath one
        vals = det3(A[idx], B[idx], C[idx], p)
        worst = max(worst, vals.min())
    assert worst <= tol, f"some point outside hull by {worst} (tol {tol})"
    return V, E, m


def test_growth_random():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((2000, 3))
        mesh, s, tetra = build_star(pts, device=DEV)
        V, E, F = _validate_star(pts, mesh, s)
        print(f"seed {seed}: star V={V} E={E} F={F}")


def test_growth_sphere():
    # points ON a sphere: all extreme, growth should already be near-complete
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1500, 3))
    pts = x / np.linalg.norm(x, axis=1, keepdims=True)
    mesh, s, tetra = build_star(pts, device=DEV)
    V, E, F = _validate_star(pts, mesh, s)
    print(f"sphere: star V={V} E={E} F={F}")


if __name__ == "__main__":
    test_growth_random()
    test_growth_sphere()
    print("growth OK")
