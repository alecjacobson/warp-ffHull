"""Unit-test the robust orient3d sign (certified filter + SoS)."""
import numpy as np
import warp as wp
import itertools

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.predicates import o3d_sign

wp.init()
DEV = "cuda:0"


@wp.kernel
def _sign_k(pts: wp.array(dtype=wp.vec3d), idx: wp.array(dtype=wp.vec4i),
            out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    ix = idx[i]
    out[i] = o3d_sign(pts[ix[0]], pts[ix[1]], pts[ix[2]], pts[ix[3]],
                      ix[0], ix[1], ix[2], ix[3])


def signs(pts, quads):
    p = wp.array(pts, dtype=wp.vec3d, device=DEV)
    q = wp.array(np.array(quads, np.int32), dtype=wp.vec4i, device=DEV)
    out = wp.zeros(len(quads), dtype=wp.int32, device=DEV)
    wp.launch(_sign_k, dim=len(quads), inputs=[p, q, out], device=DEV)
    wp.synchronize()
    return out.numpy()


def test_matches_float_sign_when_nonzero():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((60, 3))
    quads = [tuple(rng.choice(60, 4, replace=False)) for _ in range(400)]
    got = signs(pts, quads)
    for (a, b, c, d), g in zip(quads, got):
        det = np.linalg.det(np.stack([pts[a] - pts[d], pts[b] - pts[d], pts[c] - pts[d]]))
        assert g in (-1, 1)
        if abs(det) > 1e-9:
            assert g == (1 if det > 0 else -1)


def test_never_zero_and_antisymmetric():
    # 5 coplanar points + generic ones: predicate must always be +-1 and flip
    # sign under a single argument swap.
    rng = np.random.default_rng(1)
    base = rng.standard_normal((20, 3))
    base[:8, 2] = 0.0  # 8 coplanar (z=0)
    quads = [tuple(rng.choice(20, 4, replace=False)) for _ in range(300)]
    s = signs(base, quads)
    assert set(s.tolist()) <= {-1, 1}
    # antisymmetry: swap args 0,1 -> sign flips
    swapped = [(b, a, c, d) for (a, b, c, d) in quads]
    s2 = signs(base, swapped)
    assert np.all(s == -s2)


def test_coplanar_facet_consistent():
    # points exactly on planes (grid on a cube face): predicate stays nonzero
    g = np.array([[x, y, 0.0] for x in range(4) for y in range(4)], float)
    quads = list(itertools.combinations(range(len(g)), 4))[:200]
    s = signs(g, quads)
    assert set(s.tolist()) <= {-1, 1}  # never 0 despite total coplanarity


if __name__ == "__main__":
    test_matches_float_sign_when_nonzero()
    test_never_zero_and_antisymmetric()
    test_coplanar_facet_consistent()
    print("SoS predicate OK")
