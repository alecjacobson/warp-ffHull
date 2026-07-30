"""The conservative interior cull must never change the hull (only speed it up
by discarding provably-interior points)."""
import numpy as np
import warp as wp
from scipy.spatial import ConvexHull

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.hull import convex_hull

wp.init()
DEV = "cuda:0"


def _same(pts, faces_a, faces_b):
    def c(f):
        return {tuple(np.round(pts[i], 6)) for t in f for i in t}
    return c(faces_a) == c(faces_b)


def _matches_scipy(pts, faces):
    def c(idx):
        return {tuple(np.round(pts[i], 6)) for i in idx}
    return c(np.unique(faces)) == c(ConvexHull(pts).vertices)


def test_filter_matches_unfiltered_and_scipy():
    cases = {
        "gaussian": np.random.default_rng(0).standard_normal((200_000, 3)),
        "uniform_cube": np.random.default_rng(1).uniform(-1, 1, (200_000, 3)),
        "two_blobs": np.concatenate([
            np.random.default_rng(2).standard_normal((100_000, 3)) * 0.3 + [3, 0, 0],
            np.random.default_rng(3).standard_normal((100_000, 3)) * 0.3 - [3, 0, 0]]),
    }
    for name, pts in cases.items():
        pts = np.ascontiguousarray(pts)
        f_on = convex_hull(pts, device=DEV, filter=True)
        f_off = convex_hull(pts, device=DEV, filter=False)
        assert _same(pts, f_on, f_off), f"{name}: filter changed the hull"
        assert _matches_scipy(pts, f_on), f"{name}: filtered hull != scipy"


def test_filter_below_threshold_is_noop():
    # small inputs skip the filter entirely; result still correct
    pts = np.ascontiguousarray(np.random.default_rng(5).standard_normal((2000, 3)))
    assert _matches_scipy(pts, convex_hull(pts, device=DEV, filter=True))


if __name__ == "__main__":
    test_filter_matches_unfiltered_and_scipy()
    test_filter_below_threshold_is_noop()
    print("filter OK")
