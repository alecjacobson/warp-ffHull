"""Performance regression tests: assert ffHull keeps beating qhull by a safe
margin on hull-heavy inputs (catches gross slowdowns).  Also prints timings so
regressions are visible in the log.  Set FFHULL_PERF_BASELINE=1 to only print.

Baseline (L40, before fp32-filter optimization, commit 3b037b4):
    sphere   n= 200000  ffHull=133.3ms  speedup=7.31x
    sphere   n=1000000  ffHull=779.7ms  speedup=7.65x
    gaussian n= 200000  ffHull= 17.1ms  speedup=1.72x
    gaussian n=1000000  ffHull= 45.6ms  speedup=3.27x
"""
import os
import time
import numpy as np
import warp as wp
from scipy.spatial import ConvexHull

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.hull import convex_hull

wp.init()
DEV = "cuda:0"


def _time(pts, reps=3):
    convex_hull(pts, device=DEV); wp.synchronize()
    t = []
    for _ in range(reps):
        wp.synchronize(); t0 = time.perf_counter()
        f = convex_hull(pts, device=DEV)
        wp.synchronize(); t.append(time.perf_counter() - t0)
    g = min(t)
    t0 = time.perf_counter(); h = ConvexHull(pts); cpu = time.perf_counter() - t0
    ok = set(map(int, np.unique(f))) == set(h.vertices.tolist())
    return g, cpu, ok


def _sphere(n, seed=0):
    x = np.random.default_rng(seed).standard_normal((n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_sphere_speedup():
    # hull-heavy (all points extreme): ffHull should be comfortably faster.
    for n, min_speedup in [(200_000, 2.0), (1_000_000, 3.0)]:
        g, cpu, ok = _time(_sphere(n))
        print(f"[perf] sphere n={n:>8d}  ffHull={g*1e3:8.1f}ms  qhull={cpu*1e3:8.1f}ms  "
              f"speedup={cpu/g:5.2f}x  ok={ok}", flush=True)
        assert ok, "sphere hull incorrect"
        if not os.environ.get("FFHULL_PERF_BASELINE"):
            assert cpu / g >= min_speedup, f"sphere n={n} speedup {cpu/g:.2f}x < {min_speedup}x"


def test_gaussian_records():
    # tiny-hull case (qhull's best): just record, no hard assert.
    for n in (200_000, 1_000_000):
        g, cpu, ok = _time(np.random.default_rng(1).standard_normal((n, 3)))
        print(f"[perf] gaussian n={n:>8d}  ffHull={g*1e3:8.1f}ms  qhull={cpu*1e3:8.1f}ms  "
              f"speedup={cpu/g:5.2f}x  ok={ok}", flush=True)
        assert ok, "gaussian hull incorrect"


if __name__ == "__main__":
    test_sphere_speedup()
    test_gaussian_records()
    print("perf OK")
