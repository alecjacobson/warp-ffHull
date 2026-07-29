"""Benchmark ffHull (GPU) vs scipy/qhull (CPU) on the convex hull."""
import time
import numpy as np
import warp as wp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffhull.hull import convex_hull

wp.init()
DEV = "cuda:0"


def bench(pts, reps=3, label=""):
    # warm up (compile kernels) then time end-to-end (incl. host<->device)
    convex_hull(pts, device=DEV)
    wp.synchronize()
    t = []
    for _ in range(reps):
        wp.synchronize(); t0 = time.perf_counter()
        f = convex_hull(pts, device=DEV)
        wp.synchronize(); t.append(time.perf_counter() - t0)
    gpu = min(t)

    from scipy.spatial import ConvexHull
    t0 = time.perf_counter(); h = ConvexHull(pts); cpu = time.perf_counter() - t0

    ok = set(map(int, np.unique(f))) == set(h.vertices.tolist())
    print(f"{label:22s} n={len(pts):>8d}  ffHull(GPU)={gpu*1e3:8.2f} ms  "
          f"qhull(CPU)={cpu*1e3:8.2f} ms  speedup={cpu/gpu:5.2f}x  hull_v={len(h.vertices):>5d}  ok={ok}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("== Gaussian ball (few hull vertices) ==")
    for n in (10_000, 100_000, 1_000_000, 4_000_000):
        bench(rng.standard_normal((n, 3)), label="gaussian")
    print("== On sphere (all points extreme) ==")
    for n in (10_000, 100_000, 1_000_000):
        x = rng.standard_normal((n, 3)); x /= np.linalg.norm(x, axis=1, keepdims=True)
        bench(x, label="sphere")
