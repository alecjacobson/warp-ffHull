"""Benchmark ffHull vs qhull on the threedscans dataset (Oliver Laric's
high-resolution museum scans, mirrored at alecjacobson/threedscans on the
Hugging Face Hub).  Needs: huggingface_hub, trimesh, scipy.

    python3 bench_scans.py
"""
import time
import numpy as np
import warp as wp

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffhull.hull import convex_hull

wp.init()
DEV = "cuda:0"


def coords(V, idx):
    return {tuple(np.round(V[i], 4)) for i in idx}


def main():
    from huggingface_hub import hf_hub_download, list_repo_files
    import trimesh
    from scipy.spatial import ConvexHull

    files = sorted(f for f in list_repo_files("alecjacobson/threedscans", repo_type="dataset")
                   if f.lower().endswith(".stl"))
    hdr = (f"{'scan':22s} {'points':>10s} {'hullV':>6s} {'miss':>4s} {'extra':>5s} "
           f"{'status':>7s} {'ffHull':>9s} {'qhull':>9s} {'speedup':>7s}")
    print(hdr); print("-" * len(hdr), flush=True)
    tot_g = tot_c = 0.0
    for f in files:
        p = hf_hub_download("alecjacobson/threedscans", f, repo_type="dataset")
        V = np.ascontiguousarray(trimesh.load(p, process=False).vertices, dtype=np.float64)
        n = len(V)
        convex_hull(V, device=DEV); wp.synchronize()
        ts = []
        for _ in range(3):
            wp.synchronize(); t0 = time.perf_counter()
            F = convex_hull(V, device=DEV)
            wp.synchronize(); ts.append(time.perf_counter() - t0)
        g = min(ts)
        t0 = time.perf_counter(); H = ConvexHull(V); cpu = time.perf_counter() - t0
        ours = coords(V, np.unique(F)); qh = coords(V, H.vertices)
        miss = len(qh - ours); extra = len(ours - qh)
        status = "INVALID" if miss > 0 else ("EXACT" if extra == 0 else "VALID")
        tot_g += g; tot_c += cpu
        print(f"{f:22s} {n:>10,} {len(ours):>6,} {miss:>4d} {extra:>5d} {status:>7s} "
              f"{g*1e3:>8.0f}m {cpu*1e3:>8.0f}m {cpu/g:>6.2f}x", flush=True)
    print("-" * len(hdr))
    print(f"{'TOTAL':22s} {'':>10s} {'':>6s} {'':>4s} {'':>5s} {'':>7s} "
          f"{tot_g*1e3:>8.0f}m {tot_c*1e3:>8.0f}m {tot_c/tot_g:>6.2f}x")


if __name__ == "__main__":
    main()
