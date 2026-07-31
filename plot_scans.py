"""Benchmark ffHull vs qhull on the threedscans dataset and plot the results.

    python3 plot_scans.py        # writes media/scans_benchmark.png

Needs: matplotlib, huggingface_hub, trimesh, scipy.
"""
import os
import time
import numpy as np
import warp as wp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffhull.hull import convex_hull

wp.init()
DEV = "cuda:0"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "media")
os.makedirs(OUT, exist_ok=True)


def run_benchmark():
    from huggingface_hub import hf_hub_download, list_repo_files
    import trimesh
    from scipy.spatial import ConvexHull
    files = sorted(f for f in list_repo_files("alecjacobson/threedscans", repo_type="dataset")
                   if f.lower().endswith(".stl"))
    rows = []
    for f in files:
        p = hf_hub_download("alecjacobson/threedscans", f, repo_type="dataset")
        V = np.ascontiguousarray(trimesh.load(p, process=False).vertices, dtype=np.float64)
        convex_hull(V, device=DEV); wp.synchronize()
        t = []
        for _ in range(3):
            wp.synchronize(); t0 = time.perf_counter()
            F = convex_hull(V, device=DEV)
            wp.synchronize(); t.append(time.perf_counter() - t0)
        g = min(t)
        t0 = time.perf_counter(); H = ConvexHull(V); cpu = time.perf_counter() - t0
        name = f[:-4].replace("_", " ")
        rows.append((name, len(V), len(H.vertices), g * 1e3, cpu * 1e3))
        print(f"  {name:24s} n={len(V):>9,}  ffHull={g*1e3:7.1f}ms  qhull={cpu*1e3:7.1f}ms  "
              f"{cpu/g:5.2f}x", flush=True)
    return rows


def plot(rows):
    rows = sorted(rows, key=lambda r: r[1])          # by point count
    names = [f"{r[0]}\n{r[1]/1e6:.1f}M" for r in rows]
    ff = np.array([r[3] for r in rows])
    qh = np.array([r[4] for r in rows])
    speed = qh / ff
    tot_ff, tot_qh = ff.sum(), qh.sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    y = np.arange(len(rows)); h = 0.38
    ax1.barh(y + h / 2, qh, h, label="qhull (CPU)", color="#c44e52")
    ax1.barh(y - h / 2, ff, h, label="ffHull (GPU)", color="#4c72b0")
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xscale("log"); ax1.set_xlabel("hull time (ms, log scale)")
    ax1.set_title("Per-scan runtime")
    ax1.legend(loc="lower right"); ax1.grid(axis="x", alpha=0.3)
    for yi, (a, b) in enumerate(zip(ff, qh)):
        ax1.text(a * 0.9, yi - h / 2, f"{a:.0f}", va="center", ha="right", fontsize=7, color="#4c72b0")

    ax2.barh(y, speed, color="#55a868")
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("speedup (qhull / ffHull)")
    ax2.set_title("ffHull speedup vs qhull")
    ax2.axvline(tot_qh / tot_ff, ls="--", color="black", lw=1)
    ax2.text(tot_qh / tot_ff, len(rows) - 0.4, f" overall {tot_qh/tot_ff:.1f}x",
             fontsize=9, va="top")
    for yi, s in enumerate(speed):
        ax2.text(s + 0.1, yi, f"{s:.1f}x", va="center", fontsize=8)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("3D convex hull on threedscans (Oliver Laric museum scans) — NVIDIA L40 vs qhull",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(OUT, "scans_benchmark.png")
    fig.savefig(out, dpi=130)
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    print("benchmarking...", flush=True)
    plot(run_benchmark())
