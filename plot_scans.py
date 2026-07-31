"""Benchmark ffHull vs qhull on the FULL threedscans dataset (134 models, most
zipped) and produce a scatter plot of runtime vs input size.

    python3 plot_scans.py          # benchmark (resumable) + write the figure
    python3 plot_scans.py --plot   # just re-plot from the cached CSV

Results are checkpointed to media/scans_results.csv so a re-run resumes.
Needs: matplotlib, huggingface_hub, trimesh, scipy.
"""
import os
import sys
import gc
import csv
import time
import zipfile
import tempfile
import numpy as np
import warp as wp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffhull.hull import convex_hull, clear_pool

DEV = "cuda:0"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "media")
CSV = os.path.join(OUT, "scans_results.csv")
os.makedirs(OUT, exist_ok=True)
MESH_EXT = (".stl", ".obj", ".ply", ".off")


def sources():
    """Yield (model_name, hf_filename) for every mesh (loose STLs + zips)."""
    from huggingface_hub import list_repo_files
    files = list_repo_files("alecjacobson/threedscans", repo_type="dataset")
    for f in sorted(files):
        low = f.lower()
        if low.endswith(MESH_EXT):
            yield os.path.splitext(os.path.basename(f))[0], f
        elif low.endswith(".zip"):
            base = os.path.basename(f)[:-4]                 # strip .zip
            yield os.path.splitext(base)[0], f              # strip mesh ext too


def load_vertices(hf_file):
    from huggingface_hub import hf_hub_download
    import trimesh
    path = hf_hub_download("alecjacobson/threedscans", hf_file, repo_type="dataset")
    tmp = None
    if hf_file.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            inner = [i for i in zf.namelist()
                     if i.lower().endswith(MESH_EXT) and "__MACOSX" not in i]
            if not inner:
                return None
            ext = os.path.splitext(inner[0])[1]
            fd, tmp = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, "wb") as out:
                out.write(zf.read(inner[0]))
            path = tmp
    try:
        m = trimesh.load(path, process=False, force="mesh")
        V = np.ascontiguousarray(np.asarray(m.vertices, dtype=np.float64))
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return V if V.ndim == 2 and V.shape[1] == 3 and len(V) >= 4 else None


def run():
    from scipy.spatial import ConvexHull
    done = set()
    if os.path.exists(CSV):
        with open(CSV) as fh:
            done = {r["model"] for r in csv.DictReader(fh)}
    new = not os.path.exists(CSV)
    fh = open(CSV, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["model", "n", "hullV", "ffhull_ms", "qhull_ms"]); fh.flush()

    todo = [(nm, f) for nm, f in sources() if nm not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    for i, (name, hf_file) in enumerate(todo):
        try:
            V = load_vertices(hf_file)
            if V is None:
                print(f"  [skip] {name}: no mesh"); continue
            n = len(V)
            convex_hull(V, device=DEV); wp.synchronize()
            t = []
            for _ in range(3):
                wp.synchronize(); t0 = time.perf_counter()
                F = convex_hull(V, device=DEV)
                wp.synchronize(); t.append(time.perf_counter() - t0)
            g = min(t)
            t0 = time.perf_counter(); H = ConvexHull(V); cpu = time.perf_counter() - t0
            w.writerow([name, n, len(H.vertices), f"{g*1e3:.3f}", f"{cpu*1e3:.3f}"]); fh.flush()
            print(f"  [{i+1}/{len(todo)}] {name:34s} n={n:>9,} ffHull={g*1e3:7.1f}ms "
                  f"qhull={cpu*1e3:8.1f}ms {cpu/g:5.1f}x", flush=True)
        except Exception as e:
            print(f"  [err] {name}: {type(e).__name__}: {str(e)[:80]}", flush=True)
        finally:
            clear_pool(); gc.collect()
    fh.close()


def plot():
    rows = []
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["n"]), float(r["ffhull_ms"]), float(r["qhull_ms"])))
    n = np.array([r[0] for r in rows]); ff = np.array([r[1] for r in rows])
    qh = np.array([r[2] for r in rows]); sp = qh / ff
    print(f"{len(rows)} models | speedup min={sp.min():.1f}x  median={np.median(sp):.1f}x  "
          f"max={sp.max():.1f}x")

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(n, qh, s=22, c="#c44e52", alpha=0.75, label="qhull (CPU)", edgecolors="none")
    ax.scatter(n, ff, s=22, c="#4c72b0", alpha=0.75, label="ffHull (GPU)", edgecolors="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("input points"); ax.set_ylabel("hull time (ms)")
    ax.set_title(f"3D convex hull on threedscans ({len(rows)} models) — ffHull (L40) vs qhull\n"
                 f"speedup: min {sp.min():.1f}× · median {np.median(sp):.1f}× · max {sp.max():.1f}×",
                 fontsize=11)
    ax.legend(loc="upper left"); ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out = os.path.join(OUT, "scans_benchmark.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        run()
    plot()
