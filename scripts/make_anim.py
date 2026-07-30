"""Render a headless polyscope animation of points swirling on a sphere with the
semi-transparent ffHull convex hull recomputed and drawn on top each frame.

Outputs PNG frames to a temp dir, then encodes a gif and an animated webp with
ffmpeg (from imageio-ffmpeg).  Run:  python3 scripts/make_anim.py
"""
import os
import subprocess
import tempfile
import numpy as np
import polyscope as ps
import imageio_ffmpeg

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.hull import convex_hull

N = 220
FRAMES = 90          # one full loop
RES = 720
DEV = "cuda:0"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
os.makedirs(OUT, exist_ok=True)


def sphere_points(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rodrigues(p, axis, ang):
    """Rotate rows of p about a single unit `axis` by per-row angle `ang`."""
    axis = axis / np.linalg.norm(axis)
    k = axis[None, :]
    ca = np.cos(ang)[:, None]; sa = np.sin(ang)[:, None]
    return (p * ca + np.cross(np.broadcast_to(k, p.shape), p) * sa
            + k * (p @ axis)[:, None] * (1 - ca))


def rodrigues_batch(p, axes, ang):
    """Rotate each row of p about its own unit axis by its own angle."""
    ca = np.cos(ang)[:, None]; sa = np.sin(ang)[:, None]
    dot = (p * axes).sum(1)[:, None]
    return p * ca + np.cross(axes, p) * sa + axes * dot * (1 - ca)


def main():
    p0 = sphere_points(N)
    rng = np.random.default_rng(1)
    # per-point swirl: a fixed rotation axis, amplitude and phase -> periodic,
    # loopable relative motion so the hull genuinely changes each frame.
    axes = rng.standard_normal((N, 3)); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    amp = rng.uniform(0.5, 1.4, N)
    phase = rng.uniform(0, 2 * np.pi, N)
    spin_axis = np.array([0.15, 1.0, 0.1]); spin_axis /= np.linalg.norm(spin_axis)

    ps.set_allow_headless_backends(True)
    ps.set_program_name("ffHull")
    ps.init()
    ps.set_window_size(RES, RES)
    ps.set_ground_plane_mode("none")
    ps.set_transparency_mode("pretty")
    if hasattr(ps, "set_SSAA_factor"):
        ps.set_SSAA_factor(2)
    ps.set_view_projection_mode("orthographic")

    tmp = tempfile.mkdtemp()
    for f in range(FRAMES):
        t = f / FRAMES
        ang = amp * np.sin(2 * np.pi * t + phase)           # periodic per-point
        # each point rotates about its own axis, then a slow global spin
        pts = rodrigues_batch(p0, axes, ang)
        pts = rodrigues(pts, spin_axis, np.full(N, 2 * np.pi * t))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)

        faces = convex_hull(np.ascontiguousarray(pts), device=DEV)

        pc = ps.register_point_cloud("points", pts, radius=0.008,
                                     color=(0.15, 0.45, 0.9))
        m = ps.register_surface_mesh("hull", pts, np.ascontiguousarray(faces),
                                     color=(1.0, 0.55, 0.15), edge_width=1.0,
                                     edge_color=(0.25, 0.1, 0.0), transparency=0.35,
                                     back_face_policy="cull")
        m.set_material("wax")
        ps.look_at((0.0, 0.0, 3.2), (0.0, 0.0, 0.0))
        ps.screenshot(os.path.join(tmp, f"f{f:04d}.png"), transparent_bg=False)
        if f % 15 == 0:
            print(f"  frame {f}/{FRAMES}  hull_faces={len(faces)}", flush=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    pat = os.path.join(tmp, "f%04d.png")
    fps = 24
    gif = os.path.join(OUT, "sphere_hull.gif")
    webp = os.path.join(OUT, "sphere_hull.webp")
    pal = os.path.join(tmp, "palette.png")
    vf = f"fps={fps},scale=480:-1:flags=lanczos"
    subprocess.run([ffmpeg, "-y", "-i", pat, "-vf", vf + ",palettegen=stats_mode=diff", pal],
                   check=True, capture_output=True)
    subprocess.run([ffmpeg, "-y", "-i", pat, "-i", pal, "-lavfi",
                    vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", gif],
                   check=True, capture_output=True)
    subprocess.run([ffmpeg, "-y", "-i", pat, "-vf", vf, "-c:v", "libwebp_anim",
                    "-lossless", "0", "-q:v", "70", "-loop", "0", webp],
                   check=True, capture_output=True)
    print("gif  :", gif, os.path.getsize(gif) // 1024, "KB")
    print("webp :", webp, os.path.getsize(webp) // 1024, "KB")


if __name__ == "__main__":
    main()
