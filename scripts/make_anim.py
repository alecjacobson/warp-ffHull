"""Render a headless polyscope animation of points drifting gently on a sphere
with the semi-transparent ffHull convex hull recomputed and drawn on top each
frame, in a clean "Keenan Crane" style (soft contact shadow, matte clay
material, tasteful palette, perspective camera).

Outputs an animated webp (and a gif) to media/.  Run: python3 scripts/make_anim.py
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
FRAMES = 140         # one seamless loop
FPS = 25
RES = 900
DEV = "cuda:0"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
os.makedirs(OUT, exist_ok=True)

# palette (soft periwinkle surface, warm-gold points, deep-blue edges)
SURF = (0.60, 0.68, 0.86)
EDGE = (0.18, 0.22, 0.38)
PTS = (0.96, 0.72, 0.22)


def sphere_points(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rodrigues(p, axis, ang):
    axis = axis / np.linalg.norm(axis)
    k = axis[None, :]
    ca = np.cos(ang)[:, None]; sa = np.sin(ang)[:, None]
    return (p * ca + np.cross(np.broadcast_to(k, p.shape), p) * sa
            + k * (p @ axis)[:, None] * (1 - ca))


def rodrigues_batch(p, axes, ang):
    ca = np.cos(ang)[:, None]; sa = np.sin(ang)[:, None]
    dot = (p * axes).sum(1)[:, None]
    return p * ca + np.cross(axes, p) * sa + axes * dot * (1 - ca)


def main():
    p0 = sphere_points(N)
    rng = np.random.default_rng(1)
    axes = rng.standard_normal((N, 3)); axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    amp = rng.uniform(0.10, 0.38, N)          # gentle per-point swing
    phase = rng.uniform(0, 2 * np.pi, N)
    spin_axis = np.array([0.0, 1.0, 0.0])     # slow full turn about up

    ps.set_allow_headless_backends(True)
    ps.set_program_name("ffHull")
    ps.init()
    ps.set_window_size(RES, RES)
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_SSAA_factor(4)
    ps.set_view_projection_mode("perspective")
    ps.set_up_dir("y_up")
    ps.set_front_dir("z_front")
    # soft contact shadow on a clean invisible ground plane
    ps.set_ground_plane_mode("shadow_only")
    ps.set_shadow_darkness(0.32)
    ps.set_shadow_blur_iters(8)
    try:
        ps.set_ground_plane_height_mode(ps.GroundPlaneHeightMode.manual)
        ps.set_ground_plane_height(-1.02)
    except Exception:
        pass

    tmp = tempfile.mkdtemp()
    for f in range(FRAMES):
        t = f / FRAMES
        ang = amp * np.sin(2 * np.pi * t + phase)
        pts = rodrigues_batch(p0, axes, ang)
        pts = rodrigues(pts, spin_axis, np.full(N, 2 * np.pi * t))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)

        faces = convex_hull(np.ascontiguousarray(pts), device=DEV)

        m = ps.register_surface_mesh("hull", pts, np.ascontiguousarray(faces),
                                     color=SURF, edge_width=0.75, edge_color=EDGE,
                                     transparency=0.62, back_face_policy="cull",
                                     smooth_shade=False)
        m.set_material("clay")
        pc = ps.register_point_cloud("points", pts, radius=0.0115, color=PTS)
        pc.set_material("clay")

        ps.look_at((2.1, 1.5, 2.6), (0.0, -0.05, 0.0))
        ps.screenshot(os.path.join(tmp, f"f{f:04d}.png"), transparent_bg=False)
        if f % 20 == 0:
            print(f"  frame {f}/{FRAMES}  hull_faces={len(faces)}", flush=True)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    pat = os.path.join(tmp, "f%04d.png")
    gif = os.path.join(OUT, "sphere_hull.gif")
    webp = os.path.join(OUT, "sphere_hull.webp")
    pal = os.path.join(tmp, "palette.png")
    vf = f"fps={FPS},scale=520:-1:flags=lanczos"
    subprocess.run([ffmpeg, "-y", "-i", pat, "-vf", vf + ",palettegen=stats_mode=diff", pal],
                   check=True, capture_output=True)
    subprocess.run([ffmpeg, "-y", "-i", pat, "-i", pal, "-lavfi",
                    vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", gif],
                   check=True, capture_output=True)
    subprocess.run([ffmpeg, "-y", "-i", pat, "-vf", vf, "-c:v", "libwebp_anim",
                    "-lossless", "0", "-q:v", "72", "-loop", "0", webp],
                   check=True, capture_output=True)
    print("gif  :", os.path.getsize(gif) // 1024, "KB")
    print("webp :", os.path.getsize(webp) // 1024, "KB")


if __name__ == "__main__":
    main()
