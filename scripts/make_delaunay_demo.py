"""Render the lifting view of 2D Delaunay triangulation, headless polyscope.

The Delaunay triangulation of 2D points is the projection of the *lower* faces
of the 3D convex hull of the points lifted onto the paraboloid z = x^2 + y^2.
This draws all three pieces of that fact in one clean scene:

  * the lifted paraboloid hull, with its LOWER envelope (the Delaunay faces)
    colored gold and the remaining UPPER faces slate + translucent,
  * the extracted flat 2D Delaunay triangulation sitting below the bowl,
  * thin drop-lines from each lifted vertex to its flat projection.

Outputs media/delaunay_lift.png.  Run: python3 scripts/make_delaunay_demo.py
"""
import os
import sys
import numpy as np
import polyscope as ps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ffhull.delaunay import delaunay_2d

RES = 1100
DEV = "cuda:0"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media")
os.makedirs(OUT, exist_ok=True)

LOWER = (0.95, 0.70, 0.20)     # Delaunay (lower) envelope
UPPER = (0.62, 0.70, 0.88)     # rest of the lifted hull
FLAT = (0.98, 0.86, 0.58)      # flat 2D triangulation fill (light gold)
EDGE = (0.16, 0.19, 0.30)
PTC = (0.20, 0.23, 0.34)


def main():
    # a modest, legible point set (blue-noise-ish jitter on a disk)
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 2 * np.pi, 90)
    r = np.sqrt(rng.uniform(0, 1, 90))
    P = np.column_stack([r * np.cos(a), r * np.sin(a)]) * 1.25

    tris, lift, faces, is_lower = delaunay_2d(P, device=DEV, return_lifted=True)
    print(f"{len(P)} points -> {len(faces)} hull faces "
          f"({int(is_lower.sum())} lower = {len(tris)} Delaunay triangles)")

    # scale the paraboloid height down a touch so the bowl isn't too tall
    lift = lift.copy()
    lift[:, 2] *= 0.6
    base_z = -0.9                                  # flat triangulation plane
    flat = np.column_stack([P, np.full(len(P), base_z)])

    # polyscope wants CCW = front; ffHull winds faces the other way
    fl = np.ascontiguousarray(faces[is_lower][:, ::-1])
    fu = np.ascontiguousarray(faces[~is_lower][:, ::-1])

    ps.set_allow_headless_backends(True)
    ps.set_program_name("ffHull delaunay")
    ps.init()
    ps.set_window_size(RES, int(RES * 0.82))
    ps.set_background_color((1.0, 1.0, 1.0))
    ps.set_SSAA_factor(4)
    ps.set_view_projection_mode("perspective")
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_shadow_darkness(0.25)
    ps.set_shadow_blur_iters(8)
    try:
        ps.set_ground_plane_height_mode(ps.GroundPlaneHeightMode.manual)
        ps.set_ground_plane_height(base_z - 0.02)
    except Exception:
        pass

    low = ps.register_surface_mesh("lower envelope (Delaunay faces)", lift, fl,
                                   color=LOWER, edge_width=1.0, edge_color=EDGE,
                                   back_face_policy="identical", smooth_shade=False)
    low.set_material("clay")
    up = ps.register_surface_mesh("upper faces", lift, fu,
                                  color=UPPER, edge_width=1.0, edge_color=EDGE,
                                  transparency=0.40, back_face_policy="cull",
                                  smooth_shade=False)
    up.set_material("clay")

    d2 = ps.register_surface_mesh("2D Delaunay triangulation",
                                  flat, np.ascontiguousarray(tris),
                                  color=FLAT, edge_width=0.45, edge_color=EDGE,
                                  back_face_policy="identical", smooth_shade=False)
    d2.set_material("flat")

    pc = ps.register_point_cloud("lifted points", lift, radius=0.010, color=PTC)
    pc.set_material("clay")

    # drop-lines: lifted vertex -> its flat projection
    seg_nodes = np.vstack([lift, flat])
    m = len(P)
    seg_edges = np.column_stack([np.arange(m), np.arange(m) + m])
    dl = ps.register_curve_network("projection", seg_nodes, seg_edges, radius=0.0016,
                                   color=(0.7, 0.7, 0.74))
    dl.set_transparency(0.5)

    ps.look_at((2.5, -3.1, 3.7), (0.0, 0.0, -0.25))
    out = os.path.join(OUT, "delaunay_lift.png")
    ps.screenshot(out, transparent_bg=False)
    print("wrote", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
