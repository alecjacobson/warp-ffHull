"""GPU-resident seed construction: extreme-point search, affine-dimension test,
and initial tetrahedron — all via Warp reductions over the device point array.

Replaces the earlier host-numpy ``choose_tetra`` / ``analyze_dimension`` O(n)
passes (which dominated end-to-end time) with a handful of atomic reductions,
keeping the point data on the GPU.  Only a few scalars are read back.
"""

import numpy as np
import warp as wp

BIG = wp.constant(wp.float64(1.0e300))
INT_MAX = wp.constant(wp.int32(2147483647))


# --- axis extremes (6 points: min/max along x,y,z) -------------------------

@wp.kernel
def axis_minmax_vals(points: wp.array(dtype=wp.vec3d), vals: wp.array(dtype=wp.float64)):
    i = wp.tid()
    p = points[i]
    wp.atomic_min(vals, 0, p[0]); wp.atomic_max(vals, 1, p[0])
    wp.atomic_min(vals, 2, p[1]); wp.atomic_max(vals, 3, p[1])
    wp.atomic_min(vals, 4, p[2]); wp.atomic_max(vals, 5, p[2])


@wp.kernel
def axis_minmax_idx(points: wp.array(dtype=wp.vec3d), vals: wp.array(dtype=wp.float64),
                    idx: wp.array(dtype=wp.int32)):
    i = wp.tid()
    p = points[i]
    if p[0] == vals[0]: wp.atomic_min(idx, 0, i)
    if p[0] == vals[1]: wp.atomic_min(idx, 1, i)
    if p[1] == vals[2]: wp.atomic_min(idx, 2, i)
    if p[1] == vals[3]: wp.atomic_min(idx, 3, i)
    if p[2] == vals[4]: wp.atomic_min(idx, 4, i)
    if p[2] == vals[5]: wp.atomic_min(idx, 5, i)


@wp.kernel
def gather_points(points: wp.array(dtype=wp.vec3d), idx: wp.array(dtype=wp.int32),
                  out: wp.array(dtype=wp.vec3d)):
    k = wp.tid()
    out[k] = points[idx[k]]


# --- farthest-from-line and farthest-from-plane (two-pass argmax) -----------

@wp.kernel
def line_dist_val(points: wp.array(dtype=wp.vec3d), a: wp.vec3d, u: wp.vec3d,
                  best: wp.array(dtype=wp.float64)):
    i = wp.tid()
    w = points[i] - a
    t = wp.dot(w, u)
    perp = w - t * u
    wp.atomic_max(best, 0, wp.dot(perp, perp))


@wp.kernel
def line_dist_idx(points: wp.array(dtype=wp.vec3d), a: wp.vec3d, u: wp.vec3d,
                  best: wp.array(dtype=wp.float64), idx: wp.array(dtype=wp.int32)):
    i = wp.tid()
    w = points[i] - a
    t = wp.dot(w, u)
    perp = w - t * u
    if wp.dot(perp, perp) == best[0]:
        wp.atomic_min(idx, 0, i)


@wp.kernel
def plane_dist_val(points: wp.array(dtype=wp.vec3d), a: wp.vec3d, nrm: wp.vec3d,
                   best: wp.array(dtype=wp.float64)):
    i = wp.tid()
    d = wp.dot(points[i] - a, nrm)
    wp.atomic_max(best, 0, d * d)


@wp.kernel
def plane_dist_idx(points: wp.array(dtype=wp.vec3d), a: wp.vec3d, nrm: wp.vec3d,
                   best: wp.array(dtype=wp.float64), idx: wp.array(dtype=wp.int32)):
    i = wp.tid()
    d = wp.dot(points[i] - a, nrm)
    if d * d == best[0]:
        wp.atomic_min(idx, 0, i)


def _argmax_line(points, a, u, dev, n):
    best = wp.array([-1.0], dtype=wp.float64, device=dev)
    idx = wp.full(1, INT_MAX, dtype=wp.int32, device=dev)
    av = wp.vec3d(float(a[0]), float(a[1]), float(a[2]))
    uv = wp.vec3d(float(u[0]), float(u[1]), float(u[2]))
    wp.launch(line_dist_val, dim=n, inputs=[points, av, uv, best], device=dev)
    wp.launch(line_dist_idx, dim=n, inputs=[points, av, uv, best, idx], device=dev)
    return int(idx.numpy()[0]), float(best.numpy()[0])


def _argmax_plane(points, a, nrm, dev, n):
    best = wp.array([-1.0], dtype=wp.float64, device=dev)
    idx = wp.full(1, INT_MAX, dtype=wp.int32, device=dev)
    av = wp.vec3d(float(a[0]), float(a[1]), float(a[2]))
    nv = wp.vec3d(float(nrm[0]), float(nrm[1]), float(nrm[2]))
    wp.launch(plane_dist_val, dim=n, inputs=[points, av, nv, best], device=dev)
    wp.launch(plane_dist_idx, dim=n, inputs=[points, av, nv, best, idx], device=dev)
    return int(idx.numpy()[0]), float(best.numpy()[0])


def build_seed(points_wp, n, dev, tol_rel=1e-12):
    """Find 4 well-separated extreme points and classify affine dimension.

    Returns (dim, tetra_idx, seed_points) where ``seed_points`` are the host
    coordinates of the 4 chosen points (or fewer for lower dimensions).
    """
    vals = wp.array([BIG, -BIG, BIG, -BIG, BIG, -BIG], dtype=wp.float64, device=dev)
    idx6 = wp.full(6, INT_MAX, dtype=wp.int32, device=dev)
    wp.launch(axis_minmax_vals, dim=n, inputs=[points_wp, vals], device=dev)
    wp.launch(axis_minmax_idx, dim=n, inputs=[points_wp, vals, idx6], device=dev)
    ext = wp.zeros(6, dtype=wp.vec3d, device=dev)
    wp.launch(gather_points, dim=6, inputs=[points_wp, idx6, ext], device=dev)
    ext_np = ext.numpy()
    ext_idx = idx6.numpy()
    v = vals.numpy()
    scale = float(max(abs(v).max(), 1.0))
    tol = tol_rel * scale

    # p0, p1: farthest-apart pair among the six axis extremes
    best = (-1.0, 0, 0)
    for i in range(6):
        for j in range(i + 1, 6):
            d = float(np.sum((ext_np[i] - ext_np[j]) ** 2))
            if d > best[0]:
                best = (d, i, j)
    if best[0] <= tol * tol:
        return 0, [int(ext_idx[0])], ext_np[0:1]
    i0 = int(ext_idx[best[1]]); i1 = int(ext_idx[best[2]])
    p0 = ext_np[best[1]]; p1 = ext_np[best[2]]
    u = p1 - p0
    u = u / np.linalg.norm(u)

    i2, d2sq = _argmax_line(points_wp, p0, u, dev, n)
    if np.sqrt(d2sq) <= tol:
        return 1, [i0, i1], np.stack([p0, p1])
    # p2 coord
    p2 = _one_point(points_wp, i2, dev)
    vperp = p2 - p0
    vperp = vperp - np.dot(vperp, u) * u
    nrm = np.cross(u, vperp / np.linalg.norm(vperp))
    nrm = nrm / np.linalg.norm(nrm)

    i3, d3sq = _argmax_plane(points_wp, p0, nrm, dev, n)
    if np.sqrt(d3sq) <= tol:
        return 2, [i0, i1, i2], np.stack([p0, p1, p2])
    p3 = _one_point(points_wp, i3, dev)
    return 3, [i0, i1, i2, i3], np.stack([p0, p1, p2, p3])


def _one_point(points_wp, i, dev):
    out = wp.zeros(1, dtype=wp.vec3d, device=dev)
    idx = wp.array([i], dtype=wp.int32, device=dev)
    wp.launch(gather_points, dim=1, inputs=[points_wp, idx, out], device=dev)
    return out.numpy()[0]
