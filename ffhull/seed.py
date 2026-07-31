"""GPU-resident seed construction: extreme-point search, tetra-vertex selection,
and affine-dimension test — all via Warp reductions over the device point array.

``build_seed_gpu`` runs the entire construction on the device (farthest pair,
line/plane argmax, dimension classification, coordinate scale) and returns the 4
tetra vertices as **device** arrays; only two O(1) scalars (``dim`` and
``scale``) are read back, so no point coordinates ever touch the host on the
full-dimensional path.
"""

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


# ===========================================================================
# Fully GPU-resident seed: farthest pair, line/plane argmax, tetra-vertex
# selection, dimension test, and scale all run as Warp kernels reading device
# scalars.  No host O(n) work and no per-point readback -- only two O(1) scalars
# (dim, scale) come back, so the initial tetrahedron is built entirely on the
# GPU (see init_tetra_gpu).
# ===========================================================================

@wp.kernel
def farthest_pair(ext: wp.array(dtype=wp.vec3d), ext_idx: wp.array(dtype=wp.int32),
                  seed_idx: wp.array(dtype=wp.int32), seed_pts: wp.array(dtype=wp.vec3d),
                  uvec: wp.array(dtype=wp.vec3d), dsq: wp.array(dtype=wp.float64)):
    # single thread: farthest-apart pair among the 6 axis extremes -> p0, p1, u
    best = wp.float64(-1.0)
    bi = wp.int32(0)
    bj = wp.int32(1)
    for i in range(6):
        for j in range(i + 1, 6):
            d = ext[i] - ext[j]
            dd = wp.dot(d, d)
            if dd > best:
                best = dd
                bi = i
                bj = j
    seed_idx[0] = ext_idx[bi]
    seed_idx[1] = ext_idx[bj]
    seed_pts[0] = ext[bi]
    seed_pts[1] = ext[bj]
    u = ext[bj] - ext[bi]
    ln = wp.sqrt(wp.dot(u, u))
    if ln > wp.float64(0.0):
        u = u / ln
    uvec[0] = u
    dsq[0] = best


@wp.kernel
def line_dist_val_d(points: wp.array(dtype=wp.vec3d), a_arr: wp.array(dtype=wp.vec3d),
                    u_arr: wp.array(dtype=wp.vec3d), best: wp.array(dtype=wp.float64)):
    i = wp.tid()
    a = a_arr[0]
    u = u_arr[0]
    w = points[i] - a
    t = wp.dot(w, u)
    perp = w - t * u
    wp.atomic_max(best, 0, wp.dot(perp, perp))


@wp.kernel
def line_dist_idx_d(points: wp.array(dtype=wp.vec3d), a_arr: wp.array(dtype=wp.vec3d),
                    u_arr: wp.array(dtype=wp.vec3d), best: wp.array(dtype=wp.float64),
                    idx: wp.array(dtype=wp.int32)):
    i = wp.tid()
    a = a_arr[0]
    u = u_arr[0]
    w = points[i] - a
    t = wp.dot(w, u)
    perp = w - t * u
    if wp.dot(perp, perp) == best[0]:
        wp.atomic_min(idx, 0, i)


@wp.kernel
def set_p2(points: wp.array(dtype=wp.vec3d), i2: wp.array(dtype=wp.int32),
           seed_idx: wp.array(dtype=wp.int32), seed_pts: wp.array(dtype=wp.vec3d),
           uvec: wp.array(dtype=wp.vec3d), nrm: wp.array(dtype=wp.vec3d)):
    # single thread: record p2 and the plane normal of (p0, p1, p2)
    k = i2[0]
    p2 = points[k]
    seed_idx[2] = k
    seed_pts[2] = p2
    a = seed_pts[0]
    u = uvec[0]
    vperp = p2 - a
    vperp = vperp - wp.dot(vperp, u) * u
    vl = wp.sqrt(wp.dot(vperp, vperp))
    if vl > wp.float64(0.0):
        vperp = vperp / vl
    nn = wp.cross(u, vperp)
    nl = wp.sqrt(wp.dot(nn, nn))
    if nl > wp.float64(0.0):
        nn = nn / nl
    nrm[0] = nn


@wp.kernel
def plane_dist_val_d(points: wp.array(dtype=wp.vec3d), a_arr: wp.array(dtype=wp.vec3d),
                     nrm: wp.array(dtype=wp.vec3d), best: wp.array(dtype=wp.float64)):
    i = wp.tid()
    d = wp.dot(points[i] - a_arr[0], nrm[0])
    wp.atomic_max(best, 0, d * d)


@wp.kernel
def plane_dist_idx_d(points: wp.array(dtype=wp.vec3d), a_arr: wp.array(dtype=wp.vec3d),
                     nrm: wp.array(dtype=wp.vec3d), best: wp.array(dtype=wp.float64),
                     idx: wp.array(dtype=wp.int32)):
    i = wp.tid()
    d = wp.dot(points[i] - a_arr[0], nrm[0])
    if d * d == best[0]:
        wp.atomic_min(idx, 0, i)


@wp.kernel
def set_p3(points: wp.array(dtype=wp.vec3d), i3: wp.array(dtype=wp.int32),
           seed_idx: wp.array(dtype=wp.int32), seed_pts: wp.array(dtype=wp.vec3d)):
    k = i3[0]
    seed_idx[3] = k
    seed_pts[3] = points[k]


@wp.kernel
def seed_scale_k(seed_pts: wp.array(dtype=wp.vec3d), out: wp.array(dtype=wp.float64)):
    # max |coord| over the 4 seed points, + 1 (matches convex_hull's host scale)
    m = wp.float64(0.0)
    for k in range(4):
        p = seed_pts[k]
        m = wp.max(m, wp.abs(p[0]))
        m = wp.max(m, wp.abs(p[1]))
        m = wp.max(m, wp.abs(p[2]))
    out[0] = m + wp.float64(1.0)


@wp.kernel
def classify_dim(dsq: wp.array(dtype=wp.float64), d2: wp.array(dtype=wp.float64),
                 d3: wp.array(dtype=wp.float64), scale: wp.array(dtype=wp.float64),
                 tol_rel: wp.float64, dim: wp.array(dtype=wp.int32)):
    tol = tol_rel * scale[0]
    tol2 = tol * tol
    if dsq[0] <= tol2:
        dim[0] = 0
    elif d2[0] <= tol2:
        dim[0] = 1
    elif d3[0] <= tol2:
        dim[0] = 2
    else:
        dim[0] = 3


def build_seed_gpu(points_wp, n, dev, tol_rel=1e-12):
    """Fully GPU-resident seed.

    Returns ``(dim, seed_idx, seed_pts, scale)`` where ``seed_idx`` (int32[4])
    and ``seed_pts`` (vec3d[4]) are **device** arrays holding the tetra vertices
    (valid entries = ``dim + 1``), and ``dim``/``scale`` are the only values read
    back to the host.  For ``dim == 3`` no point coordinates ever touch the host.
    """
    vals = wp.array([BIG, -BIG, BIG, -BIG, BIG, -BIG], dtype=wp.float64, device=dev)
    idx6 = wp.full(6, INT_MAX, dtype=wp.int32, device=dev)
    wp.launch(axis_minmax_vals, dim=n, inputs=[points_wp, vals], device=dev)
    wp.launch(axis_minmax_idx, dim=n, inputs=[points_wp, vals, idx6], device=dev)
    ext = wp.zeros(6, dtype=wp.vec3d, device=dev)
    wp.launch(gather_points, dim=6, inputs=[points_wp, idx6, ext], device=dev)

    seed_idx = wp.full(4, INT_MAX, dtype=wp.int32, device=dev)
    seed_pts = wp.zeros(4, dtype=wp.vec3d, device=dev)
    uvec = wp.zeros(1, dtype=wp.vec3d, device=dev)
    nrm = wp.zeros(1, dtype=wp.vec3d, device=dev)
    dsq = wp.zeros(1, dtype=wp.float64, device=dev)

    # p0, p1, direction u
    wp.launch(farthest_pair, dim=1, inputs=[ext, idx6, seed_idx, seed_pts, uvec, dsq], device=dev)

    # p2: farthest from line (p0, u)
    best2 = wp.array([-1.0], dtype=wp.float64, device=dev)
    i2 = wp.full(1, INT_MAX, dtype=wp.int32, device=dev)
    wp.launch(line_dist_val_d, dim=n, inputs=[points_wp, seed_pts, uvec, best2], device=dev)
    wp.launch(line_dist_idx_d, dim=n, inputs=[points_wp, seed_pts, uvec, best2, i2], device=dev)
    wp.launch(set_p2, dim=1, inputs=[points_wp, i2, seed_idx, seed_pts, uvec, nrm], device=dev)

    # p3: farthest from plane (p0, nrm)
    best3 = wp.array([-1.0], dtype=wp.float64, device=dev)
    i3 = wp.full(1, INT_MAX, dtype=wp.int32, device=dev)
    wp.launch(plane_dist_val_d, dim=n, inputs=[points_wp, seed_pts, nrm, best3], device=dev)
    wp.launch(plane_dist_idx_d, dim=n, inputs=[points_wp, seed_pts, nrm, best3, i3], device=dev)
    wp.launch(set_p3, dim=1, inputs=[points_wp, i3, seed_idx, seed_pts], device=dev)

    scale = wp.zeros(1, dtype=wp.float64, device=dev)
    wp.launch(seed_scale_k, dim=1, inputs=[seed_pts, scale], device=dev)
    dimw = wp.zeros(1, dtype=wp.int32, device=dev)
    wp.launch(classify_dim, dim=1,
              inputs=[dsq, best2, best3, scale, wp.float64(tol_rel), dimw], device=dev)
    return int(dimw.numpy()[0]), seed_idx, seed_pts, float(scale.numpy()[0])
