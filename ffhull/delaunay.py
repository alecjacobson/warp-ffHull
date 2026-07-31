"""2D Delaunay triangulation via the lifting map.

Lift each 2D point ``(x, y)`` to the paraboloid ``(x, y, x^2 + y^2)`` in 3D.
The **lower** faces of the 3D convex hull of the lifted points (those whose
outward normal points downward) project straight back to the Delaunay
triangulation of the original 2D points.  So this is a thin wrapper over the
pure-Warp convex hull: lift, hull, keep the downward-facing faces.

``delaunay_2d_wp`` is the warp-native entry point (device 2D points in, device
triangles out -- lifting, lower-envelope classification, and CCW orientation all
run as Warp kernels).  ``delaunay_2d`` is the numpy convenience shim.
"""

import numpy as np
import warp as wp

from .hull import convex_hull, convex_hull_wp


@wp.kernel
def _cast_to_vec2d(src: wp.array(dtype=wp.vec2f), dst: wp.array(dtype=wp.vec2d)):
    i = wp.tid()
    p = src[i]
    dst[i] = wp.vec2d(wp.float64(p[0]), wp.float64(p[1]))


@wp.kernel
def _lift(pts2: wp.array(dtype=wp.vec2d), out: wp.array(dtype=wp.vec3d)):
    i = wp.tid()
    p = pts2[i]
    out[i] = wp.vec3d(p[0], p[1], p[0] * p[0] + p[1] * p[1])


@wp.kernel
def _accum(pts: wp.array(dtype=wp.vec3d), acc: wp.array(dtype=wp.float64)):
    i = wp.tid()
    p = pts[i]
    wp.atomic_add(acc, 0, p[0])
    wp.atomic_add(acc, 1, p[1])
    wp.atomic_add(acc, 2, p[2])


@wp.kernel
def _emit_lower_ccw(faces: wp.array(dtype=wp.vec3i), lift: wp.array(dtype=wp.vec3d),
                    centroid: wp.vec3d, pts2: wp.array(dtype=wp.vec2d),
                    out: wp.array(dtype=wp.vec3i), out_n: wp.array(dtype=wp.int32)):
    # a hull face is on the lower envelope iff its outward normal points down;
    # emit such faces re-wound counter-clockwise in 2D.
    t = wp.tid()
    f = faces[t]
    a = lift[f[0]]; b = lift[f[1]]; c = lift[f[2]]
    nrm = wp.cross(b - a, c - a)
    nz = nrm[2]
    if wp.dot(nrm, a - centroid) < wp.float64(0.0):   # make the normal outward
        nz = -nz
    if nz < wp.float64(0.0):                            # outward normal points down
        pa = pts2[f[0]]; pb = pts2[f[1]]; pc = pts2[f[2]]
        area2 = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        j = wp.atomic_add(out_n, 0, 1)
        if area2 < wp.float64(0.0):
            out[j] = wp.vec3i(f[0], f[2], f[1])
        else:
            out[j] = f


def _as_pts2d_wp(points_2d, device):
    if isinstance(points_2d, wp.array):
        dev = str(points_2d.device)
        if points_2d.ndim != 1:
            raise TypeError("expected a 1-D wp.array of wp.vec2d / wp.vec2f points")
        if points_2d.dtype == wp.vec2d:
            return points_2d, dev
        if points_2d.dtype == wp.vec2f:
            out = wp.empty(int(points_2d.shape[0]), dtype=wp.vec2d, device=points_2d.device)
            wp.launch(_cast_to_vec2d, dim=int(points_2d.shape[0]), inputs=[points_2d, out],
                      device=points_2d.device)
            return out, dev
        raise TypeError(f"unsupported wp.array dtype {points_2d.dtype}; "
                        "pass wp.vec2d or wp.vec2f")
    device = device or "cuda:0"
    arr = np.ascontiguousarray(points_2d, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 2, "expected an (n, 2) array"
    return wp.array(arr, dtype=wp.vec2d, device=device), device


def delaunay_2d_wp(points_2d, device=None):
    """Warp-native 2D Delaunay: **device 2D points in, device triangles out**.

    ``points_2d`` is a device ``wp.array(dtype=wp.vec2d)`` (or ``wp.vec2f``, cast
    on the device).  Returns a device ``wp.array(dtype=wp.vec3i)`` of CCW triangle
    vertex indices into ``points_2d``.  The lift, hull, lower-envelope test, and
    CCW re-winding all run on the GPU; only the lift centroid (3 scalars) and the
    final triangle count cross to the host.
    """
    P, device = _as_pts2d_wp(points_2d, device)
    n = int(P.shape[0])
    lift = wp.empty(n, dtype=wp.vec3d, device=device)
    wp.launch(_lift, dim=n, inputs=[P, lift], device=device)

    faces = convex_hull_wp(lift, device=device)          # device (m,) vec3i into P
    m = int(faces.shape[0])

    acc = wp.zeros(3, dtype=wp.float64, device=device)
    wp.launch(_accum, dim=n, inputs=[lift, acc], device=device)
    a = acc.numpy() / n
    centroid = wp.vec3d(float(a[0]), float(a[1]), float(a[2]))

    out = wp.empty(m, dtype=wp.vec3i, device=device)
    out_n = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch(_emit_lower_ccw, dim=m, inputs=[faces, lift, centroid, P, out, out_n],
              device=device)
    k = int(out_n.numpy()[0])
    return out[:k]


# ---------------------------------------------------------------------------
# numpy convenience shim (and the return_lifted path used for visualisation)
# ---------------------------------------------------------------------------

def _orient2d_ccw(pts2d, tris):
    """Reorient each triangle counter-clockwise in 2D (host)."""
    a = pts2d[tris[:, 0]]; b = pts2d[tris[:, 1]]; c = pts2d[tris[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = area2 < 0
    out = tris.copy()
    out[flip, 1], out[flip, 2] = tris[flip, 2], tris[flip, 1]
    return out


def delaunay_2d(points_2d, device="cuda:0", return_lifted=False):
    """Convenience numpy shim over :func:`delaunay_2d_wp`.

    Returns an ``(m, 3)`` int array of CCW triangle vertex indices into
    ``points_2d``.  For GPU-resident 2D points, call ``delaunay_2d_wp`` directly
    to keep the whole pipeline on the device.

    With ``return_lifted=True`` also returns ``(lifted_points, hull_faces,
    is_lower)`` for the underlying 3D hull (handy for visualising the lifting);
    that path is computed on the host.
    """
    if not return_lifted:
        return delaunay_2d_wp(points_2d, device=device).numpy()

    P = np.ascontiguousarray(points_2d, dtype=np.float64)
    assert P.ndim == 2 and P.shape[1] == 2
    lift = np.column_stack([P, (P ** 2).sum(axis=1)])          # (x, y, x^2+y^2)
    faces = convex_hull(np.ascontiguousarray(lift), device=device)   # indices into lift == P
    a = lift[faces[:, 0]]; b = lift[faces[:, 1]]; c = lift[faces[:, 2]]
    nrm = np.cross(b - a, c - a)
    centroid = lift.mean(axis=0)
    outward = np.einsum("ij,ij->i", nrm, a - centroid)         # >0 already outward
    nz = np.where(outward >= 0, nrm[:, 2], -nrm[:, 2])          # outward normal z
    is_lower = nz < 0
    tris = _orient2d_ccw(P, faces[is_lower])
    return tris, lift, faces, is_lower
