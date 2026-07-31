"""2D Delaunay triangulation via the lifting map.

Lift each 2D point ``(x, y)`` to the paraboloid ``(x, y, x^2 + y^2)`` in 3D.
The **lower** faces of the 3D convex hull of the lifted points (those whose
outward normal points downward) project straight back to the Delaunay
triangulation of the original 2D points.  So this is a thin wrapper over the
pure-Warp ``convex_hull``: lift, hull, keep the downward-facing faces.
"""

import numpy as np

from .hull import convex_hull


def _orient2d_ccw(pts2d, tris):
    """Reorient each triangle counter-clockwise in 2D."""
    a = pts2d[tris[:, 0]]; b = pts2d[tris[:, 1]]; c = pts2d[tris[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = area2 < 0
    out = tris.copy()
    out[flip, 1], out[flip, 2] = tris[flip, 2], tris[flip, 1]
    return out


def delaunay_2d(points_2d, device="cuda:0", return_lifted=False):
    """Delaunay triangulation of 2D points.

    Returns an (m, 3) int array of triangle vertex indices into ``points_2d``
    (counter-clockwise).  With ``return_lifted=True`` also returns
    ``(lifted_points, hull_faces, is_lower)`` for the underlying 3D hull, which
    is handy for visualising the lifting.
    """
    P = np.ascontiguousarray(points_2d, dtype=np.float64)
    assert P.ndim == 2 and P.shape[1] == 2
    lift = np.column_stack([P, (P ** 2).sum(axis=1)])          # (x, y, x^2+y^2)

    faces = convex_hull(np.ascontiguousarray(lift), device=device)   # indices into lift == P

    # A hull face is on the lower envelope iff its outward normal points down.
    a = lift[faces[:, 0]]; b = lift[faces[:, 1]]; c = lift[faces[:, 2]]
    nrm = np.cross(b - a, c - a)
    centroid = lift.mean(axis=0)
    outward = np.einsum("ij,ij->i", nrm, a - centroid)         # >0 already outward
    nz = np.where(outward >= 0, nrm[:, 2], -nrm[:, 2])          # outward normal z
    is_lower = nz < 0

    tris = _orient2d_ccw(P, faces[is_lower])
    if return_lifted:
        return tris, lift, faces, is_lower
    return tris
