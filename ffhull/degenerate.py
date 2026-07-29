"""Lower-dimensional input handling for the convex hull.

The GPU ffHull path assumes a full-dimensional (3D) point set.  Coincident,
collinear and coplanar inputs are detected up front on the host and dispatched
to the appropriate lower-dimensional routine.  These are corner cases, not the
performance path, so they are handled with simple exact-enough host code.
"""

import numpy as np


def analyze_dimension(pts: np.ndarray, tol_rel=1e-12):
    """Classify the affine dimension of ``pts``.

    Returns (dim, info) where dim in {0,1,2,3}.  ``info`` carries the data each
    lower-dimensional handler needs (basis vectors / representative point).
    """
    n = len(pts)
    scale = np.abs(pts).max() + 1.0
    tol = tol_rel * scale

    p0 = pts[0]
    # dimension 0: all coincident
    d = np.linalg.norm(pts - p0, axis=1)
    i1 = int(np.argmax(d))
    if d[i1] <= tol:
        return 0, {"rep": 0}

    u = pts[i1] - p0
    u = u / np.linalg.norm(u)
    # distance from line p0 + t*u
    w = pts - p0
    perp = w - np.outer(w @ u, u)
    dperp = np.linalg.norm(perp, axis=1)
    i2 = int(np.argmax(dperp))
    if dperp[i2] <= tol:
        return 1, {"p0": p0, "u": u}

    v = perp[i2] / np.linalg.norm(perp[i2])
    normal = np.cross(u, v)
    normal = normal / np.linalg.norm(normal)
    dn = np.abs(w @ normal)
    i3 = int(np.argmax(dn))
    if dn[i3] <= tol:
        return 2, {"p0": p0, "u": u, "v": v, "normal": normal}

    return 3, {}


def _hull2d_indices(pts2d: np.ndarray):
    """Andrew's monotone chain; returns CCW boundary vertex indices."""
    order = sorted(range(len(pts2d)), key=lambda i: (pts2d[i, 0], pts2d[i, 1]))
    pts = pts2d

    def cross(o, a, b):
        return ((pts[a, 0] - pts[o, 0]) * (pts[b, 1] - pts[o, 1])
                - (pts[a, 1] - pts[o, 1]) * (pts[b, 0] - pts[o, 0]))

    lower = []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    upper = []
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return lower[:-1] + upper[:-1]


def hull_lowdim(pts: np.ndarray, dim: int, info: dict):
    """Return face indices (m,3) for a degenerate hull.

    * dim 0: single point -> empty face list, vertices = {rep}.
    * dim 1: segment -> two extreme endpoints, no triangles.
    * dim 2: polygon in a plane -> triangulated (fan, both orientations) so the
      flat hull is a closed zero-volume surface.

    Returns (faces, vertex_indices).
    """
    if dim == 0:
        return np.zeros((0, 3), np.int64), np.array([info["rep"]], np.int64)

    if dim == 1:
        p0 = info["p0"]; u = info["u"]
        t = (pts - p0) @ u
        lo = int(np.argmin(t)); hi = int(np.argmax(t))
        return np.zeros((0, 3), np.int64), np.array([lo, hi], np.int64)

    # dim == 2
    p0 = info["p0"]; u = info["u"]; v = info["v"]
    w = pts - p0
    coords = np.stack([w @ u, w @ v], axis=1)
    ring = _hull2d_indices(coords)
    ring = np.array(ring, np.int64)
    # triangulate the polygon as a fan from ring[0]; emit both faces (front/back)
    faces = []
    for k in range(1, len(ring) - 1):
        faces.append([ring[0], ring[k], ring[k + 1]])
        faces.append([ring[0], ring[k + 1], ring[k]])
    return np.array(faces, np.int64).reshape(-1, 3), ring
