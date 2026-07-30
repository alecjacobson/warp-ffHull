"""Conservative interior-point cull (initial-polytope filter).

The convex hull of a big point cloud is usually a tiny fraction of its points.
This filter throws away the deep-interior points cheaply before the exact
algorithm runs, without ever discarding a true hull vertex:

  1. take a subset of the points (a strided sample plus the axis extremes);
  2. build the convex hull H0 of just that subset  (a coarse *inner* hull);
  3. discard every input point that lies strictly inside H0.

Since H0 is the hull of a subset, H0 subset of the true hull H, so any point
inside H0 is inside H and cannot be a hull vertex -- the cull is conservative.
All true hull vertices lie on/outside H0 and survive.  This is a big win for
solid/volumetric clouds (most points are deep interior) and a near-free no-op
for surface scans (most points sit on the hull boundary and survive).
"""

import numpy as np
import warp as wp
from warp.utils import array_scan

from .predicates import orient3d


@wp.kernel
def _cull(points: wp.array(dtype=wp.vec3d), scoords: wp.array(dtype=wp.vec3d),
          h0faces: wp.array(dtype=wp.vec3i), tol: wp.float64,
          keep: wp.array(dtype=wp.int32)):
    # keep point i unless it is strictly inside H0 (beneath every outward face
    # by more than tol).  H0 faces are oriented so a point outside a face has
    # orient3d(face, p) > 0.
    i = wp.tid()
    p = points[i]
    k = wp.int32(0)
    for f in range(h0faces.shape[0]):
        fc = h0faces[f]
        if orient3d(scoords[fc[0]], scoords[fc[1]], scoords[fc[2]], p) > tol:
            k = 1
            break
    keep[i] = k


@wp.kernel
def _compact(keep: wp.array(dtype=wp.int32), offset: wp.array(dtype=wp.int32),
             out_idx: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if keep[i] == 1:
        out_idx[offset[i]] = i


def cull_indices(points_wp, points_np, n, device, hull_fn, sample=8000):
    """Return original indices of the points that survive the conservative cull,
    as a numpy int array (or None if the filter is not worthwhile).

    ``points_wp`` is the points already on the device; ``points_np`` the same on
    the host (for cheap sampling).  ``hull_fn(coords)`` computes the convex hull
    of a small point set and returns its (m,3) outward faces (the inner hull H0).
    """
    # subset for the inner hull: a strided sample + the 6 axis extremes
    stride = max(1, n // sample)
    idx = np.arange(0, n, stride)
    ax = np.array([points_np[:, d].argmin() for d in range(3)]
                  + [points_np[:, d].argmax() for d in range(3)])
    idx = np.unique(np.concatenate([idx, ax]))
    S = np.ascontiguousarray(points_np[idx])
    if len(S) < 4:
        return None
    try:
        h0 = hull_fn(S)                                # faces into S
    except Exception:
        return None
    if len(h0) == 0:
        return None

    scoords = wp.array(S, dtype=wp.vec3d, device=device)
    h0faces = wp.array(np.ascontiguousarray(h0.astype(np.int32)), dtype=wp.vec3i, device=device)
    scale = float(np.abs(points_np).max() + 1.0)
    tol = wp.float64(-1e-9 * scale ** 3)   # keep points within a thin shell of H0

    keep = wp.empty(n, dtype=wp.int32, device=device)
    wp.launch(_cull, dim=n, inputs=[points_wp, scoords, h0faces, tol, keep], device=device)
    offset = wp.empty(n, dtype=wp.int32, device=device)
    array_scan(keep, offset, inclusive=False)
    total = int(offset.numpy()[n - 1]) + int(keep.numpy()[n - 1])
    if total >= n:
        return None  # nothing culled -> not worthwhile
    out_idx = wp.empty(total, dtype=wp.int32, device=device)
    wp.launch(_compact, dim=n, inputs=[keep, offset, out_idx], device=device)
    return out_idx.numpy()
