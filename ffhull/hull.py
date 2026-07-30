"""ffHull driver: initial tetrahedron, Phase A growth, (later) Phase B flips."""

import numpy as np
import warp as wp

from .mesh import Mesh, INT_MAX
from . import grow
from . import flip

# Safety cap on Flip-Flop rounds inside the device loop.  Convergence is fast
# for well-conditioned inputs; hitting this cap signals a degenerate case, which
# the robust wrapper resolves by joggling and retrying.
FLIP_MAXIT = 20000


# ----------------------------------------------------------------------------
# Host-side setup
# ----------------------------------------------------------------------------

def _orient3d_np(a, b, c, d):
    return float(np.linalg.det(np.stack([a - d, b - d, c - d])))


def choose_tetra(pts: np.ndarray):
    """Pick 4 affinely-independent, well-separated extreme points.

    Returns (indices[4], ok).  ok is False if the points are (near) coplanar,
    signalling that a lower-dimensional path is required.
    """
    n = len(pts)
    # Extremes along the 6 axis directions give a good starting spread.
    ext = set()
    for ax in range(3):
        ext.add(int(np.argmin(pts[:, ax])))
        ext.add(int(np.argmax(pts[:, ax])))
    ext = list(ext)
    # p0, p1: farthest-apart pair among extremes.
    best = (-1.0, ext[0], ext[0])
    for i in range(len(ext)):
        for j in range(i + 1, len(ext)):
            d = np.sum((pts[ext[i]] - pts[ext[j]]) ** 2)
            if d > best[0]:
                best = (d, ext[i], ext[j])
    i0, i1 = best[1], best[2]
    if best[0] == 0.0:
        return None, False  # all coincident
    # p2: farthest from line p0-p1.
    line = pts[i1] - pts[i0]
    line = line / np.linalg.norm(line)
    d2 = np.linalg.norm(np.cross(pts - pts[i0], line), axis=1)
    i2 = int(np.argmax(d2))
    if d2[i2] <= 1e-12 * np.sqrt(best[0]):
        return None, False  # collinear
    # p3: farthest from plane p0-p1-p2.  orient3d(a,b,c,p) = (p-a).((b-a)x(c-a));
    # vectorised distance-to-plane over all points.
    a, b, c = pts[i0], pts[i1], pts[i2]
    nrm = np.cross(b - a, c - a)
    vol = np.abs((pts - a) @ nrm)
    i3 = int(np.argmax(vol))
    if vol[i3] <= 1e-9 * (best[0] ** 1.5):
        return None, False  # coplanar
    return np.array([i0, i1, i2, i3], dtype=np.int64), True


def _build_adjacency(tri_v):
    """Build reciprocal adjacency for a small closed triangle mesh (host)."""
    m = len(tri_v)
    adj = np.full((m, 3), -1, dtype=np.int32)
    slot = np.full((m, 3), -1, dtype=np.int32)
    edge_map = {}
    for t in range(m):
        v = tri_v[t]
        for i in range(3):
            e = (v[(i + 1) % 3], v[(i + 2) % 3])
            key = frozenset(e)
            if key in edge_map:
                ot, oi = edge_map[key]
                adj[t][i] = ot
                slot[t][i] = oi
                adj[ot][oi] = t
                slot[ot][oi] = i
            else:
                edge_map[key] = (t, i)
    return adj, slot


def init_tetra(mesh: Mesh, seed_pts: np.ndarray, tetra_idx):
    """Build the 4 oriented seed-tetra faces and write them to the first 4
    triangle slots.  ``seed_pts`` are the host coords of the 4 tetra vertices
    (from the GPU seed search); only 4 entries are written — no full readback."""
    s = seed_pts.mean(axis=0)
    # Four faces, each the three-vertex subset opposite one apex; orient so the
    # kernel point s is beneath (orient3d(face, s) < 0).
    faces = []
    for skip in range(4):
        ks = [k for k in range(4) if k != skip]
        a, b, c = seed_pts[ks[0]], seed_pts[ks[1]], seed_pts[ks[2]]
        vs = [int(tetra_idx[ks[0]]), int(tetra_idx[ks[1]]), int(tetra_idx[ks[2]])]
        if _orient3d_np(a, b, c, s) > 0:
            vs[1], vs[2] = vs[2], vs[1]
        faces.append(vs)
    faces = np.array(faces, dtype=np.int32)
    adj, slot = _build_adjacency([list(f) for f in faces])

    dev = mesh.device
    wp.copy(mesh.tri_v[0:4], wp.array(faces, dtype=wp.vec3i, device=dev))
    wp.copy(mesh.tri_adj[0:4], wp.array(adj, dtype=wp.vec3i, device=dev))
    wp.copy(mesh.tri_adj_slot[0:4], wp.array(slot, dtype=wp.vec3i, device=dev))
    wp.copy(mesh.tri_active[0:4], wp.array([1, 1, 1, 1], dtype=wp.int32, device=dev))
    mesh.set_tri_count(4)
    return wp.vec3d(float(s[0]), float(s[1]), float(s[2]))


# ----------------------------------------------------------------------------
# Device-controlled loop helper (graph-capturable, no per-round host sync)
# ----------------------------------------------------------------------------

def _run_while(mesh: Mesh, cond, body, use_graph: bool, safety: int):
    """Repeat ``body`` while device flag ``cond[0]`` stays nonzero.

    With ``use_graph`` the loop is captured as a conditional CUDA graph and
    replayed with zero host synchronisation.  The plain-Python fallback still
    avoids per-round syncs except the single ``cond`` readback used to stop.
    """
    cond.fill_(1)
    if use_graph:
        with wp.ScopedCapture(mesh.device) as cap:
            wp.capture_while(cond, body)
        wp.capture_launch(cap.graph)
    else:
        for _ in range(safety):
            body()
            if int(cond.numpy()[0]) == 0:
                break


# ----------------------------------------------------------------------------
# Phase A driver
# ----------------------------------------------------------------------------

def grow_star(mesh: Mesh, s: wp.vec3d, tetra_idx: np.ndarray, verbose=False,
              use_graph=True):
    dev = mesh.device
    n = mesh.n
    cap = mesh.cap

    is_seed = np.zeros(n, dtype=np.int32)
    is_seed[tetra_idx] = 1
    is_seed_wp = wp.array(is_seed, dtype=wp.int32, device=dev)

    wp.launch(grow.init_associate, dim=n,
              inputs=[mesh.points, mesh.tri_v, s, 4, is_seed_wp, mesh.point_owner],
              device=dev)

    def body():
        wp.launch(grow.snapshot_count, dim=1, inputs=[mesh.tri_count, mesh.old_count], device=dev)
        wp.launch(grow.reset_growth, dim=cap,
                  inputs=[mesh.face_score, mesh.face_pivot, mesh.face_children], device=dev)
        wp.launch(grow.zero_scalar, dim=1, inputs=[mesh.scratch_i], device=dev)
        wp.launch(grow.furthest_pass_a, dim=n,
                  inputs=[mesh.points, mesh.tri_v, mesh.point_owner, mesh.face_score], device=dev)
        wp.launch(grow.furthest_pass_b, dim=n,
                  inputs=[mesh.points, mesh.tri_v, mesh.point_owner, mesh.face_score, mesh.face_pivot],
                  device=dev)
        wp.launch(grow.split_faces, dim=cap,
                  inputs=[mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot, mesh.tri_active,
                          mesh.face_pivot, mesh.face_children, mesh.tri_count, mesh.old_count],
                  device=dev)
        wp.launch(grow.fix_adjacency, dim=cap,
                  inputs=[mesh.tri_adj, mesh.tri_adj_slot, mesh.face_pivot, mesh.face_children,
                          mesh.old_count, mesh.tri_count], device=dev)
        wp.launch(grow.reassociate, dim=n,
                  inputs=[mesh.points, mesh.tri_v, s, mesh.face_pivot, mesh.face_children,
                          mesh.point_owner, mesh.scratch_i], device=dev)
        wp.launch(grow.advance_cond, dim=1,
                  inputs=[mesh.scratch_i, mesh.iter_count, n + 16, mesh.cond], device=dev)

    mesh.iter_count.zero_()
    _run_while(mesh, mesh.cond, body, use_graph, safety=n + 16)
    return mesh.get_tri_count()


def live_faces(mesh: Mesh):
    """Return an (m,3) int array of vertex indices of live triangles."""
    k = mesh.get_tri_count()
    tv = mesh.tri_v.numpy()[:k]
    act = mesh.tri_active.numpy()[:k]
    return tv[act == 1]


def build_star(points_np: np.ndarray, device="cuda:0", verbose=False, use_graph=True):
    """Convenience: run Phase A only, returning (mesh, s, tetra_idx)."""
    from . import seed as seedmod
    mesh = Mesh(points_np, device)
    dim, tetra_idx, seed_pts = seedmod.build_seed(mesh.points, mesh.n, device)
    if dim < 3:
        raise NotImplementedError("degenerate (lower-dimensional) input")
    s = init_tetra(mesh, seed_pts, tetra_idx)
    grow_star(mesh, s, tetra_idx, verbose=verbose, use_graph=use_graph)
    return mesh, s, tetra_idx


def flip_convexify(mesh: Mesh, s: wp.vec3d, verbose=False, use_graph=True):
    dev = mesh.device
    tc = mesh.tri_count  # device count (unchanged by flips)
    # Flips never grow the triangle count, so launch over the post-growth count
    # (read once) rather than the full 2n capacity -- for a small hull from a
    # large point cloud this is thousands of threads instead of millions.
    count = mesh.get_tri_count()

    def body():
        wp.launch(flip.reset_flip, dim=count,
                  inputs=[mesh.tri_claim, mesh.prop_slot, mesh.prop_type], device=dev)
        wp.launch(grow.zero_scalar, dim=1, inputs=[mesh.changed], device=dev)
        wp.launch(flip.label_kernel, dim=count,
                  inputs=[mesh.points, s, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, tc, mesh.vertex_label, mesh.changed], device=dev)
        wp.launch(flip.propose_claim, dim=count,
                  inputs=[mesh.points, s, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, tc, mesh.vertex_label,
                          mesh.tri_claim, mesh.prop_slot, mesh.prop_type], device=dev)
        wp.launch(flip.apply_flips, dim=count,
                  inputs=[mesh.points, s, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, tc, mesh.tri_claim, mesh.prop_slot,
                          mesh.prop_type, mesh.changed], device=dev)
        # `changed` records progress this round; fold in the safety cap.
        wp.launch(grow.advance_cond, dim=1,
                  inputs=[mesh.changed, mesh.iter_count, FLIP_MAXIT, mesh.cond], device=dev)

    mesh.iter_count.zero_()
    _run_while(mesh, mesh.cond, body, use_graph, safety=FLIP_MAXIT)


FILTER_THRESHOLD = 50_000   # only cull for point clouds large enough to benefit


def convex_hull(points_np: np.ndarray, device="cuda:0", verbose=False,
                return_vertices=False, use_graph=True, robust=True, filter=False):
    """Compute the 3D convex hull.

    Returns an (m,3) int array of face vertex indices into ``points_np``
    (outward-oriented triangles).  Extreme-point search and affine-dimension
    classification run on the GPU (``ffhull.seed``); genuinely lower-dimensional
    inputs (coincident, collinear, coplanar) are dispatched to a host handler.
    With ``return_vertices=True`` also returns the extreme-vertex indices.

    For large clouds a conservative interior-point cull (``ffhull.filter``) runs
    first, discarding deep-interior points before the exact hull; set
    ``filter=False`` to disable it.
    """
    from . import degenerate, seed as seedmod
    points_np = np.ascontiguousarray(points_np, dtype=np.float64)
    n = len(points_np)

    # Affine dimension is judged on the true input (a joggle would hide it).
    # Upload only the points for this check, not a full (2n-capacity) mesh.
    _pchk = wp.array(points_np, dtype=wp.vec3d, device=device)
    dim0, _, _ = seedmod.build_seed(_pchk, n, device)
    if dim0 < 3:
        del _pchk
        d, info = degenerate.analyze_dimension(points_np)
        faces, verts = degenerate.hull_lowdim(points_np, min(dim0, d), info)
        return (faces, verts) if return_vertices else faces

    # Conservative interior-point cull: discard deep-interior points, then run
    # the exact hull on the survivors and map indices back.  Never drops a true
    # hull vertex (survivors include every point on/outside the inner hull H0).
    if filter and n >= FILTER_THRESHOLD:
        from . import filter as filt
        keep = filt.cull_indices(
            _pchk, points_np, n, device,
            hull_fn=lambda c: convex_hull(c, device=device, use_graph=use_graph,
                                          robust=robust, filter=False))
        del _pchk
        if keep is not None and len(keep) < 0.6 * n:
            if verbose:
                print(f"  cull: {n} -> {len(keep)} survivors ({100*len(keep)/n:.1f}%)")
            f_local = convex_hull(points_np[keep], device=device, verbose=verbose,
                                  use_graph=use_graph, robust=robust, filter=False)
            faces = keep[f_local]
            return (faces, np.unique(faces)) if return_vertices else faces
    else:
        del _pchk

    scale = float(np.abs(points_np).max() + 1.0)
    convex_tol = 1e-6 * scale ** 3
    mesh = None
    for attempt in range(5):
        if attempt == 0:
            work = points_np
        else:
            # deterministic joggle to escape exact/near degeneracies (coplanar
            # facets etc.); magnitude grows each retry, hull indices unchanged.
            mag = 1e-8 * scale * (10.0 ** attempt)
            jog = np.random.default_rng(attempt).standard_normal(points_np.shape)
            work = points_np + mag * jog
        mesh = Mesh(work, device)
        dim, tetra_idx, seed_pts = seedmod.build_seed(mesh.points, mesh.n, device)
        if dim < 3:
            continue  # joggle collapsed dimension (shouldn't happen); retry
        s = init_tetra(mesh, seed_pts, tetra_idx)
        grow_star(mesh, s, tetra_idx, verbose=verbose, use_graph=use_graph)
        flip_convexify(mesh, s, verbose=verbose, use_graph=use_graph)
        if not robust:
            break
        mesh.convex_flag.zero_()
        count = mesh.get_tri_count()
        # Local reflex test is always cheap; for modest n also run the global
        # O(n*F) containment test, which catches tangled (non-simple) results
        # that a local test misses (the failure mode on exact-degenerate input).
        wp.launch(flip.check_convex, dim=count,
                  inputs=[mesh.points, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, mesh.tri_count, wp.float64(convex_tol),
                          mesh.convex_flag], device=device)
        if mesh.n * count <= 400_000_000:
            wp.launch(flip.check_contains, dim=mesh.n,
                      inputs=[mesh.points, mesh.tri_v, mesh.tri_active, count,
                              wp.float64(convex_tol), mesh.convex_flag], device=device)
        if int(mesh.convex_flag.numpy()[0]) == 0:
            break  # a genuine, enclosing convex hull
        if verbose:
            print(f"  invalid hull (attempt {attempt}); joggling and retrying")

    faces = live_faces(mesh)
    if return_vertices:
        return faces, np.unique(faces)
    return faces
