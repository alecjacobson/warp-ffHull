"""ffHull driver: initial tetrahedron, Phase A growth, (later) Phase B flips."""

import numpy as np
import warp as wp

from .mesh import Mesh, INT_MAX
from . import grow
from . import flip


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
        wp.launch(grow.set_cond_gt0, dim=1, inputs=[mesh.scratch_i, mesh.cond], device=dev)

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
    cap = mesh.cap
    tc = mesh.tri_count  # device count (unchanged by flips)

    def body():
        wp.launch(flip.reset_flip, dim=cap,
                  inputs=[mesh.tri_claim, mesh.prop_slot, mesh.prop_type], device=dev)
        wp.launch(grow.zero_scalar, dim=1, inputs=[mesh.changed], device=dev)
        wp.launch(flip.label_kernel, dim=cap,
                  inputs=[mesh.points, s, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, tc, mesh.vertex_label, mesh.changed], device=dev)
        wp.launch(flip.propose_claim, dim=cap,
                  inputs=[mesh.points, s, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, tc, mesh.vertex_label,
                          mesh.tri_claim, mesh.prop_slot, mesh.prop_type], device=dev)
        wp.launch(flip.apply_flips, dim=cap,
                  inputs=[mesh.points, s, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, tc, mesh.tri_claim, mesh.prop_slot,
                          mesh.prop_type, mesh.changed], device=dev)

    # `changed` doubles as the loop condition: each round zeroes it, then label/
    # apply set it if any progress was made (new label or flip).
    _run_while(mesh, mesh.changed, body, use_graph, safety=50 * mesh.n + 100)


def convex_hull(points_np: np.ndarray, device="cuda:0", verbose=False,
                return_vertices=False, use_graph=True):
    """Compute the 3D convex hull.

    Returns an (m,3) int array of face vertex indices into ``points_np``
    (outward-oriented triangles).  Extreme-point search and affine-dimension
    classification run on the GPU (``ffhull.seed``); genuinely lower-dimensional
    inputs (coincident, collinear, coplanar) are dispatched to a host handler.
    With ``return_vertices=True`` also returns the extreme-vertex indices.
    """
    from . import degenerate, seed as seedmod
    points_np = np.ascontiguousarray(points_np, dtype=np.float64)

    mesh = Mesh(points_np, device)
    dim, tetra_idx, seed_pts = seedmod.build_seed(mesh.points, mesh.n, device)
    if dim < 3:
        d, info = degenerate.analyze_dimension(points_np)
        faces, verts = degenerate.hull_lowdim(points_np, min(dim, d), info)
        return (faces, verts) if return_vertices else faces

    s = init_tetra(mesh, seed_pts, tetra_idx)
    grow_star(mesh, s, tetra_idx, verbose=verbose, use_graph=use_graph)
    flip_convexify(mesh, s, verbose=verbose, use_graph=use_graph)
    faces = live_faces(mesh)
    if return_vertices:
        return faces, np.unique(faces)
    return faces
