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
    # p3: farthest from plane p0-p1-p2.
    a, b, c = pts[i0], pts[i1], pts[i2]
    vol = np.array([abs(_orient3d_np(a, b, c, p)) for p in pts])
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


def init_tetra(mesh: Mesh, pts: np.ndarray, tetra_idx: np.ndarray):
    """Build the 4 oriented faces of the seed tetrahedron on the host."""
    s = pts[tetra_idx].mean(axis=0)
    # Four faces, each the three-vertex subset opposite one apex; orient so the
    # kernel point s is beneath (orient3d(face, s) < 0).
    faces = []
    for skip in range(4):
        vs = [tetra_idx[k] for k in range(4) if k != skip]
        a, b, c = pts[vs[0]], pts[vs[1]], pts[vs[2]]
        if _orient3d_np(a, b, c, s) > 0:
            vs[1], vs[2] = vs[2], vs[1]
        faces.append([int(vs[0]), int(vs[1]), int(vs[2])])
    faces = np.array(faces, dtype=np.int32)
    adj, slot = _build_adjacency([list(f) for f in faces])

    tv = mesh.tri_v.numpy()
    ta = mesh.tri_adj.numpy()
    ts = mesh.tri_adj_slot.numpy()
    tact = mesh.tri_active.numpy()
    tv[:4] = faces
    ta[:4] = adj
    ts[:4] = slot
    tact[:4] = 1
    mesh.tri_v.assign(tv)
    mesh.tri_adj.assign(ta)
    mesh.tri_adj_slot.assign(ts)
    mesh.tri_active.assign(tact)
    mesh.set_tri_count(4)
    return wp.vec3d(float(s[0]), float(s[1]), float(s[2]))


# ----------------------------------------------------------------------------
# Reset kernel for per-round growth scratch
# ----------------------------------------------------------------------------

@wp.kernel
def _reset_growth(face_score: wp.array(dtype=wp.float64),
                  face_pivot: wp.array(dtype=wp.int32),
                  face_children: wp.array(dtype=wp.vec3i)):
    t = wp.tid()
    face_score[t] = wp.float64(0.0)
    face_pivot[t] = grow.INT_MAX
    face_children[t] = wp.vec3i(-1, -1, -1)


# ----------------------------------------------------------------------------
# Phase A driver
# ----------------------------------------------------------------------------

def grow_star(mesh: Mesh, s: wp.vec3d, tetra_idx: np.ndarray, verbose=False):
    dev = mesh.device
    n = mesh.n

    is_seed = np.zeros(n, dtype=np.int32)
    is_seed[tetra_idx] = 1
    is_seed_wp = wp.array(is_seed, dtype=wp.int32, device=dev)

    wp.launch(grow.init_associate, dim=n,
              inputs=[mesh.points, mesh.tri_v, s, 4, is_seed_wp, mesh.point_owner],
              device=dev)

    max_iter = n + 8
    for it in range(max_iter):
        old_count = mesh.get_tri_count()
        wp.launch(_reset_growth, dim=mesh.cap,
                  inputs=[mesh.face_score, mesh.face_pivot, mesh.face_children], device=dev)
        wp.launch(grow.furthest_pass_a, dim=n,
                  inputs=[mesh.points, mesh.tri_v, mesh.point_owner, mesh.face_score], device=dev)
        wp.launch(grow.furthest_pass_b, dim=n,
                  inputs=[mesh.points, mesh.tri_v, mesh.point_owner, mesh.face_score, mesh.face_pivot],
                  device=dev)
        wp.launch(grow.split_faces, dim=old_count,
                  inputs=[mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot, mesh.tri_active,
                          mesh.face_pivot, mesh.face_children, mesh.tri_count, old_count],
                  device=dev)
        new_count = mesh.get_tri_count()
        wp.launch(grow.fix_adjacency, dim=new_count,
                  inputs=[mesh.tri_adj, mesh.tri_adj_slot, mesh.face_pivot, mesh.face_children,
                          old_count, new_count], device=dev)
        mesh.scratch_i.zero_()
        wp.launch(grow.reassociate, dim=n,
                  inputs=[mesh.points, mesh.tri_v, s, mesh.face_pivot, mesh.face_children,
                          mesh.point_owner, mesh.scratch_i], device=dev)
        active = int(mesh.scratch_i.numpy()[0])
        if verbose:
            print(f"  grow it={it} tris={new_count} active_pts={active}")
        if active == 0:
            break
    return mesh.get_tri_count()


def live_faces(mesh: Mesh):
    """Return an (m,3) int array of vertex indices of live triangles."""
    k = mesh.get_tri_count()
    tv = mesh.tri_v.numpy()[:k]
    act = mesh.tri_active.numpy()[:k]
    return tv[act == 1]


def build_star(points_np: np.ndarray, device="cuda:0", verbose=False):
    """Convenience: run Phase A only, returning (mesh, s, tetra_idx)."""
    mesh = Mesh(points_np, device)
    tetra_idx, ok = choose_tetra(points_np)
    if not ok:
        raise NotImplementedError("degenerate (lower-dimensional) input")
    s = init_tetra(mesh, points_np, tetra_idx)
    grow_star(mesh, s, tetra_idx, verbose=verbose)
    return mesh, s, tetra_idx


def flip_convexify(mesh: Mesh, s: wp.vec3d, verbose=False):
    dev = mesh.device
    count = mesh.get_tri_count()
    max_iter = 50 * mesh.n + 100
    for it in range(max_iter):
        wp.launch(flip.reset_flip, dim=mesh.cap,
                  inputs=[mesh.tri_claim, mesh.prop_slot, mesh.prop_type], device=dev)
        mesh.changed.zero_()
        wp.launch(flip.label_kernel, dim=count,
                  inputs=[mesh.points, s, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, count, mesh.vertex_label, mesh.changed], device=dev)
        wp.launch(flip.propose_claim, dim=count,
                  inputs=[mesh.points, s, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, count, mesh.vertex_label,
                          mesh.tri_claim, mesh.prop_slot, mesh.prop_type], device=dev)
        wp.launch(flip.apply_flips, dim=count,
                  inputs=[mesh.points, s, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, count, mesh.tri_claim, mesh.prop_slot,
                          mesh.prop_type, mesh.changed], device=dev)
        ch = int(mesh.changed.numpy()[0])
        if verbose and (it % 20 == 0 or ch == 0):
            print(f"  flip it={it} changed={ch}")
        if ch == 0:
            break
    return it


def convex_hull(points_np: np.ndarray, device="cuda:0", verbose=False,
                return_vertices=False):
    """Compute the 3D convex hull.

    Returns an (m,3) int array of face vertex indices into ``points_np``
    (outward-oriented triangles).  Lower-dimensional inputs (coincident,
    collinear, coplanar) are detected on the host and dispatched to the
    appropriate degenerate handler.  With ``return_vertices=True`` also returns
    the array of extreme-vertex indices.
    """
    from . import degenerate
    points_np = np.ascontiguousarray(points_np, dtype=np.float64)
    dim, info = degenerate.analyze_dimension(points_np)
    if dim < 3:
        faces, verts = degenerate.hull_lowdim(points_np, dim, info)
        return (faces, verts) if return_vertices else faces

    mesh = Mesh(points_np, device)
    tetra_idx, ok = choose_tetra(points_np)
    if not ok:
        # numerically borderline: fall back to the coplanar handler
        faces, verts = degenerate.hull_lowdim(points_np, 2,
                                              degenerate.analyze_dimension(points_np, 1e-9)[1])
        return (faces, verts) if return_vertices else faces
    s = init_tetra(mesh, points_np, tetra_idx)
    grow_star(mesh, s, tetra_idx, verbose=verbose)
    flip_convexify(mesh, s, verbose=verbose)
    faces = live_faces(mesh)
    if return_vertices:
        return faces, np.unique(faces)
    return faces
