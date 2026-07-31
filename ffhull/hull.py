"""ffHull driver: initial tetrahedron, Phase A growth, (later) Phase B flips."""

import numpy as np
import warp as wp

from .mesh import Mesh, INT_MAX
from .predicates import orient3d
from . import grow
from . import flip

# Safety cap on Flip-Flop rounds inside the device loop.  A converging hull
# needs few rounds (the worst across the 134 threedscans models is ~1700, for a
# 58k-vertex hull); hitting this cap means the float64 predicate is oscillating
# on a degenerate case (e.g. a large coplanar facet), which the robust wrapper
# then resolves by joggling and retrying.  Kept a few x above the worst legit
# count so the doomed attempt bails quickly instead of burning 20k rounds.
FLIP_MAXIT = 5000


# ----------------------------------------------------------------------------
# Host-side setup
# ----------------------------------------------------------------------------

def _orient3d_np(a, b, c, d):
    return float(np.linalg.det(np.stack([a - d, b - d, c - d])))


# ----------------------------------------------------------------------------
# GPU-resident tetra: build the 4 oriented seed faces + reciprocal adjacency +
# kernel point s directly on the device from the device seed arrays.  A single
# thread does the fixed O(1) work, so nothing about the initial tetrahedron
# touches the host.
# ----------------------------------------------------------------------------

@wp.func
def _has(tv: wp.vec3i, x: wp.int32):
    return tv[0] == x or tv[1] == x or tv[2] == x


@wp.func
def _slot_not(tv: wp.vec3i, x: wp.int32, y: wp.int32):
    # local slot of tv's vertex that is neither x nor y == the reciprocal slot
    # of the shared edge {x, y} (the edge at slot e omits vertex e).
    for e in range(3):
        if tv[e] != x and tv[e] != y:
            return e
    return wp.int32(-1)


@wp.func
def _match(tri_v: wp.array(dtype=wp.vec3i), k: wp.int32, x: wp.int32, y: wp.int32):
    # the other tetra face (besides k) sharing edge {x, y}, and its slot
    for m in range(4):
        if m != k:
            tvm = tri_v[m]
            if _has(tvm, x) and _has(tvm, y):
                return wp.vec2i(m, _slot_not(tvm, x, y))
    return wp.vec2i(-1, -1)


@wp.func
def _oriented_face(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, s: wp.vec3d,
                   ia: wp.int32, ib: wp.int32, ic: wp.int32):
    # wind so the kernel point s is beneath the face: orient3d(face, s) < 0
    if orient3d(a, b, c, s) > wp.float64(0.0):
        return wp.vec3i(ia, ic, ib)
    return wp.vec3i(ia, ib, ic)


@wp.kernel
def build_tetra(seed_pts: wp.array(dtype=wp.vec3d), seed_idx: wp.array(dtype=wp.int32),
                n: wp.int32, points: wp.array(dtype=wp.vec3d),
                tri_v: wp.array(dtype=wp.vec3i), tri_adj: wp.array(dtype=wp.vec3i),
                tri_adj_slot: wp.array(dtype=wp.vec3i), tri_active: wp.array(dtype=wp.int32),
                tri_count: wp.array(dtype=wp.int32)):
    # kernel point s = centroid, stored at point slot n
    s = (seed_pts[0] + seed_pts[1] + seed_pts[2] + seed_pts[3]) * wp.float64(0.25)
    points[n] = s

    p0 = seed_pts[0]; p1 = seed_pts[1]; p2 = seed_pts[2]; p3 = seed_pts[3]
    i0 = seed_idx[0]; i1 = seed_idx[1]; i2 = seed_idx[2]; i3 = seed_idx[3]

    # four faces, each opposite one apex, oriented so s is beneath
    tri_v[0] = _oriented_face(p1, p2, p3, s, i1, i2, i3)
    tri_v[1] = _oriented_face(p0, p2, p3, s, i0, i2, i3)
    tri_v[2] = _oriented_face(p0, p1, p3, s, i0, i1, i3)
    tri_v[3] = _oriented_face(p0, p1, p2, s, i0, i1, i2)
    for k in range(4):
        tri_active[k] = 1

    # reciprocal adjacency by edge matching (edge at slot e = verts (e+1, e+2))
    for k in range(4):
        tv = tri_v[k]
        m0 = _match(tri_v, k, tv[1], tv[2])
        m1 = _match(tri_v, k, tv[2], tv[0])
        m2 = _match(tri_v, k, tv[0], tv[1])
        tri_adj[k] = wp.vec3i(m0[0], m1[0], m2[0])
        tri_adj_slot[k] = wp.vec3i(m0[1], m1[1], m2[1])

    tri_count[0] = 4


def init_tetra_gpu(mesh: Mesh, seed_idx, seed_pts):
    """Build the seed tetrahedron entirely on the GPU from the device seed
    arrays (from ``seed.build_seed_gpu``).  No host readback."""
    wp.launch(build_tetra, dim=1,
              inputs=[seed_pts, seed_idx, mesh.n, mesh.points, mesh.tri_v,
                      mesh.tri_adj, mesh.tri_adj_slot, mesh.tri_active, mesh.tri_count],
              device=mesh.device)


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

def grow_star(mesh: Mesh, seed_idx, nseed=4, verbose=False, use_graph=True):
    """Phase A growth.  ``seed_idx`` is a device int32 array of the tetra vertex
    indices (from ``seed.build_seed_gpu`` or wrapped from a host list); seed
    membership is tested on the GPU, so there is no O(n) host pass."""
    dev = mesh.device
    n = mesh.n
    cap = mesh.cap

    wp.launch(grow.init_associate, dim=n,
              inputs=[mesh.points, mesh.tri_v, mesh.s_idx, 4, seed_idx, nseed,
                      mesh.point_owner],
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
                  inputs=[mesh.points, mesh.tri_v, mesh.s_idx, mesh.face_pivot, mesh.face_children,
                          mesh.point_owner, mesh.scratch_i], device=dev)
        wp.launch(grow.advance_cond, dim=1,
                  inputs=[mesh.scratch_i, mesh.iter_count, n + 16, mesh.cond], device=dev)

    mesh.iter_count.zero_()
    _run_while(mesh, mesh.cond, body, use_graph, safety=n + 16)
    return mesh.get_tri_count()


def live_faces(mesh: Mesh):
    """Return an (m,3) int array of vertex indices of live triangles."""
    k = mesh.get_tri_count()
    # slice on-device before copying, so we read back only k triangles, not the
    # full 2n-capacity arrays (that readback dominated small-hull cases).
    tv = mesh.tri_v[:k].numpy()
    act = mesh.tri_active[:k].numpy()
    return tv[act == 1]


def build_star(points_np: np.ndarray, device="cuda:0", verbose=False, use_graph=True):
    """Convenience: run Phase A only, returning (mesh, s, tetra_idx)."""
    from . import seed as seedmod
    mesh = Mesh(points_np, device)
    dim, seed_idx, seed_pts, scale = seedmod.build_seed_gpu(mesh.points, mesh.n, device)
    if dim < 3:
        raise NotImplementedError("degenerate (lower-dimensional) input")
    init_tetra_gpu(mesh, seed_idx, seed_pts)
    grow_star(mesh, seed_idx, 4, verbose=verbose, use_graph=use_graph)
    tetra_idx = seed_idx.numpy()
    s = wp.vec3d(*[float(x) for x in seed_pts.numpy().mean(axis=0)])
    return mesh, s, tetra_idx


def flip_convexify(mesh: Mesh, verbose=False, use_graph=True):
    dev = mesh.device
    tc = mesh.tri_count  # device count (unchanged by flips)
    # Flips never grow the triangle count, so launch over the post-growth count
    # (read once) rather than the full 2n capacity -- for a small hull from a
    # large point cloud this is thousands of threads instead of millions.
    count = mesh.get_tri_count()

    def body():
        wp.launch(flip.reset_flip, dim=count,
                  inputs=[mesh.tri_claim, mesh.prop_slot, mesh.prop_type, mesh.changed], device=dev)
        wp.launch(flip.label_kernel, dim=count,
                  inputs=[mesh.points, mesh.s_idx, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, tc, mesh.vertex_label, mesh.changed], device=dev)
        wp.launch(flip.propose_claim, dim=count,
                  inputs=[mesh.points, mesh.s_idx, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, tc, mesh.vertex_label,
                          mesh.tri_claim, mesh.prop_slot, mesh.prop_type], device=dev)
        wp.launch(flip.apply_flips, dim=count,
                  inputs=[mesh.points, mesh.s_idx, mesh.tri_v, mesh.tri_adj, mesh.tri_adj_slot,
                          mesh.tri_active, tc, mesh.tri_claim, mesh.prop_slot,
                          mesh.prop_type, mesh.changed], device=dev)
        # `changed` records progress this round; fold in the safety cap.
        wp.launch(grow.advance_cond, dim=1,
                  inputs=[mesh.changed, mesh.iter_count, FLIP_MAXIT, mesh.cond], device=dev)

    mesh.iter_count.zero_()
    _run_while(mesh, mesh.cond, body, use_graph, safety=FLIP_MAXIT)


FILTER_THRESHOLD = 50_000   # only cull for point clouds large enough to benefit

# Reusable workspace pool, keyed by (device, n): repeated hulls of the same size
# (and the joggle retries within one call) reuse the 2n-capacity arrays instead
# of re-allocating them.  Bounded so it can't grow without limit.
_MESH_POOL = {}
_POOL_MAX_PER_KEY = 2


def _acquire_mesh(points_np, device):
    key = (device, len(points_np))
    pool = _MESH_POOL.get(key)
    if pool:
        m = pool.pop()
        m.rebind(points_np)
        return m
    return Mesh(points_np, device)


def _release_mesh(m):
    pool = _MESH_POOL.setdefault((m.device, m.n), [])
    if len(pool) < _POOL_MAX_PER_KEY:
        pool.append(m)


def clear_pool():
    """Drop all cached workspaces (frees their GPU memory)."""
    _MESH_POOL.clear()


def convex_hull(points_np: np.ndarray, device="cuda:0", verbose=False,
                return_vertices=False, use_graph=True, robust=True, filter=False,
                reuse=True):
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

    # Conservative interior-point cull FIRST, on a lightweight points-only array,
    # so we never allocate the big 2n workspace for a cloud we're about to shrink.
    # Discard deep-interior points, run the exact hull on the survivors (a
    # 2*survivors workspace), and map indices back.  Never drops a true hull
    # vertex (survivors include every point on/outside the inner hull H0).
    if filter and n >= FILTER_THRESHOLD:
        from . import filter as filt
        pchk = wp.array(points_np, dtype=wp.vec3d, device=device)
        dim0, _, _, _ = seedmod.build_seed_gpu(pchk, n, device)
        if dim0 < 3:
            del pchk
            d, info = degenerate.analyze_dimension(points_np)
            faces, verts = degenerate.hull_lowdim(points_np, min(dim0, d), info)
            return (faces, verts) if return_vertices else faces
        keep = filt.cull_indices(
            pchk, points_np, n, device,
            hull_fn=lambda c: convex_hull(c, device=device, use_graph=use_graph,
                                          robust=robust, filter=False))
        del pchk
        if keep is not None and len(keep) < 0.6 * n:
            if verbose:
                print(f"  cull: {n} -> {len(keep)} survivors ({100*len(keep)/n:.1f}%)")
            f_local = convex_hull(points_np[keep], device=device, verbose=verbose,
                                  use_graph=use_graph, robust=robust, filter=False)
            faces = keep[f_local]
            return (faces, np.unique(faces)) if return_vertices else faces

    mesh = _acquire_mesh(points_np, device) if reuse else Mesh(points_np, device)
    # Affine dimension is judged on the true input (a joggle would hide it).
    # Fully GPU-resident: seed search, tetra construction, and seed membership
    # all run on the device; only `dim` and `scale` come back (2 scalars).
    dim0, seed_idx, seed_pts, scale = seedmod.build_seed_gpu(mesh.points, n, device)
    if dim0 < 3:
        _release_mesh(mesh)
        d, info = degenerate.analyze_dimension(points_np)
        faces, verts = degenerate.hull_lowdim(points_np, min(dim0, d), info)
        return (faces, verts) if return_vertices else faces

    convex_tol = 1e-6 * scale ** 3
    for attempt in range(5):
        if attempt > 0:
            # deterministic joggle to escape exact/near degeneracies (coplanar
            # facets etc.); magnitude grows each retry, hull indices unchanged.
            mag = 1e-8 * scale * (10.0 ** attempt)
            jog = np.random.default_rng(attempt).standard_normal(points_np.shape)
            mesh.rebind(points_np + mag * jog)
            dim, seed_idx, seed_pts, _ = seedmod.build_seed_gpu(mesh.points, mesh.n, device)
            if dim < 3:
                continue  # joggle collapsed dimension (shouldn't happen); retry
        init_tetra_gpu(mesh, seed_idx, seed_pts)
        grow_star(mesh, seed_idx, 4, verbose=verbose, use_graph=use_graph)
        flip_convexify(mesh, verbose=verbose, use_graph=use_graph)
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

    faces = live_faces(mesh)   # copies out of the mesh, so it can be released
    _release_mesh(mesh)
    if return_vertices:
        return faces, np.unique(faces)
    return faces
