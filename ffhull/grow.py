"""Phase A: grow a star-shaped polyhedron by parallel point insertion.

Each round:
  1. furthest_pass_a / furthest_pass_b : per active face, find the furthest
     associated point (its pivot), deterministically tie-broken by index.
  2. split_faces : replace each pivoted face by three children (one reuses the
     parent slot, two are appended), wiring internal adjacency.
  3. fix_adjacency : repair the external neighbour links (deferred from the
     split so reads and writes never race).
  4. reassociate : re-home each point onto one of its owner's three children,
     dropping points that have fallen beneath the surface.
"""

import warp as wp

from .predicates import orient3d, in_cone

INT_MAX = wp.constant(wp.int32(2147483647))


@wp.func
def _tri_pts(points: wp.array(dtype=wp.vec3d), tri_v: wp.array(dtype=wp.vec3i), t: wp.int32):
    tv = tri_v[t]
    return points[tv[0]], points[tv[1]], points[tv[2]]


@wp.kernel
def init_associate(
    points: wp.array(dtype=wp.vec3d),
    tri_v: wp.array(dtype=wp.vec3i),
    s: wp.vec3d,
    n_faces: wp.int32,
    is_seed: wp.array(dtype=wp.int32),
    point_owner: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if is_seed[i] == 1:
        point_owner[i] = -1
        return
    p = points[i]
    owner = wp.int32(-1)
    for t in range(n_faces):
        a, b, c = _tri_pts(points, tri_v, t)
        if in_cone(s, a, b, c, p):
            if orient3d(a, b, c, p) > wp.float64(0.0):
                owner = t
            else:
                owner = -1  # beneath this face => inside tetra => interior
            break
    point_owner[i] = owner


@wp.kernel
def furthest_pass_a(
    points: wp.array(dtype=wp.vec3d),
    tri_v: wp.array(dtype=wp.vec3i),
    point_owner: wp.array(dtype=wp.int32),
    face_score: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    t = point_owner[i]
    if t < 0:
        return
    a, b, c = _tri_pts(points, tri_v, t)
    d = orient3d(a, b, c, points[i])
    wp.atomic_max(face_score, t, d)


@wp.kernel
def furthest_pass_b(
    points: wp.array(dtype=wp.vec3d),
    tri_v: wp.array(dtype=wp.vec3i),
    point_owner: wp.array(dtype=wp.int32),
    face_score: wp.array(dtype=wp.float64),
    face_pivot: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    t = point_owner[i]
    if t < 0:
        return
    a, b, c = _tri_pts(points, tri_v, t)
    d = orient3d(a, b, c, points[i])
    if d == face_score[t]:
        wp.atomic_min(face_pivot, t, i)


@wp.kernel
def split_faces(
    tri_v: wp.array(dtype=wp.vec3i),
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    tri_active: wp.array(dtype=wp.int32),
    face_pivot: wp.array(dtype=wp.int32),
    face_children: wp.array(dtype=wp.vec3i),
    tri_count: wp.array(dtype=wp.int32),
    old_count: wp.int32,
):
    t = wp.tid()
    if t >= old_count:
        return
    if tri_active[t] == 0:
        return
    v = face_pivot[t]
    if v == INT_MAX:
        return

    tv = tri_v[t]
    a = tv[0]
    b = tv[1]
    c = tv[2]
    old_adj = tri_adj[t]
    old_slot = tri_adj_slot[t]

    base = wp.atomic_add(tri_count, 0, 2)
    c1 = base
    c2 = base + 1

    # child0 reuses slot t: (v, a, b), external edge (a,b) = parent slot 2
    tri_v[t] = wp.vec3i(v, a, b)
    tri_adj[t] = wp.vec3i(old_adj[2], c1, c2)
    tri_adj_slot[t] = wp.vec3i(old_slot[2], 2, 1)

    # child1 = c1: (v, b, c), external edge (b,c) = parent slot 0
    tri_v[c1] = wp.vec3i(v, b, c)
    tri_adj[c1] = wp.vec3i(old_adj[0], c2, t)
    tri_adj_slot[c1] = wp.vec3i(old_slot[0], 2, 1)
    tri_active[c1] = 1
    face_pivot[c1] = INT_MAX
    face_children[c1] = wp.vec3i(-1, -1, -1)

    # child2 = c2: (v, c, a), external edge (c,a) = parent slot 1
    tri_v[c2] = wp.vec3i(v, c, a)
    tri_adj[c2] = wp.vec3i(old_adj[1], t, c1)
    tri_adj_slot[c2] = wp.vec3i(old_slot[1], 2, 1)
    tri_active[c2] = 1
    face_pivot[c2] = INT_MAX
    face_children[c2] = wp.vec3i(-1, -1, -1)

    face_children[t] = wp.vec3i(t, c1, c2)


@wp.kernel
def fix_adjacency(
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    face_pivot: wp.array(dtype=wp.int32),
    face_children: wp.array(dtype=wp.vec3i),
    old_count: wp.int32,
    new_count: wp.int32,
):
    # Pure gather: every triangle repairs ONLY its own adjacency, so there are
    # no cross-triangle writes and no races.  If a neighbour split this round,
    # redirect to the neighbour's child that retained the shared edge (its
    # external edge is always local slot 0).
    t = wp.tid()
    if t >= new_count:
        return
    # A "child" created this round (appended, or a split parent's reused slot)
    # has final internal sibling links in slots 1,2; only its external slot 0
    # may need redirecting.  Non-split old faces may redirect any slot.
    is_child = t >= old_count
    if t < old_count and face_pivot[t] != INT_MAX:
        is_child = True

    adj = tri_adj[t]
    aslot = tri_adj_slot[t]
    for i in range(3):
        if is_child and i != 0:
            continue
        n = adj[i]
        if n < 0:
            continue
        if n < old_count and face_pivot[n] != INT_MAX:
            sl = aslot[i]                       # reciprocal slot of the shared edge in n
            fc = face_children[n]
            adj[i] = fc[(sl + 1) % 3]           # child_of_edge(n, sl)
            aslot[i] = 0
    tri_adj[t] = adj
    tri_adj_slot[t] = aslot


@wp.kernel
def reassociate(
    points: wp.array(dtype=wp.vec3d),
    tri_v: wp.array(dtype=wp.vec3i),
    s: wp.vec3d,
    face_pivot: wp.array(dtype=wp.int32),
    face_children: wp.array(dtype=wp.vec3i),
    point_owner: wp.array(dtype=wp.int32),
    active_counter: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    t = point_owner[i]
    if t < 0:
        return
    if face_pivot[t] == INT_MAX:
        # owner not split this round: keep association, still outside
        wp.atomic_add(active_counter, 0, 1)
        return
    fc = face_children[t]
    p = points[i]
    new_owner = wp.int32(-1)
    for k in range(3):
        ch = fc[k]
        a, b, c = _tri_pts(points, tri_v, ch)
        if in_cone(s, a, b, c, p):
            if orient3d(a, b, c, p) > wp.float64(0.0):
                new_owner = ch
            else:
                new_owner = -1
            break
    point_owner[i] = new_owner
    if new_owner >= 0:
        wp.atomic_add(active_counter, 0, 1)
