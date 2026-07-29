"""Phase B: Flip-Flop convexification via parallel local edge flips.

Each round runs two kernels:
  * ``propose_claim``: every triangle inspects its canonical edges, decides
    (per the V/D criteria) whether to flip one, records the proposal, and
    ``atomic_min``-claims every triangle in the flip's local footprint with its
    own index.
  * ``apply``: a proposer performs its flip only if it still owns every claimed
    triangle, guaranteeing conflict-free parallel flips.

This file currently implements the 2->2 flip and the V-criterion (flip reflex
flippable edges).  3->1 flips and the D-criterion are layered on next.
"""

import warp as wp

from .predicates import orient3d, in_cone

INT_MAX = wp.constant(wp.int32(2147483647))


@wp.func
def slot_of(tv: wp.vec3i, val: wp.int32) -> wp.int32:
    if tv[0] == val:
        return 0
    if tv[1] == val:
        return 1
    return 2


@wp.struct
class Flip22:
    ok: wp.int32          # 1 if a well-formed (non-degenerate footprint) 2-2 flip
    t1: wp.int32
    t2: wp.int32
    a: wp.int32
    b: wp.int32
    c: wp.int32
    d: wp.int32
    n_ca: wp.int32
    s_ca: wp.int32
    n_bc: wp.int32
    s_bc: wp.int32
    n_ad: wp.int32
    s_ad: wp.int32
    n_bd: wp.int32
    s_bd: wp.int32


@wp.func
def gather22(
    tri_v: wp.array(dtype=wp.vec3i),
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    t1: wp.int32,
    i1: wp.int32,
) -> Flip22:
    f = Flip22()
    f.ok = 0
    tv = tri_v[t1]
    adj1 = tri_adj[t1]
    aslot1 = tri_adj_slot[t1]
    t2 = adj1[i1]
    j1 = aslot1[i1]
    if t2 < 0:
        return f
    nv = tri_v[t2]
    adj2 = tri_adj[t2]
    aslot2 = tri_adj_slot[t2]

    sa1 = (i1 + 1) % 3
    sb1 = (i1 + 2) % 3
    c = tv[i1]
    a = tv[sa1]
    b = tv[sb1]
    d = nv[j1]
    sa2 = slot_of(nv, a)
    sb2 = slot_of(nv, b)

    # external neighbours around the quad (a,d,b,c)
    n_ca = adj1[sb1]   # edge (c,a) opposite b in t1
    s_ca = aslot1[sb1]
    n_bc = adj1[sa1]   # edge (b,c) opposite a in t1
    s_bc = aslot1[sa1]
    n_ad = adj2[sb2]   # edge (a,d) opposite b in t2
    s_ad = aslot2[sb2]
    n_bd = adj2[sa2]   # edge (b,d) opposite a in t2
    s_bd = aslot2[sa2]

    # Reject degenerate footprints (shared external neighbours / self loops):
    # these indicate a 3-1 edge or a tiny closed component, not a plain 2-2.
    if n_ca == t1 or n_ca == t2 or n_bc == t1 or n_bc == t2:
        return f
    if n_ad == t1 or n_ad == t2 or n_bd == t1 or n_bd == t2:
        return f
    if n_ca == n_bc or n_ca == n_ad or n_ca == n_bd:
        return f
    if n_bc == n_ad or n_bc == n_bd or n_ad == n_bd:
        return f

    f.ok = 1
    f.t1 = t1; f.t2 = t2
    f.a = a; f.b = b; f.c = c; f.d = d
    f.n_ca = n_ca; f.s_ca = s_ca
    f.n_bc = n_bc; f.s_bc = s_bc
    f.n_ad = n_ad; f.s_ad = s_ad
    f.n_bd = n_bd; f.s_bd = s_bd
    return f


@wp.func
def is_reflex(points: wp.array(dtype=wp.vec3d), tv: wp.vec3i, d: wp.int32) -> bool:
    # t1 stored outward (s beneath) => reflex iff neighbour apex d is outside t1's plane
    return orient3d(points[tv[0]], points[tv[1]], points[tv[2]], points[d]) > wp.float64(0.0)


@wp.func
def flippable22(points: wp.array(dtype=wp.vec3d), s: wp.vec3d,
                a: wp.int32, b: wp.int32, c: wp.int32, d: wp.int32) -> bool:
    pa = points[a]; pb = points[b]; pc = points[c]; pd = points[d]
    # union of cones == CH of the 4 rays  <=>  neither edge endpoint lies inside
    # the cone of the triangle formed by the other three vertices.
    if in_cone(s, pb, pc, pd, pa):
        return False
    if in_cone(s, pa, pc, pd, pb):
        return False
    return True


@wp.kernel
def propose_claim(
    points: wp.array(dtype=wp.vec3d),
    s: wp.vec3d,
    tri_v: wp.array(dtype=wp.vec3i),
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    tri_active: wp.array(dtype=wp.int32),
    tri_count: wp.int32,
    tri_claim: wp.array(dtype=wp.int32),
    prop_slot: wp.array(dtype=wp.int32),
    prop_type: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    if t >= tri_count or tri_active[t] == 0:
        return
    adj = tri_adj[t]
    for i in range(3):
        n = adj[i]
        if n < 0 or t > n:
            continue  # only the smaller-index triangle proposes an edge
        f = gather22(tri_v, tri_adj, tri_adj_slot, t, i)
        if f.ok == 0:
            continue
        tv = tri_v[t]
        if not is_reflex(points, tv, f.d):
            continue
        if not flippable22(points, s, f.a, f.b, f.c, f.d):
            continue
        # V-criterion: flip this reflex flippable 2-2 edge.
        prop_slot[t] = i
        prop_type[t] = 2
        wp.atomic_min(tri_claim, f.t1, t)
        wp.atomic_min(tri_claim, f.t2, t)
        wp.atomic_min(tri_claim, f.n_ca, t)
        wp.atomic_min(tri_claim, f.n_bc, t)
        wp.atomic_min(tri_claim, f.n_ad, t)
        wp.atomic_min(tri_claim, f.n_bd, t)
        return


@wp.kernel
def apply_flips(
    tri_v: wp.array(dtype=wp.vec3i),
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    tri_count: wp.int32,
    tri_claim: wp.array(dtype=wp.int32),
    prop_slot: wp.array(dtype=wp.int32),
    prop_type: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    if t >= tri_count or prop_type[t] != 2:
        return
    i1 = prop_slot[t]
    f = gather22(tri_v, tri_adj, tri_adj_slot, t, i1)
    if f.ok == 0:
        return
    # ownership: every footprint triangle still claimed by us
    if tri_claim[f.t1] != t or tri_claim[f.t2] != t:
        return
    if tri_claim[f.n_ca] != t or tri_claim[f.n_bc] != t:
        return
    if tri_claim[f.n_ad] != t or tri_claim[f.n_bd] != t:
        return

    a = f.a; b = f.b; c = f.c; d = f.d
    t1 = f.t1; t2 = f.t2

    # new_t1 (slot t1) = (a, d, c); new_t2 (slot t2) = (d, b, c); new edge (c,d)
    tri_v[t1] = wp.vec3i(a, d, c)
    tri_adj[t1] = wp.vec3i(t2, f.n_ca, f.n_ad)
    tri_adj_slot[t1] = wp.vec3i(1, f.s_ca, f.s_ad)

    tri_v[t2] = wp.vec3i(d, b, c)
    tri_adj[t2] = wp.vec3i(f.n_bc, t1, f.n_bd)
    tri_adj_slot[t2] = wp.vec3i(f.s_bc, 0, f.s_bd)

    # external back-pointers (we own these triangles this round)
    na = tri_adj[f.n_ca]; na[f.s_ca] = t1; tri_adj[f.n_ca] = na
    sa = tri_adj_slot[f.n_ca]; sa[f.s_ca] = 1; tri_adj_slot[f.n_ca] = sa

    nb = tri_adj[f.n_ad]; nb[f.s_ad] = t1; tri_adj[f.n_ad] = nb
    sb = tri_adj_slot[f.n_ad]; sb[f.s_ad] = 2; tri_adj_slot[f.n_ad] = sb

    nc = tri_adj[f.n_bc]; nc[f.s_bc] = t2; tri_adj[f.n_bc] = nc
    sc = tri_adj_slot[f.n_bc]; sc[f.s_bc] = 0; tri_adj_slot[f.n_bc] = sc

    nd = tri_adj[f.n_bd]; nd[f.s_bd] = t2; tri_adj[f.n_bd] = nd
    sd = tri_adj_slot[f.n_bd]; sd[f.s_bd] = 2; tri_adj_slot[f.n_bd] = sd

    changed[0] = 1


@wp.kernel
def reset_flip(tri_claim: wp.array(dtype=wp.int32),
               prop_slot: wp.array(dtype=wp.int32),
               prop_type: wp.array(dtype=wp.int32)):
    t = wp.tid()
    tri_claim[t] = INT_MAX
    prop_slot[t] = -1
    prop_type[t] = 0
