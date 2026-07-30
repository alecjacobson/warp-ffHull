"""Phase B: Flip-Flop convexification via parallel local edge flips.

Implements Algorithm 1 of the paper with both flip families and both criteria:

  * V-criterion: flip a reflex flippable edge (increases volume).
  * D-criterion: remove non-extreme vertices (labelled via unflippable reflex
    2-2 edges) by reducing their degree with 2-2 "flops" and finishing with a
    3-1 flip, prioritising the smallest-index non-extreme vertex locally.

Each round runs three kernels:
  1. ``label_kernel``    : label non-extreme vertices (Fig 4a config).
  2. ``propose_claim``   : per canonical edge decide a flip, record it, and
     ``atomic_min``-claim every triangle in its footprint with the proposer id.
  3. ``apply_flips``     : a proposer flips only if it still owns its whole
     footprint, giving conflict-free parallel flips.

Orientation invariant: every triangle is stored so the kernel point ``s`` is
beneath it (orient3d(face, s) < 0); a point is outside a face iff
orient3d(face, p) > 0, and an edge is reflex iff the neighbour apex is outside.
"""

import warp as wp

from .predicates import orient3d, in_cone

INT_MAX = wp.constant(wp.int32(2147483647))

# FlipInfo.kind values
KIND_NONE = wp.constant(wp.int32(0))
KIND_22 = wp.constant(wp.int32(2))
KIND_31A = wp.constant(wp.int32(31))   # 3-1 removing apex a
KIND_31B = wp.constant(wp.int32(32))   # 3-1 removing apex b


@wp.func
def slot_of(tv: wp.vec3i, val: wp.int32) -> wp.int32:
    if tv[0] == val:
        return 0
    if tv[1] == val:
        return 1
    return 2


@wp.struct
class FlipInfo:
    kind: wp.int32
    t1: wp.int32
    t2: wp.int32
    a: wp.int32
    b: wp.int32
    c: wp.int32
    d: wp.int32
    # 2-2 external ring, and their reciprocal slots
    n_ca: wp.int32
    s_ca: wp.int32
    n_bc: wp.int32
    s_bc: wp.int32
    n_ad: wp.int32
    s_ad: wp.int32
    n_bd: wp.int32
    s_bd: wp.int32
    # 3-1 extras: third triangle and the cd-edge external neighbour
    t3: wp.int32
    n_cd: wp.int32
    s_cd: wp.int32


@wp.func
def gather(
    tri_v: wp.array(dtype=wp.vec3i),
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    t1: wp.int32,
    i1: wp.int32,
) -> FlipInfo:
    f = FlipInfo()
    f.kind = KIND_NONE
    adj1 = tri_adj[t1]
    aslot1 = tri_adj_slot[t1]
    t2 = adj1[i1]
    j1 = aslot1[i1]
    if t2 < 0:
        return f
    tv = tri_v[t1]
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

    n_ca = adj1[sb1]; s_ca = aslot1[sb1]   # edge (c,a) opposite b in t1
    n_bc = adj1[sa1]; s_bc = aslot1[sa1]   # edge (b,c) opposite a in t1
    n_ad = adj2[sb2]; s_ad = aslot2[sb2]   # edge (a,d) opposite b in t2
    n_bd = adj2[sa2]; s_bd = aslot2[sa2]   # edge (b,d) opposite a in t2

    f.t1 = t1; f.t2 = t2
    f.a = a; f.b = b; f.c = c; f.d = d
    f.n_ca = n_ca; f.s_ca = s_ca
    f.n_bc = n_bc; f.s_bc = s_bc
    f.n_ad = n_ad; f.s_ad = s_ad
    f.n_bd = n_bd; f.s_bd = s_bd

    # 3-1 detection: a shared external neighbour means the link edge (c,d)
    # already forms a triangle, and the corresponding endpoint has degree 3.
    if n_ca == n_ad and n_ca != t1 and n_ca != t2 and n_ca >= 0:
        t3 = n_ca                         # triangle a-c-d
        sa3 = slot_of(tri_v[t3], a)
        f.kind = KIND_31A
        f.t3 = t3
        f.n_cd = tri_adj[t3][sa3]
        f.s_cd = tri_adj_slot[t3][sa3]
        if _bad31(f, n_bc, n_bd):
            f.kind = KIND_NONE
        return f
    if n_bc == n_bd and n_bc != t1 and n_bc != t2 and n_bc >= 0:
        t3 = n_bc                         # triangle b-c-d
        sb3 = slot_of(tri_v[t3], b)
        f.kind = KIND_31B
        f.t3 = t3
        f.n_cd = tri_adj[t3][sb3]
        f.s_cd = tri_adj_slot[t3][sb3]
        if _bad31(f, n_ca, n_ad):
            f.kind = KIND_NONE
        return f

    # plain 2-2 edge: reject degenerate footprints (tiny closed components)
    if n_ca == t1 or n_ca == t2 or n_bc == t1 or n_bc == t2:
        return f
    if n_ad == t1 or n_ad == t2 or n_bd == t1 or n_bd == t2:
        return f
    if n_ca == n_bc or n_ca == n_bd or n_bc == n_ad or n_ad == n_bd:
        return f
    f.kind = KIND_22
    return f


@wp.func
def _bad31(f: FlipInfo, ex1: wp.int32, ex2: wp.int32) -> bool:
    # the three surviving external neighbours (ex1, ex2, n_cd) must be distinct
    # and disjoint from the three subcomplex triangles (t1, t2, t3).
    ncd = f.n_cd
    if ncd < 0:
        return True
    if ncd == f.t1 or ncd == f.t2 or ncd == f.t3:
        return True
    if ex1 == ex2 or ex1 == ncd or ex2 == ncd:
        return True
    if ex1 == f.t1 or ex1 == f.t2 or ex1 == f.t3:
        return True
    if ex2 == f.t1 or ex2 == f.t2 or ex2 == f.t3:
        return True
    return False


@wp.func
def is_reflex(points: wp.array(dtype=wp.vec3d), tv: wp.vec3i, d: wp.int32) -> bool:
    return orient3d(points[tv[0]], points[tv[1]], points[tv[2]], points[d]) > wp.float64(0.0)


@wp.func
def flippable22(points: wp.array(dtype=wp.vec3d), s: wp.vec3d,
                a: wp.int32, b: wp.int32, c: wp.int32, d: wp.int32) -> bool:
    pa = points[a]; pb = points[b]; pc = points[c]; pd = points[d]
    if in_cone(s, pb, pc, pd, pa):
        return False
    if in_cone(s, pa, pc, pd, pb):
        return False
    return True


@wp.func
def _same_side(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, p: wp.vec3d, q: wp.vec3d) -> bool:
    dp = orient3d(a, b, c, p)
    dq = orient3d(a, b, c, q)
    if dp > wp.float64(0.0) and dq < wp.float64(0.0):
        return False
    if dp < wp.float64(0.0) and dq > wp.float64(0.0):
        return False
    return True


@wp.func
def s_in_tetra(points: wp.array(dtype=wp.vec3d), s: wp.vec3d,
               a: wp.int32, b: wp.int32, c: wp.int32, d: wp.int32) -> bool:
    pa = points[a]; pb = points[b]; pc = points[c]; pd = points[d]
    if not _same_side(pa, pb, pc, pd, s):
        return False
    if not _same_side(pa, pb, pd, pc, s):
        return False
    if not _same_side(pa, pc, pd, pb, s):
        return False
    if not _same_side(pb, pc, pd, pa, s):
        return False
    return True


@wp.func
def smallest_nonextreme(label: wp.array(dtype=wp.int32),
                        a: wp.int32, b: wp.int32, c: wp.int32, d: wp.int32) -> wp.int32:
    x = INT_MAX
    if label[a] == 1 and a < x:
        x = a
    if label[b] == 1 and b < x:
        x = b
    if label[c] == 1 and c < x:
        x = c
    if label[d] == 1 and d < x:
        x = d
    return x


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@wp.kernel
def label_kernel(
    points: wp.array(dtype=wp.vec3d),
    s: wp.vec3d,
    tri_v: wp.array(dtype=wp.vec3i),
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    tri_active: wp.array(dtype=wp.int32),
    tri_count: wp.array(dtype=wp.int32),
    vertex_label: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    if t >= tri_count[0] or tri_active[t] == 0:
        return
    adj = tri_adj[t]
    for i in range(3):
        n = adj[i]
        if n < 0 or t > n:
            continue
        f = gather(tri_v, tri_adj, tri_adj_slot, t, i)
        if f.kind != KIND_22:
            continue
        if flippable22(points, s, f.a, f.b, f.c, f.d):
            continue
        # unflippable 2-2: label the endpoint inside the cone of the other three
        if not is_reflex(points, tri_v[t], f.d):
            continue
        inside = f.a
        if not in_cone(s, points[f.b], points[f.c], points[f.d], points[f.a]):
            inside = f.b
        if vertex_label[inside] == 0:
            vertex_label[inside] = 1
            changed[0] = 1


@wp.kernel
def propose_claim(
    points: wp.array(dtype=wp.vec3d),
    s: wp.vec3d,
    tri_v: wp.array(dtype=wp.vec3i),
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    tri_active: wp.array(dtype=wp.int32),
    tri_count: wp.array(dtype=wp.int32),
    vertex_label: wp.array(dtype=wp.int32),
    tri_claim: wp.array(dtype=wp.int32),
    prop_slot: wp.array(dtype=wp.int32),
    prop_type: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    if t >= tri_count[0] or tri_active[t] == 0:
        return
    adj = tri_adj[t]
    for i in range(3):
        n = adj[i]
        if n < 0 or t > n:
            continue
        f = gather(tri_v, tri_adj, tri_adj_slot, t, i)
        if f.kind == KIND_NONE:
            continue
        reflex = is_reflex(points, tri_v[t], f.d)

        do_flip = False
        if f.kind == KIND_31A:
            if vertex_label[f.a] == 1 or reflex:
                do_flip = True
        elif f.kind == KIND_31B:
            if vertex_label[f.b] == 1 or reflex:
                do_flip = True
        else:  # KIND_22
            if flippable22(points, s, f.a, f.b, f.c, f.d):
                x = smallest_nonextreme(vertex_label, f.a, f.b, f.c, f.d)
                if x == INT_MAX:
                    if reflex:
                        do_flip = True
                elif (x == f.a or x == f.b) and (not s_in_tetra(points, s, f.a, f.b, f.c, f.d)):
                    do_flip = True

        if not do_flip:
            continue

        prop_slot[t] = i
        prop_type[t] = f.kind
        wp.atomic_min(tri_claim, f.t1, t)
        wp.atomic_min(tri_claim, f.t2, t)
        if f.kind == KIND_22:
            wp.atomic_min(tri_claim, f.n_ca, t)
            wp.atomic_min(tri_claim, f.n_bc, t)
            wp.atomic_min(tri_claim, f.n_ad, t)
            wp.atomic_min(tri_claim, f.n_bd, t)
        else:
            wp.atomic_min(tri_claim, f.t3, t)
            wp.atomic_min(tri_claim, f.n_cd, t)
            if f.kind == KIND_31A:
                wp.atomic_min(tri_claim, f.n_bc, t)
                wp.atomic_min(tri_claim, f.n_bd, t)
            else:
                wp.atomic_min(tri_claim, f.n_ca, t)
                wp.atomic_min(tri_claim, f.n_ad, t)
        return


@wp.func
def _set_adj(tri_adj: wp.array(dtype=wp.vec3i), tri_adj_slot: wp.array(dtype=wp.vec3i),
             t: wp.int32, k: wp.int32, nb: wp.int32, recip: wp.int32):
    a = tri_adj[t]; a[k] = nb; tri_adj[t] = a
    sl = tri_adj_slot[t]; sl[k] = recip; tri_adj_slot[t] = sl


@wp.func
def _wire_new_tri(
    tri_v: wp.array(dtype=wp.vec3i),
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    nt: wp.int32, u0: wp.int32, u1: wp.int32, u2: wp.int32,
    e0a: wp.int32, e0b: wp.int32, nb0: wp.int32, sl0: wp.int32,
    e1a: wp.int32, e1b: wp.int32, nb1: wp.int32, sl1: wp.int32,
    e2a: wp.int32, e2b: wp.int32, nb2: wp.int32, sl2: wp.int32,
):
    # Write triangle (u0,u1,u2) and connect its three edges to given neighbours,
    # matching by vertex pair (three neighbours = edges {e0},{e1},{e2}).
    tri_v[nt] = wp.vec3i(u0, u1, u2)
    for k in range(3):
        p = u1
        q = u2
        if k == 1:
            p = u2; q = u0
        if k == 2:
            p = u0; q = u1
        nb = wp.int32(-1)
        recip = wp.int32(-1)
        if (p == e0a and q == e0b) or (p == e0b and q == e0a):
            nb = nb0; recip = sl0
        elif (p == e1a and q == e1b) or (p == e1b and q == e1a):
            nb = nb1; recip = sl1
        else:
            nb = nb2; recip = sl2
        _set_adj(tri_adj, tri_adj_slot, nt, k, nb, recip)
        # reciprocal back-pointer (we own nb this round)
        _set_adj(tri_adj, tri_adj_slot, nb, recip, nt, k)


@wp.kernel
def apply_flips(
    points: wp.array(dtype=wp.vec3d),
    s: wp.vec3d,
    tri_v: wp.array(dtype=wp.vec3i),
    tri_adj: wp.array(dtype=wp.vec3i),
    tri_adj_slot: wp.array(dtype=wp.vec3i),
    tri_active: wp.array(dtype=wp.int32),
    tri_count: wp.array(dtype=wp.int32),
    tri_claim: wp.array(dtype=wp.int32),
    prop_slot: wp.array(dtype=wp.int32),
    prop_type: wp.array(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    if t >= tri_count[0] or prop_type[t] == 0:
        return
    i1 = prop_slot[t]
    f = gather(tri_v, tri_adj, tri_adj_slot, t, i1)
    if f.kind == KIND_NONE or f.kind != prop_type[t]:
        return

    # ownership
    if tri_claim[f.t1] != t or tri_claim[f.t2] != t:
        return
    if f.kind == KIND_22:
        if tri_claim[f.n_ca] != t or tri_claim[f.n_bc] != t:
            return
        if tri_claim[f.n_ad] != t or tri_claim[f.n_bd] != t:
            return
        _apply_22(tri_v, tri_adj, tri_adj_slot, f)
    else:
        if tri_claim[f.t3] != t or tri_claim[f.n_cd] != t:
            return
        if f.kind == KIND_31A:
            if tri_claim[f.n_bc] != t or tri_claim[f.n_bd] != t:
                return
        else:
            if tri_claim[f.n_ca] != t or tri_claim[f.n_ad] != t:
                return
        _apply_31(points, s, tri_v, tri_adj, tri_adj_slot, tri_active, f)
    changed[0] = 1


@wp.func
def _apply_22(tri_v: wp.array(dtype=wp.vec3i), tri_adj: wp.array(dtype=wp.vec3i),
              tri_adj_slot: wp.array(dtype=wp.vec3i), f: FlipInfo):
    a = f.a; b = f.b; c = f.c; d = f.d
    t1 = f.t1; t2 = f.t2
    # new_t1 (slot t1) = (a,d,c); new_t2 (slot t2) = (d,b,c); new edge (c,d)
    tri_v[t1] = wp.vec3i(a, d, c)
    tri_adj[t1] = wp.vec3i(t2, f.n_ca, f.n_ad)
    tri_adj_slot[t1] = wp.vec3i(1, f.s_ca, f.s_ad)

    tri_v[t2] = wp.vec3i(d, b, c)
    tri_adj[t2] = wp.vec3i(f.n_bc, t1, f.n_bd)
    tri_adj_slot[t2] = wp.vec3i(f.s_bc, 0, f.s_bd)

    _set_adj(tri_adj, tri_adj_slot, f.n_ca, f.s_ca, t1, 1)
    _set_adj(tri_adj, tri_adj_slot, f.n_ad, f.s_ad, t1, 2)
    _set_adj(tri_adj, tri_adj_slot, f.n_bc, f.s_bc, t2, 0)
    _set_adj(tri_adj, tri_adj_slot, f.n_bd, f.s_bd, t2, 2)


@wp.func
def _apply_31(points: wp.array(dtype=wp.vec3d), s: wp.vec3d,
              tri_v: wp.array(dtype=wp.vec3i), tri_adj: wp.array(dtype=wp.vec3i),
              tri_adj_slot: wp.array(dtype=wp.vec3i), tri_active: wp.array(dtype=wp.int32),
              f: FlipInfo):
    # Remove the degree-3 apex; the three subcomplex triangles collapse to one.
    if f.kind == KIND_31A:
        # keep triangle (b,c,d), edges bc->n_bc, cd->n_cd, db->n_bd
        u0 = f.b; u1 = f.c; u2 = f.d
        if orient3d(points[u0], points[u1], points[u2], s) > wp.float64(0.0):
            u1 = f.d; u2 = f.c
        _wire_new_tri(tri_v, tri_adj, tri_adj_slot, f.t1, u0, u1, u2,
                      f.b, f.c, f.n_bc, f.s_bc,
                      f.c, f.d, f.n_cd, f.s_cd,
                      f.d, f.b, f.n_bd, f.s_bd)
    else:
        # keep triangle (a,c,d), edges ac->n_ca, cd->n_cd, da->n_ad
        u0 = f.a; u1 = f.c; u2 = f.d
        if orient3d(points[u0], points[u1], points[u2], s) > wp.float64(0.0):
            u1 = f.d; u2 = f.c
        _wire_new_tri(tri_v, tri_adj, tri_adj_slot, f.t1, u0, u1, u2,
                      f.a, f.c, f.n_ca, f.s_ca,
                      f.c, f.d, f.n_cd, f.s_cd,
                      f.d, f.a, f.n_ad, f.s_ad)
    tri_active[f.t2] = 0
    tri_active[f.t3] = 0


@wp.kernel
def reset_flip(tri_claim: wp.array(dtype=wp.int32),
               prop_slot: wp.array(dtype=wp.int32),
               prop_type: wp.array(dtype=wp.int32)):
    t = wp.tid()
    tri_claim[t] = INT_MAX
    prop_slot[t] = -1
    prop_type[t] = 0


@wp.kernel
def check_contains(points: wp.array(dtype=wp.vec3d),
                   tri_v: wp.array(dtype=wp.vec3i),
                   tri_active: wp.array(dtype=wp.int32),
                   tri_count: wp.int32,
                   tol: wp.float64,
                   flag: wp.array(dtype=wp.int32)):
    # O(n*F) global validity: set flag if point i is strictly outside any face
    # (i.e. the surface does not enclose it) -> the hull is invalid/tangled.
    i = wp.tid()
    p = points[i]
    for t in range(tri_count):
        if tri_active[t] == 0:
            continue
        tv = tri_v[t]
        if orient3d(points[tv[0]], points[tv[1]], points[tv[2]], p) > tol:
            flag[0] = 1
            return


@wp.kernel
def check_convex(points: wp.array(dtype=wp.vec3d),
                 tri_v: wp.array(dtype=wp.vec3i),
                 tri_adj: wp.array(dtype=wp.vec3i),
                 tri_adj_slot: wp.array(dtype=wp.vec3i),
                 tri_active: wp.array(dtype=wp.int32),
                 tri_count: wp.array(dtype=wp.int32),
                 tol: wp.float64,
                 flag: wp.array(dtype=wp.int32)):
    # O(F) convexity test: set flag if any edge is reflex beyond tol (neighbour
    # apex strictly outside this face's plane).
    t = wp.tid()
    if t >= tri_count[0] or tri_active[t] == 0:
        return
    tv = tri_v[t]
    a = points[tv[0]]; b = points[tv[1]]; c = points[tv[2]]
    adj = tri_adj[t]
    aslot = tri_adj_slot[t]
    for i in range(3):
        n = adj[i]
        if n < 0:
            continue
        d = tri_v[n][aslot[i]]
        if orient3d(a, b, c, points[d]) > tol:
            flag[0] = 1
