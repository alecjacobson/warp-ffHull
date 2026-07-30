"""Geometric predicates for ffHull, implemented as Warp device functions.

All predicates are built on the single 3D orientation determinant ``orient3d``.
For now this is the fast float64 path; an exact Shewchuk-style fallback and
Simulation-of-Simplicity tie-breaking are layered on later without changing the
combinatorial code that calls these functions.

Sign convention
---------------
``orient3d(a, b, c, d)`` returns the determinant

    | ax-dx  ay-dy  az-dz |
    | bx-dx  by-dy  bz-dz |
    | cx-dx  cy-dy  cz-dz |

which is the signed volume (x6) of tetrahedron (a, b, c, d).  It is positive
when ``d`` sees triangle ``(a, b, c)`` wound clockwise, i.e. when (a, b, c, d)
is a negatively oriented tetrahedron in the usual right-handed sense.  The
combinatorial code never relies on the absolute sign: every face is stored so
that the kernel point ``s`` is *beneath* it, meaning ``orient3d(a, b, c, s) <
0``.  A point ``p`` is then *outside* (beyond) face ``(a, b, c)`` iff
``orient3d(a, b, c, p) > 0``.
"""

import warp as wp

# First-stage float64 error bound for the orient3d determinant expression
# (Shewchuk).  |det| > ERRBOUND * permanent  =>  the sign is certified.
ORIENT3D_ERRBOUND = wp.constant(wp.float64(7.771561172376103e-16))
# Sentinel index for the constructed kernel point s (sorts after all input
# points, i.e. least-dominant SoS perturbation).
S_INDEX = wp.constant(wp.int32(2000000000))


@wp.func
def orient3d(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d) -> wp.float64:
    ax = a[0] - d[0]
    ay = a[1] - d[1]
    az = a[2] - d[2]
    bx = b[0] - d[0]
    by = b[1] - d[1]
    bz = b[2] - d[2]
    cx = c[0] - d[0]
    cy = c[1] - d[1]
    cz = c[2] - d[2]
    return (
        ax * (by * cz - bz * cy)
        - ay * (bx * cz - bz * cx)
        + az * (bx * cy - by * cx)
    )


# ---------------------------------------------------------------------------
# Exact-sign orient3d: certified float64 filter + Simulation of Simplicity
# ---------------------------------------------------------------------------

@wp.func
def _isign(x: wp.float64) -> wp.int32:
    if x > wp.float64(0.0):
        return 1
    return -1


@wp.func
def _o2d(ux: wp.float64, uy: wp.float64, vx: wp.float64, vy: wp.float64,
         wx: wp.float64, wy: wp.float64) -> wp.float64:
    # det[[ux,uy,1],[vx,vy,1],[wx,wy,1]]  (2D orientation of u,v,w)
    return (vx - ux) * (wy - uy) - (vy - uy) * (wx - ux)


@wp.func
def _sos_sign(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d,
              ia: wp.int32, ib: wp.int32, ic: wp.int32, id: wp.int32) -> wp.int32:
    """Simulation-of-Simplicity sign when the determinant is (near) zero.

    Perturb coordinate k of point i by an infinitesimal ordered so the point
    with the smallest index dominates (and x>y>z within a point).  For coplanar
    (but otherwise generic) inputs the answer is the sign of the first nonzero
    first-order cofactor, taken over the index-sorted points.
    """
    # sort the four (point, index) by index with a 5-comparator network,
    # tracking permutation parity in `par`.
    q0 = a; q1 = b; q2 = c; q3 = d
    j0 = ia; j1 = ib; j2 = ic; j3 = id
    par = wp.int32(1)
    # (0,1)
    if j0 > j1:
        t = q0; q0 = q1; q1 = t;  u = j0; j0 = j1; j1 = u;  par = -par
    # (2,3)
    if j2 > j3:
        t = q2; q2 = q3; q3 = t;  u = j2; j2 = j3; j3 = u;  par = -par
    # (0,2)
    if j0 > j2:
        t = q0; q0 = q2; q2 = t;  u = j0; j0 = j2; j2 = u;  par = -par
    # (1,3)
    if j1 > j3:
        t = q1; q1 = q3; q3 = t;  u = j1; j1 = j3; j3 = u;  par = -par
    # (1,2)
    if j1 > j2:
        t = q1; q1 = q2; q2 = t;  u = j1; j1 = j2; j2 = u;  par = -par

    z = wp.float64(0.0)
    # cofactors in dominance order (q0.x, q0.y, q0.z, q1.x, ...); sign (-1)^(i+k)
    # k=0(x)->_o2d over (y,z); k=1(y)->(x,z) negated; k=2(z)->(x,y)
    v = _o2d(q1[1], q1[2], q2[1], q2[2], q3[1], q3[2])            # (q0,x)  +
    if v != z:
        return par * _isign(v)
    v = -_o2d(q1[0], q1[2], q2[0], q2[2], q3[0], q3[2])           # (q0,y)  -
    if v != z:
        return par * _isign(v)
    v = _o2d(q1[0], q1[1], q2[0], q2[1], q3[0], q3[1])            # (q0,z)  +
    if v != z:
        return par * _isign(v)
    v = -_o2d(q0[1], q0[2], q2[1], q2[2], q3[1], q3[2])           # (q1,x)  -
    if v != z:
        return par * _isign(v)
    v = _o2d(q0[0], q0[2], q2[0], q2[2], q3[0], q3[2])            # (q1,y)  +
    if v != z:
        return par * _isign(v)
    v = -_o2d(q0[0], q0[1], q2[0], q2[1], q3[0], q3[1])           # (q1,z)  -
    if v != z:
        return par * _isign(v)
    v = _o2d(q0[1], q0[2], q1[1], q1[2], q3[1], q3[2])            # (q2,x)  +
    if v != z:
        return par * _isign(v)
    v = -_o2d(q0[0], q0[2], q1[0], q1[2], q3[0], q3[2])           # (q2,y)  -
    if v != z:
        return par * _isign(v)
    v = _o2d(q0[0], q0[1], q1[0], q1[1], q3[0], q3[1])            # (q2,z)  +
    if v != z:
        return par * _isign(v)
    v = -_o2d(q0[1], q0[2], q1[1], q1[2], q2[1], q2[2])           # (q3,x)  -
    if v != z:
        return par * _isign(v)
    v = _o2d(q0[0], q0[2], q1[0], q1[2], q2[0], q2[2])            # (q3,y)  +
    if v != z:
        return par * _isign(v)
    v = -_o2d(q0[0], q0[1], q1[0], q1[1], q2[0], q2[1])           # (q3,z)  -
    if v != z:
        return par * _isign(v)
    # fully degenerate (collinear): deterministic nonzero fallback
    return par


@wp.func
def o3d_sign(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d,
             ia: wp.int32, ib: wp.int32, ic: wp.int32, id: wp.int32) -> wp.int32:
    """Robust sign of orient3d(a,b,c,d) in {-1,+1}.

    Certified float64 filter first; on an uncertain (near-degenerate) result,
    fall back to Simulation of Simplicity keyed on the point indices, so the
    combinatorial algorithm sees a consistent, tie-free orientation.
    """
    adx = a[0] - d[0]; ady = a[1] - d[1]; adz = a[2] - d[2]
    bdx = b[0] - d[0]; bdy = b[1] - d[1]; bdz = b[2] - d[2]
    cdx = c[0] - d[0]; cdy = c[1] - d[1]; cdz = c[2] - d[2]
    bdxcdy = bdx * cdy; cdxbdy = cdx * bdy
    cdxady = cdx * ady; adxcdy = adx * cdy
    adxbdy = adx * bdy; bdxady = bdx * ady
    det = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady)
    perm = ((wp.abs(bdxcdy) + wp.abs(cdxbdy)) * wp.abs(adz)
            + (wp.abs(cdxady) + wp.abs(adxcdy)) * wp.abs(bdz)
            + (wp.abs(adxbdy) + wp.abs(bdxady)) * wp.abs(cdz))
    eb = ORIENT3D_ERRBOUND * perm
    if det > eb:
        return 1
    if det < -eb:
        return -1
    return _sos_sign(a, b, c, d, ia, ib, ic, id)


@wp.func
def in_cone_i(s: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, p: wp.vec3d,
              ia: wp.int32, ib: wp.int32, ic: wp.int32, ip: wp.int32) -> bool:
    """Robust (SoS) in-cone test.  ``s`` is the kernel point (index S_INDEX);
    a,b,c,p are input points with the given indices."""
    si = S_INDEX
    if o3d_sign(s, a, b, p, si, ia, ib, ip) != o3d_sign(s, a, b, c, si, ia, ib, ic):
        return False
    if o3d_sign(s, b, c, p, si, ib, ic, ip) != o3d_sign(s, b, c, a, si, ib, ic, ia):
        return False
    if o3d_sign(s, c, a, p, si, ic, ia, ip) != o3d_sign(s, c, a, b, si, ic, ia, ib):
        return False
    return True


@wp.func
def in_cone(
    s: wp.vec3d,
    a: wp.vec3d,
    b: wp.vec3d,
    c: wp.vec3d,
    p: wp.vec3d,
) -> bool:
    """True if point ``p`` lies inside the cone C_s(triangle abc).

    The cone is the convex hull of the three rays s->a, s->b, s->c.  A point is
    inside iff it lies on the same side of each of the three planes
    (s,a,b), (s,b,c), (s,c,a) as the remaining vertex of the triangle.  The
    base plane (a,b,c) is *not* tested because the cone extends to infinity.
    """
    d_ab_p = orient3d(s, a, b, p)
    d_ab_c = orient3d(s, a, b, c)
    if _diff_sign(d_ab_p, d_ab_c):
        return False
    d_bc_p = orient3d(s, b, c, p)
    d_bc_a = orient3d(s, b, c, a)
    if _diff_sign(d_bc_p, d_bc_a):
        return False
    d_ca_p = orient3d(s, c, a, p)
    d_ca_b = orient3d(s, c, a, b)
    if _diff_sign(d_ca_p, d_ca_b):
        return False
    return True


@wp.func
def _diff_sign(x: wp.float64, y: wp.float64) -> bool:
    """True if x and y have strictly opposite signs.

    A zero on either side is treated as "same side" (boundary counts as
    inside) so that a point exactly on a cone wall is still claimed by the
    cone; ownership ambiguity on walls is resolved later by tie-breaking.
    """
    z = wp.float64(0.0)
    if x > z and y < z:
        return True
    if x < z and y > z:
        return True
    return False
