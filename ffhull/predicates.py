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
