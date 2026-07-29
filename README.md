# ffHull-warp

GPU-resident 3D convex hull in **pure [NVIDIA Warp](https://github.com/NVIDIA/warp)**,
implementing the **Flip-Flop / ffHull** algorithm of Gao, Cao, Tan & Huang
(*"Flip-Flop: Convex Hull Construction via Star-Shaped Polyhedron in 3D"*,
[paper](https://www.comp.nus.edu.sg/~tants/flipflop_files/flipflop.pdf)).

Everything but the initial tetrahedron and lower-dimensional fallbacks runs on
the GPU as Warp kernels. The only geometric predicate is `orient3d`.

## Algorithm

Two fully data-parallel phases:

1. **Grow a star-shaped polyhedron** (`ffhull/grow.py`). Start from a seed
   tetrahedron with kernel point `s` = centroid. Each point is associated to the
   face whose cone (from `s`) contains it. Each round: pick the furthest point
   per active face, split that face into three (`vab, vbc, vca`), repair
   adjacency, and re-home points onto the three children (dropping points that
   fell beneath the surface). Repeat until no points remain.

2. **Flip-Flop convexification** (`ffhull/flip.py`). Repeatedly flip edges until
   no reflex edge remains — the convex hull. Uses both flip families and both
   criteria from Algorithm 1 of the paper:
   - **V-criterion**: flip reflex flippable 2→2 / 3→1 edges (increase volume).
   - **D-criterion**: label non-extreme vertices (via unflippable reflex 2→2
     edges), reduce their degree with 2→2 "flops", and remove them with a 3→1
     flip, prioritising the smallest-index non-extreme vertex locally.
   Parallel flips are made conflict-free with an `atomic_min` claim over each
   flip's triangle footprint; a flip applies only if its proposer still owns the
   whole footprint (guaranteeing progress + no races).

### Design notes
- Fixed-capacity SoA topology (`ffhull/mesh.py`), ≤ `2n-4` triangles; the two
  new triangles per split are allocated with one atomic counter.
- Reciprocal adjacency slots stored explicitly → constant-time split/flip
  updates. Adjacency repair is a **pure gather** (each triangle rewrites only its
  own links) so there are no cross-triangle write races.
- Faces oriented so `orient3d(face, s) < 0`; then "p outside face" ⇔
  `orient3d(face, p) > 0`, and an edge is reflex ⇔ the neighbour apex is outside.

## Status
- [x] Phase A: parallel star-shaped growth
- [x] Phase B: Flip-Flop convexification (2→2 + 3→1, V & D criteria)
- [x] Lower-dimensional inputs (coincident / collinear / coplanar) + duplicates
- [ ] Exact Shewchuk `orient3d` + Simulation of Simplicity (see
  `warp_orient3d_plan.md`) — needed for exactly-coplanar facets and aspect
  ratios beyond ~1e6.

Matches `scipy.spatial.ConvexHull` exactly on sphere, gaussian, uniform-cube,
and clustered inputs up to n = 50k in the test suite (and larger in `bench.py`).

## Usage
```python
import numpy as np
from ffhull.hull import convex_hull

pts = np.random.standard_normal((100_000, 3))
faces = convex_hull(pts, device="cuda:0")          # (m, 3) outward triangles
faces, verts = convex_hull(pts, return_vertices=True)
```

## Test & benchmark
```
python3 tests/test_predicates.py
python3 tests/test_growth.py
python3 tests/test_hull.py
python3 tests/test_degenerate.py       # or: pytest tests/
python3 bench.py
```
