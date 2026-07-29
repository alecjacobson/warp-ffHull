# ffHull-warp

GPU-resident exact 3D convex hull in **pure [NVIDIA Warp](https://github.com/NVIDIA/warp)**,
implementing the **Flip-Flop / ffHull** algorithm of Gao, Cao, Tan & Huang
(*"Flip-Flop: Convex Hull Construction via Star-Shaped Polyhedron in 3D"*).

Two phases, both fully data-parallel on the GPU:

1. **Grow** a star-shaped polyhedron from a seed tetrahedron by inserting the
   furthest point into every active face in parallel.
2. **Flip-Flop**: convexify via parallel local 2->2 and 3->1 edge flips
   (V-criterion for volume, D-criterion to remove non-extreme vertices),
   with `atomic_min` conflict resolution.

The only geometric predicate is `orient3d`.

## Status
- [x] Phase A: parallel star-shaped growth (`ffhull/grow.py`)
- [x] Phase B: Flip-Flop convexification (2->2 + 3->1, V & D criteria)
- [ ] Exact predicate + Simulation of Simplicity
- [ ] Degenerate (coincident / collinear / coplanar) handling

## Test
```
python3 tests/test_predicates.py
python3 tests/test_growth.py
```
