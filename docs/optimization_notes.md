# Optimization notes

Profiling target: `alecjacobson/threedscans` (1.8–6.4 M points, small hulls) and
synthetic sphere/gaussian, on an NVIDIA L40 (Ada). All numbers end-to-end,
including host↔device transfer.

## What worked (committed)

Almost all the wall-clock was host↔device overhead, not hull compute:

1. **GPU-resident seed + slice-write init** — extreme-point search and the
   affine-dimension test as Warp reductions; the seed tetra written to 4 slots
   instead of reading back the whole triangle array. Removed a ~0.7 s host
   preamble at n = 1 M.
2. **Sync-free CUDA-graph loops** — device-side counts + `wp.capture_while`, so
   growth and flip run to convergence with zero per-round host syncs.
3. **`wp.empty` allocation** — the large topology / scratch arrays are fully
   written before read (per-round reset kernels, `init_tetra`, `split`), so skip
   the ~0.3 GB memset. Plus a lightweight dimension pre-check (upload only the
   points, not a second full mesh). ≈ 13 % on the scans.
4. **Flip over the live triangle count** — flips never grow the triangle count,
   so launching over the post-growth count instead of the 2n capacity turns
   millions of no-op early-return threads per round into thousands. Flip time
   for a 1.8 M-point scan: 42 → 20 ms; ≈ another 20 % overall.

5. **Workspace reuse** — a small pool keyed by point count reuses the
   `2n`-capacity arrays across calls (batch workloads, and the joggle retries
   within one call). `Mesh.rebind()` re-uploads points + resets counters;
   `wp.empty` arrays are reinitialized by the per-round kernels anyway. On the
   3-reps-per-scan benchmark this took the 9 scans **1.90 s → 1.35 s**.

Net on the 9 scans: **2.71 s → 1.35 s (2× faster), 2.8× → 5.8× vs qhull.**

## What didn't work: fp32-first predicate filter

An fp32 certified filter (Shewchuk bound) with fp64 fallback is ~6× faster per
`orient3d` on Ada (fp64 is 1/64 rate) and did speed up the flip arithmetic
(c1 flip 45 → 26 ms; sphere-200k 2.2×). **But it was a net loss and was
reverted**, because near-degenerate predicates are pervasive:

- Flat sampled facets (sculpture bases) and dense hulls (uniform sphere at
  n ≥ 1 M) produce `orient3d` values fp32 cannot sign reliably.
- A wrong-but-"certified" sign makes the parallel flip give inconsistent reflex
  decisions and **oscillate to the iteration cap**, forcing an fp64 redo — and,
  because growth's `atomic_add` indexing is nondeterministic, whether fp32
  converges is a per-run coin flip, giving **nondeterministic runtime**
  (Hermanubis 592 → 704 → 1006 ms across runs).
- A conservative bound with zero mis-certifications sends ~93 % of dense-hull
  predicates to fp64 anyway → no speedup.

Gotchas if revisited: compile the filter module with
`wp.set_module_options({"fuse_fp": False, "fast_math": False})` (FP contraction
breaks the error bound); the fp64 fallback must use the **plain** sign, not SoS
(SoS in the hot path also oscillates — the algorithm + joggle wrapper isn't
built for it). The tested `predicates.o3d_sign` (certified fp64 + SoS) remains
as a standalone robust primitive.

## Conservative interior cull (`ffhull/filter.py`, opt-in `filter=True`)

Build a coarse *inner* hull `H0` from a subset of the points (a strided sample +
the axis extremes), then discard every input point strictly inside `H0`. Because
`H0 ⊆ H` (hull of a subset), inside-`H0` ⟹ inside-`H` ⟹ not a hull vertex — so
it never drops a true vertex. It culls **97–99.7 %** of a solid/volumetric cloud
(gaussian, uniform ball/cube).

It culls a big fraction, but the win is only modest and *situational*:

- The cull cost is `O(n·F0)` — testing each of `n` points against `F0` faces of
  `H0` (≈50 ms for a 1.8 M scan, `F0≈470`), plus ~19 ms to build `H0`. That
  overhead is comparable to the whole hull, so it's a net loss below ~5 M points.
- **Surface scans don't benefit.** With a tight `H0` the cull rate looks high,
  but their points sit near the hull boundary, and — critically — even a small
  survivor set doesn't shrink the survivor hull's *fixed* costs (allocation,
  growth) enough to beat the cull overhead.
- A **grid cull** was implemented and measured (test grid *cells* against `H0`,
  classify points by cell in `O(n)`, only occupied cells tested, cell = center
  eroded by its circumradius): it was **slower** than the per-point cull here.
  At a useful resolution `G³` rivals `n`, and a coarse grid leaves a thick
  1-cell survivor shell → lower cull rate → bigger survivor hull. So the grid
  didn't pay off at n ≤ 6 M either.

The cull is left **opt-in** (`filter=True`); it's a real win only for very large
*solid/volumetric* clouds where the hull time dominates the cull overhead.

## Remaining levers (not done)

- **Active-point compaction in growth** — growth reprocesses all n points every
  round (O(n·rounds)); a compacted active set would cut that. Small payoff here
  (growth is ~12 % of a scan's time).
- A cull that actually helps surface scans would likely need to cut the survivor
  hull's fixed costs too (e.g. cull + tight-capacity workspace for the
  survivors), or target much larger n where hull time ≫ cull overhead.
