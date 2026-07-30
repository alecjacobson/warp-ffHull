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

Net on the 9 scans: **2.71 s → 1.90 s (~30 % faster), 2.8× → 4.0× vs qhull.**

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

## Remaining levers (not done)

- **Buffer reuse across calls** — a persistent workspace so batch/throughput
  workloads skip re-allocating the 2n arrays (allocation is now the largest
  single cost for a small-hull scan). Doesn't help a one-off call.
- **Active-point compaction in growth** — growth reprocesses all n points every
  round (O(n·rounds)); a compacted active set would cut that. Small payoff here
  (growth is ~12 % of a scan's time).
