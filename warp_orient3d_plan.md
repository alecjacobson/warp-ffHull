# Robust `orient3d` in NVIDIA Warp: Implementation Plan

## Goal

Implement a robust and efficient 3D orientation predicate for NVIDIA Warp that:

- returns the correct sign of

  \[
  \operatorname{orient3d}(a,b,c,d)
  =
  \det
  \begin{pmatrix}
  a_x-d_x & a_y-d_y & a_z-d_z \\
  b_x-d_x & b_y-d_y & b_z-d_z \\
  c_x-d_x & c_y-d_y & c_z-d_z
  \end{pmatrix},
  \]

- can be called from Warp kernels,
- does not require a custom CUDA extension or rebuilding Warp,
- handles nearly coplanar inputs robustly,
- avoids imposing strict floating-point settings on unrelated kernels,
- remains efficient for both batched predicates and small-hull construction.

The recommended architecture is:

```text
float32 certified filter
        |
        v
float64 certified filter
        |
        v
exact expansion fallback
```

The common case should exit from one of the inexpensive filters. Only uncertain predicates should use exact arithmetic.

---

## 1. Public interface

Expose two logically separate predicates:

```python
@wp.func
def orient3d_exact_sign(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
) -> wp.int32:
    ...
```

Returns:

```text
-1   negative orientation
 0   exactly coplanar
+1   positive orientation
```

For algorithms that require a strictly simplicial combinatorial result, provide a second layer:

```python
@wp.func
def orient3d_sos(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
    ia: wp.int32,
    ib: wp.int32,
    ic: wp.int32,
    id: wp.int32,
) -> wp.int32:
    ...
```

This applies a deterministic symbolic-perturbation rule when the exact determinant is zero.

Keep exact geometric classification and degeneracy resolution separate.

---

## 2. Floating-point compilation requirements

Error-free transformations such as `two_sum`, `two_diff`, and `two_product` require controlled floating-point evaluation.

Use:

```python
STRICT_FP_OPTIONS = {
    "fast_math": False,
    "fuse_fp": False,
}
```

### Important Warp limitation

These are module-level compilation options, not `@wp.func` options.

There is no supported equivalent of:

```python
@wp.func(fuse_fp=False)
def orient3d(...):
    ...
```

A `@wp.func` inherits the floating-point compilation behavior of the kernel module into which it is compiled.

### Recommended enforcement

Define strict kernels using a wrapper:

```python
import warp as wp


STRICT_FP_OPTIONS = {
    "fast_math": False,
    "fuse_fp": False,
}


def robust_kernel(fn):
    kernel = wp.kernel(
        module="unique",
        module_options=STRICT_FP_OPTIONS,
        enable_backward=False,
    )(fn)

    options = wp.get_module_options(module=kernel.module)
    assert options["fast_math"] is False
    assert options["fuse_fp"] is False

    return kernel
```

Use:

```python
@robust_kernel
def build_small_hulls(...):
    sign = orient3d_exact_sign(...)
```

This guarantees that `orient3d` and every helper it reaches are compiled without contraction in that kernel.

### Scope of the strict setting

For a sequential one-thread-per-hull kernel, compile the whole hull kernel with strict settings.

For a more parallel hull algorithm, keep ordinary kernels separate and evaluate uncertain predicates in a dedicated strict kernel.

---

## 3. Stage one: `float32` certified filter

The first filter evaluates the determinant in the input precision.

After translating relative to `d`:

```text
adx = ax - dx
ady = ay - dy
adz = az - dz

bdx = bx - dx
bdy = by - dy
bdz = bz - dz

cdx = cx - dx
cdy = cy - dy
cdz = cz - dz
```

Evaluate:

```text
det =
    adz * (bdx*cdy - cdx*bdy)
  + bdz * (cdx*ady - adx*cdy)
  + cdz * (adx*bdy - bdx*ady)
```

Also evaluate the permanent:

```text
permanent =
    (abs(bdx*cdy) + abs(cdx*bdy)) * abs(adz)
  + (abs(cdx*ady) + abs(adx*cdy)) * abs(bdz)
  + (abs(adx*bdy) + abs(bdx*ady)) * abs(cdz)
```

Certify the sign when:

```text
abs(det) > error_bound * permanent
```

For IEEE binary32, use the appropriate Shewchuk `orient3d` first-stage bound.

Implementation notes:

- write the scalar expression explicitly;
- do not use `wp.dot()` or `wp.cross()` in the certified filter;
- keep the expression ordering fixed;
- reject or separately handle non-finite coordinates.

The result of this function should distinguish:

```text
certified positive
certified negative
uncertain
```

For example:

```python
@wp.struct
class FilterResult:
    sign: wp.int32
    certain: wp.int32
```

A simpler alternative is:

```text
-1, +1   certified sign
 0       uncertain
```

but this makes “uncertain” indistinguishable from “exactly coplanar.” That is acceptable only inside a filter helper, not as the final public result.

---

## 4. Stage two: `float64` certified filter

If the `float32` result is uncertain, reevaluate the same expression in `float64`.

Convert each coordinate before subtraction:

```python
adx = wp.float64(a[0]) - wp.float64(d[0])
```

Do not write:

```python
adx = wp.float64(a[0] - d[0])
```

because that performs the subtraction in `float32` first.

Use the binary64 first-stage error bound:

```python
ORIENT3D_ERRBOUND_A_F64 = wp.float64(
    7.771561172376103e-16
)
```

The exact literal should be checked against the final implementation of the arithmetic expression and the reference constants used by the chosen Shewchuk port.

A sketch:

```python
@wp.func
def orient3d_filter64(
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    d: wp.vec3,
) -> wp.int32:
    adx = wp.float64(a[0]) - wp.float64(d[0])
    ady = wp.float64(a[1]) - wp.float64(d[1])
    adz = wp.float64(a[2]) - wp.float64(d[2])

    bdx = wp.float64(b[0]) - wp.float64(d[0])
    bdy = wp.float64(b[1]) - wp.float64(d[1])
    bdz = wp.float64(b[2]) - wp.float64(d[2])

    cdx = wp.float64(c[0]) - wp.float64(d[0])
    cdy = wp.float64(c[1]) - wp.float64(d[1])
    cdz = wp.float64(c[2]) - wp.float64(d[2])

    bdxcdy = bdx * cdy
    cdxbdy = cdx * bdy
    cdxady = cdx * ady
    adxcdy = adx * cdy
    adxbdy = adx * bdy
    bdxady = bdx * ady

    det = (
        adz * (bdxcdy - cdxbdy)
        + bdz * (cdxady - adxcdy)
        + cdz * (adxbdy - bdxady)
    )

    permanent = (
        (wp.abs(bdxcdy) + wp.abs(cdxbdy)) * wp.abs(adz)
        + (wp.abs(cdxady) + wp.abs(adxcdy)) * wp.abs(bdz)
        + (wp.abs(adxbdy) + wp.abs(bdxady)) * wp.abs(cdz)
    )

    errbound = (
        wp.float64(7.771561172376103e-16) * permanent
    )

    if det > errbound:
        return wp.int32(1)
    if det < -errbound:
        return wp.int32(-1)

    return wp.int32(0)
```

For `float32` input data, this stage should resolve almost every nondegenerate predicate encountered in ordinary geometry.

It is still not a mathematical guarantee.

---

## 5. Stage three: exact expansion arithmetic

If the `float64` filter is uncertain, evaluate the determinant exactly using floating-point expansions.

### Required primitives

Implement and test these helpers independently:

```text
fast_two_sum
two_sum
two_diff
split
two_product
two_product_presplit
fast_expansion_sum_zeroelim
scale_expansion_zeroelim
expansion_product
expansion_negate
expansion_diff
```

The exact fallback should use `float64` expansion components.

### Exact determinant construction

Represent each coordinate difference exactly using `two_diff`:

```text
adx = a.x - d.x
```

becomes an expansion containing the rounded result and its exact roundoff tail.

Then construct:

```text
bc = bdx*cdy - cdx*bdy
ca = cdx*ady - adx*cdy
ab = adx*bdy - bdx*ady
```

as exact expansions.

Finally compute:

```text
det = adz*bc + bdz*ca + cdz*ab
```

using expansion multiplication, scaling, and summation.

The sign is the sign of the highest-magnitude nonzero component of the final nonoverlapping expansion.

### Initial implementation choice

Do not initially port the entire hand-optimized adaptive `orient3dadapt()` routine.

Instead:

1. run the `float32` filter;
2. run the `float64` filter;
3. if still uncertain, construct the complete exact determinant mechanically.

This produces a simpler implementation with a smaller verification surface.

Once correctness is established, profile whether porting the more specialized adaptive path is worthwhile.

---

## 6. Local storage strategy

Warp functions can use fixed-capacity local storage, but large expansion buffers may cause register spilling or local-memory traffic.

Use fixed capacities, for example:

```python
fin = wp.zeros(shape=192, dtype=wp.float64)
```

The exact capacity should be derived from the chosen expansion construction and guarded with assertions during development.

Prefer several small purpose-specific buffers over many simultaneously live maximum-size buffers.

Document the maximum possible length of every helper result.

Example:

```text
two_diff:                    <= 2
two_product:                 <= 2
2x2 exact minor:             <= ...
scaled minor:                <= ...
final orient3d expansion:    <= ...
```

Never silently truncate an expansion.

---

## 7. Execution architecture

### Option A: small independent hulls

For many small hulls, use one Warp thread per hull.

Compile the complete hull-construction kernel with:

```text
fast_math = False
fuse_fp   = False
```

Each orientation call follows:

```text
float32 filter
float64 filter
exact fallback
```

Advantages:

- simple control flow;
- no intermediate predicate buffers;
- no extra launches;
- no synchronization inside one hull;
- exact result available immediately for topology decisions.

Tradeoff:

- the entire hull kernel runs without floating-point contraction;
- exact fallback code may increase per-thread register pressure even when rarely executed.

A useful variant is to abort and retry:

```text
first kernel:
    use only float32 and float64 filters
    if any predicate is uncertain:
        mark this hull and stop

second kernel:
    rerun only marked hulls with exact fallback enabled
```

This keeps the common kernel smaller.

### Option B: large parallel hull

For ffHull or another bulk-parallel algorithm:

1. evaluate predicates with inexpensive certified filters;
2. store uncertain predicate IDs;
3. compact the uncertain IDs;
4. evaluate only those predicates in a dedicated strict exact kernel;
5. continue the topology operation after exact signs are available.

This isolates strict floating-point settings to the predicate kernel and avoids carrying large expansion workspaces in general topology kernels.

---

## 8. Degeneracy policy

A robust predicate can return exact zero. The hull algorithm still needs a deterministic policy.

Handle these separately:

```text
duplicate points
all points coincident
all points collinear
all points coplanar
coplanar points on a 3D hull facet
```

Recommended behavior:

- remove exact duplicate input points or assign a stable representative;
- detect lower-dimensional input before starting the 3D hull;
- use a projected robust 2D hull for coplanar data;
- use symbolic perturbation for zero `orient3d` results inside the simplicial 3D hull.

Symbolic perturbation should depend only on stable point IDs, not execution order, thread ID, or atomic winner.

---

## 9. Validation plan

### Primitive arithmetic tests

Compare every expansion helper against exact integer or rational arithmetic.

Test:

```text
two_sum
two_diff
two_product
expansion sum
expansion scale
expansion product
```

Include random exponent ranges and cancellation-heavy cases.

### Predicate reference

Build a CPU reference using one of:

- Python arbitrary-precision integers for exactly representable integer inputs;
- `fractions.Fraction`;
- a trusted robust-predicate implementation used only by tests.

### Test families

#### Random ordinary inputs

Uniform and Gaussian coordinates over several scales.

#### Nearly coplanar inputs

Construct:

```text
d = point_on_plane + epsilon * normal
```

and sweep `epsilon` toward the representable limit.

#### Large translated coordinates

Use points clustered near a large offset:

```text
offset ~ 1e8
local scale ~ 1
```

This catches subtraction and casting mistakes.

#### Mixed scales

Coordinates with very different exponents in different axes.

#### Exact coplanarity

Generate `d` as an exact affine combination of `a`, `b`, and `c` using integer coordinates.

#### Permutation identities

Verify:

```text
orient(a,b,c,d) = -orient(b,a,c,d)
orient(a,b,c,d) = -orient(a,b,d,c)
```

and all expected parity relationships.

#### CPU/GPU agreement

Run identical batches on Warp CPU and CUDA devices.

#### Compiler-option tests

For every kernel that may execute expansion arithmetic:

```python
options = wp.get_module_options(module=kernel.module)

assert options["fast_math"] is False
assert options["fuse_fp"] is False
```

Make these tests part of CI so a later refactor cannot accidentally compile the predicate under unsafe settings.

---

## 10. Benchmarking plan

Measure each level separately:

```text
float32 certified rate
float64 certified rate
exact fallback rate
average predicate time
worst-case predicate time
```

Use at least three datasets:

1. ordinary random points;
2. points sampled near common planes;
3. deliberately adversarial near-coplanar points.

For hull workloads, additionally measure:

```text
hulls per second
predicate calls per hull
uncertain predicates per hull
exact-retry hull fraction
register usage
occupancy
local-memory traffic
```

Compare:

```text
float32 only
float64 only
float32 + float64 filters
filters + inline exact fallback
filters + separate exact retry
```

The expected best design may differ between:

- many tiny hulls,
- medium batched hulls,
- one large parallel hull.

---

## 11. Suggested implementation milestones

### Milestone 1: certified `float64` predicate

Implement:

```text
orient3d_filter64
```

Return an explicit uncertain state.

Validate against high-precision CPU arithmetic.

This is not fully exact, but it establishes expression ordering, sign conventions, and test infrastructure.

### Milestone 2: `float32` front filter

Add the inexpensive input-precision filter.

Measure how often it avoids `float64`.

### Milestone 3: exact expansion primitives

Implement and independently verify every error-free transformation and expansion operation.

Do not integrate them into the hull yet.

### Milestone 4: complete exact fallback

Construct the exact determinant mechanically from exact coordinate differences.

Verify exact zeros and adversarial near-coplanar cases.

### Milestone 5: strict-kernel wrapper

Add the `@robust_kernel` wrapper and CI assertions for module options.

### Milestone 6: hull integration

For small hulls, begin with:

```text
strict whole-hull kernel
+ inline filters
+ exact fallback
```

Then benchmark the exact-retry variant.

### Milestone 7: symbolic perturbation

Add deterministic handling for exact zeros based on point IDs.

### Milestone 8: optimization

Only after correctness:

- reduce expansion buffer sizes;
- shorten live ranges;
- specialize common expansion lengths;
- consider a closer port of adaptive `orient3dadapt`;
- consider a dedicated exact-predicate kernel;
- consider native snippets only if profiling shows a compelling benefit.

---

## 12. Recommended initial design

For the first robust implementation:

```text
Pure Warp source
No custom CUDA extension
No Warp rebuild

Dedicated robust kernel modules
fast_math=False
fuse_fp=False

float32 certified filter
float64 certified filter
full exact expansion fallback

exact zero returned explicitly
symbolic perturbation in a separate layer
```

For the first small-hull integration:

```text
one thread per hull
strict compilation for the whole hull kernel
abort-and-retry exact path if register pressure is excessive
```

This is the simplest design that provides a defensible correctness story while staying entirely within Warp's Python-facing programming model.

