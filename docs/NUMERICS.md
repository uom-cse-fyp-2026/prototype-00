# Numerical and data contracts

## Shapes and storage

`X` and `Y` are row-major contiguous FP32 vectors at the public boundary.
`W` has shape `[output,input]`. All four dimensions, including group size,
must be positive and within generated limits and the working-set admission
budget. Arbitrary strides are not supported. Partial final groups and odd
input widths are supported.

Each weight row contains `ceil(input/2)` packed bytes. Column `k` occupies the
low nibble when `k` is even and the high nibble when `k` is odd. Nibbles use
signed two's complement; unpacking accepts `[-8,7]`. Quantization itself uses
the symmetric range `[-7,7]`:

```text
scale[n,g] = max(abs(W[n, group g])) / 7
q[n,k]     = clamp(round_away_from_zero(W[n,k] / scale[n,g]), -7, 7)
W4[n,k]    = q[n,k] * scale[n,g]
```

An all-zero group uses scale 1. Underflowing nonzero scales are bounded below
by the smallest positive FP32 subnormal. Scales are FP32, one per output row
and input group. Padding nibbles are zero and never read as valid weights.
Payload bytes and actual vector capacity bytes are recorded separately.

## Activation paths

**BF16:** convert IEEE binary32 to actual `uint16_t` BF16 bits using
round-to-nearest, ties-to-even. Infinity/NaN conversion is defined by the helper,
but the operator API rejects nonfinite inputs. Convert the BF16 bits back to
FP32 inside the kernel and multiply by dequantized W4 weights, accumulating in
FP32. No native BF16 instruction requirement or low-bit arithmetic claim.

**A8:** use one fixed calibration-derived scale for the operator policy:

```text
clip   = configured empirical quantile of typical calibration magnitudes
scaleA = clip / 127
a8     = round_away_from_zero(clamp(X / scaleA, -127, 127))
dot_g  = sum_k_in_group(int32(a8[k]) * int32(unpacked_w4[k]))
Y      = sum_g(float(dot_g) * scaleA * scaleW[g])
```

Rounding is implemented in double precision in the conversion helpers before
storing the result. Both execution backends consume identical stored activation
and packed-weight values. The INT32 dot resets for each group, and dimension
validation bounds its maximum magnitude using `127 * 8 * min(group,input)`.
The final weighted sum is FP32. This differs numerically and computationally
from converting A8 immediately back to FP32 for a float dot product.

Inputs must fit the finite BF16 storage domain. Extremely large finite inputs
that would overflow BF16 storage are rejected. Projection output overflow is
also an explicit error. No implicit NaN/Inf fallback is attempted.

## References and errors

- Original FP32 weights + FP32 activations: scalar FP32 accumulation reference.
- Actual W4 integers/scales + FP32 activations: FP64 accumulation for an
  accurate numerical anchor that isolates activation error.
- Independent per-variant reference: explicitly dequantized stored BF16/A8
  values, evaluated with FP64 accumulation. This checks implementation
  correctness independently of whether quantization quality meets the budget.
- Scalar BF16/A8 kernels: separate execution backends for comparisons with the
  generated MLIR kernels.

The activation-risk budget is applied to:

```text
relative_L2 = norm(Y_variant - Y_W4_reference) /
              max(norm(Y_W4_reference), epsilon)
```

The budget and absolute denominator floor `epsilon` are configured before
calibration. Near-zero references can make relative errors unstable; the floor
is explicit and exact zero/near-zero cases are covered by the test source.
Benchmark CSVs also report error against original FP32 weights, which includes
weight error. Neither metric is perplexity, task accuracy or a token-level
quality guarantee.

## Calibration, splits and risk

Four seeded synthetic regimes cover normal activations, rare large outliers,
scale shift and mixtures. An outlier input explicitly contains at least one
large value so tiny-shape demonstrations remain meaningful. Scale fitting uses
only typical calibration inputs with split indices `100000..100015` and at
most 65536 collected magnitudes. Guard fitting uses indices `0..samples-1`.
The warm-up scale-fitting input uses index `200000`.

Calibration and held-out seeds differ by a reserved high bit before an
invertible SplitMix64 permutation. This produces disjoint seeds for the same
base seed and permitted indices. Held-out benchmarks use test indices
`0..samples-1`; demo and single-run examples use separate test indices.
Standard-library normal-distribution generation is repeatable within the
recorded library/toolchain environment, not promised bit-identical across all
C++ standard-library implementations.

The interpretable guard searches maximum-magnitude and clipping-fraction
thresholds using calibration output errors only. It maximizes accepted
calibration inputs subject to the configured false-safe fraction and minimum
accept count over a deterministic grid of at most 64 observed values per
descriptor. This bounds fitting cost on larger calibration sets. If no rule
qualifies, it disables A8 risk acceptance. The
minimum count is a feasibility heuristic, not a statistical reliability proof.
The frozen policy is never refit from held-out results.

Full inspection reads every activation once and uses a temporary magnitude
array plus `nth_element` for p99. Sampled inspection reads evenly spaced
locations including endpoints, bounded by the configured sample size. The
descriptor records bytes read and sample count. Additional conversion and
validation scans are part of invocation cost. Sampling can miss isolated
outliers; resulting false-safe decisions are an expected research risk to
measure, not something the cache can certify away.

## Guard metrics

Guard metrics use unique held-out inputs, independent of whether the cost
decision ultimately selects A8:

- False-safe rate: guard accepts but A8 exceeds the budget, divided by the
  number of actually unsafe inputs.
- False-fallback rate: guard rejects but A8 meets the budget, divided by the
  number of actually safe inputs.
- Unsafe fraction among accepted: unsafe accepted inputs divided by all guard
  accepts. This is distinct from the first rate.

Undefined denominators are saved as JSON `null`. Forced A8 and guard-disabled
ablation results are labeled separately from deployable adaptive decisions.
