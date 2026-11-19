#include "neurominds/inspector.hpp"
#include "neurominds/quantization.hpp"
#include "nm_config.hpp"
#include <cmath>

namespace nm {
#if NM_HAS_MLIR
// C-compatible entry point for the risk predicate emitted by the MLIR pass.
// Arguments: observed max, observed clipping fraction, their two limits, enabled flag.
// Descriptor reductions below remain C++; only this predicate is compiler-generated.
extern "C" std::int32_t _mlir_ciface_nm_risk_guard(double, double, double, double, std::int32_t);
#endif
// Summarize the current X[M,K] activations without running a numerical reference.
// clip is the frozen A8 clipping range learned during calibration, not fitted here.
Descriptor inspect(const std::vector<float>& x, const Shape& shape, float clip, bool sampled) {
  // Check positive dimensions, compiled limits and working-set/accumulator bounds.
  validate_shape(shape);
  // The contiguous input must have M*K values; a clipping range must be finite and positive.
  if (x.size() != shape.inputs() || !std::isfinite(clip) || clip <= 0) throw std::invalid_argument("invalid inspector input");
  // Default member initializers start every statistic and counter at zero.
  Descriptor d;
  // Full mode observes all values; sampled mode observes at most sample_size (default 128).
  const auto count = sampled ? std::min(x.size(), generated::sample_size) : x.size();
  // Record what actually happened: requesting sampling on a small tensor still gives a full scan.
  d.sampled = count < x.size();
  d.observed = count;
  // Count activation payload bytes only, not temporary-array or later conversion traffic.
  d.bytes_read = count * sizeof(float);
  // Retain magnitudes so p99 can be found after the scan without sorting the original input.
  std::vector<double> magnitudes;
  magnitudes.reserve(count);
  std::size_t clipping = 0, outliers = 0;
  // A wider running sum reduces rounding loss while accumulating absolute magnitudes.
  long double total = 0;
  for (std::size_t i = 0; i < count; ++i) {
    // Full mode uses consecutive indices; samples are evenly spaced and include both endpoints.
    // Generated configuration requires sample_size >= 2, so a partial scan has count > 1.
    const auto index = count == x.size() ? i : (i * (x.size() - 1)) / (count - 1);
    // Magnitude treats large negative and positive activations as equally relevant to clipping.
    const double value = std::abs(static_cast<double>(x[index]));
    // NaN/Inf cannot produce meaningful descriptors. Unobserved values are checked by run_kernel.
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite inspected activation");
    d.max_abs = std::max(d.max_abs, value);
    total += value;
    // Adding a boolean adds 1 when true and 0 when false; values exactly at clip do not count.
    clipping += value > clip;
    // This project's outlier proxy is >75% of clip, not a standard-deviation test.
    outliers += value > clip * 0.75;
    magnitudes.push_back(value);
  }
  // Convert integer counts to fractions in [0,1]; double division avoids integer truncation.
  d.mean_abs = static_cast<double>(total / count);
  d.clipping_fraction = static_cast<double>(clipping) / count;
  d.outlier_fraction = static_cast<double>(outliers) / count;
  // Nearest-rank p99: ceil(0.99*count) is a one-based rank, so subtract 1 for a vector index.
  const auto q = static_cast<std::size_t>(std::ceil(count * 0.99)) - 1;
  // nth_element places the desired rank at q without fully sorting the magnitude array.
  std::nth_element(magnitudes.begin(), magnitudes.begin() + q, magnitudes.end());
  d.percentile99 = magnitudes[q];
  // Deliberately coarse: cache hits never waive the exact current risk predicate.
  // Quantize max/clip into steps of 1/16 and mean/clip into steps of 1/64.
  // Clamp each magnitude bucket to fit 16 bits; clipping_fraction already lies in [0,1].
  const auto max_bucket = static_cast<std::uint64_t>(std::min(65535.0, std::floor(d.max_abs / clip * 16)));
  const auto clip_bucket = static_cast<std::uint64_t>(std::floor(d.clipping_fraction * 65535));
  const auto mean_bucket = static_cast<std::uint64_t>(std::min(65535.0, std::floor(d.mean_abs / clip * 64)));
  // Pack max in bits 0..15, clipping in 16..31, mean in 32..47, and sampled in bit 63.
  // p99 and outlier_fraction are diagnostics; neither participates in this key or the risk rule.
  d.bucket = max_bucket | (clip_bucket << 16) | (mean_bucket << 32) | (static_cast<std::uint64_t>(d.sampled) << 63);
  return d;
}
// Empirical risk prediction: acceptance does not certify the current output error.
bool risk_accepts(const Descriptor& d, const Policy& p) {
  // Reject empty/malformed summaries before calling either backend's predicate.
  if (!d.observed || !std::isfinite(d.max_abs) || !std::isfinite(d.clipping_fraction) ||
      d.max_abs < 0 || d.clipping_fraction < 0 || d.clipping_fraction > 1) return false;
#if NM_HAS_MLIR
  // Convert the policy boolean to the compiled predicate's i32 ABI; nonzero means accepted.
  if (p.backend == Backend::Mlir)
    return _mlir_ciface_nm_risk_guard(d.max_abs, d.clipping_fraction, p.max_abs_limit, p.clipping_limit, p.guard_enabled ? 1 : 0) != 0;
#endif
  // Scalar equivalent: all three conditions must hold, with inclusive threshold comparisons.
  return p.guard_enabled && d.max_abs <= p.max_abs_limit && d.clipping_fraction <= p.clipping_limit;
}
// The unnamed Descriptor parameter is intentionally unused: costs are calibrated per policy,
// shape and backend, rather than predicted from activation values. No per-call benchmarking occurs.
bool cost_accepts(const Descriptor&, const Policy& p, double inspection_us) {
  if (!std::isfinite(inspection_us) || inspection_us < 0 || p.a8_us <= 0 || p.bf16_us <= 0) return false;
  // Charge this call's measured inspection plus calibration's estimated decision/cache overhead.
  const double overhead = inspection_us + p.decision_us;
  // Require both a mean improvement margin (default 5%) and a p95 improvement.
  // The second comparison has no extra percentage margin. Full measured total_us tests the estimate.
  return p.a8_us + overhead < p.bf16_us * (1 - generated::performance_margin) &&
    p.a8_p95_us + overhead < p.bf16_p95_us;
}
// Aggregate initializer order follows PlanKey in types.hpp. Separate ablation and guarded entries.
// No weight-content ID is stored: weights must stay fixed for this calibrated policy.
PlanKey plan_key(const Shape& s, const Descriptor& d, const Policy& p, Backend b, bool guard_disabled) {
  return {s, d.bucket, generated::config_hash, p.id, b, guard_disabled};
}
// Hash plan metadata to detect stale/inconsistent internal fields, not adversarial tampering.
std::uint64_t validity_key(const Plan& p) {
  // FNV-style offset basis; the unsigned 64-bit arithmetic intentionally wraps on overflow.
  std::uint64_t h = 14695981039346656037ULL;
  // Exclude validity_key itself so the checksum can be recomputed and compared later.
  const std::uint64_t values[] = {p.key.shape.batch, p.key.shape.input, p.key.shape.output, p.key.shape.group,
    p.key.feature_bucket, p.key.config_hash, p.key.policy_id, static_cast<std::uint64_t>(p.key.backend),
    p.key.guard_disabled, static_cast<std::uint64_t>(p.variant), static_cast<std::uint64_t>(p.fallback),
    p.config_version, p.activation_policy};
  // XOR mixes in each field, then multiplication spreads its influence through the checksum.
  for (auto value : values) { h ^= value; h *= 1099511628211ULL; }
  return h;
}
}
