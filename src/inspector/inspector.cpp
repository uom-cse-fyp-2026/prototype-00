#include "neurominds/inspector.hpp"
#include "neurominds/quantization.hpp"
#include "nm_config.hpp"
#include <cmath>

namespace nm {
#if NM_HAS_MLIR
extern "C" std::int32_t _mlir_ciface_nm_risk_guard(double, double, double, double, std::int32_t);
#endif
Descriptor inspect(const std::vector<float>& x, const Shape& shape, float clip, bool sampled) {
  validate_shape(shape);
  if (x.size() != shape.inputs() || !std::isfinite(clip) || clip <= 0) throw std::invalid_argument("invalid inspector input");
  Descriptor d;
  const auto count = sampled ? std::min(x.size(), generated::sample_size) : x.size();
  d.sampled = count < x.size();
  d.observed = count;
  d.bytes_read = count * sizeof(float);
  std::vector<double> magnitudes;
  magnitudes.reserve(count);
  std::size_t clipping = 0, outliers = 0;
  long double total = 0;
  for (std::size_t i = 0; i < count; ++i) {
    const auto index = count == x.size() ? i : (i * (x.size() - 1)) / (count - 1);
    const double value = std::abs(static_cast<double>(x[index]));
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite inspected activation");
    d.max_abs = std::max(d.max_abs, value);
    total += value;
    clipping += value > clip;
    outliers += value > clip * 0.75;
    magnitudes.push_back(value);
  }
  d.mean_abs = static_cast<double>(total / count);
  d.clipping_fraction = static_cast<double>(clipping) / count;
  d.outlier_fraction = static_cast<double>(outliers) / count;
  const auto q = static_cast<std::size_t>(std::ceil(count * 0.99)) - 1;
  std::nth_element(magnitudes.begin(), magnitudes.begin() + q, magnitudes.end());
  d.percentile99 = magnitudes[q];
  // Deliberately coarse: cache hits never waive the exact current risk predicate.
  const auto max_bucket = static_cast<std::uint64_t>(std::min(65535.0, std::floor(d.max_abs / clip * 16)));
  const auto clip_bucket = static_cast<std::uint64_t>(std::floor(d.clipping_fraction * 65535));
  const auto mean_bucket = static_cast<std::uint64_t>(std::min(65535.0, std::floor(d.mean_abs / clip * 64)));
  d.bucket = max_bucket | (clip_bucket << 16) | (mean_bucket << 32) | (static_cast<std::uint64_t>(d.sampled) << 63);
  return d;
}
bool risk_accepts(const Descriptor& d, const Policy& p) {
  if (!d.observed || !std::isfinite(d.max_abs) || !std::isfinite(d.clipping_fraction) ||
      d.max_abs < 0 || d.clipping_fraction < 0 || d.clipping_fraction > 1) return false;
#if NM_HAS_MLIR
  if (p.backend == Backend::Mlir)
    return _mlir_ciface_nm_risk_guard(d.max_abs, d.clipping_fraction, p.max_abs_limit, p.clipping_limit, p.guard_enabled ? 1 : 0) != 0;
#endif
  return p.guard_enabled && d.max_abs <= p.max_abs_limit && d.clipping_fraction <= p.clipping_limit;
}
bool cost_accepts(const Descriptor&, const Policy& p, double inspection_us) {
  if (!std::isfinite(inspection_us) || inspection_us < 0 || p.a8_us <= 0 || p.bf16_us <= 0) return false;
  const double overhead = inspection_us + p.decision_us;
  return p.a8_us + overhead < p.bf16_us * (1 - generated::performance_margin) &&
    p.a8_p95_us + overhead < p.bf16_p95_us;
}
PlanKey plan_key(const Shape& s, const Descriptor& d, const Policy& p, Backend b, bool guard_disabled) {
  return {s, d.bucket, generated::config_hash, p.id, b, guard_disabled};
}
std::uint64_t validity_key(const Plan& p) {
  std::uint64_t h = 14695981039346656037ULL;
  const std::uint64_t values[] = {p.key.shape.batch, p.key.shape.input, p.key.shape.output, p.key.shape.group,
    p.key.feature_bucket, p.key.config_hash, p.key.policy_id, static_cast<std::uint64_t>(p.key.backend),
    p.key.guard_disabled, static_cast<std::uint64_t>(p.variant), static_cast<std::uint64_t>(p.fallback),
    p.config_version, p.activation_policy};
  for (auto value : values) { h ^= value; h *= 1099511628211ULL; }
  return h;
}
}
