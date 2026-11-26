#include "neurominds/inspector.hpp"
#include "neurominds/quantization.hpp"
#include "nm_config.hpp"
#include <cmath>
#include <array>
#include "neurominds/risk_math.hpp"

namespace nm {

#if NM_HAS_MLIR
// C entry point emitted by the MLIR pass: observed max/clipping, their limits, enabled flag.
// The feature scan is C++; this small predicate is compiler-generated.
extern "C" std::int32_t _mlir_ciface_nm_risk_guard(double, double, double, double, double, double, std::int32_t);
#endif

namespace {

// Bucket similar descriptors together for cache lookup. Exact risk is still checked on every call.
std::uint64_t feature_bucket(const Descriptor& descriptor, float clip) {
  // Max/clip uses steps of 1/16; mean/clip uses steps of 1/64. Each bin fits in 16 bits.
  const auto max_bucket = static_cast<std::uint64_t>(
      std::min(65535.0, std::floor(descriptor.max_abs / clip * 16)));
  const auto clipping_bucket = static_cast<std::uint64_t>(
      std::floor(descriptor.clipping_fraction * 65535));
  const auto mean_bucket = static_cast<std::uint64_t>(
      std::min(65535.0, std::floor(descriptor.mean_abs / clip * 64)));

  // Bits 0..15: max; 16..31: clipping; 32..47: mean; bit 63: effective sampling mode.
  return max_bucket | (clipping_bucket << 16) | (mean_bucket << 32) |
         (static_cast<std::uint64_t>(descriptor.sampled) << 63);
}

}  // namespace

// Summarize current X[M,K] activations. clip is fixed by calibration, never fitted here.
Descriptor inspect(const std::vector<float>& input, const Shape& shape, float clip, bool sampled) {
  // 1. Validate the shape, contiguous buffer length and clipping range.
  validate_shape(shape);
  if (input.size() != shape.inputs() || !std::isfinite(clip) || clip <= 0) {
    throw std::invalid_argument("invalid inspector input");
  }

  // 2. Choose how many values to observe. Small inputs are fully scanned even in sampled mode.
  // Sampling can miss finite outliers; later domain validation does not recompute these features.
  const auto observed_count = sampled ? std::min(input.size(), generated::sample_size) : input.size();
  Descriptor descriptor;
  descriptor.sampled = observed_count < input.size();
  descriptor.observed = observed_count;
  descriptor.bytes_read = observed_count * sizeof(float);  // Activation payload bytes only.

  // Avoid heap allocation for small inspected tensors.
  std::array<double, 256> small_magnitudes;
  std::vector<double> large_magnitudes;
  if (observed_count > small_magnitudes.size()) large_magnitudes.resize(observed_count);
  double* magnitudes = observed_count <= small_magnitudes.size() ? small_magnitudes.data() : large_magnitudes.data();
  const float scale = clip / 127;
  if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("invalid inspector scale");
  risk_math::Reconstruction reconstruction;

  std::size_t clipping_count = 0;
  std::size_t outlier_count = 0;
  long double magnitude_sum = 0;  // Wider accumulation reduces rounding loss in the mean.

  // 3. Scan consecutive values, or evenly spaced samples including both endpoints.
  for (std::size_t sample = 0; sample < observed_count; ++sample) {
    std::size_t input_index = sample;
    if (descriptor.sampled) {
      // Configuration requires sample_size >= 2, so a partial scan cannot divide by zero.
      input_index = (sample * (input.size() - 1)) / (observed_count - 1);
    }

    const double magnitude = std::abs(static_cast<double>(input[input_index]));
    if (!std::isfinite(magnitude)) {
      throw std::invalid_argument("nonfinite inspected activation");
    }

    descriptor.max_abs = std::max(descriptor.max_abs, magnitude);
    magnitude_sum += magnitude;
    magnitudes[sample] = magnitude;
    reconstruction.add(magnitude, scale);

    if (magnitude > clip) {
      ++clipping_count;  // A value exactly at clip is still inside the A8 range.
    }
    if (magnitude > clip * 0.75) {
      ++outlier_count;  // Project-specific proxy, not a standard-deviation test.
    }
  }

  // 4. Convert totals to statistics. Cast before division to retain fractional counts.
  descriptor.mean_abs = static_cast<double>(magnitude_sum / observed_count);
  descriptor.clipping_fraction = static_cast<double>(clipping_count) / observed_count;
  descriptor.outlier_fraction = static_cast<double>(outlier_count) / observed_count;

  // Nearest-rank p99: subtract one to turn the one-based rank into a vector index.
  const auto percentile_index = static_cast<std::size_t>(std::ceil(observed_count * 0.99)) - 1;
  // Place the desired rank at that index without fully sorting the magnitude array.
  std::nth_element(magnitudes, magnitudes + percentile_index, magnitudes + observed_count);
  descriptor.percentile99 = magnitudes[percentile_index];
  descriptor.reconstruction_error = reconstruction.relative();

  // 5. Add the coarse reuse key. p99/outlier_fraction are diagnostics, not guard/key inputs.
  descriptor.bucket = feature_bucket(descriptor, clip);
  return descriptor;
}

// An empirical prediction, not proof that the current output meets the quality budget.
bool risk_accepts(const Descriptor& descriptor, const Policy& policy) {
  const bool valid_descriptor = risk_math::valid_feature(descriptor.max_abs,
      descriptor.clipping_fraction, descriptor.reconstruction_error, descriptor.observed);

  if (!valid_descriptor) {
    return false;
  }

#if NM_HAS_MLIR
  if (policy.backend == Backend::Mlir) {
    // The compiled predicate uses i32 for booleans; any nonzero result means accepted.
    return _mlir_ciface_nm_risk_guard(
        descriptor.max_abs, descriptor.clipping_fraction, descriptor.reconstruction_error,
        policy.max_abs_limit, policy.clipping_limit, policy.reconstruction_limit,
        policy.guard_enabled ? 1 : 0) != 0;
  }
#endif

  // Equivalent scalar predicate. All conditions must hold; comparisons include the limits.
  return policy.guard_enabled && descriptor.max_abs <= policy.max_abs_limit &&
         descriptor.clipping_fraction <= policy.clipping_limit &&
         descriptor.reconstruction_error <= policy.reconstruction_limit;
}

// Descriptor is intentionally unused: execution costs come from the frozen shape/backend policy.
bool cost_accepts(const Descriptor&, const Policy& policy, double inspection_us) {
  return risk_math::profitable(policy.bf16_us, policy.a8_us, policy.bf16_p95_us,
      policy.a8_p95_us, inspection_us, policy.decision_us, generated::performance_margin);
}

PlanKey plan_key(const Shape& shape, const Descriptor& descriptor, const Policy& policy,
                 Backend backend, bool guard_disabled) {
  PlanKey key;
  key.shape = shape;
  key.feature_bucket = descriptor.bucket;
  key.config_hash = generated::config_hash;
  key.policy_id = policy.id;
  key.backend = backend;
  key.guard_disabled = guard_disabled;  // Keep ablation plans separate from normal plans.
  return key;
}

// Internal metadata checksum only: it does not hash tensors or protect against malicious changes.
std::uint64_t validity_key(const Plan& plan) {
  constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;
  constexpr std::uint64_t fnv_prime = 1099511628211ULL;
  std::uint64_t checksum = fnv_offset_basis;

  // Order is part of the checksum contract. Exclude validity_key itself to avoid self-reference.
  const std::uint64_t fields[] = {
      plan.key.shape.batch,
      plan.key.shape.input,
      plan.key.shape.output,
      plan.key.shape.group,
      plan.key.feature_bucket,
      plan.key.config_hash,
      plan.key.policy_id,
      static_cast<std::uint64_t>(plan.key.backend),
      plan.key.guard_disabled,
      static_cast<std::uint64_t>(plan.variant),
      static_cast<std::uint64_t>(plan.fallback),
      plan.config_version,
      plan.activation_policy,
  };

  for (auto field : fields) {
    checksum ^= field;
    checksum *= fnv_prime;  // Unsigned 64-bit wraparound is intentional.
  }
  return checksum;
}

}  // namespace nm
