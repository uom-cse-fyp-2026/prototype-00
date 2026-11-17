#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nm {
using Clock = std::chrono::steady_clock;
inline double elapsed_us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
inline std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::invalid_argument("shape product overflow");
  return a * b;
}
inline std::size_t ceil_div(std::size_t a, std::size_t b) {
  if (!b) throw std::invalid_argument("zero divisor");
  return a / b + (a % b != 0);
}
struct Shape {
  std::size_t batch = 1, input = 512, output = 256, group = 32;
  bool operator==(const Shape& s) const {
    return batch == s.batch && input == s.input && output == s.output && group == s.group;
  }
  bool operator!=(const Shape& s) const { return !(*this == s); }
  std::size_t inputs() const { return product(batch, input); }
  std::size_t outputs() const { return product(batch, output); }
};
enum class Variant : std::uint32_t { Bf16 = 1, A8 = 2 };
enum class Backend { Scalar, Mlir };
inline const char* name(Variant v) { return v == Variant::Bf16 ? "w4_bf16" : v == Variant::A8 ? "w4_a8" : "invalid"; }
inline const char* name(Backend b) { return b == Backend::Mlir ? "mlir_llvm" : "scalar_cpp"; }
struct PackedWeights {
  std::size_t input = 0, output = 0, group = 0;
  std::vector<std::uint8_t> data;
  std::vector<float> scales;
  std::size_t bytes() const { return data.size() + scales.size() * sizeof(float); }
  std::size_t capacity_bytes() const { return data.capacity() + scales.capacity() * sizeof(float); }
  std::size_t groups() const { return ceil_div(input, group); }
  void validate() const;
};
struct Descriptor {
  double max_abs = 0, mean_abs = 0, percentile99 = 0, outlier_fraction = 0, clipping_fraction = 0;
  std::size_t observed = 0, bytes_read = 0;
  std::uint64_t bucket = 0;
  bool sampled = false;
};
struct Policy {
  std::uint64_t id = 0;
  std::uint64_t config_hash = 0;
  Shape shape;
  Backend backend = Backend::Mlir;
  float clip = 1, scale = 1.0f / 127;
  double max_abs_limit = 0, clipping_limit = 0;
  double bf16_us = 0, a8_us = 0, inspection_us = 0, decision_us = 0;
  double bf16_p95_us = 0, a8_p95_us = 0;
  bool guard_enabled = false;
  bool sampled = false;
  std::uint64_t calibration_seed = 0;
  std::size_t calibration_count = 0;
  std::string environment_key;
};
struct PlanKey {
  Shape shape;
  std::uint64_t feature_bucket = 0, config_hash = 0, policy_id = 0;
  Backend backend = Backend::Mlir;
  bool guard_disabled = false;
  bool operator==(const PlanKey& k) const {
    return shape == k.shape && feature_bucket == k.feature_bucket && config_hash == k.config_hash &&
      policy_id == k.policy_id && backend == k.backend && guard_disabled == k.guard_disabled;
  }
};
struct Plan {
  PlanKey key;
  Variant variant = Variant::Bf16, fallback = Variant::Bf16;
  std::uint32_t config_version = 0, activation_policy = 1;
  std::uint64_t validity_key = 0;
};
struct Counters { std::size_t hits = 0, misses = 0, invalidations = 0, fallbacks = 0; };
struct Invocation {
  std::vector<float> output;
  Descriptor descriptor;
  Plan plan;
  bool cache_hit = false, fallback = false;
  std::string reason;
  double inspector_us = 0, validity_us = 0, decision_us = 0, dispatch_us = 0;
  double preparation_us = 0, kernel_us = 0, total_us = 0;
};
}
