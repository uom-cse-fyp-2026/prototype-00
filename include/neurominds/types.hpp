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
// A monotonic clock measures durations without being affected by wall-clock corrections.
using Clock = std::chrono::steady_clock;
inline double elapsed_us(Clock::time_point start) {
  // std::micro expresses the elapsed steady-clock duration as floating-point microseconds.
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}
// Checked size multiplication prevents an overflowing allocation/index calculation from wrapping.
inline std::size_t product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::invalid_argument("shape product overflow");
  return a * b;
}
// Integer ceiling division counts partial groups/bytes without using floating-point arithmetic.
inline std::size_t ceil_div(std::size_t a, std::size_t b) {
  if (!b) throw std::invalid_argument("zero divisor");
  return a / b + (a % b != 0);
}
// One contiguous bias-free projection: X[M,K] * W[N,K]^T -> Y[M,N].
// batch=M, input=K, output=N, group=G weight values per quantization group in each row.
struct Shape {
  std::size_t batch = 1, input = 512, output = 256, group = 32;
  bool operator==(const Shape& s) const {
    return batch == s.batch && input == s.input && output == s.output && group == s.group;
  }
  bool operator!=(const Shape& s) const { return !(*this == s); }
  // Element counts for public activation and output vectors, not byte counts.
  std::size_t inputs() const { return product(batch, input); }
  std::size_t outputs() const { return product(batch, output); }
};
// Both variants share packed W4 weights; their activation storage/arithmetic differs.
// Bf16: 16-bit activation storage with FP32 math. A8: INT8 storage, INT32 group dots, FP32 scaling.
enum class Variant : std::uint32_t { Bf16 = 1, A8 = 2 };
// Backend selects the kernel implementation, independently of the numerical Variant.
enum class Backend { Scalar, Mlir };
inline const char* name(Variant v) { return v == Variant::Bf16 ? "w4_bf16" : v == Variant::A8 ? "w4_a8" : "invalid"; }
inline const char* name(Backend b) { return b == Backend::Mlir ? "mlir_llvm" : "scalar_cpp"; }
// Shared immutable weight representation for a calibrated experiment; content identity is not cached.
struct PackedWeights {
  std::size_t input = 0, output = 0, group = 0;
  // Two signed 4-bit nibbles per byte: even k in the low nibble, odd k in the high nibble.
  std::vector<std::uint8_t> data;
  // FP32 scale per output-row/input-group; quantized weights themselves use [-7,7].
  std::vector<float> scales;
  // Payload size versus reserved vector storage; neither is whole-process resident memory (RSS).
  std::size_t bytes() const { return data.size() + scales.size() * sizeof(float); }
  std::size_t capacity_bytes() const { return data.capacity() + scales.capacity() * sizeof(float); }
  std::size_t groups() const { return ceil_div(input, group); }
  // Verify packed buffer lengths and positive finite scales against the shape contract.
  void validate() const;
};
// Facts observed from the current input, recomputed for each adaptive invocation.
struct Descriptor {
  // Magnitudes, nearest-rank p99, fraction >0.75*clip, and fraction >clip, respectively.
  // Only max_abs and clipping_fraction enter the current risk predicate.
  double max_abs = 0, mean_abs = 0, percentile99 = 0, outlier_fraction = 0, clipping_fraction = 0;
  // Number of observed FP32 values and their payload bytes; exclude later full validation/conversion.
  std::size_t observed = 0, bytes_read = 0;
  // Coarse max/clipping/mean bins plus sampling flag; equal buckets can have different exact risk.
  std::uint64_t bucket = 0;
  // True only if fewer than all values were actually observed, regardless of requested mode.
  bool sampled = false;
};
// Frozen calibration rulebook. This differs from Descriptor (current facts) and Plan (current choice).
struct Policy {
  // Loaded payload checksum identifies the policy; zero is treated as missing identity at runtime.
  std::uint64_t id = 0;
  // Configuration hash also includes frontend provenance; it must match generated nm_config.hpp.
  std::uint64_t config_hash = 0;
  // Costs/thresholds apply only to this exact shape and execution backend.
  Shape shape;
  Backend backend = Backend::Mlir;
  // A8 uses one fixed symmetric scale: scale=clip/127, no per-input refitting.
  float clip = 1, scale = 1.0f / 127;
  // Maximum permitted observed magnitude and clipping fraction for empirical risk acceptance.
  double max_abs_limit = 0, clipping_limit = 0;
  // Calibration means in microseconds: complete forced variants, inspection and decision/cache work.
  double bf16_us = 0, a8_us = 0, inspection_us = 0, decision_us = 0;
  // Tail-latency estimates provide a second condition beyond the mean cost margin.
  double bf16_p95_us = 0, a8_p95_us = 0;
  // False if guard fitting found no rule meeting calibration support/risk requirements.
  bool guard_enabled = false;
  // Effective calibration inspection mode must agree with the current Descriptor.
  bool sampled = false;
  // Provenance for reproducible, disjoint calibration/evaluation splits.
  std::uint64_t calibration_seed = 0;
  std::size_t calibration_count = 0;
  // CPU/OS/code/compiler identity; it excludes transient scheduling load.
  std::string environment_key;
};
// Exact cache identity except for the deliberately coarse activation feature bucket.
// There is no weight-content, live hardware-state or KV-cache identity in the current prototype.
struct PlanKey {
  Shape shape;
  std::uint64_t feature_bucket = 0, config_hash = 0, policy_id = 0;
  Backend backend = Backend::Mlir;
  // Prevent risk-disabled experimental runs from sharing entries with normal guarded runs.
  bool guard_disabled = false;
  bool operator==(const PlanKey& k) const {
    // Every field must match; Shape equality includes group size as well as M/K/N.
    return shape == k.shape && feature_bucket == k.feature_bucket && config_hash == k.config_hash &&
      policy_id == k.policy_id && backend == k.backend && guard_disabled == k.guard_disabled;
  }
};
// Small execution choice consumed by Executor; it is not cached output or a generated kernel.
struct Plan {
  PlanKey key;
  // BF16 is conservative for activations only: fallback still retains W4 weight quantization error.
  Variant variant = Variant::Bf16, fallback = Variant::Bf16;
  // Version must match generated metadata; activation_policy=1 is the supported fixed-scale policy.
  std::uint32_t config_version = 0, activation_policy = 1;
  // Checksum of the other plan fields detects inconsistency, not malicious modifications.
  std::uint64_t validity_key = 0;
};
// Lifetime counters: invalidations include eviction and changed choices, not only corrupt plans.
struct Counters { std::size_t hits = 0, misses = 0, invalidations = 0, fallbacks = 0; };
// Numerical result plus evidence explaining one invocation's decision and cost.
struct Invocation {
  // FP32 output vector of M*N elements, whichever internal activation format was used.
  std::vector<float> output;
  Descriptor descriptor;
  Plan plan;
  // A hit can still fall back after current checks. Cost-only BF16 selection normally is not fallback.
  bool cache_hit = false, fallback = false;
  // Human-readable selection/fallback explanation printed by the demonstration.
  std::string reason;
  // Microseconds: descriptor scan, cache/final validity, selection, and execution residual.
  double inspector_us = 0, validity_us = 0, decision_us = 0, dispatch_us = 0;
  // Preparation includes full validation/allocation/conversion; total_us covers the complete call.
  double preparation_us = 0, kernel_us = 0, total_us = 0;
};
}
