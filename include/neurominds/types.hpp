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

// A monotonic clock measures durations without wall-clock corrections.
using Clock = std::chrono::steady_clock;

inline double elapsed_us(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

// Check element-count multiplication before using it for allocations or indexing.
inline std::size_t product(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::invalid_argument("shape product overflow");
  }
  return a * b;
}

// Ceiling division counts a partial final group/byte without floating-point arithmetic.
inline std::size_t ceil_div(std::size_t a, std::size_t b) {
  if (b == 0) {
    throw std::invalid_argument("zero divisor");
  }
  return a / b + (a % b != 0);
}

// Projection contract: X[M,K] * W[N,K]^T -> Y[M,N], contiguous and without bias.
struct Shape {
  std::size_t batch = 1;     // M: input rows.
  std::size_t input = 512;   // K: features in each row.
  std::size_t output = 256;  // N: projected features.
  std::size_t group = 32;    // G: weights per quantization group within each weight row.

  bool operator==(const Shape& other) const {
    return batch == other.batch && input == other.input &&
           output == other.output && group == other.group;
  }

  bool operator!=(const Shape& other) const {
    return !(*this == other);
  }

  std::size_t inputs() const {
    return product(batch, input);  // Number of FP32 input elements, not bytes.
  }

  std::size_t outputs() const {
    return product(batch, output);
  }
};

// Both variants keep W4 weights; only activation storage and numerical computation change.
enum class Variant : std::uint32_t {
  Bf16 = 1,  // BF16 activation storage; FP32 arithmetic.
  A8 = 2,    // INT8 activation storage; INT32 group dots followed by FP32 scaling.
};

enum class Backend {
  Scalar,  // C++ numerical implementation.
  Mlir,    // MLIR/LLVM-generated native implementation.
};

inline const char* name(Variant variant) {
  if (variant == Variant::Bf16) {
    return "w4_bf16";
  }
  if (variant == Variant::A8) {
    return "w4_a8";
  }
  return "invalid";
}

inline const char* name(Backend backend) {
  return backend == Backend::Mlir ? "mlir_llvm" : "scalar_cpp";
}

// Shared weight representation. Its contents must stay fixed for the calibrated policy.
struct PackedWeights {
  std::size_t input = 0;
  std::size_t output = 0;
  std::size_t group = 0;

  // Two signed 4-bit nibbles per byte: even k in the low nibble, odd k in the high nibble.
  std::vector<std::uint8_t> data;
  std::vector<float> scales;  // One FP32 scale per output row/input group.

  std::size_t bytes() const {
    return data.size() + scales.size() * sizeof(float);  // Actual payload.
  }

  std::size_t capacity_bytes() const {
    return data.capacity() + scales.capacity() * sizeof(float);  // Reserved vector storage.
  }

  std::size_t groups() const {
    return ceil_div(input, group);
  }

  void validate() const;  // Check buffer sizes and positive finite group scales.
};

// Current input facts, recomputed on every adaptive call, including cache hits.
struct Descriptor {
  double max_abs = 0;
  double mean_abs = 0;
  double percentile99 = 0;       // Nearest-rank p99 of observed magnitudes.
  double outlier_fraction = 0;   // Observed fraction with magnitude >0.75*clip.
  double clipping_fraction = 0;  // Observed fraction with magnitude >clip.
  double reconstruction_error = 0; // Relative L2 error of reconstructed inspected activations.

  std::size_t observed = 0;
  std::size_t bytes_read = 0;  // Observed FP32 payload only; excludes later full scans.
  std::uint64_t bucket = 0;   // Coarse max/clipping/mean bins plus sampling flag.
  bool sampled = false;      // True only when fewer than all input values were observed.
};

// Frozen calibration rulebook, distinct from current Descriptor facts and a chosen Plan.
struct Policy {
  std::uint64_t id = 0;           // Payload checksum; zero means missing identity at runtime.
  std::uint64_t config_hash = 0;  // Must match generated configuration/frontend provenance.
  Shape shape;
  Backend backend = Backend::Mlir;

  float clip = 1;
  float scale = 1.0f / 127;  // Fixed A8 scale = clip/127; never refitted per invocation.
  double max_abs_limit = 0;
  double clipping_limit = 0;
  double reconstruction_limit = std::numeric_limits<double>::max();

  // Calibration means for complete forced variants and runtime overhead, in microseconds.
  double bf16_us = 0;
  double a8_us = 0;
  double inspection_us = 0;
  double decision_us = 0;
  double bf16_p95_us = 0;
  double a8_p95_us = 0;

  bool guard_enabled = false;  // Fitting enables only a rule with sufficient accepted support.
  bool sampled = false;        // Effective inspection mode used during calibration.
  std::uint64_t calibration_seed = 0;
  std::size_t calibration_count = 0;
  std::string environment_key;  // CPU/OS/code/compiler identity; excludes transient load.
};

// Exact reuse context except for the deliberately coarse feature bucket.
// No weight-content, live hardware-state or KV-cache identity is represented here.
struct PlanKey {
  Shape shape;
  std::uint64_t feature_bucket = 0;
  std::uint64_t config_hash = 0;
  std::uint64_t policy_id = 0;
  Backend backend = Backend::Mlir;
  bool guard_disabled = false;  // Separate ablation entries from normal guarded entries.

  bool operator==(const PlanKey& other) const {
    return shape == other.shape && feature_bucket == other.feature_bucket &&
           config_hash == other.config_hash && policy_id == other.policy_id &&
           backend == other.backend && guard_disabled == other.guard_disabled;
  }
};

// Small execution choice, not a cached output, weight buffer or generated kernel.
struct Plan {
  PlanKey key;
  Variant variant = Variant::Bf16;
  Variant fallback = Variant::Bf16;  // Conservative for activations; W4 weight error remains.
  std::uint32_t config_version = 0;
  std::uint32_t activation_policy = 1;  // The supported fixed calibration-derived scale policy.
  std::uint64_t validity_key = 0;       // Internal metadata checksum, not a security guarantee.
};

struct Counters {
  std::size_t hits = 0;
  std::size_t misses = 0;
  std::size_t invalidations = 0;  // Includes eviction and changed choices, not just corrupt plans.
  std::size_t fallbacks = 0;
};

// Output plus traceable decision metadata and cost for one invocation.
struct Invocation {
  std::vector<float> output;  // M*N FP32 values for either activation path.
  Descriptor descriptor;
  Plan plan;

  bool cache_hit = false;  // A candidate hit can still change choice or trigger fallback.
  bool fallback = false;   // Cost-only BF16 selection normally does not count as fallback.
  std::string reason;

  // All durations are microseconds. Components need not sum exactly to complete elapsed time.
  double inspector_us = 0;
  double validity_us = 0;  // Cache handling plus final plan checks.
  double decision_us = 0;
  double dispatch_us = 0;  // Execution residual, including output validation/cleanup.
  double preparation_us = 0;  // Full validation, allocation and activation conversion.
  double kernel_us = 0;
  double total_us = 0;
};

}  // namespace nm
