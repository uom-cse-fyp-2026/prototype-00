#include "neurominds/experiments.hpp"
#include "neurominds/quantization.hpp"
#include "nm_config.hpp"
#include <cmath>
#include "neurominds/risk_math.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>

namespace nm {
namespace {

constexpr const char* policy_file_version = "NM_POLICY_V2";

// Identify the serialized payload and detect accidental changes, not malicious tampering.
std::uint64_t hash_text(const std::string& text) {
  constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;
  constexpr std::uint64_t fnv_prime = 1099511628211ULL;
  std::uint64_t checksum = fnv_offset_basis;

  for (unsigned char character : text) {
    checksum ^= character;
    checksum *= fnv_prime;  // Unsigned wraparound is part of this checksum algorithm.
  }
  return checksum;
}

// Keep field order, spacing and precision stable: existing files/checksums depend on this format.
// Exclude policy.id because the ID is the checksum of the resulting payload.
std::string payload(const Policy& policy) {
  std::ostringstream output;
  output << std::setprecision(17);  // Enough significant digits for double round trips.

  // Configuration, exact shape and backend.
  output << policy.config_hash << ' ' << policy.shape.batch << ' ' << policy.shape.input << ' '
         << policy.shape.output << ' ' << policy.shape.group << ' '
         << static_cast<int>(policy.backend) << '\n';

  // Fixed activation scale and fitted risk thresholds.
  output << policy.clip << ' ' << policy.scale << ' '
         << policy.max_abs_limit << ' ' << policy.clipping_limit << ' ' << policy.reconstruction_limit << '\n';

  // Calibrated complete execution costs and overhead, all in microseconds.
  output << policy.bf16_us << ' ' << policy.a8_us << ' '
         << policy.inspection_us << ' ' << policy.decision_us << ' '
         << policy.bf16_p95_us << ' ' << policy.a8_p95_us << '\n';

  // Calibration provenance. Quoting preserves spaces/newlines in the environment string.
  output << policy.guard_enabled << ' ' << policy.sampled << ' '
         << policy.calibration_seed << ' ' << policy.calibration_count << '\n';
  output << std::quoted(policy.environment_key) << '\n';
  return output.str();
}

// Sort, deduplicate and bound one list of candidate thresholds before the grid search.
void prepare_threshold_candidates(std::vector<double>& candidates) {
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

  constexpr std::size_t candidate_limit = 64;
  if (candidates.size() <= candidate_limit) {
    return;
  }

  // Deterministic evenly spaced positions in the sorted list, including both endpoints.
  std::vector<double> bounded_candidates;
  for (std::size_t index = 0; index < candidate_limit; ++index) {
    const auto source_index = index * (candidates.size() - 1) / (candidate_limit - 1);
    bounded_candidates.push_back(candidates[source_index]);
  }
  candidates = std::move(bounded_candidates);
}

}  // namespace

// OFFLINE fitting uses calibration output errors. Runtime sees only the resulting frozen policy.
void fit_guard(Policy& policy, const std::vector<CalibrationRow>& rows) {
  // 1. Start disabled. If no useful rule qualifies, normal adaptive execution stays conservative.
  policy.guard_enabled = false;
  policy.max_abs_limit = 0;
  policy.clipping_limit = 0;
  policy.reconstruction_limit = 0;

  // 2. Safe calibration examples supply candidate maximum/clipping threshold values.
  std::vector<double> maximum_candidates;
  std::vector<double> clipping_candidates;
  std::vector<risk_math::Observation> reconstruction_rows;
  for (const auto& row : rows) {
    const auto& d = row.descriptor;
    if (!risk_math::valid_feature(d.max_abs, d.clipping_fraction, d.reconstruction_error, d.observed)) continue;
    reconstruction_rows.push_back({d.max_abs, d.clipping_fraction, d.reconstruction_error, row.a8_error});
    const bool a8_is_safe = std::isfinite(row.a8_error) && row.a8_error <= generated::error_budget;
    if (a8_is_safe) {
      maximum_candidates.push_back(row.descriptor.max_abs);
      clipping_candidates.push_back(row.descriptor.clipping_fraction);
    }
  }

  prepare_threshold_candidates(maximum_candidates);
  prepare_threshold_candidates(clipping_candidates);

  // 3. Test at most 64*64 threshold pairs against ALL rows, including unsafe examples.
  std::size_t best_accepted_count = 0;
  for (double maximum_limit : maximum_candidates) {
    for (double clipping_limit : clipping_candidates) {
      std::size_t accepted_count = 0;
      std::size_t unsafe_count = 0;

      for (const auto& row : rows) {
        const auto& d = row.descriptor;
        const bool accepted = risk_math::valid_feature(d.max_abs, d.clipping_fraction, d.reconstruction_error, d.observed) &&
                              row.descriptor.max_abs <= maximum_limit &&
                              row.descriptor.clipping_fraction <= clipping_limit;
        if (!accepted) {
          continue;
        }

        ++accepted_count;
        if (!std::isfinite(row.a8_error) || row.a8_error > generated::error_budget) {
          ++unsafe_count;
        }
      }

      // 4. Prefer larger coverage with enough support. Strict improvement preserves the first tie.
      if (accepted_count < generated::minimum_guard_accepts || accepted_count <= best_accepted_count) {
        continue;
      }

      // Denominator is accepted examples, NOT all unsafe examples. Default allowed fraction is zero.
      const double unsafe_fraction = static_cast<double>(unsafe_count) / accepted_count;
      if (unsafe_fraction > generated::max_calibration_false_safe_rate) {
        continue;
      }

      best_accepted_count = accepted_count;
      policy.guard_enabled = true;
      policy.max_abs_limit = maximum_limit;
      policy.clipping_limit = clipping_limit;
      policy.reconstruction_limit = std::numeric_limits<double>::max();
    }
  }
  // Preserve the baseline rule on equal coverage.
  const auto rounded = risk_math::best_reconstruction_rule(std::move(reconstruction_rows),
      generated::error_budget, generated::minimum_guard_accepts, generated::max_calibration_false_safe_rate);
  if (rounded.accepted > best_accepted_count) {
    policy.guard_enabled = true;
    policy.max_abs_limit = rounded.maximum;
    policy.clipping_limit = rounded.clipping;
    policy.reconstruction_limit = rounded.reconstruction;
  }
  // Passing finite calibration data is an empirical result, never a per-input quality guarantee.
}

// Validate once at load/calibration; runtime assumes the policy stays immutable afterward.
void validate_policy(const Policy& policy, bool require_environment) {
  // 1. Check shape, compiled configuration and supported backend.
  validate_shape(policy.shape);
  if (policy.config_hash != generated::config_hash) {
    throw std::invalid_argument("policy belongs to another compiled configuration; recalibrate");
  }
  if (policy.backend != Backend::Scalar && policy.backend != Backend::Mlir) {
    throw std::invalid_argument("unknown policy backend");
  }
  if (policy.calibration_count == 0 || policy.calibration_count > 1000000 || policy.environment_key.empty()) {
    throw std::invalid_argument("incomplete calibration provenance");
  }

  // 2. A8 stores [-127,127], so scale must agree with clip/127 within a small relative tolerance.
  if (!std::isfinite(policy.clip) || !std::isfinite(policy.scale) || policy.clip <= 0 || policy.scale <= 0 ||
      policy.scale != policy.clip / 127) {
    throw std::invalid_argument("inconsistent activation scale");
  }
  if (!std::isfinite(policy.max_abs_limit) || policy.max_abs_limit < 0 ||
      !std::isfinite(policy.clipping_limit) || policy.clipping_limit < 0 || policy.clipping_limit > 1 ||
      !std::isfinite(policy.reconstruction_limit) || policy.reconstruction_limit < 0) {
    throw std::invalid_argument("invalid calibrated guard");
  }

  // 3. Execution costs must be positive. Overhead can be zero, but never negative/nonfinite.
  for (double cost : {policy.bf16_us, policy.a8_us, policy.bf16_p95_us, policy.a8_p95_us}) {
    if (!std::isfinite(cost) || cost <= 0) {
      throw std::invalid_argument("missing/invalid measured execution costs");
    }
  }
  for (double overhead : {policy.inspection_us, policy.decision_us}) {
    if (!std::isfinite(overhead) || overhead < 0) {
      throw std::invalid_argument("invalid measured overhead");
    }
  }

  // 4. Match recorded CPU/OS/code/compiler identity. Transient scheduling load is not captured.
  // CMake fingerprints source bytes, so source edits require fresh policies after rebuilding.
  if (require_environment && policy.environment_key != environment_key()) {
    throw std::invalid_argument("policy environment/code/compiler fingerprint changed; recalibrate on this VM");
  }
}

// Saving is const: reload the file to obtain its final checksum-derived ID in memory.
void save_policy(const Policy& policy, const std::string& path) {
  validate_policy(policy);
  const auto text = payload(policy);

  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot write policy: " + path);
  }

  // Existing format: version marker, checksum/ID, then the canonical payload.
  output << policy_file_version << '\n' << hash_text(text) << '\n' << text;
  if (!output) {
    throw std::runtime_error("failed writing policy");
  }
}

// Loading a bad policy raises an error; it does not silently invent a fallback policy.
Policy load_policy(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open policy; run ./nm calibrate first: " + path);
  }

  // 1. Parse in the same order as save_policy()/payload(). Use integers to validate enums/flags.
  Policy policy;
  std::string version;
  int backend_value = -1;
  int guard_value = -1;
  int sampled_value = -1;

  input >> version >> policy.id;
  input >> policy.config_hash >> policy.shape.batch >> policy.shape.input
        >> policy.shape.output >> policy.shape.group >> backend_value;
  input >> policy.clip >> policy.scale >> policy.max_abs_limit >> policy.clipping_limit >> policy.reconstruction_limit;
  input >> policy.bf16_us >> policy.a8_us >> policy.inspection_us >> policy.decision_us
        >> policy.bf16_p95_us >> policy.a8_p95_us;
  input >> guard_value >> sampled_value >> policy.calibration_seed >> policy.calibration_count
        >> std::quoted(policy.environment_key);

  // 2. Reject incomplete fields, unsupported file versions and invalid enum/boolean encodings.
  if (!input || version != policy_file_version || backend_value < 0 || backend_value > 1 ||
      guard_value < 0 || guard_value > 1 || sampled_value < 0 || sampled_value > 1) {
    throw std::invalid_argument("malformed policy file");
  }

  input >> std::ws;  // Trailing whitespace is allowed; extra data is not.
  if (!input.eof()) {
    throw std::invalid_argument("unexpected trailing policy data");
  }

  policy.backend = static_cast<Backend>(backend_value);
  policy.guard_enabled = guard_value != 0;
  policy.sampled = sampled_value != 0;

  // 3. Check payload consistency, then semantic/environment compatibility.
  if (policy.id != hash_text(payload(policy))) {
    throw std::invalid_argument("policy checksum mismatch");
  }
  validate_policy(policy);
  return policy;
}

}  // namespace nm
