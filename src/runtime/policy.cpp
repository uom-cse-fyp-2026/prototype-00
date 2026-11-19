#include "neurominds/experiments.hpp"
#include "neurominds/quantization.hpp"
#include "nm_config.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace nm {
namespace {
// File-local helper: checksum serialized policy text to detect accidental changes.
// This FNV-style checksum is an identity/consistency check, not cryptographic authentication.
std::uint64_t hash_text(const std::string& text) {
  std::uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : text) { h ^= c; h *= 1099511628211ULL; }
  return h;
}
// Serialize fields in a stable order shared by saving and checksum verification after loading.
// The ID is excluded because it is computed from this payload; including it would be circular.
std::string payload(const Policy& p) {
  std::ostringstream out;
  // Seventeen significant digits preserve double values through text save/load round trips.
  out << std::setprecision(17) << p.config_hash << ' ' << p.shape.batch << ' ' << p.shape.input << ' '
      << p.shape.output << ' ' << p.shape.group << ' ' << static_cast<int>(p.backend) << '\n'
      << p.clip << ' ' << p.scale << ' ' << p.max_abs_limit << ' ' << p.clipping_limit << '\n'
      << p.bf16_us << ' ' << p.a8_us << ' ' << p.inspection_us << ' ' << p.decision_us << ' '
      << p.bf16_p95_us << ' ' << p.a8_p95_us << '\n'
      << p.guard_enabled << ' ' << p.sampled << ' ' << p.calibration_seed << ' ' << p.calibration_count << '\n'
      // Quote/escape the environment string because it can contain whitespace and newlines.
      << std::quoted(p.environment_key) << '\n';
  return out.str();
}
}
// OFFLINE calibration only: use labelled output errors to fit a simple two-threshold risk rule.
// Runtime risk_accepts() consumes the frozen thresholds without seeing reference-output errors.
void fit_guard(Policy& policy, const std::vector<CalibrationRow>& rows) {
  // Start conservatively. If no rule qualifies, normal adaptive execution cannot risk-accept A8.
  policy.guard_enabled = false;
  policy.max_abs_limit = policy.clipping_limit = 0;
  std::vector<double> maxima, clips;
  for (const auto& row : rows) {
    // Only actually safe calibration rows supply candidate threshold values.
    // a8_error is relative L2 error against the actual W4 reference (default budget 0.03).
    if (std::isfinite(row.a8_error) && row.a8_error <= generated::error_budget) {
      maxima.push_back(row.descriptor.max_abs);
      clips.push_back(row.descriptor.clipping_fraction);
    }
  }
  // sort + unique + erase retains one copy of each observed threshold value.
  // unique alone only moves duplicates to the end; erase removes that redundant tail.
  std::sort(maxima.begin(), maxima.end());
  maxima.erase(std::unique(maxima.begin(), maxima.end()), maxima.end());
  std::sort(clips.begin(), clips.end());
  clips.erase(std::unique(clips.begin(), clips.end()), clips.end());
  // Bound fitting work: at most 64 max thresholds * 64 clipping thresholds, each checking all rows.
  const auto bounded_grid = [](std::vector<double>& values) {
    constexpr std::size_t limit = 64;
    if (values.size() <= limit) return;
    std::vector<double> grid;
    // Deterministically choose evenly spaced positions in the sorted candidates, including endpoints.
    for (std::size_t i = 0; i < limit; ++i) grid.push_back(values[i * (values.size() - 1) / (limit - 1)]);
    values = std::move(grid);
  };
  bounded_grid(maxima); bounded_grid(clips);
  std::size_t best = 0;
  // Each pair defines a rectangular accept region: max_abs <= maximum AND clipping_fraction <= clipping.
  for (double maximum : maxima) for (double clipping : clips) {
    std::size_t accepted = 0, unsafe = 0;
    // Evaluate every calibration row, including unsafe rows that supplied no candidate thresholds.
    for (const auto& row : rows) if (row.descriptor.max_abs <= maximum && row.descriptor.clipping_fraction <= clipping) {
      ++accepted;
      unsafe += !std::isfinite(row.a8_error) || row.a8_error > generated::error_budget;
    }
    // Prefer the greatest coverage subject to minimum support and an unsafe-accepted fraction cap.
    // Defaults: >=8 accepts and zero unsafe/accepted on calibration; this is no held-out guarantee.
    // The first qualifying pair wins ties because accepted must be strictly greater than best.
    if (accepted >= generated::minimum_guard_accepts && accepted > best &&
        static_cast<double>(unsafe) / accepted <= generated::max_calibration_false_safe_rate) {
      best = accepted;
      policy.guard_enabled = true;
      policy.max_abs_limit = maximum;
      policy.clipping_limit = clipping;
    }
  }
}
// Validate a policy once at calibration/load. The fast runtime path assumes it then stays immutable.
void validate_policy(const Policy& p, bool require_environment) {
  validate_shape(p.shape);
  if (p.config_hash != generated::config_hash) throw std::invalid_argument("policy belongs to another compiled configuration; recalibrate");
  if (p.backend != Backend::Scalar && p.backend != Backend::Mlir) throw std::invalid_argument("unknown policy backend");
  // Require enough provenance to identify a real calibration context, not just threshold numbers.
  if (!p.calibration_count || p.calibration_count > 1000000 || p.environment_key.empty()) throw std::invalid_argument("incomplete calibration provenance");
  // A8 uses a fixed symmetric range [-127,127], so its scale must agree with clip/127.
  // The relative tolerance allows small text/floating-point round-trip differences.
  if (!std::isfinite(p.clip) || !std::isfinite(p.scale) || p.clip <= 0 || p.scale <= 0 ||
      std::abs(static_cast<double>(p.clip) / 127 - p.scale) > static_cast<double>(p.scale) * 1e-5)
    throw std::invalid_argument("inconsistent activation scale");
  // Magnitude limits cannot be negative and a clipping fraction cannot exceed one.
  if (!std::isfinite(p.max_abs_limit) || p.max_abs_limit < 0 || !std::isfinite(p.clipping_limit) ||
      p.clipping_limit < 0 || p.clipping_limit > 1) throw std::invalid_argument("invalid calibrated guard");
  // Execution costs must be positive; measured overheads may validly be zero but never negative.
  for (double value : {p.bf16_us, p.a8_us, p.bf16_p95_us, p.a8_p95_us})
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("missing/invalid measured execution costs");
  for (double value : {p.inspection_us, p.decision_us})
    if (!std::isfinite(value) || value < 0) throw std::invalid_argument("invalid measured overhead");
  // Reject costs fitted under another recorded CPU/OS/code/compiler identity.
  // This identity excludes transient scheduler load; matching it does not guarantee stable latency.
  // CMake fingerprints source bytes, including comments, so a rebuild after annotation can change it.
  if (require_environment && p.environment_key != environment_key())
    throw std::invalid_argument("policy environment/code/compiler fingerprint changed; recalibrate on this VM");
}
// Write a versioned policy file. The input is const: saving does not update input.id in memory.
void save_policy(const Policy& input, const std::string& path) {
  validate_policy(input);
  const auto text = payload(input);
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write policy: " + path);
  // Format: version marker, checksum/ID, then payload. Calibration reloads to obtain the final ID.
  out << "NM_POLICY_V1\n" << hash_text(text) << '\n' << text;
  // Check again after writing to catch failures such as insufficient disk space.
  if (!out) throw std::runtime_error("failed writing policy");
}
// Policy-load failures reject the file; they do not silently turn it into a fallback policy.
Policy load_policy(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open policy; run ./nm calibrate first: " + path);
  std::string magic;
  Policy p;
  // Parse enums/booleans into integers first, so unsupported encodings can be rejected explicitly.
  int backend = -1, guard = -1, sampled = -1;
  in >> magic >> p.id >> p.config_hash >> p.shape.batch >> p.shape.input >> p.shape.output >> p.shape.group >> backend
     >> p.clip >> p.scale >> p.max_abs_limit >> p.clipping_limit >> p.bf16_us >> p.a8_us >> p.inspection_us >> p.decision_us
     >> p.bf16_p95_us >> p.a8_p95_us >> guard >> sampled >> p.calibration_seed >> p.calibration_count >> std::quoted(p.environment_key);
  // !in catches missing/malformed fields; magic and ranges enforce the supported file schema.
  if (!in || magic != "NM_POLICY_V1" || backend < 0 || backend > 1 || guard < 0 || guard > 1 || sampled < 0 || sampled > 1)
    throw std::invalid_argument("malformed policy file");
  // Permit trailing whitespace only; additional tokens would otherwise escape checksum validation.
  in >> std::ws;
  if (!in.eof()) throw std::invalid_argument("unexpected trailing policy data");
  p.backend = static_cast<Backend>(backend);
  p.guard_enabled = guard != 0;
  p.sampled = sampled != 0;
  // Recreate the canonical payload from parsed values and check it matches the stored ID.
  if (p.id != hash_text(payload(p))) throw std::invalid_argument("policy checksum mismatch");
  // Then check semantic consistency and compatibility with this executable and environment.
  validate_policy(p);
  return p;
}
}
