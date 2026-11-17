#include "neurominds/experiments.hpp"
#include "neurominds/quantization.hpp"
#include "nm_config.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace nm {
namespace {
std::uint64_t hash_text(const std::string& text) {
  std::uint64_t h = 14695981039346656037ULL;
  for (unsigned char c : text) { h ^= c; h *= 1099511628211ULL; }
  return h;
}
std::string payload(const Policy& p) {
  std::ostringstream out;
  out << std::setprecision(17) << p.config_hash << ' ' << p.shape.batch << ' ' << p.shape.input << ' '
      << p.shape.output << ' ' << p.shape.group << ' ' << static_cast<int>(p.backend) << '\n'
      << p.clip << ' ' << p.scale << ' ' << p.max_abs_limit << ' ' << p.clipping_limit << '\n'
      << p.bf16_us << ' ' << p.a8_us << ' ' << p.inspection_us << ' ' << p.decision_us << ' '
      << p.bf16_p95_us << ' ' << p.a8_p95_us << '\n'
      << p.guard_enabled << ' ' << p.sampled << ' ' << p.calibration_seed << ' ' << p.calibration_count << '\n'
      << std::quoted(p.environment_key) << '\n';
  return out.str();
}
}
void fit_guard(Policy& policy, const std::vector<CalibrationRow>& rows) {
  policy.guard_enabled = false;
  policy.max_abs_limit = policy.clipping_limit = 0;
  std::vector<double> maxima, clips;
  for (const auto& row : rows) {
    if (std::isfinite(row.a8_error) && row.a8_error <= generated::error_budget) {
      maxima.push_back(row.descriptor.max_abs);
      clips.push_back(row.descriptor.clipping_fraction);
    }
  }
  std::sort(maxima.begin(), maxima.end());
  maxima.erase(std::unique(maxima.begin(), maxima.end()), maxima.end());
  std::sort(clips.begin(), clips.end());
  clips.erase(std::unique(clips.begin(), clips.end()), clips.end());
  const auto bounded_grid = [](std::vector<double>& values) {
    constexpr std::size_t limit = 64;
    if (values.size() <= limit) return;
    std::vector<double> grid;
    for (std::size_t i = 0; i < limit; ++i) grid.push_back(values[i * (values.size() - 1) / (limit - 1)]);
    values = std::move(grid);
  };
  bounded_grid(maxima); bounded_grid(clips);
  std::size_t best = 0;
  for (double maximum : maxima) for (double clipping : clips) {
    std::size_t accepted = 0, unsafe = 0;
    for (const auto& row : rows) if (row.descriptor.max_abs <= maximum && row.descriptor.clipping_fraction <= clipping) {
      ++accepted;
      unsafe += !std::isfinite(row.a8_error) || row.a8_error > generated::error_budget;
    }
    if (accepted >= generated::minimum_guard_accepts && accepted > best &&
        static_cast<double>(unsafe) / accepted <= generated::max_calibration_false_safe_rate) {
      best = accepted;
      policy.guard_enabled = true;
      policy.max_abs_limit = maximum;
      policy.clipping_limit = clipping;
    }
  }
}
void validate_policy(const Policy& p, bool require_environment) {
  validate_shape(p.shape);
  if (p.config_hash != generated::config_hash) throw std::invalid_argument("policy belongs to another compiled configuration; recalibrate");
  if (p.backend != Backend::Scalar && p.backend != Backend::Mlir) throw std::invalid_argument("unknown policy backend");
  if (!p.calibration_count || p.calibration_count > 1000000 || p.environment_key.empty()) throw std::invalid_argument("incomplete calibration provenance");
  if (!std::isfinite(p.clip) || !std::isfinite(p.scale) || p.clip <= 0 || p.scale <= 0 ||
      std::abs(static_cast<double>(p.clip) / 127 - p.scale) > static_cast<double>(p.scale) * 1e-5)
    throw std::invalid_argument("inconsistent activation scale");
  if (!std::isfinite(p.max_abs_limit) || p.max_abs_limit < 0 || !std::isfinite(p.clipping_limit) ||
      p.clipping_limit < 0 || p.clipping_limit > 1) throw std::invalid_argument("invalid calibrated guard");
  for (double value : {p.bf16_us, p.a8_us, p.bf16_p95_us, p.a8_p95_us})
    if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("missing/invalid measured execution costs");
  for (double value : {p.inspection_us, p.decision_us})
    if (!std::isfinite(value) || value < 0) throw std::invalid_argument("invalid measured overhead");
  if (require_environment && p.environment_key != environment_key())
    throw std::invalid_argument("policy environment/code/compiler fingerprint changed; recalibrate on this VM");
}
void save_policy(const Policy& input, const std::string& path) {
  validate_policy(input);
  const auto text = payload(input);
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write policy: " + path);
  out << "NM_POLICY_V1\n" << hash_text(text) << '\n' << text;
  if (!out) throw std::runtime_error("failed writing policy");
}
Policy load_policy(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open policy; run ./nm calibrate first: " + path);
  std::string magic;
  Policy p;
  int backend = -1, guard = -1, sampled = -1;
  in >> magic >> p.id >> p.config_hash >> p.shape.batch >> p.shape.input >> p.shape.output >> p.shape.group >> backend
     >> p.clip >> p.scale >> p.max_abs_limit >> p.clipping_limit >> p.bf16_us >> p.a8_us >> p.inspection_us >> p.decision_us
     >> p.bf16_p95_us >> p.a8_p95_us >> guard >> sampled >> p.calibration_seed >> p.calibration_count >> std::quoted(p.environment_key);
  if (!in || magic != "NM_POLICY_V1" || backend < 0 || backend > 1 || guard < 0 || guard > 1 || sampled < 0 || sampled > 1)
    throw std::invalid_argument("malformed policy file");
  in >> std::ws;
  if (!in.eof()) throw std::invalid_argument("unexpected trailing policy data");
  p.backend = static_cast<Backend>(backend);
  p.guard_enabled = guard != 0;
  p.sampled = sampled != 0;
  if (p.id != hash_text(payload(p))) throw std::invalid_argument("policy checksum mismatch");
  validate_policy(p);
  return p;
}
}
