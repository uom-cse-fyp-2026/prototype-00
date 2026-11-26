#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nm::risk_math {
// Same rounding and saturation as to_a8, using absolute magnitudes.
struct Reconstruction {
  double input_sq = 0, error_sq = 0;
  void add(double magnitude, float scale) {
    const double q = std::round(std::min(magnitude / scale, 127.0));
    const double delta = magnitude - q * static_cast<double>(scale);
    input_sq += magnitude * magnitude;
    error_sq += delta * delta;
  }
  double relative() const {
    return input_sq == 0 ? 0 : std::sqrt(error_sq / input_sq);
  }
};
inline bool valid_feature(double maximum, double clipping, double reconstruction, std::size_t observed) {
  return observed && std::isfinite(maximum) && maximum >= 0 &&
      std::isfinite(clipping) && clipping >= 0 && clipping <= 1 &&
      std::isfinite(reconstruction) && reconstruction >= 0;
}
struct Observation { double maximum, clipping, reconstruction, output_error; };
struct Rule {
  std::size_t accepted = 0, unsafe = 0;
  double maximum = 0, clipping = 0, reconstruction = 0;
};
// Group equal scores to agree with inclusive runtime comparisons.
inline Rule best_reconstruction_rule(std::vector<Observation> rows, double budget,
                                     std::size_t min_accepts, double max_unsafe_fraction) {
  rows.erase(std::remove_if(rows.begin(), rows.end(), [](const Observation& r) {
    return !valid_feature(r.maximum, r.clipping, r.reconstruction, 1);
  }), rows.end());
  std::sort(rows.begin(), rows.end(), [](const Observation& a, const Observation& b) {
    return a.reconstruction < b.reconstruction;
  });
  Rule best, current;
  for (std::size_t i = 0; i < rows.size();) {
    const double threshold = rows[i].reconstruction;
    do {
      const auto& r = rows[i++];
      ++current.accepted;
      current.unsafe += !std::isfinite(r.output_error) || r.output_error > budget;
      current.maximum = std::max(current.maximum, r.maximum);
      current.clipping = std::max(current.clipping, r.clipping);
    } while (i < rows.size() && rows[i].reconstruction == threshold);
    current.reconstruction = threshold;
    if (current.accepted >= min_accepts &&
        static_cast<double>(current.unsafe) / current.accepted <= max_unsafe_fraction)
      best = current;
  }
  return best;
}
inline bool accepts(const Rule& r, const Observation& x) {
  return valid_feature(x.maximum, x.clipping, x.reconstruction, 1) &&
      x.maximum <= r.maximum && x.clipping <= r.clipping && x.reconstruction <= r.reconstruction;
}
// Tighten on a separate offline split, never on held-out evaluation labels.
inline Rule tighten_rule(Rule rule, const std::vector<Observation>& training,
                         const std::vector<Observation>& validation, double budget,
                         std::size_t min_accepts) {
  if (rule.accepted < min_accepts) return {};
  for (const auto* rows : {&training, &validation}) for (const auto& r : *rows) {
    if (accepts(rule, r) && (!std::isfinite(r.output_error) || r.output_error > budget)) {
      if (r.reconstruction == 0) return {};
      rule.reconstruction = std::nextafter(r.reconstruction, -std::numeric_limits<double>::infinity());
    }
  }
  auto count = [&](const std::vector<Observation>& rows) {
    std::size_t accepted = 0;
    for (const auto& r : rows) if (accepts(rule, r)) {
      if (!std::isfinite(r.output_error) || r.output_error > budget) return std::size_t{0};
      ++accepted;
    }
    return accepted;
  };
  rule.accepted = count(training);
  rule.unsafe = 0;
  if (rule.accepted < min_accepts || count(validation) < min_accepts) return {};
  return rule;
}
struct ScaleScore {
  float clip;
  std::size_t safe = 0;
  double total_error = 0;
  void add(double output_error, double budget) {
    safe += std::isfinite(output_error) && output_error <= budget;
    total_error += std::isfinite(output_error) ? output_error : std::numeric_limits<double>::infinity();
  }
};
inline std::vector<ScaleScore> scale_candidates(const std::vector<float>& sorted, double baseline_quantile) {
  if (sorted.empty()) throw std::invalid_argument("empty scale-fit inputs");
  std::vector<ScaleScore> result;
  auto add = [&](float clip) {
    clip = std::max(clip, 1e-6f);
    if (std::isfinite(clip) && std::none_of(result.begin(), result.end(),
        [&](const ScaleScore& x) { return x.clip == clip; })) result.push_back({clip});
  };
  auto quantile = [&](double q) {
    return sorted[static_cast<std::size_t>(std::ceil(sorted.size() * q)) - 1];
  };
  const float baseline = std::max(quantile(baseline_quantile), 1e-6f);
  add(baseline);
  for (double q : {.90, .95, .99, 1.0}) add(quantile(q));
  for (float factor : {.25f, .5f, 2.f}) add(baseline * factor);
  return result;
}
inline float select_clip(const std::vector<ScaleScore>& scores) {
  if (scores.empty()) throw std::invalid_argument("no candidate activation scale");
  const ScaleScore* best = &scores.front();
  for (const auto& s : scores) {
    if (s.safe > best->safe || (s.safe == best->safe && s.total_error < best->total_error)) best = &s;
  }
  return best->clip;
}
inline bool profitable(double bf16_mean, double a8_mean, double bf16_p95, double a8_p95,
                       double inspection, double decision, double margin) {
  if (!std::isfinite(inspection) || inspection < 0 || !std::isfinite(decision) || decision < 0 ||
      !std::isfinite(a8_mean) || !std::isfinite(bf16_mean) || !std::isfinite(a8_p95) ||
      !std::isfinite(bf16_p95) || a8_mean <= 0 || bf16_mean <= 0 || a8_p95 <= 0 || bf16_p95 <= 0) return false;
  const double overhead = inspection + decision;
  return a8_mean + overhead < bf16_mean * (1 - margin) && a8_p95 + overhead < bf16_p95;
}
} // namespace nm::risk_math
