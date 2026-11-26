#include "neurominds/experiments.hpp"
#include "neurominds/kernels.hpp"
#include "nm_config.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

namespace nm {
namespace {
volatile double observed_checksum = 0;
void consume(const std::vector<float>& values) {
  double sum = 0;
  for (float value : values) sum += value;
  observed_checksum = observed_checksum + sum;
}
double mean(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); }
double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[static_cast<std::size_t>(std::ceil(v.size() * q)) - 1];
}
const char* regime_for(std::size_t index) {
  static const char* regimes[] = {"normal", "outliers", "shifted", "mixed"};
  return regimes[index % 4];
}
void directory(const std::string& path) {
  std::filesystem::create_directories(path);
}
void check_options(const Options& o) {
  validate_shape(o.shape);
  if (!o.samples || o.samples > 4096 || !o.repeats || o.repeats > 10000 || o.warmup > 10000)
    throw std::invalid_argument("samples must be [1,4096], repeats [1,10000], warmup [0,10000]");
  if (o.backend == Backend::Mlir && !mlir_available()) throw std::invalid_argument("MLIR build required or explicitly use --backend scalar");
}
void require_matching(const Options& o, const Policy& p) {
  if (o.shape != p.shape || o.backend != p.backend || (o.sampled && o.shape.inputs() > generated::sample_size) != p.sampled)
    throw std::invalid_argument("shape/backend/inspection mode differs from calibration; recalibrate with the same options");
  if (o.seed != p.calibration_seed) throw std::invalid_argument("evaluation must use the policy's base seed (held-out split is separate)");
}
Invocation forced(const std::vector<float>& x, const PackedWeights& w, const Shape& shape, Variant variant, float scale, Backend b) {
  Invocation r;
  r.plan.variant = variant;
  r.reason = "forced variant (not an adaptive selection)";
  const auto start = Clock::now();
  run_kernel(x, w, shape, variant, scale, b, r.output, r.preparation_us, r.kernel_us);
  r.total_us = elapsed_us(start);
  r.dispatch_us = std::max(0.0, r.total_us - r.preparation_us - r.kernel_us);
  return r;
}
void print_invocation(const std::string& regime, const Invocation& r, const std::vector<float>& reference) {
  const auto& d = r.descriptor;
  const bool forced_mode = r.reason.rfind("forced variant", 0) == 0;
  std::cout << std::fixed << std::setprecision(3) << "regime=" << regime << " max=" << d.max_abs << " mean=" << d.mean_abs
    << " p99=" << d.percentile99 << " clipping=" << d.clipping_fraction << " outliers=" << d.outlier_fraction
    << " inspected=" << d.observed << (d.sampled ? "(sampled)" : "(full)")
    << " cache=" << (forced_mode ? "n/a" : r.cache_hit ? "hit" : "miss") << " selected=" << name(r.plan.variant)
    << " fallback=" << r.fallback << " error_w4=" << std::setprecision(6) << relative_l2(r.output, reference)
    << std::setprecision(2) << " total_us=" << r.total_us << " kernel_us=" << r.kernel_us
    << " inspector_us=" << r.inspector_us << " reason=\"" << r.reason << "\"\n";
}
}
std::uint64_t split_seed(std::uint64_t base, bool held_out, std::size_t index) {
  if (index >= (1ULL << 63)) throw std::invalid_argument("split index too large");
  // SplitMix64 is a bijection; the high-bit split gives disjoint seeds for a fixed base.
  std::uint64_t z = (base ^ static_cast<std::uint64_t>(index) ^ (held_out ? (1ULL << 63) : 0)) + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
std::vector<float> synthetic_weights(const Shape& s, std::uint64_t seed) {
  validate_shape(s);
  std::mt19937_64 engine(seed ^ 0x8b8b8b8b01020304ULL);
  std::normal_distribution<float> normal(0, 1.0f / std::sqrt(static_cast<float>(s.input)));
  std::vector<float> values(product(s.input, s.output));
  for (auto& v : values) v = normal(engine);
  return values;
}
std::vector<float> synthetic_input(const Shape& s, const std::string& regime, std::uint64_t seed) {
  validate_shape(s);
  if (regime != "normal" && regime != "outliers" && regime != "shifted" && regime != "mixed" && regime != "zero")
    throw std::invalid_argument("regime must be normal, outliers, shifted, mixed or zero");
  std::mt19937_64 engine(seed);
  std::normal_distribution<float> normal(0, 1);
  std::uniform_real_distribution<float> uniform(0, 1);
  std::vector<float> x(s.inputs());
  for (auto& value : x) {
    value = normal(engine);
    if (regime == "outliers" && uniform(engine) < 0.005f) value *= 40;
    if (regime == "shifted") value *= 3.5f;
    if (regime == "mixed") value *= uniform(engine) < 0.1f ? 6.0f : 0.5f;
    if (regime == "zero") value = 0;
  }
  // Ensure the supervisor's outlier demonstration contains a real outlier even at tiny shapes.
  if (regime == "outliers") x[x.size() / 3] = 40;
  return x;
}
Policy calibrate(const Options& o) {
  check_options(o);
  const auto weights = quantize_weights(synthetic_weights(o.shape, o.seed), o.shape);
  return calibrate_layer(o, weights, [&](const std::string& regime, std::uint64_t seed) {
    return synthetic_input(o.shape, regime, seed);
  });
}
Policy calibrate_layer(const Options& o, const PackedWeights& weights, const CalibrationInput& input) {
  check_options(o);
  if (o.samples < generated::minimum_guard_accepts) throw std::invalid_argument("too few calibration samples");
  directory(o.output);
  Policy p;
  p.config_hash = generated::config_hash; p.shape = o.shape; p.backend = o.backend;
  p.sampled = o.sampled && o.shape.inputs() > generated::sample_size;
  p.calibration_seed = o.seed; p.calibration_count = o.samples; p.environment_key = environment_key();
  std::vector<float> magnitudes;
  // Scale fitting uses typical calibration inputs only, separate from held-out evaluation.
  const std::size_t cap = 65536;
  for (std::size_t i = 0; i < 16 && magnitudes.size() < cap; ++i) {
    const auto x = input("normal", split_seed(o.seed, false, 100000 + i));
    const auto take = std::min(x.size(), cap - magnitudes.size());
    for (std::size_t j = 0; j < take; ++j) magnitudes.push_back(std::abs(x[j * x.size() / take]));
  }
  std::sort(magnitudes.begin(), magnitudes.end());
  const auto qi = static_cast<std::size_t>(std::ceil(magnitudes.size() * generated::activation_quantile)) - 1;
  p.clip = std::max(magnitudes[qi], 1e-6f); p.scale = p.clip / 127;
  auto warm_input = input("normal", split_seed(o.seed, false, 200000));
  for (std::size_t i = 0; i < o.warmup; ++i) for (auto v : {Variant::Bf16, Variant::A8})
    consume(forced(warm_input, weights, o.shape, v, p.scale, o.backend).output);
  std::ofstream csv(o.output + "/calibration.csv");
  if (!csv) throw std::runtime_error("cannot write calibration.csv");
  csv << "sample,seed,regime,max_abs,mean_abs,p99,clipping_fraction,outlier_fraction,observed,inspector_us,bf16_error_w4,a8_error_w4\n";
  csv << std::setprecision(17);
  std::ofstream timings(o.output + "/calibration_timings.csv");
  if (!timings) throw std::runtime_error("cannot write calibration_timings.csv");
  timings << "sample,repeat,variant,total_us,preparation_us,kernel_us\n" << std::setprecision(17);
  std::vector<CalibrationRow> rows;
  std::vector<double> bf16_times, a8_times, inspector_times;
  for (std::size_t i = 0; i < o.samples; ++i) {
    const auto seed = split_seed(o.seed, false, i);
    const auto x = input(regime_for(i), seed);
    const auto reference = w4_reference(x, weights, o.shape);
    const auto start = Clock::now();
    auto d = inspect(x, o.shape, p.clip, o.sampled);
    const double inspect_us = elapsed_us(start);
    inspector_times.push_back(inspect_us);
    double errors[2] = {};
    for (std::size_t rep = 0; rep < o.repeats; ++rep) for (std::size_t order = 0; order < 2; ++order) {
      const auto index = (order + rep + i) % 2;
      const auto variant = index == 0 ? Variant::Bf16 : Variant::A8;
      auto r = forced(x, weights, o.shape, variant, p.scale, o.backend);
      (index == 0 ? bf16_times : a8_times).push_back(r.total_us);
      if (!rep) errors[index] = relative_l2(r.output, reference);
      timings << i << ',' << rep << ',' << name(variant) << ',' << r.total_us << ',' << r.preparation_us << ',' << r.kernel_us << '\n';
      consume(r.output);
    }
    rows.push_back({d, errors[1], errors[0]});
    csv << i << ',' << seed << ',' << regime_for(i) << ',' << d.max_abs << ',' << d.mean_abs << ',' << d.percentile99
      << ',' << d.clipping_fraction << ',' << d.outlier_fraction << ',' << d.observed << ',' << inspect_us << ',' << errors[0] << ',' << errors[1] << '\n';
    if ((i + 1) % 16 == 0 || i + 1 == o.samples) std::cout << "Calibrated input " << i + 1 << '/' << o.samples << '\n';
  }
  fit_guard(p, rows);
  p.bf16_us = mean(bf16_times); p.a8_us = mean(a8_times);
  p.bf16_p95_us = percentile(bf16_times, .95); p.a8_p95_us = percentile(a8_times, .95);
  p.inspection_us = mean(inspector_times);
  PlanCache cache(generated::cache_capacity); Counters counters;
  const auto decisions = Clock::now();
  for (const auto& row : rows) {
    const auto key = plan_key(o.shape, row.descriptor, p, o.backend);
    Plan plan;
    const bool hit = cache.find(key, plan, counters);
    const auto variant = risk_accepts(row.descriptor, p) && cost_accepts(row.descriptor, p, p.inspection_us) ? Variant::A8 : Variant::Bf16;
    if (!hit) {
      plan = {key, variant, Variant::Bf16, generated::config_version, 1, 0}; plan.validity_key = validity_key(plan); cache.put(plan, counters);
    }
    observed_checksum = observed_checksum + static_cast<double>(static_cast<unsigned>(variant));
  }
  p.decision_us = elapsed_us(decisions) / rows.size();
  if (!csv || !timings) throw std::runtime_error("failed writing calibration measurements");
  csv.close(); timings.close();
  std::filesystem::create_directories(std::filesystem::path(o.policy).parent_path().empty() ? "." : std::filesystem::path(o.policy).parent_path());
  save_policy(p, o.policy); p = load_policy(o.policy);
  write_manifest(o, &p, "calibration_only", o.output + "/calibration_manifest.json");
  std::size_t accepts = 0, unsafe = 0;
  for (const auto& r : rows) if (risk_accepts(r.descriptor, p)) { ++accepts; unsafe += r.a8_error > generated::error_budget; }
  std::cout << "Calibration only: " << rows.size() << " inputs; clip=" << p.clip << " scale=" << p.scale
    << " guard_enabled=" << p.guard_enabled << " accepts=" << accepts << " unsafe_accepted=" << unsafe
    << "\nMeasured static BF16=" << p.bf16_us << " us A8=" << p.a8_us << " us; inspector=" << p.inspection_us
    << " us; decision/cache=" << p.decision_us << " us\nSaved " << o.policy << '\n';
  return p;
}
void demonstrate(const Options& o) {
  check_options(o);
  const auto p = load_policy(o.policy); require_matching(o, p);
  const auto original = synthetic_weights(o.shape, o.seed); const auto w = quantize_weights(original, o.shape);
  const auto ordinary = synthetic_input(o.shape, "normal", split_seed(o.seed, true, 500000));
  const auto outliers = synthetic_input(o.shape, "outliers", split_seed(o.seed, true, 500001));
  Executor executor(generated::cache_capacity);
  std::cout << "Adaptive sequence (references are evaluation-only; they never enter the inspector):\n";
  for (std::size_t i = 0; i < 5; ++i) {
    const bool outlier = i == 2;
    const auto& x = outlier ? outliers : ordinary;
    const auto ref = w4_reference(x, w, o.shape);
    print_invocation(outlier ? "outliers" : "normal", executor.adaptive(x, w, o.shape, p, o.backend, o.sampled), ref);
  }
  std::cout << "Forced implementations (separate from adaptive decisions):\n";
  for (auto v : {Variant::Bf16, Variant::A8}) {
    auto result = forced(ordinary, w, o.shape, v, p.scale, o.backend);
    result.descriptor = inspect(ordinary, o.shape, p.clip, o.sampled);
    print_invocation("normal/forced", result, w4_reference(ordinary, w, o.shape));
  }
  const auto c = executor.counters();
  std::cout << "cache_hits=" << c.hits << " misses=" << c.misses << " invalidations=" << c.invalidations << " fallbacks=" << c.fallbacks << '\n';
}
void benchmark(const Options& o) {
  check_options(o); directory(o.output);
  const auto p = load_policy(o.policy); require_matching(o, p);
  const auto original = synthetic_weights(o.shape, o.seed); const auto w = quantize_weights(original, o.shape);
  std::ofstream csv(o.output + "/benchmark.csv");
  if (!csv) throw std::runtime_error("cannot write benchmark.csv");
  csv << "sample,repeat,seed,regime,mode,selected,cache_hit,cache_miss,invalidations,fallback,total_us,preparation_us,kernel_us,inspector_us,validity_us,decision_us,dispatch_us,error_w4,error_fp32,actual_a8_safe,guard_accept,max_abs,clipping_fraction,inspected_bytes,weight_bytes,weight_capacity_bytes,metadata_bytes,resident_bytes,peak_resident_bytes\n";
  csv << std::setprecision(17);
  const char* modes[] = {"fp32", "static_bf16", "static_a8", "adaptive", "guard_disabled"};
  Executor adaptive(generated::cache_capacity), ablation(generated::cache_capacity);
  std::vector<double> latencies[5];
  std::size_t safe_count = 0, unsafe_count = 0, false_safe = 0, false_fallback = 0, accepted_count = 0;
  for (std::size_t i = 0; i < o.samples; ++i) {
    const auto seed = split_seed(o.seed, true, i);
    const auto x = synthetic_input(o.shape, regime_for(i), seed);
    // Held-out labels are evaluation data only. No policy mutation occurs here.
    const auto full_reference = fp32_reference(x, original, o.shape);
    const auto reference = w4_reference(x, w, o.shape);
    const auto a8 = forced(x, w, o.shape, Variant::A8, p.scale, o.backend);
    const bool actual_safe = relative_l2(a8.output, reference) <= generated::error_budget;
    const auto d = inspect(x, o.shape, p.clip, o.sampled);
    const bool predicted = risk_accepts(d, p);
    safe_count += actual_safe; unsafe_count += !actual_safe; accepted_count += predicted;
    false_safe += predicted && !actual_safe; false_fallback += !predicted && actual_safe;
    Executor warm(generated::cache_capacity);
    for (std::size_t j = 0; j < o.warmup; ++j) {
      consume(fp32_reference(x, original, o.shape));
      consume(forced(x, w, o.shape, Variant::Bf16, p.scale, o.backend).output);
      consume(forced(x, w, o.shape, Variant::A8, p.scale, o.backend).output);
      consume(warm.adaptive(x, w, o.shape, p, o.backend, o.sampled).output);
      consume(warm.adaptive(x, w, o.shape, p, o.backend, o.sampled, true).output);
    }
    for (std::size_t rep = 0; rep < o.repeats; ++rep) for (std::size_t order = 0; order < 5; ++order) {
      const auto mode = (order + rep + i) % 5; // rotate trial order, deterministic
      Invocation r;
      std::size_t invalidations = 0;
      if (mode == 0) {
        const auto begin = Clock::now(); r.output = fp32_reference(x, original, o.shape); r.total_us = elapsed_us(begin);
        r.kernel_us = 0; // only whole-invocation timing is available for this reference
      } else if (mode < 3) r = forced(x, w, o.shape, mode == 1 ? Variant::Bf16 : Variant::A8, p.scale, o.backend);
      else {
        auto& executor = mode == 3 ? adaptive : ablation;
        const auto before = executor.counters().invalidations;
        r = executor.adaptive(x, w, o.shape, p, o.backend, o.sampled, mode == 4);
        invalidations = executor.counters().invalidations - before;
      }
      latencies[mode].push_back(r.total_us);
      const auto error = relative_l2(r.output, reference), full_error = relative_l2(r.output, full_reference);
      consume(r.output); // observable consumption outside the timed interval
      csv << i << ',' << rep << ',' << seed << ',' << regime_for(i) << ',' << modes[mode] << ','
        << (mode == 0 ? "fp32" : name(r.plan.variant)) << ',' << r.cache_hit << ',' << (mode >= 3 && !r.cache_hit)
        << ',' << invalidations << ',' << r.fallback << ',' << r.total_us << ',' << r.preparation_us << ',' << r.kernel_us
        << ',' << r.inspector_us << ',' << r.validity_us << ',' << r.decision_us << ',' << r.dispatch_us
        << ',' << error << ',' << full_error << ',' << actual_safe << ',' << predicted << ',' << d.max_abs << ',' << d.clipping_fraction
        << ',' << (mode >= 3 ? d.bytes_read : 0) << ',' << w.bytes() << ',' << w.capacity_bytes() << ','
        << sizeof(Policy) + p.environment_key.capacity() + 3 * sizeof(Executor) + 3 * sizeof(Plan) * generated::cache_capacity
        << ',' << resident_bytes() << ',' << peak_resident_bytes() << '\n';
    }
    if ((i + 1) % 4 == 0 || i + 1 == o.samples) std::cout << "Evaluated held-out input " << i + 1 << '/' << o.samples << '\n';
  }
  if (!csv) throw std::runtime_error("failed writing benchmark measurements");
  csv.close();
  write_manifest(o, &p, "held_out_evaluation_policy_frozen", o.output + "/benchmark_manifest.json");
  std::ofstream metrics(o.output + "/guard_metrics.json");
  if (!metrics) throw std::runtime_error("cannot write guard metrics");
  metrics << "{\"held_out_inputs\":" << o.samples << ",\"actual_safe\":" << safe_count << ",\"actual_unsafe\":" << unsafe_count
    << ",\"accepted\":" << accepted_count << ",\"false_safe\":" << false_safe << ",\"false_fallback\":" << false_fallback
    << ",\"false_safe_rate_among_unsafe\":" << (unsafe_count ? std::to_string(static_cast<double>(false_safe) / unsafe_count) : "null")
    << ",\"false_fallback_rate_among_safe\":" << (safe_count ? std::to_string(static_cast<double>(false_fallback) / safe_count) : "null")
    << ",\"unsafe_fraction_among_accepted\":" << (accepted_count ? std::to_string(static_cast<double>(false_safe) / accepted_count) : "null") << "}\n";
  if (!metrics) throw std::runtime_error("failed writing guard metrics");
  std::cout << "Held-out operator benchmark, " << name(o.backend) << "; microseconds (preliminary VM performance):\n";
  for (std::size_t mode = 0; mode < 5; ++mode)
    std::cout << modes[mode] << ": mean=" << mean(latencies[mode]) << " median=" << percentile(latencies[mode], .5)
      << " p95=" << percentile(latencies[mode], .95) << '\n';
  std::cout << "False-safe=" << false_safe << '/' << unsafe_count << " unsafe; false-fallback=" << false_fallback << '/' << safe_count
    << " safe; packed+scales=" << w.bytes() << " bytes; checksum=" << observed_checksum << "\nSaved " << o.output << "/benchmark.csv\n";
}
void projection(const Options& o) {
  check_options(o);
  const auto original = synthetic_weights(o.shape, o.seed); const auto w = quantize_weights(original, o.shape);
  const auto x = synthetic_input(o.shape, o.regime, split_seed(o.seed, true, 700000));
  const auto reference = w4_reference(x, w, o.shape);
  if (o.variant == "fp32" || o.variant == "w4-reference") {
    const auto start = Clock::now();
    const auto result = o.variant == "fp32" ? fp32_reference(x, original, o.shape) : w4_reference(x, w, o.shape);
    const double us = elapsed_us(start);
    consume(result);
    std::cout << "reference=" << o.variant << " outputs=" << result.size() << " total_us=" << us
      << " relative_error_to_w4=" << relative_l2(result, reference) << '\n';
  } else {
    const auto p = load_policy(o.policy); require_matching(o, p);
    Invocation r;
    if (o.variant == "adaptive") { Executor executor(generated::cache_capacity); r = executor.adaptive(x, w, o.shape, p, o.backend, o.sampled); }
    else if (o.variant == "bf16" || o.variant == "a8") {
      r = forced(x, w, o.shape, o.variant == "bf16" ? Variant::Bf16 : Variant::A8, p.scale, o.backend);
      r.descriptor = inspect(x, o.shape, p.clip, o.sampled);
    } else throw std::invalid_argument("variant must be fp32, w4-reference, bf16, a8 or adaptive");
    print_invocation(o.regime + (o.variant == "adaptive" ? "" : "/forced"), r, reference);
  }
}
}
