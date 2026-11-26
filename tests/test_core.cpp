#include "neurominds/experiments.hpp"
#include "neurominds/kernels.hpp"
#include "nm_config.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>

namespace {
void check(bool condition, const char* expression, int line) {
  if (!condition) throw std::runtime_error(std::string("line ") + std::to_string(line) + ": " + expression);
}
#define CHECK(expression) check((expression), #expression, __LINE__)
template<typename F> void rejects(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  CHECK(rejected);
}
float bits(std::uint32_t value) { float f; std::memcpy(&f, &value, sizeof(f)); return f; }
void close(float a, float b, float tolerance = 1e-5f) { CHECK(std::abs(a - b) <= tolerance); }
nm::Policy fixture(nm::Shape s, nm::Backend backend) {
  nm::Policy p;
  p.id = 123; p.config_hash = nm::generated::config_hash; p.shape = s; p.backend = backend;
  p.clip = 4; p.scale = 4.0f / 127; p.max_abs_limit = 4; p.clipping_limit = 0;
  p.bf16_us = 100000; p.bf16_p95_us = 100001; p.a8_us = .001; p.a8_p95_us = .002;
  p.inspection_us = 1; p.decision_us = 1; p.guard_enabled = true;
  p.calibration_seed = 42; p.calibration_count = 64; p.environment_key = nm::environment_key();
  return p;
}
void packing() {
  for (int low = -8; low <= 7; ++low) for (int high = -8; high <= 7; ++high) {
    auto packed = nm::pack_nibbles(low, high);
    CHECK(nm::unpack_nibble(packed, false) == low);
    CHECK(nm::unpack_nibble(packed, true) == high);
  }
  CHECK(nm::pack_nibbles(-8, 7) == 0x78);
  rejects([] { nm::pack_nibbles(-9, 0); });
  rejects([] { nm::pack_nibbles(0, 8); });
}
void weight_scales_and_tails() {
  nm::Shape s{1, 5, 2, 3};
  std::vector<float> original{7, -7, 0, 14, -14, 0, 0, 0, 0, 0};
  auto w = nm::quantize_weights(original, s);
  CHECK(w.data.size() == 6 && w.scales.size() == 4);
  CHECK(w.bytes() == 22);
  close(w.scales[0], 1); close(w.scales[1], 2); close(w.scales[2], 1); close(w.scales[3], 1);
  CHECK(nm::weight_value(w, 0, 0) == 7 && nm::weight_value(w, 0, 1) == -7);
  CHECK(nm::unpack_nibble(w.data[2], true) == 0); // odd-row padding cannot spill into the next row
  const auto dequantized = nm::dequantize_weights(w);
  CHECK(dequantized == original);
  w.data.pop_back(); rejects([&] { w.validate(); });
  rejects([&] { nm::quantize_weights({1}, s); });
}
void bf16_conversion() {
  CHECK(nm::to_bf16(1.0f) == 0x3f80);
  CHECK(nm::to_bf16(-1.0f) == 0xbf80);
  CHECK(nm::to_bf16(bits(0x3f808000)) == 0x3f80); // tie, even retained LSB
  CHECK(nm::to_bf16(bits(0x3f818000)) == 0x3f82); // tie, odd retained LSB
  CHECK(nm::to_bf16(bits(0xbf808000)) == 0xbf80);
  CHECK(nm::to_bf16(bits(0x80000000)) == 0x8000); // negative zero
  CHECK(std::isinf(nm::from_bf16(nm::to_bf16(std::numeric_limits<float>::infinity()))));
  CHECK(std::isnan(nm::from_bf16(nm::to_bf16(bits(0x7f800001)))));
  CHECK(nm::from_bf16(nm::to_bf16(1e-40f)) >= 0);
}
void a8_conversion() {
  CHECK(nm::to_a8(.5f, 1) == 1 && nm::to_a8(-.5f, 1) == -1);
  CHECK(nm::to_a8(10000, 1) == 127 && nm::to_a8(-10000, 1) == -127);
  CHECK(nm::to_a8(0, 1) == 0);
  rejects([] { nm::to_a8(1, 0); });
  rejects([] { nm::to_a8(std::numeric_limits<float>::quiet_NaN(), 1); });
}
void integer_accumulation() {
  nm::Shape s{1, 3, 1, 2};
  nm::PackedWeights w{3, 1, 2, {nm::pack_nibbles(-8, 7), nm::pack_nibbles(-1, 0)}, {.5f, 2.0f}};
  w.validate();
  std::vector<float> y(1);
  nm::a8_scalar({-127, 127, -1}, w, s, .25f, y);
  close(y[0], 238.625f);
}
void dimensions_and_extremes() {
  rejects([] { nm::validate_shape({0, 4, 3, 2}); });
  rejects([] { nm::validate_shape({1, 0, 3, 2}); });
  rejects([] { nm::validate_shape({1, 4, 3, 0}); });
  rejects([] { nm::validate_shape({1, nm::generated::max_input + 1, 3, 2}); });
  rejects([] { nm::product(std::numeric_limits<std::size_t>::max(), 2); });
  nm::Shape s{1, 3, 1, 2};
  auto w = nm::quantize_weights({1, -1, .5f}, s);
  std::vector<float> y; double prep = 0, kernel = 0;
  nm::run_kernel({0, 0, 0}, w, s, nm::Variant::A8, 1, nm::Backend::Scalar, y, prep, kernel);
  CHECK(y.size() == 1 && y[0] == 0);
  close(static_cast<float>(nm::relative_l2({0}, {0})), 0);
  nm::run_kernel({1e-30f, -1e-30f, 1e-30f}, w, s, nm::Variant::Bf16, 1, nm::Backend::Scalar, y, prep, kernel);
  CHECK(std::isfinite(y[0]));
  rejects([&] { nm::run_kernel({1, 2}, w, s, nm::Variant::A8, 1, nm::Backend::Scalar, y, prep, kernel); });
  rejects([&] { nm::run_kernel({1, 2, 3}, w, s, nm::Variant::A8, 1, static_cast<nm::Backend>(99), y, prep, kernel); });
  rejects([&] { nm::run_kernel({std::numeric_limits<float>::max(), 0, 0}, w, s, nm::Variant::Bf16, 1, nm::Backend::Scalar, y, prep, kernel); });
  w.scales[0] = 0; rejects([&] { w.validate(); });
}
void kernels(nm::Backend backend) {
  const nm::Shape shapes[] = {{1, 5, 3, 3}, {4, 33, 7, 16}, {8, 65, 9, 128}, {1, 128, 17, 32}};
  for (auto shape : shapes) {
    auto x = nm::synthetic_input(shape, "mixed", 1234);
    auto w = nm::quantize_weights(nm::synthetic_weights(shape, 987), shape);
    for (auto variant : {nm::Variant::Bf16, nm::Variant::A8}) {
      std::vector<float> y, scalar; double prep = 0, kernel = 0;
      nm::run_kernel(x, w, shape, variant, .04f, backend, y, prep, kernel);
      CHECK(y.size() == shape.outputs());
      const auto reference = nm::variant_reference(x, w, shape, variant, .04f);
      CHECK(nm::relative_l2(y, reference) < 2e-5);
      nm::run_kernel(x, w, shape, variant, .04f, nm::Backend::Scalar, scalar, prep, kernel);
      CHECK(nm::relative_l2(y, scalar) < 2e-5);
    }
  }
}
void inspector() {
  nm::Shape s{1, 4, 1, 2};
  auto d = nm::inspect({0, -1, 2, -5}, s, 2);
  close(d.max_abs, 5); close(d.mean_abs, 2); close(d.percentile99, 5);
  close(d.clipping_fraction, .25f); close(d.outlier_fraction, .5f);
  CHECK(d.observed == 4 && d.bytes_read == 16 && !d.sampled);
  auto sampled = nm::inspect({0, -1, 2, -5}, s, 2, true);
  CHECK(!sampled.sampled && sampled.bucket == d.bucket);
  s.input = nm::generated::sample_size * 2;
  auto large = nm::inspect(std::vector<float>(s.input, 1), s, 2, true);
  CHECK(large.sampled && large.observed == nm::generated::sample_size);
  rejects([&] { nm::inspect({1}, s, 2); });
}
void guard_calibration() {
  nm::Policy p;
  std::vector<nm::CalibrationRow> rows;
  for (unsigned i = 0; i < 12; ++i) { nm::Descriptor d; d.max_abs = 1 + i * .01; d.observed = 100; rows.push_back({d, .001, .001}); }
  nm::Descriptor unsafe; unsafe.max_abs = 50; unsafe.clipping_fraction = .1; unsafe.observed = 100;
  rows.push_back({unsafe, .8, .001});
  nm::fit_guard(p, rows);
  CHECK(p.guard_enabled && nm::risk_accepts(rows[0].descriptor, p) && !nm::risk_accepts(unsafe, p));
  rows.push_back({rows[0].descriptor, .8, .001});
  nm::fit_guard(p, rows); // indistinguishable unsafe input prevents any zero-false-safe acceptance
  CHECK(!p.guard_enabled);
}
void deterministic_splits() {
  std::set<std::uint64_t> calibration, heldout;
  for (unsigned i = 0; i < 1000; ++i) {
    auto a = nm::split_seed(42, false, i), b = nm::split_seed(42, true, i);
    CHECK(a == nm::split_seed(42, false, i));
    calibration.insert(a); heldout.insert(b);
  }
  CHECK(calibration.size() == 1000 && heldout.size() == 1000);
  for (auto value : calibration) CHECK(heldout.count(value) == 0);
  nm::Shape shape{1, 16, 3, 4};
  CHECK(nm::synthetic_input(shape, "normal", 42) == nm::synthetic_input(shape, "normal", 42));
}
void quality_is_separate() {
  nm::Shape shape{1, 4, 1, 4};
  auto w = nm::quantize_weights({1, 1, 1, 1}, shape);
  std::vector<float> x{40, 0, 0, 0}, y; double prep, kernel;
  const auto reference = nm::w4_reference(x, w, shape);
  nm::run_kernel(x, w, shape, nm::Variant::A8, 1.0f / 127, nm::Backend::Scalar, y, prep, kernel);
  CHECK(nm::relative_l2(y, reference) > .9);
  CHECK(nm::relative_l2(y, nm::variant_reference(x, w, shape, nm::Variant::A8, 1.0f / 127)) < 1e-6);
}
void cache_and_executor(nm::Backend backend) {
  nm::Shape s{1, 64, 8, 32};
  auto p = fixture(s, backend);
  auto w = nm::quantize_weights(nm::synthetic_weights(s, 42), s);
  std::vector<float> x(s.inputs(), .1f);
  nm::Executor executor(2);
  auto first = executor.adaptive(x, w, s, p, backend);
  CHECK(!first.cache_hit && first.plan.variant == nm::Variant::A8);
  auto second = executor.adaptive(x, w, s, p, backend);
  CHECK(second.cache_hit && second.plan.variant == nm::Variant::A8);
  auto outliers = x; outliers[4] = 50;
  auto rejected = executor.adaptive(outliers, w, s, p, backend);
  CHECK(rejected.fallback && rejected.plan.variant == nm::Variant::Bf16);
  auto returned = executor.adaptive(x, w, s, p, backend);
  CHECK(returned.cache_hit);
  auto bad = first.plan; ++bad.config_version;
  auto fallback = executor.execute(x, w, s, p, backend, bad, first.descriptor);
  CHECK(fallback.fallback && fallback.plan.variant == nm::Variant::Bf16);
  bad = first.plan; bad.variant = static_cast<nm::Variant>(99); bad.validity_key = nm::validity_key(bad);
  CHECK(executor.execute(x, w, s, p, backend, bad, first.descriptor).fallback);
  const auto old_invalidations = executor.counters().invalidations;
  ++p.id;
  executor.adaptive(x, w, s, p, backend);
  CHECK(executor.counters().invalidations > old_invalidations);
  p.bf16_us = p.bf16_p95_us = .000001;
  CHECK(executor.adaptive(x, w, s, p, backend).plan.variant == nm::Variant::Bf16);
}
void same_bucket_rechecks_guard() {
  nm::Shape s{1, 64, 2, 32};
  auto p = fixture(s, nm::Backend::Scalar); p.max_abs_limit = 1.03;
  auto w = nm::quantize_weights(nm::synthetic_weights(s, 42), s);
  nm::Executor e;
  std::vector<float> low(s.inputs(), 1.02f), high(s.inputs(), 1.04f);
  auto a = e.adaptive(low, w, s, p, p.backend), b = e.adaptive(high, w, s, p, p.backend);
  CHECK(a.descriptor.bucket == b.descriptor.bucket);
  CHECK(a.plan.variant == nm::Variant::A8);
  CHECK(b.cache_hit && b.fallback && b.plan.variant == nm::Variant::Bf16);
}
void generated_guard_equivalence() {
  nm::Shape shape{1, 64, 2, 32};
  auto scalar = fixture(shape, nm::Backend::Scalar), compiled = fixture(shape, nm::Backend::Mlir);
  scalar.max_abs_limit = compiled.max_abs_limit = 3;
  scalar.clipping_limit = compiled.clipping_limit = .02;
  for (bool enabled : {false, true}) for (double maximum : {0., 2.999, 3., 3.001, 50.})
    for (double clipping : {0., .019, .02, .021, 1.}) {
      scalar.guard_enabled = compiled.guard_enabled = enabled;
      nm::Descriptor d; d.observed = 100; d.max_abs = maximum; d.clipping_fraction = clipping;
      CHECK(nm::risk_accepts(d, scalar) == nm::risk_accepts(d, compiled));
    }
}
void bounded_cache() {
  nm::PlanCache c(1); nm::Counters counters;
  nm::Plan a; a.key.policy_id = 1; a.validity_key = nm::validity_key(a);
  c.put(a, counters);
  auto b = a; b.key.feature_bucket = 1; b.validity_key = nm::validity_key(b);
  c.put(b, counters);
  CHECK(c.size() == 1 && counters.invalidations == 1);
  nm::Plan found;
  CHECK(!c.find(a.key, found, counters) && c.find(b.key, found, counters));
  c.invalidate(99, 2, counters);
  CHECK(c.size() == 0);
  rejects([] { nm::PlanCache invalid(0); });
}
void cost_guard_boundaries() {
  auto policy = fixture({1, 64, 8, 32}, nm::Backend::Scalar);
  policy.bf16_us = 100;
  policy.a8_us = 70;
  policy.bf16_p95_us = 120;
  policy.a8_p95_us = 85;
  policy.decision_us = 2;
  nm::Descriptor descriptor;

  CHECK(nm::cost_accepts(descriptor, policy, 8));

  // Equal to the 5% mean-improvement boundary is insufficient: the rule is strict.
  policy.a8_us = 85;
  CHECK(!nm::cost_accepts(descriptor, policy, 8));

  // A mean improvement alone must not override a failed tail-latency condition.
  policy.a8_us = 70;
  policy.a8_p95_us = 110;
  CHECK(!nm::cost_accepts(descriptor, policy, 8));
  CHECK(!nm::cost_accepts(descriptor, policy, -1));
  CHECK(!nm::cost_accepts(descriptor, policy, std::numeric_limits<double>::quiet_NaN()));
}
void fifo_hit_preserves_eviction_order() {
  nm::PlanCache cache(2);
  nm::Counters counters;
  nm::Plan first;
  first.key.feature_bucket = 1;
  first.validity_key = nm::validity_key(first);

  auto second = first;
  second.key.feature_bucket = 2;
  second.validity_key = nm::validity_key(second);
  auto third = first;
  third.key.feature_bucket = 3;
  third.validity_key = nm::validity_key(third);

  cache.put(first, counters);
  cache.put(second, counters);
  nm::Plan found;
  CHECK(cache.find(first.key, found, counters));
  cache.put(third, counters);

  // Unlike LRU, hitting the first insertion does not protect it from FIFO eviction.
  CHECK(!cache.find(first.key, found, counters));
  CHECK(cache.find(second.key, found, counters));
  CHECK(cache.find(third.key, found, counters));
  CHECK(cache.size() == 2 && counters.invalidations == 1);
}
void guard_candidate_grid() {
  std::vector<nm::CalibrationRow> rows;
  for (unsigned i = 0; i < 130; ++i) {
    nm::Descriptor descriptor;
    descriptor.observed = 100;
    descriptor.max_abs = 1 + i * .01;
    descriptor.clipping_fraction = i * .001;
    rows.push_back({descriptor, .001, .001});
  }

  // More than 64 distinct candidates exercises bounded-grid fitting, including both endpoints.
  nm::Policy policy;
  nm::fit_guard(policy, rows);
  CHECK(policy.guard_enabled);
  CHECK(policy.max_abs_limit == rows.back().descriptor.max_abs);
  CHECK(policy.clipping_limit == rows.back().descriptor.clipping_fraction);
  for (const auto& row : rows) {
    CHECK(nm::risk_accepts(row.descriptor, policy));
  }

  // Every useful threshold pair also accepts this indistinguishable unsafe example.
  rows.push_back({rows.front().descriptor, .8, .001});
  nm::fit_guard(policy, rows);
  CHECK(!policy.guard_enabled && policy.max_abs_limit == 0 && policy.clipping_limit == 0);
}
void policy_roundtrip() {
  auto p = fixture({1, 64, 8, 32}, nm::Backend::Scalar);
  auto path = std::filesystem::temp_directory_path() / ("nm-policy-" + std::to_string(nm::Clock::now().time_since_epoch().count()) + ".nm");
  nm::save_policy(p, path.string());
  auto loaded = nm::load_policy(path.string());
  CHECK(loaded.shape == p.shape && loaded.scale == p.scale && loaded.id != 0);
  { std::ofstream out(path, std::ios::app); out << "trailing garbage"; }
  rejects([&] { nm::load_policy(path.string()); });
  std::filesystem::remove(path);
  p.config_hash ^= 1; rejects([&] { nm::validate_policy(p, false); });
  p = fixture({1, 64, 8, 32}, nm::Backend::Scalar); p.scale = 0;
  rejects([&] { nm::validate_policy(p, false); });
}
}
int main() {
  std::vector<std::pair<const char*, std::function<void()>>> tests = {
    {"all signed W4 packing pairs", packing}, {"group scales, odd rows and tails", weight_scales_and_tails},
    {"BF16 ties, NaN, infinity and signed zero", bf16_conversion}, {"A8 rounding and saturation", a8_conversion},
    {"INT32 group accumulation and scale placement", integer_accumulation}, {"invalid dimensions and extreme inputs", dimensions_and_extremes},
    {"scalar kernels vs independent numerical references", [] { kernels(nm::Backend::Scalar); }},
    {"full and sampled inspector descriptors", inspector}, {"calibration-only guard fit and unreliable discrimination", guard_calibration},
    {"deterministic disjoint splits", deterministic_splits}, {"implementation correctness differs from quantization quality", quality_is_separate},
    {"scalar dispatch, reuse, invalidation and safe fallback", [] { cache_and_executor(nm::Backend::Scalar); }},
    {"cache hit still rechecks current guard", same_bucket_rechecks_guard}, {"bounded cache eviction", bounded_cache},
    {"versioned policy roundtrip and rejection", policy_roundtrip},
    {"strict mean and p95 cost boundaries", cost_guard_boundaries},
    {"FIFO hits preserve eviction order", fifo_hit_preserves_eviction_order},
    {"bounded calibration grid and indistinguishable unsafe input", guard_candidate_grid}
  };
  if (nm::mlir_available()) {
    tests.push_back({"MLIR/LLVM kernels vs scalar and independent references", [] { kernels(nm::Backend::Mlir); }});
    tests.push_back({"compiler-generated guard and MLIR executor dispatch", [] { cache_and_executor(nm::Backend::Mlir); }});
    tests.push_back({"generated guard boundary/disabled equivalence", generated_guard_equivalence});
  }
  std::size_t failed = 0;
  for (const auto& test : tests) {
    try { test.second(); std::cout << "PASS " << test.first << '\n'; }
    catch (const std::exception& e) { ++failed; std::cerr << "FAIL " << test.first << ": " << e.what() << '\n'; }
  }
  std::cout << tests.size() - failed << '/' << tests.size() << " tests passed; MLIR compiled=" << nm::mlir_available() << '\n';
  return failed ? 1 : 0;
}
