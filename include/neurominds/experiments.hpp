#pragma once
#include "runtime.hpp"
namespace nm {
struct Options {
  Shape shape;
  std::uint64_t seed = 4533;
  std::size_t samples = 64, warmup = 3, repeats = 30;
  Backend backend = Backend::Mlir;
  bool sampled = false;
  std::string output = "results", policy = "results/policy.nm", regime = "normal", variant = "adaptive";
};
struct CalibrationRow {
  Descriptor descriptor;
  double a8_error = 0, bf16_error = 0;
};
std::vector<float> synthetic_input(const Shape&, const std::string& regime, std::uint64_t seed);
std::vector<float> synthetic_weights(const Shape&, std::uint64_t seed);
std::uint64_t split_seed(std::uint64_t base, bool held_out, std::size_t index);
void fit_guard(Policy&, const std::vector<CalibrationRow>&);
Policy calibrate(const Options&);
void demonstrate(const Options&);
void benchmark(const Options&);
void projection(const Options&);
void print_environment();
void print_metadata();
std::string environment_json();
std::string json_quote(const std::string&);
std::size_t resident_bytes();
std::size_t peak_resident_bytes();
void write_manifest(const Options&, const Policy*, const std::string& kind, const std::string& path);
}
