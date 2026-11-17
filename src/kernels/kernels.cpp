#include "neurominds/kernels.hpp"
#include <cmath>

namespace nm {
namespace {
void validate(const std::vector<float>& x, const PackedWeights& w, const Shape& s) {
  validate_activations(x, s);
  w.validate();
  if (w.input != s.input || w.output != s.output || w.group != s.group) throw std::invalid_argument("packed weight/shape mismatch");
}
template<typename T> struct MemRef1D {
  T* allocated;
  T* aligned;
  std::int64_t offset;
  std::int64_t sizes[1];
  std::int64_t strides[1];
};
template<typename T> MemRef1D<T> descriptor(const std::vector<T>& values) {
  // MLIR's C ABI has no const qualifier; input descriptors are used only by loads.
  auto* ptr = const_cast<T*>(values.data());
  return {ptr, ptr, 0, {static_cast<std::int64_t>(values.size())}, {1}};
}
#if NM_HAS_MLIR
extern "C" void _mlir_ciface_nm_bf16(MemRef1D<std::uint16_t>*, MemRef1D<std::uint8_t>*, MemRef1D<float>*,
                                     MemRef1D<float>*, std::int64_t, std::int64_t, std::int64_t, std::int64_t);
extern "C" void _mlir_ciface_nm_a8(MemRef1D<std::int8_t>*, MemRef1D<std::uint8_t>*, MemRef1D<float>*,
                                   MemRef1D<float>*, std::int64_t, std::int64_t, std::int64_t, std::int64_t, float);
#endif
}
bool mlir_available() { return NM_HAS_MLIR != 0; }
std::vector<float> fp32_reference(const std::vector<float>& x, const std::vector<float>& w, const Shape& s) {
  validate_activations(x, s);
  if (w.size() != product(s.input, s.output)) throw std::invalid_argument("FP32 weight shape mismatch");
  for (float value : w) if (!std::isfinite(value)) throw std::invalid_argument("nonfinite FP32 weight");
  std::vector<float> y(s.outputs());
  for (std::size_t m = 0; m < s.batch; ++m)
    for (std::size_t n = 0; n < s.output; ++n) {
      float acc = 0;
      for (std::size_t k = 0; k < s.input; ++k) acc += x[m * s.input + k] * w[n * s.input + k];
      y[m * s.output + n] = acc;
    }
  return y;
}
std::vector<float> w4_reference(const std::vector<float>& x, const PackedWeights& w, const Shape& s) {
  validate(x, w, s);
  std::vector<float> y(s.outputs());
  for (std::size_t m = 0; m < s.batch; ++m)
    for (std::size_t n = 0; n < s.output; ++n) {
      double acc = 0;
      for (std::size_t k = 0; k < s.input; ++k)
        acc += static_cast<double>(x[m * s.input + k]) * weight_value(w, n, k) * weight_scale(w, n, k);
      y[m * s.output + n] = static_cast<float>(acc);
    }
  return y;
}
std::vector<float> variant_reference(const std::vector<float>& x, const PackedWeights& w, const Shape& s, Variant v, float scale) {
  validate(x, w, s);
  if (v != Variant::Bf16 && v != Variant::A8) throw std::invalid_argument("invalid reference variant");
  std::vector<float> y(s.outputs());
  for (std::size_t m = 0; m < s.batch; ++m)
    for (std::size_t n = 0; n < s.output; ++n) {
      double acc = 0;
      for (std::size_t k = 0; k < s.input; ++k) {
        const float value = x[m * s.input + k];
        const double activation = v == Variant::Bf16 ? from_bf16(to_bf16(value)) : static_cast<double>(to_a8(value, scale)) * scale;
        acc += activation * weight_value(w, n, k) * weight_scale(w, n, k);
      }
      y[m * s.output + n] = static_cast<float>(acc);
    }
  return y;
}
void bf16_scalar(const std::vector<std::uint16_t>& x, const PackedWeights& w, const Shape& s, std::vector<float>& y) {
  if (x.size() != s.inputs() || y.size() != s.outputs()) throw std::invalid_argument("BF16 buffer size mismatch");
  for (std::size_t m = 0; m < s.batch; ++m)
    for (std::size_t n = 0; n < s.output; ++n) {
      float acc = 0;
      for (std::size_t k = 0; k < s.input; ++k) {
        const float dequantized = weight_value(w, n, k) * weight_scale(w, n, k);
        acc += from_bf16(x[m * s.input + k]) * dequantized;
      }
      y[m * s.output + n] = acc;
    }
}
void a8_scalar(const std::vector<std::int8_t>& x, const PackedWeights& w, const Shape& s, float scale, std::vector<float>& y) {
  if (x.size() != s.inputs() || y.size() != s.outputs()) throw std::invalid_argument("A8 buffer size mismatch");
  for (std::size_t m = 0; m < s.batch; ++m)
    for (std::size_t n = 0; n < s.output; ++n) {
      float acc = 0;
      for (std::size_t g = 0; g < w.groups(); ++g) {
        std::int32_t dot = 0;
        const auto end = std::min((g + 1) * s.group, s.input);
        for (auto k = g * s.group; k < end; ++k) dot += static_cast<std::int32_t>(x[m * s.input + k]) * weight_value(w, n, k);
        const float scaled = static_cast<float>(dot) * scale;
        acc += scaled * w.scales[n * w.groups() + g];
      }
      y[m * s.output + n] = acc;
    }
}
void run_kernel(const std::vector<float>& x, const PackedWeights& w, const Shape& s, Variant v, float scale, Backend backend,
                std::vector<float>& y, double& preparation_us, double& kernel_us) {
  const auto prep = Clock::now();
  validate(x, w, s);
  if (backend != Backend::Scalar && backend != Backend::Mlir) throw std::invalid_argument("unsupported execution backend");
  if (backend == Backend::Mlir && !mlir_available()) throw std::runtime_error("MLIR backend was not built; select scalar explicitly");
  y.resize(s.outputs());
  if (v == Variant::Bf16) {
    std::vector<std::uint16_t> stored(x.size());
    std::transform(x.begin(), x.end(), stored.begin(), to_bf16);
    preparation_us = elapsed_us(prep);
    const auto start = Clock::now();
    if (backend == Backend::Scalar) bf16_scalar(stored, w, s, y);
    else {
#if NM_HAS_MLIR
      auto xd = descriptor(stored); auto wd = descriptor(w.data); auto sd = descriptor(w.scales); auto yd = descriptor(y);
      _mlir_ciface_nm_bf16(&xd, &wd, &sd, &yd, s.batch, s.output, s.input, s.group);
#endif
    }
    kernel_us = elapsed_us(start);
  } else if (v == Variant::A8) {
    if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("invalid A8 scale");
    std::vector<std::int8_t> stored(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) stored[i] = to_a8(x[i], scale);
    preparation_us = elapsed_us(prep);
    const auto start = Clock::now();
    if (backend == Backend::Scalar) a8_scalar(stored, w, s, scale, y);
    else {
#if NM_HAS_MLIR
      auto xd = descriptor(stored); auto wd = descriptor(w.data); auto sd = descriptor(w.scales); auto yd = descriptor(y);
      _mlir_ciface_nm_a8(&xd, &wd, &sd, &yd, s.batch, s.output, s.input, s.group, scale);
#endif
    }
    kernel_us = elapsed_us(start);
  } else throw std::invalid_argument("unsupported executor variant");
  for (float value : y) if (!std::isfinite(value)) throw std::overflow_error("projection output overflowed FP32");
}
}
