#include "neurominds/quantization.hpp"
#include "nm_config.hpp"
#include <cmath>
#include <cstring>

namespace nm {
void validate_shape(const Shape& s) {
  if (!s.batch || !s.input || !s.output || !s.group) throw std::invalid_argument("dimensions and group must be positive");
  if (s.batch > generated::max_batch || s.input > generated::max_input ||
      s.output > generated::max_output || s.group > generated::max_group)
    throw std::invalid_argument("shape exceeds compiled configuration limits");
  // Worst-case unpacked signed nibble is -8, activation magnitude is at most 127.
  if (std::min(s.group, s.input) > static_cast<std::size_t>(INT32_MAX / (127 * 8)))
    throw std::invalid_argument("INT32 per-group accumulator could overflow");
  const auto weights = product(s.input, s.output);
  const auto elements = s.inputs() + s.outputs();
  // Conservative working-set admission: original/dequantized weights, output/reference,
  // activation copies, packed storage, and group scales. RSS is measured separately.
  const long double estimate = static_cast<long double>(weights) * 12 +
    static_cast<long double>(elements) * 24 +
    static_cast<long double>(ceil_div(s.input, s.group)) * s.output * 4;
  if (estimate > generated::memory_budget_bytes) throw std::invalid_argument("shape exceeds configured operator working-set budget");
}
void PackedWeights::validate() const {
  validate_shape(Shape{1, input, output, group});
  if (data.size() != product(output, ceil_div(input, 2)) || scales.size() != product(output, groups()))
    throw std::invalid_argument("malformed packed W4 buffers");
  for (float scale : scales) if (!std::isfinite(scale) || scale <= 0) throw std::invalid_argument("invalid W4 scale");
}
std::uint8_t pack_nibbles(int low, int high) {
  if (low < -8 || low > 7 || high < -8 || high > 7) throw std::invalid_argument("signed INT4 outside [-8,7]");
  return static_cast<std::uint8_t>((low & 15) | ((high & 15) << 4));
}
int unpack_nibble(std::uint8_t byte, bool high) {
  const int nibble = high ? byte >> 4 : byte & 15;
  return nibble >= 8 ? nibble - 16 : nibble;
}
std::uint16_t to_bf16(float value) {
  static_assert(sizeof(float) == sizeof(std::uint32_t), "IEEE binary32 required");
  static_assert(std::numeric_limits<float>::is_iec559, "IEEE binary32 required");
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7f800000U) == 0x7f800000U) {
    // Preserve infinity; keep every NaN a NaN, including payloads only in low bits.
    return static_cast<std::uint16_t>((bits >> 16) | ((bits & 0x007fffffU) ? 0x0040U : 0U));
  }
  bits += 0x7fffU + ((bits >> 16) & 1U);  // round-to-nearest, ties-to-even
  return static_cast<std::uint16_t>(bits >> 16);
}
float from_bf16(std::uint16_t value) {
  std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
std::int8_t to_a8(float value, float scale) {
  if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0) throw std::invalid_argument("A8 requires finite value and positive scale");
  const double normalized = std::clamp(static_cast<double>(value) / scale, -127.0, 127.0);
  return static_cast<std::int8_t>(std::round(normalized)); // ties away from zero
}
PackedWeights quantize_weights(const std::vector<float>& weights, const Shape& s) {
  validate_shape(s);
  if (weights.size() != product(s.output, s.input)) throw std::invalid_argument("weight shape mismatch");
  PackedWeights p{s.input, s.output, s.group, {}, {}};
  const auto stride = ceil_div(s.input, 2), groups = p.groups();
  p.data.assign(product(s.output, stride), 0);
  p.scales.resize(product(s.output, groups));
  for (std::size_t n = 0; n < s.output; ++n) {
    for (std::size_t g = 0; g < groups; ++g) {
      const auto begin = g * s.group, end = std::min(begin + s.group, s.input);
      double maximum = 0;
      for (auto k = begin; k < end; ++k) {
        const float value = weights[n * s.input + k];
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite weight");
        maximum = std::max(maximum, std::abs(static_cast<double>(value)));
      }
      const float scale = maximum == 0 ? 1.0f : std::max(static_cast<float>(maximum / 7), std::numeric_limits<float>::denorm_min());
      p.scales[n * groups + g] = scale;
      for (auto k = begin; k < end; ++k) {
        const int q = static_cast<int>(std::clamp(std::round(static_cast<double>(weights[n * s.input + k]) / scale), -7.0, 7.0));
        auto& byte = p.data[n * stride + k / 2];
        if (k % 2) byte = static_cast<std::uint8_t>((byte & 15) | ((q & 15) << 4));
        else byte = static_cast<std::uint8_t>((byte & 240) | (q & 15));
      }
    }
  }
  p.validate();
  return p;
}
int weight_value(const PackedWeights& p, std::size_t n, std::size_t k) {
  if (n >= p.output || k >= p.input) throw std::out_of_range("W4 index");
  return unpack_nibble(p.data.at(n * ceil_div(p.input, 2) + k / 2), k % 2 != 0);
}
float weight_scale(const PackedWeights& p, std::size_t n, std::size_t k) {
  if (n >= p.output || k >= p.input) throw std::out_of_range("scale index");
  return p.scales.at(n * p.groups() + k / p.group);
}
std::vector<float> dequantize_weights(const PackedWeights& p) {
  p.validate();
  std::vector<float> w(product(p.input, p.output));
  for (std::size_t n = 0; n < p.output; ++n)
    for (std::size_t k = 0; k < p.input; ++k) w[n * p.input + k] = weight_value(p, n, k) * weight_scale(p, n, k);
  return w;
}
void validate_activations(const std::vector<float>& x, const Shape& s) {
  validate_shape(s);
  if (x.size() != s.inputs()) throw std::invalid_argument("activation shape mismatch");
  for (float v : x)
    if (!std::isfinite(v) || std::abs(static_cast<double>(v)) > 3.3895313892515355e38)
      throw std::invalid_argument("activation outside finite BF16 storage domain");
}
double relative_l2(const std::vector<float>& actual, const std::vector<float>& reference) {
  if (actual.size() != reference.size() || actual.empty()) throw std::invalid_argument("error metric shape mismatch");
  long double difference = 0, magnitude = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) return std::numeric_limits<double>::infinity();
    const long double d = static_cast<long double>(actual[i]) - reference[i];
    difference += d * d;
    magnitude += static_cast<long double>(reference[i]) * reference[i];
  }
  return static_cast<double>(std::sqrt(difference) / std::max(std::sqrt(magnitude), static_cast<long double>(generated::l2_epsilon)));
}
}
