#pragma once
#include "types.hpp"
namespace nm {
std::uint8_t pack_nibbles(int low, int high);
int unpack_nibble(std::uint8_t byte, bool high);
std::uint16_t to_bf16(float value);
float from_bf16(std::uint16_t bits);
std::int8_t to_a8(float value, float scale);
PackedWeights quantize_weights(const std::vector<float>& weights, const Shape& shape);
int weight_value(const PackedWeights&, std::size_t row, std::size_t column);
float weight_scale(const PackedWeights&, std::size_t row, std::size_t column);
std::vector<float> dequantize_weights(const PackedWeights&);
void validate_shape(const Shape&);
void validate_activations(const std::vector<float>&, const Shape&);
double relative_l2(const std::vector<float>& actual, const std::vector<float>& reference);
}
