#pragma once
#include "quantization.hpp"
namespace nm {
std::vector<float> fp32_reference(const std::vector<float>& x, const std::vector<float>& weights, const Shape&);
std::vector<float> w4_reference(const std::vector<float>& x, const PackedWeights&, const Shape&);
// Independent FP64 dequantized references for implementation checks, not policy selection.
std::vector<float> variant_reference(const std::vector<float>& x, const PackedWeights&, const Shape&, Variant, float scale);
void bf16_scalar(const std::vector<std::uint16_t>&, const PackedWeights&, const Shape&, std::vector<float>&);
void a8_scalar(const std::vector<std::int8_t>&, const PackedWeights&, const Shape&, float scale, std::vector<float>&);
void run_kernel(const std::vector<float>& x, const PackedWeights&, const Shape&, Variant, float scale, Backend,
                std::vector<float>& output, double& preparation_us, double& kernel_us);
bool mlir_available();
}
