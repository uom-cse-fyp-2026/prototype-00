#pragma once
#include "types.hpp"
namespace nm {
Descriptor inspect(const std::vector<float>& input, const Shape&, float clip, bool sampled = false);
bool risk_accepts(const Descriptor&, const Policy&);
bool cost_accepts(const Descriptor&, const Policy&, double measured_inspector_us);
PlanKey plan_key(const Shape&, const Descriptor&, const Policy&, Backend, bool guard_disabled = false);
std::uint64_t validity_key(const Plan&);
}
