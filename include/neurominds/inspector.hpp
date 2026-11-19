#pragma once
#include "types.hpp"
namespace nm {
// Build current activation statistics. Sampling, when requested, is deterministic and bounded.
// Full kernel preparation later validates every activation even if inspection observed a subset.
Descriptor inspect(const std::vector<float>& input, const Shape&, float clip, bool sampled = false);
// Empirical two-feature risk predicate; no numerical reference/output error is available here.
bool risk_accepts(const Descriptor&, const Policy&);
// Require calibrated mean/p95 A8 cost plus overhead to beat BF16; Descriptor is currently unused.
bool cost_accepts(const Descriptor&, const Policy&, double measured_inspector_us);
// Group reusable plan candidates by shape, coarse features, configuration, policy, backend and mode.
PlanKey plan_key(const Shape&, const Descriptor&, const Policy&, Backend, bool guard_disabled = false);
// Internal consistency checksum over plan fields; not a hash of tensors or a security guarantee.
std::uint64_t validity_key(const Plan&);
}
