#pragma once
#include "types.hpp"

namespace nm {

// Compute current features using a full scan or deterministic bounded sampling.
// Kernel preparation later validates every activation even if inspection sampled only a subset.
Descriptor inspect(const std::vector<float>& input, const Shape& shape, float clip,
                   bool sampled = false);

// Calibrated maximum/clipping/reconstruction predicate; empirical prediction, not quality certification.
bool risk_accepts(const Descriptor& descriptor, const Policy& policy);

// Require mean/p95 cost benefit including overhead. Descriptor is presently unused by the model.
bool cost_accepts(const Descriptor& descriptor, const Policy& policy, double measured_inspector_us);

// Build the shape/features/config/policy/backend/mode identity used for plan reuse.
PlanKey plan_key(const Shape& shape, const Descriptor& descriptor, const Policy& policy,
                 Backend backend, bool guard_disabled = false);

// Checksum plan metadata, excluding the checksum itself. Tensor contents are not hashed.
std::uint64_t validity_key(const Plan& plan);

}  // namespace nm
