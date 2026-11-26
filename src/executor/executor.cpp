#include "neurominds/runtime.hpp"
#include "neurominds/kernels.hpp"
#include "nm_config.hpp"
#include <cmath>

namespace nm {
namespace {

// Both new choices and repaired fallbacks use the same metadata construction rules.
Plan make_plan(const PlanKey& key, Variant variant) {
  Plan plan;
  plan.key = key;
  plan.variant = variant;
  plan.fallback = Variant::Bf16;
  plan.config_version = generated::config_version;
  plan.activation_policy = 1;  // The one supported fixed calibration-derived scale policy.
  plan.validity_key = validity_key(plan);  // Compute last, after all other fields are set.
  return plan;
}

}  // namespace

// One Executor owns mutable cache/counters; use it from one thread at a time.
Executor::Executor(std::size_t capacity) : cache_(capacity) {}

// Final dispatch layer. Direct callers must provide a fresh descriptor for this input.
// The plan is copied so fallback can repair it locally without changing the caller's copy.
Invocation Executor::execute(const std::vector<float>& input, const PackedWeights& weights,
                             const Shape& shape, const Policy& policy, Backend backend,
                             Plan plan, const Descriptor& current, bool guard_disabled) {
  const auto execution_start = Clock::now();

  // 1. Reject broken tensor contracts: changing precision cannot repair a shape mismatch.
  validate_shape(shape);
  if (input.size() != shape.inputs() || weights.input != shape.input ||
      weights.output != shape.output || weights.group != shape.group) {
    throw std::invalid_argument("executor tensor/shape mismatch");
  }

  Invocation result;
  result.descriptor = current;

  // 2. Check the current policy context and every plan validity field.
  const auto validity_start = Clock::now();
  const auto expected_key = plan_key(shape, current, policy, backend, guard_disabled);

  // Full policy structure/environment validation happens once at calibration/load.
  const bool compatible_policy =
      policy.config_hash == generated::config_hash &&
      policy.shape == shape &&
      policy.backend == backend &&
      policy.sampled == current.sampled &&
      policy.id != 0 &&
      std::isfinite(policy.scale) && policy.scale > 0 &&
      std::isfinite(policy.clip) && policy.clip > 0 && policy.scale == policy.clip / 127;

  const bool legal_plan =
      compatible_policy &&
      plan.key == expected_key &&
      plan.config_version == generated::config_version &&
      plan.fallback == Variant::Bf16 &&
      plan.activation_policy == 1 &&
      plan.validity_key == validity_key(plan) &&
      (plan.variant == Variant::Bf16 || plan.variant == Variant::A8);

  // 3. Recheck A8 approval/current risk immediately before dispatch, even for cached plans.
  // Cost selection belongs to adaptive(); the explicit ablation bypasses only this risk rule.
  bool a8_requires_fallback = false;
  if (legal_plan && plan.variant == Variant::A8) {
    a8_requires_fallback = !generated::a8_approved ||
        (!guard_disabled && !risk_accepts(current, policy));
  }

  if (!legal_plan || a8_requires_fallback) {
    result.fallback = true;
    if (!legal_plan) {
      ++counters_.invalidations;
      result.reason = "invalid/stale/unsupported execution plan";
    } else {
      result.reason = "current input rejected by risk guard";
    }
    plan = make_plan(expected_key, Variant::Bf16);
  }

  if (!generated::bf16_approved) {
    throw std::runtime_error("compiled conservative path is not approved");
  }
  result.validity_us = elapsed_us(validity_start);
  result.plan = plan;

  // 4. Validate all activations/weights, convert storage, run the kernel and check finite output.
  // BF16 remains a W4-weight path. Neither variant compares output error to a reference here.
  const auto dispatch_start = Clock::now();
  run_kernel(input, weights, shape, plan.variant, policy.scale, backend,
             result.output, result.preparation_us, result.kernel_us);

  const double dispatch_total_us = elapsed_us(dispatch_start);
  result.dispatch_us = std::max(
      0.0, dispatch_total_us - result.preparation_us - result.kernel_us);
  // The residual includes output checks/cleanup outside the preparation and kernel timers.

  if (result.fallback) {
    ++counters_.fallbacks;
  }
  result.total_us = elapsed_us(execution_start);
  return result;
}

// Main runtime flow: inspect -> lookup -> decide -> update plan -> validate/execute -> report.
// Selection sees input features and frozen policy values, never reference-output error.
Invocation Executor::adaptive(const std::vector<float>& input, const PackedWeights& weights,
                              const Shape& shape, const Policy& policy, Backend backend,
                              bool sampled, bool guard_disabled) {
  const auto invocation_start = Clock::now();

  // Skip inspection and cache lookup only when A8 cannot win even with zero inspection cost.
  validate_shape(shape);
  const bool effective_sampled = sampled && shape.inputs() > generated::sample_size;
  const bool compatible = policy.shape == shape && policy.backend == backend &&
      policy.config_hash == generated::config_hash && policy.sampled == effective_sampled &&
      policy.id != 0 && generated::a8_approved && std::isfinite(policy.clip) && policy.clip > 0 &&
      std::isfinite(policy.scale) && policy.scale == policy.clip / 127;
  if (compatible && (policy.guard_enabled || guard_disabled) && !cost_accepts(Descriptor{}, policy, 0)) {
    Descriptor skipped;
    skipped.sampled = effective_sampled;
    auto plan = make_plan(plan_key(shape, skipped, policy, backend, guard_disabled), Variant::Bf16);
    auto result = execute(input, weights, shape, policy, backend, plan, skipped, guard_disabled);
    if (result.reason.empty()) result.reason = "cost precheck rejected A8; risk not evaluated";
    result.total_us = elapsed_us(invocation_start);
    return result;
  }
  // 1. Inspect every potentially profitable invocation. A coarse cache bucket cannot certify current risk.
  const auto inspection_start = Clock::now();
  const auto descriptor = inspect(input, shape, policy.clip, sampled);
  const double inspector_us = elapsed_us(inspection_start);

  // 2. Invalidate old config/policy entries and look up a candidate plan.
  const auto cache_start = Clock::now();
  cache_.invalidate(generated::config_hash, policy.id, counters_);
  const auto key = plan_key(shape, descriptor, policy, backend, guard_disabled);
  Plan plan;
  const bool cache_hit = cache_.find(key, plan, counters_);
  const double cache_us = elapsed_us(cache_start);

  // 3. Make a fresh decision even on a hit: A8 needs support, low predicted risk AND cost benefit.
  const auto decision_start = Clock::now();
  const bool supported =
      policy.shape == shape &&
      policy.backend == backend &&
      policy.config_hash == generated::config_hash &&
      policy.sampled == descriptor.sampled &&
      policy.id != 0 &&
      generated::a8_approved;

  const bool risk_ok = guard_disabled || risk_accepts(descriptor, policy);
  const bool faster = cost_accepts(descriptor, policy, inspector_us);

  Variant selected_variant = Variant::Bf16;
  if (supported && risk_ok && faster) {
    selected_variant = Variant::A8;
  }

  // 4. Reuse the old plan only while its chosen variant still agrees with the current decision.
  const bool cached_choice_changed = cache_hit && plan.variant != selected_variant;
  if (cached_choice_changed) {
    ++counters_.invalidations;
  }
  if (!cache_hit || cached_choice_changed) {
    plan = make_plan(key, selected_variant);
    cache_.put(plan, counters_);
  }
  const double decision_us = elapsed_us(decision_start);

  // 5. Final execution performs its own validity/risk checks before numerical dispatch.
  auto result = execute(input, weights, shape, policy, backend, plan, descriptor, guard_disabled);
  result.cache_hit = cache_hit;  // A candidate hit may still lead to a changed choice or fallback.
  result.inspector_us = inspector_us;
  result.validity_us += cache_us;
  result.decision_us = decision_us;

  // 6. Preserve a final-dispatch error reason; otherwise explain the adaptive decision.
  if (result.reason.empty()) {
    if (!supported) {
      result.reason = "policy shape/configuration/backend/inspection mode mismatch";
    } else if (!risk_ok) {
      result.reason = "calibrated risk guard rejected A8";
    } else if (!faster) {
      result.reason = "no measured total-cost advantage for A8";
    } else if (guard_disabled) {
      result.reason = "GUARD-DISABLED ABLATION: measured A8 cost advantage";
    } else {
      result.reason = "risk guard accepted; measured A8 total-cost advantage";
    }
  }

  // Risk/support rejection counts as fallback; choosing BF16 only for cost normally does not.
  // execute() may already have counted a fallback, so never count that invocation twice.
  const bool selection_requires_fallback = !supported || !risk_ok;
  if (selection_requires_fallback && !result.fallback) {
    result.fallback = true;
    ++counters_.fallbacks;
  }

  // Replace execute()'s inner duration with the whole adaptive invocation, including inspection.
  result.total_us = elapsed_us(invocation_start);
  return result;
}

}  // namespace nm
