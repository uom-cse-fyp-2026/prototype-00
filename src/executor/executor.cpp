#include "neurominds/runtime.hpp"
#include "neurominds/kernels.hpp"
#include "nm_config.hpp"
#include <cmath>

namespace nm {
// One Executor owns its mutable cache and counters; use it from one thread at a time.
Executor::Executor(std::size_t capacity) : cache_(capacity) {}
// Final plan-consuming layer. adaptive() normally supplies a freshly inspected Descriptor.
// Direct callers must supply a descriptor for this input: execute() does not rescan it here.
// Passing plan by value lets fallback repair this local copy without altering the caller's copy.
Invocation Executor::execute(const std::vector<float>& x, const PackedWeights& w, const Shape& s, const Policy& policy,
                             Backend backend, Plan plan, const Descriptor& current, bool guard_disabled) {
  // This timer covers execute(); adaptive() later overwrites total_us with the outer-call duration.
  const auto start = Clock::now();
  validate_shape(s);
  // Broken tensors cannot be repaired by precision fallback; reject before dispatch.
  if (x.size() != s.inputs() || w.input != s.input || w.output != s.output || w.group != s.group)
    throw std::invalid_argument("executor tensor/shape mismatch");
  Invocation result;
  // Return the features used for the decision alongside the numerical output for traceability.
  result.descriptor = current;
  const auto check = Clock::now();
  // Reconstruct the key we expect for the current shape, descriptor, policy and backend.
  const auto expected = plan_key(s, current, policy, backend, guard_disabled);
  // Policy loading checks full structure/environment once; these are cheap current-call checks.
  const bool compatible_policy = policy.config_hash == generated::config_hash && policy.shape == s && policy.backend == backend &&
    policy.sampled == current.sampled && policy.id != 0 && std::isfinite(policy.scale) && policy.scale > 0 &&
    std::isfinite(policy.clip) && policy.clip > 0;
  // A legal plan must match the key, approved metadata schema and checksum, and name a real variant.
  // activation_policy=1 denotes the one supported fixed calibration-derived activation policy.
  const bool legal_plan = compatible_policy && plan.key == expected && plan.config_version == generated::config_version &&
    plan.fallback == Variant::Bf16 && plan.activation_policy == 1 && plan.validity_key == validity_key(plan) &&
    (plan.variant == Variant::Bf16 || plan.variant == Variant::A8);
  // Recheck A8 approval and current risk immediately before dispatch. The explicit ablation
  // bypasses only risk; cost selection belongs to adaptive(), not this low-level method.
  if (!legal_plan || (plan.variant == Variant::A8 &&
      (!generated::a8_approved || (!guard_disabled && !risk_accepts(current, policy))))) {
    result.fallback = true;
    if (!legal_plan) ++counters_.invalidations;
    result.reason = legal_plan ? "current input rejected by risk guard" : "invalid/stale/unsupported execution plan";
    // Repair metadata to a current BF16 plan. Both paths still use the same W4 weights.
    plan.key = expected;
    plan.variant = plan.fallback = Variant::Bf16;
    plan.config_version = generated::config_version;
    plan.activation_policy = 1;
    // Recompute the checksum after modifying the plan, otherwise its own validity check would fail.
    plan.validity_key = validity_key(plan);
  }
  // An unavailable conservative implementation is an error; there is no third numerical path.
  if (!generated::bf16_approved) throw std::runtime_error("compiled conservative path is not approved");
  result.validity_us = elapsed_us(check);
  result.plan = plan;
  const auto dispatch = Clock::now();
  // Invocation boundary includes conversion, allocation, numerical kernel and output checks.
  // run_kernel validates all activations/weights, converts storage, dispatches, and rejects overflow.
  // It does not compare this output to a reference or retry BF16 after a quality-budget failure.
  run_kernel(x, w, s, plan.variant, policy.scale, backend, result.output, result.preparation_us, result.kernel_us);
  const double execution_us = elapsed_us(dispatch);
  // Residual includes work outside preparation/kernel timers, including output checks and cleanup.
  // Clamp tiny negative residuals to zero rather than reporting negative time.
  result.dispatch_us = std::max(0.0, execution_us - result.preparation_us - result.kernel_us);
  if (result.fallback) ++counters_.fallbacks;
  result.total_us = elapsed_us(start);
  return result;
}
// Public adaptive path: inspect -> cache lookup -> fresh selection -> guarded execution.
// Selection uses only current input features and the frozen policy, never reference-output error.
Invocation Executor::adaptive(const std::vector<float>& x, const PackedWeights& w, const Shape& s, const Policy& policy,
                              Backend backend, bool sampled, bool guard_disabled) {
  const auto invocation = Clock::now();
  // The policy is structurally validated once at load/calibration. Matching below is cheap.
  const auto inspection = Clock::now();
  // Inspect every call, including cache hits; a coarse bucket cannot certify current risk.
  auto d = inspect(x, s, policy.clip, sampled);
  const double inspector_us = elapsed_us(inspection);
  const auto checking = Clock::now();
  // Remove obsolete-policy/configuration entries before looking up a candidate reusable plan.
  cache_.invalidate(generated::config_hash, policy.id, counters_);
  const auto key = plan_key(s, d, policy, backend, guard_disabled);
  Plan p;
  // hit means a valid candidate key was found, not that its old variant will necessarily execute.
  const bool hit = cache_.find(key, p, counters_);
  const double cache_us = elapsed_us(checking);
  const auto decision = Clock::now();
  // The policy must describe this exact shape/backend and the effective full/sampled mode.
  const bool supported = policy.shape == s && policy.backend == backend && policy.config_hash == generated::config_hash &&
    policy.sampled == d.sampled && policy.id != 0 && generated::a8_approved;
  // Guard-disabled is an evaluation ablation, not a command to force A8 execution.
  const bool risk_ok = guard_disabled || risk_accepts(d, policy);
  // Use calibrated complete execution costs plus this call's inspection overhead estimate.
  const bool faster = cost_accepts(d, policy, inspector_us);
  // A8 needs all three permissions; risk acceptance alone is insufficient.
  const auto selected = supported && risk_ok && faster ? Variant::A8 : Variant::Bf16;
  // Two inputs can share a coarse key yet cross an exact threshold: count the old choice as invalid.
  if (hit && p.variant != selected) ++counters_.invalidations;
  if (!hit || p.variant != selected) {
    // Aggregate order: key, selected variant, fallback, config version, activation-policy ID, checksum.
    p = {key, selected, Variant::Bf16, generated::config_version, 1, 0};
    p.validity_key = validity_key(p);
    cache_.put(p, counters_);
  }
  const double decision_us = elapsed_us(decision);
  // Even a cached/reselected plan must pass the final validity and current-risk checks.
  auto result = execute(x, w, s, policy, backend, p, d, guard_disabled);
  result.cache_hit = hit;
  result.inspector_us = inspector_us;
  // Report cache time together with execute()'s final validity-check time.
  result.validity_us += cache_us;
  result.decision_us = decision_us;
  // Preserve a final-dispatch failure reason; otherwise explain why adaptive selection chose its path.
  if (result.reason.empty()) {
    if (!supported) result.reason = "policy shape/configuration/backend/inspection mode mismatch";
    else if (!risk_ok) result.reason = "calibrated risk guard rejected A8";
    else if (!faster) result.reason = "no measured total-cost advantage for A8";
    else result.reason = guard_disabled ? "GUARD-DISABLED ABLATION: measured A8 cost advantage" : "risk guard accepted; measured A8 total-cost advantage";
  }
  // Count unsupported/risk-rejected BF16 choices as fallbacks, but avoid counting a fallback twice.
  // A BF16 choice made only because A8 lacked a cost advantage normally has fallback=false.
  const bool already_counted = result.fallback;
  result.fallback = result.fallback || !supported || !risk_ok;
  if (result.fallback && !already_counted) ++counters_.fallbacks;
  // Primary latency includes inspection, cache, decision, preparation, kernel and remaining overhead.
  result.total_us = elapsed_us(invocation);
  return result;
}
}
