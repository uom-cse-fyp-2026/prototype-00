#include "neurominds/runtime.hpp"
#include "neurominds/kernels.hpp"
#include "nm_config.hpp"
#include <cmath>

namespace nm {
Executor::Executor(std::size_t capacity) : cache_(capacity) {}
Invocation Executor::execute(const std::vector<float>& x, const PackedWeights& w, const Shape& s, const Policy& policy,
                             Backend backend, Plan plan, const Descriptor& current, bool guard_disabled) {
  const auto start = Clock::now();
  validate_shape(s);
  // Broken tensors cannot be repaired by precision fallback; reject before dispatch.
  if (x.size() != s.inputs() || w.input != s.input || w.output != s.output || w.group != s.group)
    throw std::invalid_argument("executor tensor/shape mismatch");
  Invocation result;
  result.descriptor = current;
  const auto check = Clock::now();
  const auto expected = plan_key(s, current, policy, backend, guard_disabled);
  const bool compatible_policy = policy.config_hash == generated::config_hash && policy.shape == s && policy.backend == backend &&
    policy.sampled == current.sampled && policy.id != 0 && std::isfinite(policy.scale) && policy.scale > 0 &&
    std::isfinite(policy.clip) && policy.clip > 0;
  const bool legal_plan = compatible_policy && plan.key == expected && plan.config_version == generated::config_version &&
    plan.fallback == Variant::Bf16 && plan.activation_policy == 1 && plan.validity_key == validity_key(plan) &&
    (plan.variant == Variant::Bf16 || plan.variant == Variant::A8);
  if (!legal_plan || (plan.variant == Variant::A8 &&
      (!generated::a8_approved || (!guard_disabled && !risk_accepts(current, policy))))) {
    result.fallback = true;
    if (!legal_plan) ++counters_.invalidations;
    result.reason = legal_plan ? "current input rejected by risk guard" : "invalid/stale/unsupported execution plan";
    plan.key = expected;
    plan.variant = plan.fallback = Variant::Bf16;
    plan.config_version = generated::config_version;
    plan.activation_policy = 1;
    plan.validity_key = validity_key(plan);
  }
  if (!generated::bf16_approved) throw std::runtime_error("compiled conservative path is not approved");
  result.validity_us = elapsed_us(check);
  result.plan = plan;
  const auto dispatch = Clock::now();
  // Invocation boundary includes conversion, allocation, numerical kernel and output checks.
  run_kernel(x, w, s, plan.variant, policy.scale, backend, result.output, result.preparation_us, result.kernel_us);
  const double execution_us = elapsed_us(dispatch);
  result.dispatch_us = std::max(0.0, execution_us - result.preparation_us - result.kernel_us);
  if (result.fallback) ++counters_.fallbacks;
  result.total_us = elapsed_us(start);
  return result;
}
Invocation Executor::adaptive(const std::vector<float>& x, const PackedWeights& w, const Shape& s, const Policy& policy,
                              Backend backend, bool sampled, bool guard_disabled) {
  const auto invocation = Clock::now();
  // The policy is structurally validated once at load/calibration. Matching below is cheap.
  const auto inspection = Clock::now();
  auto d = inspect(x, s, policy.clip, sampled);
  const double inspector_us = elapsed_us(inspection);
  const auto checking = Clock::now();
  cache_.invalidate(generated::config_hash, policy.id, counters_);
  const auto key = plan_key(s, d, policy, backend, guard_disabled);
  Plan p;
  const bool hit = cache_.find(key, p, counters_);
  const double cache_us = elapsed_us(checking);
  const auto decision = Clock::now();
  const bool supported = policy.shape == s && policy.backend == backend && policy.config_hash == generated::config_hash &&
    policy.sampled == d.sampled && policy.id != 0 && generated::a8_approved;
  const bool risk_ok = guard_disabled || risk_accepts(d, policy);
  const bool faster = cost_accepts(d, policy, inspector_us);
  const auto selected = supported && risk_ok && faster ? Variant::A8 : Variant::Bf16;
  if (hit && p.variant != selected) ++counters_.invalidations;
  if (!hit || p.variant != selected) {
    p = {key, selected, Variant::Bf16, generated::config_version, 1, 0};
    p.validity_key = validity_key(p);
    cache_.put(p, counters_);
  }
  const double decision_us = elapsed_us(decision);
  auto result = execute(x, w, s, policy, backend, p, d, guard_disabled);
  result.cache_hit = hit;
  result.inspector_us = inspector_us;
  result.validity_us += cache_us;
  result.decision_us = decision_us;
  if (result.reason.empty()) {
    if (!supported) result.reason = "policy shape/configuration/backend/inspection mode mismatch";
    else if (!risk_ok) result.reason = "calibrated risk guard rejected A8";
    else if (!faster) result.reason = "no measured total-cost advantage for A8";
    else result.reason = guard_disabled ? "GUARD-DISABLED ABLATION: measured A8 cost advantage" : "risk guard accepted; measured A8 total-cost advantage";
  }
  const bool already_counted = result.fallback;
  result.fallback = result.fallback || !supported || !risk_ok;
  if (result.fallback && !already_counted) ++counters_.fallbacks;
  result.total_us = elapsed_us(invocation);
  return result;
}
}
