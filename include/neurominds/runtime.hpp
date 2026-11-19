#pragma once
#include "inspector.hpp"
#include <deque>
namespace nm {
// Bounded FIFO cache of plans. Reusing a plan never skips adaptive()'s current inspection/selection.
class PlanCache {
 public:
  // Capacity must lie in [1,4096]; Executor's normal default is 16 entries.
  explicit PlanCache(std::size_t capacity);
  // Return true and fill the output Plan on a matching key with a consistent checksum.
  bool find(const PlanKey&, Plan&, Counters&);
  // Update a matching entry or append a new one, evicting the oldest insertion if necessary.
  void put(const Plan&, Counters&);
  // Remove plans belonging to a different compiled configuration or policy.
  void invalidate(std::uint64_t config, std::uint64_t policy, Counters&);
  std::size_t size() const { return entries_.size(); }
 private:
  std::size_t capacity_;
  // Front is oldest, back is newest; lookups do not reorder entries as an LRU cache would.
  std::deque<Plan> entries_;
};
// Stateful runtime for one thread. Policy/weights must remain fixed for their calibration contract.
class Executor {
 public:
  explicit Executor(std::size_t capacity = 16);
  // Normal entry point: inspect input, recheck support/risk/cost, select/reuse plan, then execute.
  // guard_disabled is an evaluation-only risk ablation; it does not bypass cost/tensor checks.
  Invocation adaptive(const std::vector<float>&, const PackedWeights&, const Shape&, const Policy&, Backend,
                      bool sampled = false, bool guard_disabled = false);
  // Lower-level entry: validate a supplied plan and dispatch, repairing invalid plans to BF16.
  // Caller must supply a fresh Descriptor for this input; this method does not rerun inspect().
  // Full cost selection occurs in adaptive(), not here. Malformed tensors still raise errors.
  Invocation execute(const std::vector<float>&, const PackedWeights&, const Shape&, const Policy&, Backend,
                     Plan plan, const Descriptor& current, bool guard_disabled = false);
  // Lifetime totals for this Executor, returned by const reference so callers cannot modify them.
  const Counters& counters() const { return counters_; }
 private:
  PlanCache cache_;
  Counters counters_;
};
// Full structural/numerical validation; environment matching is enabled except for explicit callers.
void validate_policy(const Policy&, bool require_environment = true);
// Persist a checksummed policy. Reload it to obtain the saved payload's checksum-derived ID.
void save_policy(const Policy&, const std::string& path);
// Reject malformed, changed or environment-incompatible policy files before inference.
Policy load_policy(const std::string& path);
// Recorded CPU/OS/code/compiler identity used to detect policies from a different environment.
std::string environment_key();
}
