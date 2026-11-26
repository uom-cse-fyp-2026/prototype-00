#pragma once
#include "inspector.hpp"
#include <deque>

namespace nm {

// Bounded FIFO cache. A hit never skips adaptive()'s fresh inspection/risk/cost decision.
class PlanCache {
 public:
  explicit PlanCache(std::size_t capacity);  // Valid range: [1,4096].

  // On a key/checksum match, copy the candidate into output and return true.
  bool find(const PlanKey& key, Plan& output, Counters& counters);
  void put(const Plan& plan, Counters& counters);
  void invalidate(std::uint64_t config_hash, std::uint64_t policy_id, Counters& counters);

  std::size_t size() const {
    return entries_.size();
  }

 private:
  std::size_t capacity_;
  std::deque<Plan> entries_;  // Oldest insertion at front; hits do not reorder entries.
};

// Stateful runtime for one thread, assuming immutable calibrated policy/weight contents.
class Executor {
 public:
  explicit Executor(std::size_t capacity = 16);

  // Normal path: inspect -> lookup -> decide -> update plan -> validate/execute -> report.
  // guard_disabled is an evaluation ablation, not a bypass of cost or tensor validation.
  Invocation adaptive(const std::vector<float>& input, const PackedWeights& weights,
                      const Shape& shape, const Policy& policy, Backend backend,
                      bool sampled = false, bool guard_disabled = false);

  // Lower-level dispatch: caller supplies the plan and a fresh descriptor for this input.
  // Does not rerun inspect() or cost selection. Invalid plans fall back; broken tensors raise errors.
  Invocation execute(const std::vector<float>& input, const PackedWeights& weights,
                     const Shape& shape, const Policy& policy, Backend backend,
                     Plan plan, const Descriptor& current, bool guard_disabled = false);

  const Counters& counters() const {
    return counters_;
  }

 private:
  PlanCache cache_;
  Counters counters_;
};

// Full policy validation at load/calibration, including environment identity by default.
void validate_policy(const Policy& policy, bool require_environment = true);

// Persist the canonical payload/checksum. Reload to obtain the saved policy ID in memory.
void save_policy(const Policy& policy, const std::string& path);
Policy load_policy(const std::string& path);

// Recorded CPU/OS/code/compiler identity; changing scheduler load is deliberately excluded.
std::string environment_key();

}  // namespace nm
