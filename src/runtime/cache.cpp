#include "neurominds/runtime.hpp"
namespace nm {
PlanCache::PlanCache(std::size_t capacity) : capacity_(capacity) {
  if (!capacity || capacity > 4096) throw std::invalid_argument("cache capacity outside [1,4096]");
}
bool PlanCache::find(const PlanKey& key, Plan& output, Counters& counters) {
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    if (it->key == key) {
      if (it->validity_key != validity_key(*it)) {
        entries_.erase(it); ++counters.invalidations; ++counters.misses; return false;
      }
      output = *it; ++counters.hits; return true;
    }
  }
  ++counters.misses;
  return false;
}
void PlanCache::put(const Plan& plan, Counters& counters) {
  for (auto& entry : entries_) if (entry.key == plan.key) { entry = plan; return; }
  if (entries_.size() == capacity_) { entries_.pop_front(); ++counters.invalidations; }
  entries_.push_back(plan); // deterministic FIFO, not an assumed token reuse horizon
}
void PlanCache::invalidate(std::uint64_t config, std::uint64_t policy, Counters& counters) {
  for (auto it = entries_.begin(); it != entries_.end();) {
    if (it->key.config_hash != config || it->key.policy_id != policy) { it = entries_.erase(it); ++counters.invalidations; }
    else ++it;
  }
}
}
