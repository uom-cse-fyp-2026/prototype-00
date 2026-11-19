#include "neurominds/runtime.hpp"
namespace nm {
// Store a bounded number of small execution plans, not outputs, weights or native kernels.
PlanCache::PlanCache(std::size_t capacity) : capacity_(capacity) {
  // Refuse an empty cache and bound memory/linear-lookup work even for custom capacities.
  if (!capacity || capacity > 4096) throw std::invalid_argument("cache capacity outside [1,4096]");
}
// output is an out-parameter: a valid hit copies its plan into the caller's object.
bool PlanCache::find(const PlanKey& key, Plan& output, Counters& counters) {
  // Linear search is simple for this small FIFO cache (default capacity 16).
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    if (it->key == key) {
      // A matching key is insufficient when the stored plan's own fields are inconsistent.
      if (it->validity_key != validity_key(*it)) {
        // A corrupted matching entry counts as an invalidation and a miss, never as a hit.
        entries_.erase(it); ++counters.invalidations; ++counters.misses; return false;
      }
      // A hit does not move the entry: replacement order is FIFO, not least-recently-used (LRU).
      // Executor::adaptive still rechecks the current exact risk and cost after this lookup.
      output = *it; ++counters.hits; return true;
    }
  }
  // No compatible entry was found; the caller must construct a current plan.
  ++counters.misses;
  return false;
}
void PlanCache::put(const Plan& plan, Counters& counters) {
  // Update a same-key plan in place, preserving its original FIFO position.
  for (auto& entry : entries_) if (entry.key == plan.key) { entry = plan; return; }
  // When full, evict the oldest inserted entry. Evictions share the invalidations counter.
  if (entries_.size() == capacity_) { entries_.pop_front(); ++counters.invalidations; }
  entries_.push_back(plan); // deterministic FIFO, not an assumed token reuse horizon
}
// Remove entries built under another compiled configuration or frozen policy.
// Other key differences, such as shape/backend/bucket, naturally fail equality in find().
void PlanCache::invalidate(std::uint64_t config, std::uint64_t policy, Counters& counters) {
  for (auto it = entries_.begin(); it != entries_.end();) {
    // erase returns the next iterator; incrementing it again here would skip an entry.
    if (it->key.config_hash != config || it->key.policy_id != policy) { it = entries_.erase(it); ++counters.invalidations; }
    else ++it;
  }
}
}
