#include "neurominds/runtime.hpp"

namespace nm {

// Store small execution plans, not weights, numerical outputs or compiled kernels.
PlanCache::PlanCache(std::size_t capacity) : capacity_(capacity) {
  if (capacity == 0 || capacity > 4096) {
    throw std::invalid_argument("cache capacity outside [1,4096]");
  }
}

bool PlanCache::find(const PlanKey& key, Plan& output, Counters& counters) {
  // Linear lookup is adequate for the small bounded cache (default 16 entries).
  for (auto entry = entries_.begin(); entry != entries_.end(); ++entry) {
    if (!(entry->key == key)) {
      continue;
    }

    // Matching key, but inconsistent plan: discard it and report a miss.
    if (entry->validity_key != validity_key(*entry)) {
      entries_.erase(entry);
      ++counters.invalidations;
      ++counters.misses;
      return false;
    }

    // Copy the candidate to the caller. Hits never refresh FIFO insertion order.
    output = *entry;
    ++counters.hits;
    return true;
  }

  ++counters.misses;
  return false;
}

void PlanCache::put(const Plan& plan, Counters& counters) {
  // Replace an existing key in place, preserving its original FIFO position.
  for (auto& entry : entries_) {
    if (entry.key == plan.key) {
      entry = plan;
      return;
    }
  }

  // A new key at capacity evicts the oldest insertion. This is FIFO, not LRU.
  if (entries_.size() == capacity_) {
    entries_.pop_front();
    ++counters.invalidations;
  }

  entries_.push_back(plan);
}

void PlanCache::invalidate(std::uint64_t config_hash, std::uint64_t policy_id, Counters& counters) {
  for (auto entry = entries_.begin(); entry != entries_.end();) {
    const bool same_config = entry->key.config_hash == config_hash;
    const bool same_policy = entry->key.policy_id == policy_id;

    if (!same_config || !same_policy) {
      // erase already returns the next iterator; incrementing again would skip an entry.
      entry = entries_.erase(entry);
      ++counters.invalidations;
    } else {
      ++entry;
    }
  }
}

}  // namespace nm
