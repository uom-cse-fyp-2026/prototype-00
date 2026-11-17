#pragma once
#include "inspector.hpp"
#include <deque>
namespace nm {
class PlanCache {
 public:
  explicit PlanCache(std::size_t capacity);
  bool find(const PlanKey&, Plan&, Counters&);
  void put(const Plan&, Counters&);
  void invalidate(std::uint64_t config, std::uint64_t policy, Counters&);
  std::size_t size() const { return entries_.size(); }
 private:
  std::size_t capacity_;
  std::deque<Plan> entries_;
};
class Executor {
 public:
  explicit Executor(std::size_t capacity = 16);
  Invocation adaptive(const std::vector<float>&, const PackedWeights&, const Shape&, const Policy&, Backend,
                      bool sampled = false, bool guard_disabled = false);
  Invocation execute(const std::vector<float>&, const PackedWeights&, const Shape&, const Policy&, Backend,
                     Plan plan, const Descriptor& current, bool guard_disabled = false);
  const Counters& counters() const { return counters_; }
 private:
  PlanCache cache_;
  Counters counters_;
};
void validate_policy(const Policy&, bool require_environment = true);
void save_policy(const Policy&, const std::string& path);
Policy load_policy(const std::string& path);
std::string environment_key();
}
