#pragma once
#include "experiments.hpp"

namespace nm {

struct ModelOptions {
  Options execution;
  std::string model_path;
  std::string input_path;
  std::size_t evaluation_samples = 16;
};

// Fit layer policies and compare complete model forwards on held-out inputs.
void run_model_experiment(const ModelOptions& options);

// Run guarded inference with the model snapshot and policies saved in execution.output.
void infer_model(const ModelOptions& options);

}  // namespace nm
