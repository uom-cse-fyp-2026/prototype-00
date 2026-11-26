#include "neurominds/experiments.hpp"
#include "neurominds/model.hpp"
#include "nm_config.hpp"
#include <iostream>

namespace {
std::size_t integer(const std::string& text) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) throw std::invalid_argument("expected unsigned integer: " + text);
  std::size_t consumed = 0;
  const auto value = std::stoull(text, &consumed);
  if (consumed != text.size() || value > std::numeric_limits<std::size_t>::max()) throw std::invalid_argument("integer out of range");
  return static_cast<std::size_t>(value);
}
void help() {
  std::cout << "NeuroMinds operator prototype\n"
    "neurominds_demo env|metadata|calibrate|demo|run|bench [options]\n"
    "  --batch N --input K --output N --group G --seed N\n"
    "  --backend mlir|scalar --sampled --samples N --warmup N --repeats N\n"
    "  --results DIR --policy FILE --regime normal|outliers|shifted|mixed|zero\n"
    "  --variant fp32|w4-reference|bf16|a8|adaptive\n"
    "neurominds_demo model-run --model FILE --results NEW_DIR [--batch N --group G --samples N --evaluation-samples N]\n"
    "neurominds_demo model-infer --results RUN_DIR --input-file FILE\n"
    "Use ./nm test for automated CTest coverage and ./nm all for the VM workflow.\n";
}
}
int main(int argc, char** argv) {
  try {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "help") { help(); return 0; }
    const std::string command = argv[1];
    nm::ModelOptions model_options;
    auto& options = model_options.execution;
    options.shape = {nm::generated::default_batch, nm::generated::default_input, nm::generated::default_output, nm::generated::default_group};
    options.seed = nm::generated::default_seed;
    if (command == "bench") options.samples = 16;
    const bool model_run = command == "model-run";
    const bool model_infer = command == "model-infer";
    if (model_run || model_infer) options.output = "results/model-run";
    for (int i = 2; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--sampled" && !model_infer) { options.sampled = true; continue; }
      if (arg == "--help") { help(); return 0; }
      if (i + 1 == argc) throw std::invalid_argument("missing option value: " + arg);
      const std::string value = argv[++i];
      if (model_infer && arg != "--results" && arg != "--input-file")
        throw std::invalid_argument("model-infer reads execution settings from its saved policies: " + arg);
      if (model_run && (arg == "--input" || arg == "--output" || arg == "--policy" ||
                        arg == "--regime" || arg == "--variant"))
        throw std::invalid_argument("option does not apply to model-run: " + arg);
      if (arg == "--batch") options.shape.batch = integer(value);
      else if (arg == "--input") options.shape.input = integer(value);
      else if (arg == "--output") options.shape.output = integer(value);
      else if (arg == "--group") options.shape.group = integer(value);
      else if (arg == "--seed") options.seed = integer(value);
      else if (arg == "--samples") options.samples = integer(value);
      else if (arg == "--warmup") options.warmup = integer(value);
      else if (arg == "--repeats") options.repeats = integer(value);
      else if (arg == "--results") options.output = value;
      else if (arg == "--policy") options.policy = value;
      else if (arg == "--regime") options.regime = value;
      else if (arg == "--variant") options.variant = value;
      else if (model_run && arg == "--model") model_options.model_path = value;
      else if (model_run && arg == "--evaluation-samples") model_options.evaluation_samples = integer(value);
      else if (model_infer && arg == "--input-file") model_options.input_path = value;
      else if (arg == "--backend") {
        if (value == "mlir") options.backend = nm::Backend::Mlir;
        else if (value == "scalar") options.backend = nm::Backend::Scalar;
        else throw std::invalid_argument("backend must be mlir or scalar");
      } else throw std::invalid_argument("unknown option: " + arg);
    }
    if (command == "env") nm::print_environment();
    else if (command == "metadata") nm::print_metadata();
    else if (command == "calibrate") nm::calibrate(options);
    else if (command == "demo") nm::demonstrate(options);
    else if (command == "run") nm::projection(options);
    else if (command == "bench") nm::benchmark(options);
    else if (model_run) nm::run_model_experiment(model_options);
    else if (model_infer) nm::infer_model(model_options);
    else throw std::invalid_argument("unknown command: " + command);
    return 0;
  } catch (const std::exception& error) { std::cerr << "ERROR: " << error.what() << '\n'; return 1; }
}
