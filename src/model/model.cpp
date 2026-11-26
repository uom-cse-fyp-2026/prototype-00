#include "neurominds/model.hpp"
#include "neurominds/kernels.hpp"
#include "nm_config.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace nm {
namespace {

namespace fs = std::filesystem;

// This prototype accepts a bounded chain of affine layers with optional ReLU.
constexpr std::size_t max_layers = 16;
constexpr std::size_t max_parameters = 1000000;
constexpr const char* model_version = "NM_MODEL_V1";
constexpr const char* model_snapshot = "model.txt";
constexpr const char* policy_index_version = "NM_MODEL_POLICIES_V1";

struct Layer {
  Shape shape;
  bool relu = false;
  std::vector<float> weights;
  std::vector<float> bias;
  PackedWeights packed;
  Policy policy;
};

using Model = std::vector<Layer>;

enum class Mode { Fp32, Bf16, A8, Adaptive };
constexpr std::array<Mode, 4> modes{Mode::Fp32, Mode::Bf16, Mode::A8, Mode::Adaptive};
constexpr std::array<const char*, 4> mode_names{"fp32", "static_bf16", "static_a8", "adaptive"};
constexpr std::array<const char*, 4> regimes{"normal", "outliers", "shifted", "mixed"};

void require_end(std::istream& stream, const char* message) {
  stream >> std::ws;
  if (!stream.eof()) {
    throw std::invalid_argument(message);
  }
}

std::vector<float> read_values(std::istream& stream, std::size_t count, const char* field) {
  std::vector<float> values(count);
  for (auto& value : values) {
    if (!(stream >> value) || !std::isfinite(value)) {
      throw std::invalid_argument(std::string("missing, malformed or nonfinite ") + field);
    }
  }
  return values;
}

Model load_model(const fs::path& path, std::size_t batch, std::size_t group) {
  std::ifstream stream(path);
  std::string version;
  std::size_t count = 0;
  if (!(stream >> version >> count) || version != model_version || count == 0 || count > max_layers) {
    throw std::invalid_argument("expected NM_MODEL_V1 and 1..16 layers in " + path.string());
  }

  Model model;
  std::size_t parameters = 0;
  for (std::size_t index = 0; index < count; ++index) {
    Layer layer;
    layer.shape.batch = batch;
    layer.shape.group = group;
    std::string operation, post_op, bias_format;
    if (!(stream >> operation >> layer.shape.input >> layer.shape.output >> post_op >> bias_format) ||
        operation != "linear" || (post_op != "relu" && post_op != "identity") ||
        (bias_format != "bias" && bias_format != "no_bias")) {
      throw std::invalid_argument("invalid linear layer header at index " + std::to_string(index));
    }
    validate_shape(layer.shape);
    if (index > 0 && model.back().shape.output != layer.shape.input) {
      throw std::invalid_argument("adjacent model layer dimensions do not match");
    }
    const auto weight_count = product(layer.shape.input, layer.shape.output);
    const auto bias_count = bias_format == "bias" ? layer.shape.output : 0;
    if (weight_count + bias_count > max_parameters - parameters) {
      throw std::invalid_argument("model exceeds the 1,000,000 parameter limit");
    }
    parameters += weight_count + bias_count;
    layer.relu = post_op == "relu";
    layer.weights = read_values(stream, weight_count, "model weights");
    layer.bias = read_values(stream, bias_count, "model bias");
    layer.packed = quantize_weights(layer.weights, layer.shape);
    model.push_back(std::move(layer));
  }
  require_end(stream, "unexpected data after final model layer");
  return model;
}

void save_model(const Model& model, const fs::path& path) {
  std::ofstream stream(path);
  stream << model_version << ' ' << model.size() << '\n';
  stream << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (const auto& layer : model) {
    stream << "linear " << layer.shape.input << ' ' << layer.shape.output << ' '
           << (layer.relu ? "relu" : "identity") << ' '
           << (layer.bias.empty() ? "no_bias" : "bias") << '\n';
    for (float weight : layer.weights) stream << weight << ' ';
    stream << '\n';
    for (float bias : layer.bias) stream << bias << ' ';
    stream << '\n';
  }
  stream.close();
  if (!stream) throw std::runtime_error("cannot write model snapshot");
}

fs::path layer_directory(const fs::path& run, std::size_t index) {
  return run / ("layer-" + std::to_string(index));
}

std::uint64_t snapshot_checksum(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open saved model snapshot");
  // Bind policy IDs to the exact saved snapshot, including its biases and activations.
  // This detects accidental changes. The run directory is trusted local data.
  std::uint64_t checksum = 14695981039346656037ULL;
  char byte;
  while (stream.get(byte)) {
    checksum ^= static_cast<unsigned char>(byte);
    checksum *= 1099511628211ULL;
  }
  if (!stream.eof() || stream.bad()) throw std::runtime_error("cannot read saved model snapshot");
  return checksum;
}

void postprocess(std::vector<float>& values, const Layer& layer) {
  // Bias and ReLU are identical across modes and operate outside the quantized dot product.
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (!layer.bias.empty()) values[index] += layer.bias[index % layer.shape.output];
    if (!std::isfinite(values[index])) throw std::overflow_error("nonfinite model layer output");
    if (layer.relu) values[index] = std::max(0.0f, values[index]);
  }
}

struct LayerTrace {
  Variant variant;
  bool cache_hit;
  bool fallback;
  double inspector_us;
  std::string reason;
};

std::vector<float> forward_layer(const std::vector<float>& input, const Layer& layer,
                                 Mode mode, Executor* executor = nullptr, LayerTrace* trace = nullptr) {
  std::vector<float> output;
  if (mode == Mode::Fp32) {
    output = fp32_reference(input, layer.weights, layer.shape);
  } else if (mode == Mode::Adaptive) {
    auto result = executor->adaptive(input, layer.packed, layer.shape, layer.policy,
                                     layer.policy.backend, layer.policy.sampled);
    *trace = {result.plan.variant, result.cache_hit, result.fallback,
              result.inspector_us, std::move(result.reason)};
    output = std::move(result.output);
  } else {
    double preparation_us = 0;
    double kernel_us = 0;
    const auto variant = mode == Mode::A8 ? Variant::A8 : Variant::Bf16;
    run_kernel(input, layer.packed, layer.shape, variant, layer.policy.scale,
               layer.policy.backend, output, preparation_us, kernel_us);
  }
  postprocess(output, layer);
  return output;
}

std::vector<float> forward(const Model& model, const std::vector<float>& input, Mode mode,
                           std::vector<Executor>& executors, std::vector<LayerTrace>& traces) {
  auto values = forward_layer(input, model.front(), mode,
      mode == Mode::Adaptive ? &executors[0] : nullptr, mode == Mode::Adaptive ? &traces[0] : nullptr);
  for (std::size_t index = 1; index < model.size(); ++index) {
    values = forward_layer(values, model[index], mode,
        mode == Mode::Adaptive ? &executors[index] : nullptr,
        mode == Mode::Adaptive ? &traces[index] : nullptr);
  }
  return values;
}

void calibrate_model(Model& model, const Options& options, const fs::path& run) {
  for (std::size_t index = 0; index < model.size(); ++index) {
    auto& layer = model[index];
    auto layer_options = options;
    layer_options.shape = layer.shape;
    layer_options.output = layer_directory(run, index).string();
    layer_options.policy = (layer_directory(run, index) / "policy.nm").string();
    std::cout << "Calibrating layer " << index << " (" << layer.shape.input
              << " -> " << layer.shape.output << ")\n";
    layer.policy = calibrate_layer(layer_options, layer.packed, [&](const std::string& regime, std::uint64_t seed) {
      auto values = synthetic_input(model.front().shape, regime, seed);
      // Fit each policy on activations propagated by the conservative upstream model.
      // Evaluation runs the actual selected paths, so downstream activations can differ.
      for (std::size_t upstream = 0; upstream < index; ++upstream) {
        values = forward_layer(values, model[upstream], Mode::Bf16);
      }
      return values;
    });
  }
}

void evaluate_model(const Model& model, const ModelOptions& options, const fs::path& run) {
  const auto& execution = options.execution;
  std::ofstream measurements(run / "model_benchmark.csv");
  std::ofstream decisions(run / "model_decisions.csv");
  if (!measurements || !decisions) throw std::runtime_error("cannot open model benchmark files");
  measurements << "sample,repeat,seed,regime,mode,total_us,error_fp32,a8_layers,fallback_layers,cache_hits,inspector_us\n";
  decisions << "sample,repeat,regime,layer,selected,cache_hit,fallback,inspector_us,reason\n";
  measurements << std::setprecision(17);
  decisions << std::setprecision(17);

  std::vector<Executor> executors(model.size(), Executor(generated::cache_capacity));
  std::vector<Executor> warmup_executors = executors;
  std::vector<LayerTrace> traces(model.size());
  std::array<double, modes.size()> total_us{};
  std::array<double, modes.size()> error_sum{};
  std::size_t a8_choices = 0;
  std::size_t fallback_choices = 0;

  for (std::size_t sample = 0; sample < options.evaluation_samples; ++sample) {
    const auto seed = split_seed(execution.seed, true, sample);
    const auto regime = regimes[sample % regimes.size()];
    const auto input = synthetic_input(model.front().shape, regime, seed);
    const auto reference = forward(model, input, Mode::Fp32, executors, traces);

    // Warmup uses a separate cache so it cannot create hits in the measured stream.
    for (std::size_t warmup = 0; warmup < execution.warmup; ++warmup) {
      for (auto mode : modes) forward(model, input, mode, warmup_executors, traces);
    }

    for (std::size_t repeat = 0; repeat < execution.repeats; ++repeat) {
      for (std::size_t order = 0; order < modes.size(); ++order) {
        const auto mode_index = (order + repeat + sample) % modes.size();
        const auto mode = modes[mode_index];
        const auto start = Clock::now();
        const auto output = forward(model, input, mode, executors, traces);
        const double duration = elapsed_us(start);
        const double error = relative_l2(output, reference);
        total_us[mode_index] += duration;
        error_sum[mode_index] += error;

        std::size_t a8_layers = mode == Mode::A8 ? model.size() : 0;
        std::size_t fallback_layers = 0;
        std::size_t cache_hits = 0;
        double inspector_us = 0;
        if (mode == Mode::Adaptive) {
          for (std::size_t index = 0; index < traces.size(); ++index) {
            const auto& trace = traces[index];
            a8_layers += trace.variant == Variant::A8;
            fallback_layers += trace.fallback;
            cache_hits += trace.cache_hit;
            inspector_us += trace.inspector_us;
            decisions << sample << ',' << repeat << ',' << regime << ',' << index << ','
                      << name(trace.variant) << ',' << trace.cache_hit << ',' << trace.fallback << ','
                      << trace.inspector_us << ',' << trace.reason << '\n';
          }
          a8_choices += a8_layers;
          fallback_choices += fallback_layers;
        }
        measurements << sample << ',' << repeat << ',' << seed << ',' << regime << ','
                     << mode_names[mode_index] << ',' << duration << ',' << error << ','
                     << a8_layers << ',' << fallback_layers << ',' << cache_hits << ',' << inspector_us << '\n';
      }
    }
  }
  measurements.close();
  decisions.close();
  if (!measurements || !decisions) throw std::runtime_error("failed writing model benchmark files");

  const auto forwards = options.evaluation_samples * execution.repeats;
  for (std::size_t index = 0; index < modes.size(); ++index) {
    std::cout << mode_names[index] << " mean_us=" << total_us[index] / forwards
              << " mean_error_fp32=" << error_sum[index] / forwards << '\n';
  }
  std::cout << "Adaptive A8 layer choices=" << a8_choices << '/' << forwards * model.size()
            << " BF16 fallbacks=" << fallback_choices << '\n';
}

}  // namespace

void run_model_experiment(const ModelOptions& options) {
  const auto& execution = options.execution;
  const fs::path run = execution.output;
  if (options.model_path.empty()) throw std::invalid_argument("model-run requires --model FILE");
  if (options.evaluation_samples == 0 || options.evaluation_samples > 4096) {
    throw std::invalid_argument("evaluation-samples must be in [1,4096]");
  }
  if (execution.samples < generated::minimum_guard_accepts || execution.samples > 4096 ||
      execution.repeats == 0 || execution.repeats > 10000 || execution.warmup > 10000) {
    throw std::invalid_argument("invalid calibration samples, repeats or warmup");
  }
  if (execution.backend == Backend::Mlir && !mlir_available()) {
    throw std::invalid_argument("MLIR build required or explicitly use --backend scalar");
  }

  auto model = load_model(options.model_path, execution.shape.batch, execution.shape.group);
  if (!fs::create_directories(run)) {
    throw std::invalid_argument("model-run requires a new results directory: " + run.string());
  }
  save_model(model, run / model_snapshot);
  calibrate_model(model, execution, run);
  evaluate_model(model, options, run);

  // Publish the ordered policy IDs last. Inference rejects incomplete runs and swapped policies.
  const auto pending_index = run / "model_policies.pending";
  std::ofstream ready(pending_index);
  ready << policy_index_version << ' ' << snapshot_checksum(run / model_snapshot) << '\n';
  for (const auto& layer : model) ready << layer.policy.id << '\n';
  ready.close();
  if (!ready) throw std::runtime_error("cannot write model policy index");
  fs::rename(pending_index, run / "model_policies.txt");
  std::cout << "Saved model experiment to " << run.string() << '\n';
}

void infer_model(const ModelOptions& options) {
  if (options.input_path.empty()) throw std::invalid_argument("model-infer requires --input-file FILE");
  const fs::path run = options.execution.output;
  std::ifstream policy_index(run / "model_policies.txt");
  if (!policy_index) throw std::invalid_argument("model run is missing its completed policy index");
  std::string index_version;
  std::uint64_t model_checksum = 0;
  if (!(policy_index >> index_version >> model_checksum) || index_version != policy_index_version ||
      model_checksum != snapshot_checksum(run / model_snapshot)) {
    throw std::invalid_argument("saved policy index does not match model snapshot");
  }
  const auto first_policy = load_policy((layer_directory(run, 0) / "policy.nm").string());
  auto model = load_model(run / model_snapshot, first_policy.shape.batch, first_policy.shape.group);

  for (std::size_t index = 0; index < model.size(); ++index) {
    auto& layer = model[index];
    layer.policy = index == 0 ? first_policy : load_policy((layer_directory(run, index) / "policy.nm").string());
    std::uint64_t expected_id = 0;
    if (!(policy_index >> expected_id) || layer.policy.id != expected_id ||
        layer.policy.shape != layer.shape || layer.policy.backend != first_policy.backend) {
      throw std::invalid_argument("saved policy does not match model layer " + std::to_string(index));
    }
  }
  require_end(policy_index, "unexpected entries in model policy index");

  std::ifstream stream(options.input_path);
  auto input = read_values(stream, model.front().shape.inputs(), "model input values");
  require_end(stream, "unexpected values after complete model input");

  std::vector<Executor> executors(model.size(), Executor(generated::cache_capacity));
  std::vector<LayerTrace> traces(model.size());
  const auto output = forward(model, input, Mode::Adaptive, executors, traces);
  for (std::size_t index = 0; index < traces.size(); ++index) {
    std::cerr << "layer " << index << ' ' << name(traces[index].variant) << ' '
              << traces[index].reason << '\n';
  }
  std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
  const auto output_width = model.back().shape.output;
  for (std::size_t index = 0; index < output.size(); ++index) {
    std::cout << output[index] << ((index + 1) % output_width == 0 ? '\n' : ' ');
  }
}

}  // namespace nm
