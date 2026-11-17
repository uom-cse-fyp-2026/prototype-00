#include "neurominds/experiments.hpp"
#include "nm_config.hpp"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>
#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace nm {
namespace {
std::string file_text(const std::string& path) {
  std::ifstream stream(path);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
std::string cpu_line(const std::string& key) {
  std::istringstream cpu(file_text("/proc/cpuinfo"));
  std::string line;
  while (std::getline(cpu, line)) if (line.rfind(key, 0) == 0) return line;
  return "unavailable";
}
std::string timestamp() {
  const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &now);
#else
  gmtime_r(&now, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}
}
std::string json_quote(const std::string& text) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c == '\n') out << "\\n";
    else if (c == '\r') out << "\\r";
    else if (c == '\t') out << "\\t";
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}
std::string environment_key() {
  // Excludes transient scheduler load. Costs must still be recalibrated after material
  // resource changes, even if this identity check passes.
  return std::string(NM_GIT_REV) + "|" + NM_SOURCE_HASH + "|" + NM_COMPILER + "|" + NM_BUILD_TYPE + "|" +
    cpu_line("model name") + "|" + cpu_line("flags") + "|" + std::to_string(std::thread::hardware_concurrency()) +
    "|" + file_text("/etc/os-release");
}
std::size_t resident_bytes() {
#if defined(__linux__)
  std::ifstream stream("/proc/self/statm");
  std::size_t pages = 0, resident = 0;
  if (stream >> pages >> resident) return resident * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#endif
  return 0;
}
std::size_t peak_resident_bytes() {
#if defined(__linux__)
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) == 0) return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
  return 0;
}
std::string environment_json() {
  std::ostringstream out;
  out << "{\"os_release\":" << json_quote(file_text("/etc/os-release"))
      << ",\"cpu\":" << json_quote(cpu_line("model name")) << ",\"isa_flags\":" << json_quote(cpu_line("flags"))
      << ",\"logical_cores\":" << std::thread::hardware_concurrency()
      << ",\"memory_info\":" << json_quote(file_text("/proc/meminfo"))
      << ",\"compiler\":" << json_quote(NM_COMPILER) << ",\"build_type\":" << json_quote(NM_BUILD_TYPE)
      << ",\"code_revision\":" << json_quote(NM_GIT_REV)
      << ",\"source_build_hash\":" << json_quote(NM_SOURCE_HASH)
      << ",\"hardware_threads_used\":1,\"energy_counters\":\"not collected\",\"performance_counters\":\"not collected\"}";
  return out.str();
}
void print_environment() { std::cout << environment_json() << '\n'; }
void print_metadata() {
  std::cout << "configuration_version=" << generated::config_version << " hash=" << generated::config_hash
    << "\nW4: two signed nibbles/byte, quantized range [-7,7], FP32 scale per output-row/group"
    << "\nBF16: stored uint16, RNE, FP32 multiply/accumulation"
    << "\nA8: stored int8 [-127,127], fixed calibration scale, INT32 per-group dot, FP32 rescaling"
    << "\nerror_budget=" << generated::error_budget << " fallback=w4_bf16"
    << "\nMLIR executors compiled=" << NM_HAS_MLIR << "\nfrontend=" << generated::frontend_kind << '\n';
}
void write_manifest(const Options& o, const Policy* p, const std::string& kind, const std::string& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write manifest: " + path);
  out << std::setprecision(17) << "{\n\"timestamp_utc\":" << json_quote(timestamp()) << ",\n\"kind\":" << json_quote(kind)
    << ",\n\"environment\":" << environment_json() << ",\n\"backend\":" << json_quote(name(o.backend))
    << ",\n\"shape\":{\"batch\":" << o.shape.batch << ",\"input\":" << o.shape.input << ",\"output\":" << o.shape.output
    << ",\"group\":" << o.shape.group << "},\n\"seed\":" << o.seed << ",\"samples\":" << o.samples
    << ",\"warmup\":" << o.warmup << ",\"repeats\":" << o.repeats << ",\"sampled_inspection\":" << (o.sampled ? "true" : "false")
    << ",\n\"configuration_hash\":" << generated::config_hash << ",\"error_budget\":" << generated::error_budget
    << ",\n\"relative_l2_epsilon\":" << generated::l2_epsilon
    << ",\n\"split_scheme\":\"SplitMix64(base XOR index XOR held_out_high_bit)\""
    << ",\n\"scale_fit_calibration_indices\":\"100000..100015, typical inputs, at most 65536 magnitudes\""
    << ",\n\"resident_bytes\":" << resident_bytes() << ",\"peak_resident_bytes\":" << peak_resident_bytes();
  if (p) out << ",\n\"policy\":{\"id\":" << p->id << ",\"clip\":" << p->clip << ",\"scale\":" << p->scale
    << ",\"guard_enabled\":" << (p->guard_enabled ? "true" : "false") << ",\"max_abs_limit\":" << p->max_abs_limit
    << ",\"clipping_limit\":" << p->clipping_limit << ",\"bf16_us\":" << p->bf16_us << ",\"a8_us\":" << p->a8_us << "}";
  out << ",\n\"limitations\":\"VM measurements are preliminary; no energy, thermal or counter claims; operator error is not model quality\"\n}\n";
  if (!out) throw std::runtime_error("failed writing manifest");
}
}
