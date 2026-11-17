#!/usr/bin/env python3
"""Validate the approved configuration and emit C++ metadata plus executable MLIR."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from mlir_kernels import emit_kernels


def positive_integer(value, field):
    if type(value) is not int or value <= 0:
        raise ValueError(field + " must be a positive integer")


def validate(c):
    expected = {"schema_version", "configuration_version", "operator", "weight_format", "quantized_weight_range",
                "activation_policy", "activation_range", "activation_calibration_quantile", "error_budget",
                "relative_l2_epsilon", "max_calibration_false_safe_rate", "minimum_guard_accepts", "performance_margin",
                "cache_capacity", "sample_size", "memory_budget_bytes", "shape_limits", "defaults", "variants",
                "fallback_variant", "guard_policy", "plan_validity"}
    if set(c) != expected:
        raise ValueError("unknown or missing configuration fields: " + str(set(c) ^ expected))
    if type(c["schema_version"]) is not int or c["schema_version"] != 1 or c["operator"] != "linear_without_bias":
        raise ValueError("only schema 1 / linear_without_bias is implemented")
    for key in ("configuration_version", "cache_capacity", "sample_size", "memory_budget_bytes", "minimum_guard_accepts"):
        positive_integer(c[key], key)
    if c["configuration_version"] >= 2 ** 32:
        raise ValueError("configuration_version must fit uint32")
    if c["memory_budget_bytes"] >= 2 ** 64 or c["minimum_guard_accepts"] > 4096:
        raise ValueError("memory/minimum calibration count exceeds implementation limits")
    if c["sample_size"] < 2 or c["cache_capacity"] > 4096 or c["sample_size"] > 1048576:
        raise ValueError("invalid sample/cache capacity")
    if c["weight_format"] != "signed_w4_row_packed_group_symmetric" or c["quantized_weight_range"] != [-7, 7]:
        raise ValueError("unsupported weight representation")
    if c["activation_policy"] != "calibration_fixed_symmetric_a8" or c["activation_range"] != [-127, 127]:
        raise ValueError("unsupported activation policy")
    if c["guard_policy"] != "calibrated_max_abs_and_clipping_fraction" or c["plan_validity"] != "shape_config_policy_backend_and_current_descriptor":
        raise ValueError("unsupported guard or validity policy")
    for key, low, high in (("error_budget", 0, 1), ("relative_l2_epsilon", 0, 1),
                           ("activation_calibration_quantile", 0, 1), ("performance_margin", 0, 1)):
        value = c[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low < value < high:
            raise ValueError("invalid " + key)
    rate = c["max_calibration_false_safe_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("invalid max_calibration_false_safe_rate")
    limits = c["shape_limits"]
    if set(limits) != {"batch", "input", "output", "group"}:
        raise ValueError("unsupported shape limit fields")
    for key, value in limits.items():
        positive_integer(value, "shape_limits." + key)
        if value > 1048576:
            raise ValueError("shape limit too large")
    if min(limits["input"], limits["group"]) > (2 ** 31 - 1) // (127 * 8):
        raise ValueError("INT32 accumulator overflow risk")
    if set(c["defaults"]) != {"batch", "input", "output", "group", "seed"}:
        raise ValueError("invalid defaults")
    for key, value in c["defaults"].items():
        if key == "seed":
            if type(value) is not int or not 0 <= value < 2 ** 64:
                raise ValueError("invalid seed")
        else:
            positive_integer(value, "defaults." + key)
            if value > limits[key]:
                raise ValueError("default shape exceeds limit")
    d = c["defaults"]
    estimated = d["input"] * d["output"] * 12 + d["batch"] * (d["input"] + d["output"]) * 24 + \
        ((d["input"] + d["group"] - 1) // d["group"]) * d["output"] * 4
    if estimated > c["memory_budget_bytes"]:
        raise ValueError("default shape exceeds operator memory budget")
    expected_variants = [
        {"id": 1, "name": "w4_bf16", "activation": "bf16", "accumulator": "f32", "approved": True},
        {"id": 2, "name": "w4_a8", "activation": "i8", "accumulator": "i32_per_group_then_f32", "approved": True},
    ]
    if c["variants"] != expected_variants or c["fallback_variant"] != 1:
        raise ValueError("only the implemented BF16/A8 family with approved BF16 fallback is supported")
    if type(c["fallback_variant"]) is not int or any(type(v["approved"]) is not bool or type(v["id"]) is not int for v in c["variants"]):
        raise ValueError("variant IDs must be integers and approval flags must be booleans")
    return c


def generate(config, output, frontend=None, scalar_only=False):
    c = validate(json.loads(Path(config).read_text(encoding="utf-8")))
    provenance = {"kind": "scalar_only"}
    if not scalar_only:
        if frontend is None:
            raise ValueError("torch-mlir frontend contract is required")
        provenance = json.loads(Path(frontend).read_text(encoding="utf-8"))
        if provenance.get("operator") != "linear_without_bias" or provenance.get("call_targets") != ["aten.t.default", "aten.mm.default"]:
            raise ValueError("frontend graph is not the supported linear projection")
        if provenance.get("kind") != "torch_mlir.fx" or not provenance.get("linalg_sha256"):
            raise ValueError("missing torch-mlir lowering provenance")
    canonical = json.dumps({"config": c, "frontend": provenance}, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    config_hash = int(digest[:16], 16)
    header = ["// Generated; edit configs/prototype.json instead.", "#pragma once", "#include <cstddef>",
              "#include <cstdint>", "namespace nm::generated {"]
    constants = {
        "config_version": c["configuration_version"], "config_hash": config_hash,
        "max_batch": c["shape_limits"]["batch"], "max_input": c["shape_limits"]["input"],
        "max_output": c["shape_limits"]["output"], "max_group": c["shape_limits"]["group"],
        "memory_budget_bytes": c["memory_budget_bytes"], "cache_capacity": c["cache_capacity"],
        "sample_size": c["sample_size"], "minimum_guard_accepts": c["minimum_guard_accepts"],
        **{"default_" + key: value for key, value in c["defaults"].items()},
    }
    for name, value in constants.items():
        header.append("inline constexpr std::uint64_t %s = %dULL;" % (name, value))
    for name, key in (("error_budget", "error_budget"), ("l2_epsilon", "relative_l2_epsilon"),
                      ("performance_margin", "performance_margin"), ("activation_quantile", "activation_calibration_quantile"),
                      ("max_calibration_false_safe_rate", "max_calibration_false_safe_rate")):
        header.append("inline constexpr double %s = %s;" % (name, repr(float(c[key]))))
    header += ["inline constexpr bool bf16_approved = true;", "inline constexpr bool a8_approved = true;",
               "inline constexpr const char* frontend_kind = %s;" % json.dumps(provenance["kind"]), "}"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "nm_config.hpp").write_text("\n".join(header) + "\n", encoding="utf-8")
    (output / "executors.mlir").write_text(emit_kernels(), encoding="utf-8")
    (output / "build_manifest.json").write_text(json.dumps({"configuration": c, "frontend": provenance,
        "configuration_sha256": digest, "configuration_hash": config_hash}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--frontend")
    p.add_argument("--scalar-only", action="store_true")
    args = p.parse_args()
    generate(args.config, args.output, args.frontend, args.scalar_only)
