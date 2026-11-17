#!/usr/bin/env python3
"""Summarize measured CSV files only. Does not synthesize benchmark observations."""
import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * q) - 1)]


def confidence_interval(pairs):
    if len(pairs) < 2:
        return None
    rng = random.Random(4533)
    draws = []
    for _ in range(2000):
        selected = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        baseline = statistics.mean(item[0] for item in selected)
        adaptive = statistics.mean(item[1] for item in selected)
        draws.append(baseline / adaptive)
    return [percentile(draws, .025), percentile(draws, .975)]


def summarize(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("No measurements in " + str(path))
    manifest = json.loads((path.parent / "benchmark_manifest.json").read_text())
    shape = manifest["shape"]
    operations = 2 * shape["batch"] * shape["input"] * shape["output"]
    modes = defaultdict(list)
    sample_means = defaultdict(lambda: defaultdict(list))
    for row in rows:
        latency = float(row["total_us"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("Invalid measured latency")
        modes[row["mode"]].append(row)
        sample_means[row["mode"]][row["sample"]].append(latency)
    summaries = {}
    for mode, group in modes.items():
        times = [float(row["total_us"]) for row in group]
        entry = {"invocations": len(group), "mean_us": statistics.mean(times), "median_us": statistics.median(times),
                 "p95_us": percentile(times, .95), "stdev_us": statistics.stdev(times) if len(times) > 1 else None,
                 "gops_equivalent": operations / (statistics.mean(times) * 1000),
                 "cache_hits": sum(int(row["cache_hit"]) for row in group),
                 "cache_misses": sum(int(row["cache_miss"]) for row in group),
                 "invalidations": sum(int(row["invalidations"]) for row in group),
                 "fallbacks": sum(int(row["fallback"]) for row in group),
                 "a8_selections": sum(row["selected"] == "w4_a8" for row in group)}
        for metric in ("preparation_us", "kernel_us", "inspector_us", "validity_us", "decision_us", "dispatch_us", "error_w4", "error_fp32"):
            values = [float(row[metric]) for row in group]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite measurement in " + metric)
            entry["mean_" + metric] = statistics.mean(values)
            if metric.startswith("error_"):
                entry["max_" + metric] = max(values)
        entry["max_resident_bytes"] = max(int(row["resident_bytes"]) for row in group)
        entry["max_peak_resident_bytes"] = max(int(row["peak_resident_bytes"]) for row in group)
        summaries[mode] = entry
    paired = []
    for index, static in sample_means["static_bf16"].items():
        adaptive = sample_means["adaptive"].get(index)
        if adaptive:
            paired.append((statistics.mean(static), statistics.mean(adaptive)))
    interval = confidence_interval(paired)
    guard = json.loads((path.parent / "guard_metrics.json").read_text())
    result = {"manifest": manifest, "modes": summaries, "guard": guard,
              "adaptive_speedup_vs_static_bf16_paired_bootstrap_95_interval": interval,
              "uncertainty_unit": "held-out input, resampling its paired mean latencies; not independent VM sessions"}
    (path.parent / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Measured operator results", "", "Backend: `%s`; shape: `%s`." % (manifest["backend"], shape), "",
             "Preliminary VM measurements. Quantization quality is operator-level; no model perplexity or energy result is implied.", "",
             "| Mode | Mean us | Median us | p95 us | SD us | Equivalent GOP/s | Mean error vs W4 | Mean error vs FP32 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("fp32", "static_bf16", "static_a8", "adaptive", "guard_disabled"):
        if mode in summaries:
            s = summaries[mode]
            sd = "%.2f" % s["stdev_us"] if s["stdev_us"] is not None else "n/a"
            lines.append("| %s | %.2f | %.2f | %.2f | %s | %.3f | %.5g | %.5g |" %
                         (mode, s["mean_us"], s["median_us"], s["p95_us"], sd, s["gops_equivalent"], s["mean_error_w4"], s["mean_error_fp32"]))
    lines += ["", "| Mode | Prep us | Kernel us | Inspector us | Validity/cache us | Decision us | Dispatch residual us | Cache hits/misses | Fallbacks | A8 selections |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, s in summaries.items():
        kernel = "n/a" if mode == "fp32" else "%.2f" % s["mean_kernel_us"]
        lines.append("| %s | %.2f | %s | %.2f | %.2f | %.2f | %.2f | %d/%d | %d | %d |" %
                     (mode, s["mean_preparation_us"], kernel, s["mean_inspector_us"], s["mean_validity_us"], s["mean_decision_us"],
                      s["mean_dispatch_us"], s["cache_hits"], s["cache_misses"], s["fallbacks"], s["a8_selections"]))
    lines += ["", "Guard errors (unique held-out inputs, independent of performance selection):", "",
              "- False-safe: %s / %s actually unsafe inputs." % (guard["false_safe"], guard["actual_unsafe"]),
              "- False-fallback: %s / %s actually safe inputs." % (guard["false_fallback"], guard["actual_safe"]),
              "- Unsafe among accepted: %s / %s accepted inputs." % (guard["false_safe"], guard["accepted"]), ""]
    if interval:
        lines += [f"Paired bootstrap 95% interval for static-BF16/adaptive latency ratio: **{interval[0]:.3f} to {interval[1]:.3f}**. Values below 1 favor static BF16.", ""]
    memory_row = rows[-1]
    lines += ["- Packed weights + FP32 group scales: %s bytes (vector capacity: %s bytes)." % (memory_row["weight_bytes"], memory_row["weight_capacity_bytes"]),
              "- Metadata reservation estimate: %s bytes; allocator bookkeeping is covered only by measured RSS." % memory_row["metadata_bytes"],
              "- Process peak RSS: %s bytes. This includes references, resident variants, caches and benchmark state." % max(int(r["peak_resident_bytes"]) for r in rows),
              "", "Timing fields are instrumented, non-overlapping components where practical. Dispatch is a residual that includes output validation and activation-buffer destruction. The FP32 reference has only a whole-invocation measurement.",
              "", "Seeded workloads repeat within the same process. Repeat on separate VM sessions before claiming a stable speedup.", ""]
    (path.parent / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", type=Path, nargs="?", default=Path("results"))
    a = p.parse_args()
    paths = [a.path] if a.path.is_file() else sorted(a.path.rglob("benchmark.csv"))
    if not paths:
        p.error("no benchmark.csv files found; run ./nm bench or ./nm all on the VM")
    for path in paths:
        summarize(path)
