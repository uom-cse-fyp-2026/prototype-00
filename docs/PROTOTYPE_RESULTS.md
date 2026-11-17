# Prototype results status

No build or test was performed on the Windows authoring PC, as requested.
No Ubuntu VM execution results have been received. This file contains no
fabricated latency, correctness, guard-accuracy or speedup measurements.

| Validation item | Current evidence |
|---|---|
| C++ / MLIR source | Implemented; source review only |
| Native Ubuntu build | Pending VM execution |
| torch-mlir frontend and compiler pass | Implemented; pending compilation/execution |
| Automated correctness tests | Written; not run |
| Numerical error / quality budget | Measurement code written; not measured |
| Calibration guard | Fit/evaluation code written; no fitted policy committed |
| Kernel and complete-invocation latency | Benchmark code written; not measured |
| Cache/fallback behavior | Test and demonstration source written; not executed |
| Reproducibility | CLI and provenance capture provided; end-to-end reproduction pending |

On the VM, run:

```bash
just all
```

The timestamped results directory contains:

- `calibration.csv`: descriptors and actual BF16/A8 errors against the W4 reference.
- `calibration_timings.csv`: complete forced-variant timing samples and components.
- `policy.nm`: versioned, checksummed, frozen shape/backend/environment-specific policy.
- `benchmark.csv`: raw held-out measurements for all five modes.
- `guard_metrics.json`: confusion counts and explicitly defined conditional rates.
- `summary.md` / `summary.json`: configuration, correctness/quality, latency,
  overhead, fallback, memory and paired bootstrap uncertainty summaries.
- `run.log`, environment metadata and compiler IR/manifests.

The summary distinguishes error from original FP32 weights from error caused
by activation precision relative to the actual W4 representation. Implementation
correctness is established by the test suite, not by meeting a quantization
quality threshold.

After a successful run, report the generated measured summary together with
CTest output. If any step fails, retain its log and fix that failure before
claiming a working prototype. `just collect results/<run-directory>` packages
the recorded artifacts for review.

## Findings that must remain visible

A8 may be slower than BF16 because of packing/unpacking, scaling or compiler
code quality. Inspection may erase a kernel gain. The sampled guard may miss
outliers. A full-scan guard may have high overhead. A policy may legitimately
select only BF16. Report these outcomes without retuning thresholds on held-out
data or calling forced A8 execution an adaptive success.

RSS is process-level and includes reference weights, variants, caches and
benchmark data. Packed-weight payload/capacity and metadata reservation are
separate quantities. The metadata reservation is an estimate and does not
measure allocator bookkeeping. The paired bootstrap uses distinct held-out
inputs within one VM session; independent sessions are still needed to assess
host scheduling variability.
