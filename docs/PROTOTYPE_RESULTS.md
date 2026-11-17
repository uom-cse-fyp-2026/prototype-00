# Prototype results status

No build or test was performed on the Windows authoring PC, as requested.
The supplied VM log for job `20261117T173353858666Z-all` confirms benchmark
execution completed and saved `experiment/benchmark.csv`. The job then failed
in Markdown report formatting because a literal `95%` was interpreted by
percent-style string interpolation. That formatting defect is corrected in
source; the existing CSV can be summarized again without repeating measurements.
Full raw artifacts and detailed test output have not yet been reviewed here.
No missing measurements are inferred or invented.

| Validation item | Current evidence |
|---|---|
| C++ / MLIR source | Implemented; source review only |
| Native Ubuntu build | VM workflow reached benchmark execution; full build log pending review |
| torch-mlir frontend and compiler pass | Integrated in the workflow; detailed compiler artifacts pending review |
| Automated correctness tests | VM workflow proceeded past its test stage; detailed results pending review |
| Numerical error / quality budget | Full numerical-error CSV pending review |
| Calibration guard | VM benchmark printed held-out guard counts; policy/artifacts pending review |
| Kernel and complete-invocation latency | Benchmark completion excerpt received; full CSV pending review |
| Cache/fallback behavior | Detailed trace pending review |
| Reproducibility | Original run failed in reporting; summary recovery available from saved CSV |

Recover the report for that existing run on the VM:

```bash
just summarize results/jobs/20261117T173353858666Z-all/experiment
```

This writes `summary.md` and `summary.json` from saved data. The original job's
failed status remains an accurate historical record of its reporting failure.

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
