# Mapping to the NeuroMinds methodology

This maps the source implementation to the supplied project proposal, the
Inspector–Executor Supervisor Review methodology and the Prototype 1 brief.
Implementation status means code exists; execution validation is pending the
Ubuntu VM run.

| Research component | Prototype implementation | Boundary |
|---|---|---|
| Phase I: numerical characterization | Real packed W4, BF16/A8 conversions, FP32/W4/per-variant references, calibration errors | Synthetic operator inputs; no real model sensitivity or perplexity |
| Phase I: hardware characterization | Whole-operator and kernel timing, process RSS, format payloads, toolchain manifest | One CPU backend per policy; no reliable power/thermal/counter claim |
| Phase I: quality-safe family | Fixed error budget and calibration-only threshold fitting | Empirical guard; no per-input accuracy certificate |
| Phase II: PyTorch import | torch.export and torch-mlir FX produce Torch/Linalg IR and checked linear contract | Only bias-free projection; not arbitrary graph compilation |
| Phase II: specialization | Python emits packed-W4 MLIR loops and generated C++ metadata | Restricted dynamic-shape operator templates |
| Phase II: compiler pass | Custom MLIR pass validates executor ABI, annotates approvals/version and emits risk guard | Does not rewrite arbitrary matmuls or generate feature reductions |
| Phase II: native preparation | MLIR lowering → LLVM IR → native object linked with C++ | Ahead-of-time compiled variants; no adaptive JIT |
| Phase III: late-bound information | Full/sampled C++ activation descriptors | Operator inputs; no full device-monitoring controller |
| Phase III: plan production | Typed legal variant, scale-policy ID, shape/feature key, version and checksum | Two supported paths only |
| Phase III: guarded execution | Current guard, policy/shape/backend/version checks and BF16 fallback | Invalid tensors are rejected rather than silently reinterpreted |
| Phase III: reuse | Bounded FIFO cache; exact current guard checked on cache hits | Inspection remains per invocation and is charged |
| Evaluation | Five execution modes, held-out risk errors, raw timing samples, summaries and paired uncertainty | Operator-level feasibility and overhead, not end-to-end LLM benefit |

## Relation to the original MDE design

The original proposal retains system-level monitoring, configuration selection,
hysteresis, dwell time and legal switching boundaries. This prototype provides
the local input-aware plan/executor mechanism that could later sit under that
controller. It does not replace MDE by renaming the same loop.

Current hardware measurements are environment/provenance and calibration
inputs. They are not live hardware-state buckets fed into a global adaptation
policy. Policy files reject changes in recorded CPU/OS/code/compiler identity;
contention and VM host changes can still invalidate cost estimates and require
recalibration.

## Relation to staged feasibility gates

- G0 / Prototype A: test both numerical paths and safe plan transitions.
- G1 / Prototype B: measure whether descriptors separate quantization risk on
  disjoint held-out synthetic inputs.
- Partial G2 / Prototype C: measure complete-invocation overhead, cache reuse
  and benefit/non-benefit on this operator. Fused inspection remains future work.
- G3: full-model quality/latency/energy and comparison to original MDE remain
  future experiments.

Source implementation alone does not establish any gate as passed.

## Deferred capabilities

Real 1B/3B LLM import and traces; layer/block coordination; fused inspectors;
real graph-to-quantized-kernel MLIR transformations; sensitivity/Pareto search;
MDE telemetry and hysteresis; weight-profile changes; KV-cache block formats;
LLVM ORC JIT; online kernel generation/cache; GPU execution; AArch64 transfer;
model perplexity/tasks; energy and physical edge-hardware evaluation.

The supervisor memorandum's improvement and overhead targets remain research
targets. A correct implementation with no speedup, a weak guard or no useful
A8 selection is a valid feasibility finding and must be reported as such.
