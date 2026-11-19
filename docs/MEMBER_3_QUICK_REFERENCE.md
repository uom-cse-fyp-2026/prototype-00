# Member 3 quick reference

**Inspector and guarded executor | NeuroMinds | 19 November 2026**

## Say this first

> I own the runtime decision layer: inspect the current activations, apply the frozen risk and cost rules, construct or reuse a versioned plan, then validate and dispatch W4+A8 or W4+BF16. A cache hit still checks the current input. The guard predicts risk empirically; it does not certify output accuracy.

## The operation and formats

`Y[M,N] = X[M,K] * W[N,K]^T`, no bias. Default M=1, K=512, N=256, G=32.

| Path | Stored activations | Computation |
|---|---|---|
| W4+BF16 | 16-bit BF16 | FP32 multiply/accumulate |
| W4+A8 | INT8 [-127,127] | INT32 within group, then FP32 scaling/sum |

Both use the same packed W4 weights. BF16 fallback still retains W4 weight error.

## Memorize these rules

```text
risk = enabled AND max_abs <= max_limit AND clipping_fraction <= clip_limit

overhead = current_inspection_us + calibrated_decision_us
cost = A8_mean + overhead < BF16_mean * 0.95
       AND A8_p95 + overhead < BF16_p95

select A8 = supported AND risk AND cost
otherwise select BF16
```

Guard-disabled ablation bypasses risk, not cost/support/tensor checks. It does not force A8.

Risk labels: relative L2 error versus actual W4 reference <=0.03. This is not 3% task error. Runtime selection never reads the current reference-output error.

## Runtime sequence

```text
inspect -> invalidate/lookup cache -> recheck support/risk/cost
        -> reuse or update plan -> validate -> convert -> kernel -> check output
```

Descriptor: max, mean, p99, clipping (>clip), outlier proxy (>0.75*clip), count, bytes, sampled flag, bucket. Risk uses max and clipping only. Sampling is evenly spaced, up to 128 values; it can miss finite outliers.

PlanKey: shape, feature bucket, config hash, policy ID, backend, guard-disabled flag. Plan: key, variant, BF16 fallback, config version, activation-policy ID, checksum.

Cache: plans only; capacity 16 by default; FIFO, not LRU. It skips no current inspection or decision. No weight-content identity: weights must remain fixed for the policy.

## Your strongest example

`tests/test_core.cpp:191`: clip=4, exact max threshold=1.03. Inputs filled with 1.02 and 1.04 share the coarse bucket. The second hits the cache but fails the fresh exact guard and selects BF16. The test's cost values are artificial fixtures, not benchmark evidence.

## Outcomes to distinguish

| Situation | Outcome |
|---|---|
| Risk and cost accept, supported | A8 |
| Risk rejects / incompatible plan | BF16 fallback |
| Risk accepts, no cost advantage | BF16, normally no fallback flag |
| Bad/corrupt policy at load | Error |
| Malformed/nonfinite/domain-invalid tensor or output overflow | Error |

## Code navigation

| File | Anchors |
|---|---|
| `src/inspector/inspector.cpp` | 10 inspect, 45 risk, 54 cost, 60 key, 63 checksum |
| `src/runtime/policy.cpp` | 28 fit, 66 validate, 83 save, 91 load |
| `src/runtime/cache.cpp` | 3 constructor, 6 find, 18 put, 23 invalidate |
| `src/executor/executor.cpp` | 48 adaptive, 8 final execute |
| `src/kernels/kernels.cpp` | 99 prepare/dispatch/output check |
| `src/benchmark/experiments.cpp` | 100 calibrate, 186 demo, 209 benchmark |

## Demo on the Ubuntu VM

```bash
just status
just test
just metadata
just calibrate --results results/member3-review --policy results/member3-review/policy.nm
just demo --results results/member3-review --policy results/member3-review/policy.nm
```

Actual sequence: ordinary -> ordinary -> outlier -> ordinary -> ordinary, then forced BF16/A8. Read descriptor, cache, selected, fallback and reason together. Error printed by the demo is evaluation-only. All-BF16 adaptive selection is possible; explain its reason.

Use `just bench` with the same results/policy options, then `just summarize results/member3-review` for evaluation. Full `just run`/`just all` also collect build/test and compiler provenance. Match shape, backend, base seed and inspection mode to calibration.

## What to measure / what not to overclaim

Primary comparison: complete `total_us`. Also record inspection, validity/cache, decision, preparation, kernel and dispatch residual. Kernel-only wins can disappear after overhead.

Held-out rates: false-safe/actual-unsafe; false-fallback/actual-safe; unsafe-accepted/all-accepted. State denominators. Calibration constrains the last type of fraction, with default zero observed unsafe accepts and at least eight accepts.

Current implementation: C++ inspector, compiler-generated risk predicate, prepared numerical paths, frozen policy, FIFO cache and guarded dispatch. Future work: fused reductions, global MDE/state coordination, hysteresis/dwell, KV/weight-profile switching, full LLM quality/energy and JIT.

<5% controller overhead and >=1.2x efficiency are research targets, not verified results. The separate 5% cost-decision margin is not an overhead measurement.

**Evidence status:** preparation is source/document review only. No project build/tests/benchmarks were run for this guide; local raw results are absent. Show a verified VM run before claiming success or numerical performance.

## Handoff

> The runtime produces traceable decisions and itemized costs. Member 4 will show whether the measured risk discrimination and complete invocation time justify input-aware selection.

For foundations, the five-minute script, 40 viva answers, troubleshooting and full sources, read `docs/MEMBER_3_PREPARATION_GUIDE.md`.
