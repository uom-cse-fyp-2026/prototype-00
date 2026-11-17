# Measured operator results

Backend: `mlir_llvm`; shape: `{'batch': 1, 'input': 512, 'output': 256, 'group': 32}`.

Preliminary VM measurements. Quantization quality is operator-level; no model perplexity or energy result is implied.

| Mode | Mean us | Median us | p95 us | SD us | Equivalent GOP/s | Mean error vs W4 | Mean error vs FP32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 475.52 | 538.23 | 580.97 | 131.95 | 0.551 | 0.096562 | 0 |
| static_bf16 | 2853.68 | 3281.11 | 3414.10 | 607.81 | 0.092 | 0.001555 | 0.096952 |
| static_a8 | 566.50 | 647.13 | 664.91 | 133.08 | 0.463 | 0.47824 | 0.50236 |
| adaptive | 2240.31 | 2086.48 | 3337.38 | 1088.20 | 0.117 | 0.003046 | 0.097052 |
| guard_disabled | 569.53 | 659.55 | 684.15 | 123.64 | 0.460 | 0.47824 | 0.50236 |

| Mode | Prep us | Kernel us | Inspector us | Validity/cache us | Decision us | Dispatch residual us | Cache hits/misses | Fallbacks | A8 selections |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.00 | n/a | 0.00 | 0.00 | 0.00 | 0.00 | 0/0 | 0 | 0 |
| static_bf16 | 14.13 | 2838.72 | 0.00 | 0.00 | 0.00 | 0.83 | 0/0 | 0 | 0 |
| static_a8 | 20.15 | 545.63 | 0.00 | 0.00 | 0.00 | 0.71 | 0/0 | 0 | 240 |
| adaptive | 13.75 | 2212.56 | 10.75 | 0.64 | 0.27 | 0.97 | 225/15 | 180 | 60 |
| guard_disabled | 19.82 | 536.78 | 10.49 | 0.54 | 0.14 | 0.69 | 225/15 | 0 | 240 |

Guard errors (unique held-out inputs, independent of performance selection):

- False-safe: 0 / 12 actually unsafe inputs.
- False-fallback: 0 / 4 actually safe inputs.
- Unsafe among accepted: 0 / 4 accepted inputs.

Paired bootstrap 95% interval for static-BF16/adaptive latency ratio: **1.058 to 1.673**. Values below 1 favor static BF16.

- Packed weights + FP32 group scales: 81920 bytes (vector capacity: 81920 bytes).
- Metadata reservation estimate: 6688 bytes; allocator bookkeeping is covered only by measured RSS.
- Process peak RSS: 21491712 bytes. This includes references, resident variants, caches and benchmark state.

Timing fields are instrumented, non-overlapping components where practical. Dispatch is a residual that includes output validation and activation-buffer destruction. The FP32 reference has only a whole-invocation measurement.

Seeded workloads repeat within the same process. Repeat on separate VM sessions before claiming a stable speedup.
