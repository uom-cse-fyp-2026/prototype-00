# NeuroMinds Prototype 00

Prototype 00 evaluates adaptive activation quantization for a single CPU linear
projection, `Y = XWᵀ`. The runtime inspects each input and selects one of two
compiled executors. Both use the same packed 4-bit weights (W4).

| Executor | Activation storage | Arithmetic |
| --- | --- | --- |
| W4/A8 | Signed INT8 | INT32 group dot products, followed by FP32 scaling and accumulation |
| W4/BF16 | BF16 | FP32 arithmetic with dequantized W4 weights |

## Architecture

![Compiler preparation and offline calibration produce native kernels and a frozen policy.](docs/figures/compiler-and-calibration.svg)

The torch-mlir frontend imports and checks a bias-free PyTorch projection.
Templates emit the two quantized kernels. The custom MLIR pass validates their
interfaces and emits the risk predicate. LLVM compiles both kernels and the
predicate ahead of time. Offline calibration measures error and latency and saves
the activation scale, risk limits, and timing estimates in a frozen policy.

![The inspector measures the input, looks up a plan, and checks risk, cost, and plan validity before dispatch.](docs/figures/runtime-dispatch.svg)

Each call measures the current activations, looks up a cached plan, and checks
whether A8 meets the calibrated risk and cost criteria. A plan may be reused
after these checks. The executor uses BF16 when A8 is unsuitable. See the
[architecture notes](docs/ARCHITECTURE.md) for the cache, validity checks, and
fallback rules.

## Benchmark results

One Ubuntu 20.04 run on an Intel Xeon E7-4820 v3 used one CPU thread, batch size
1, 512 input features, 256 output features, and weight groups of 32. The policy
was fitted on 64 synthetic inputs. Evaluation used 16 separate synthetic
inputs with 15 timed repetitions per mode. All six automated tests in that run
passed.

| Mode | Mean latency (µs) | Mean relative L2 vs W4 reference |
| --- | ---: | ---: |
| FP32 reference | 475.52 | 9.656% |
| Static W4/BF16 | 2,853.68 | 0.156% |
| Static W4/A8 | 566.50 | 47.824% |
| Adaptive W4/A8 or W4/BF16 | 2,240.31 | 0.305% |
| Adaptive with risk guard disabled | 569.53 | 47.824% |

Adaptive execution reduced mean latency by **21.5%** compared with static
W4/BF16. The FP32 C++ reference was the fastest measured mode. The guard accepted
all four held-out inputs that met the 3% A8 error budget and rejected all twelve
that exceeded it.

The 3% budget uses the same W4 weights with FP32 activations as its reference.
Against the original FP32 projection, the adaptive output had **9.71%** mean
relative L2 error. These measurements cover one synthetic operator workload on
one CPU. Model quality, GPU performance, and energy use are outside this
evaluation.

[Run summary][summary] · [Raw measurements][benchmark] · [Build and test log][log]

## Run on an Ubuntu VM

The VM needs `just`, tmux, Git, curl, tar, bzip2, CA certificates, and system
Python 3.8 or later. See the [setup guide](docs/UBUNTU_20_04.md) for installation
and troubleshooting. Start the one-time toolchain setup from the repository root.

```bash
just setup-bg
just status
```

Setup builds LLVM, MLIR, and torch-mlir from source on the first run. It keeps
dependencies under `.deps/` and `.venv/` in this checkout. Once setup reports
`succeeded` with exit status `0`, launch the experiment.

```bash
just start
```

The experiment builds the prototype, runs tests, calibrates the policy,
demonstrates A8 → BF16 → A8 switching, and benchmarks held-out inputs. Both
commands run in tmux and continue when the SSH session closes.

| Command | Use |
| --- | --- |
| `just status` | Check the latest job and its exit status |
| `just logs` | Follow the job log |
| `just attach` | Reconnect to the tmux session. Press Ctrl+B, then D to detach. |
| `just stop` | Stop the active job and retain its logs |
| `just` | List all recipes |

Managed runs write to `results/jobs/<job-id>/experiment/` by default.
Start with `summary.md` for the report, `benchmark.csv` for individual
measurements, and `policy.nm` for the frozen calibration policy.

## Further reading

- [Numerical formats and error definitions](docs/NUMERICS.md)
- [Mapping to the NeuroMinds research methodology](docs/METHODOLOGY_MAPPING.md)

[summary]: results/jobs/20261117T173353858666Z-all/experiment/summary.md
[benchmark]: results/jobs/20261117T173353858666Z-all/experiment/benchmark.csv
[log]: results/jobs/20261117T173353858666Z-all/experiment/run.log
