# NeuroMinds Prototype 1

A compiler-directed Inspector–Executor experiment for one CPU linear projection,
`Y = X W^T`. Both executable paths share genuinely packed group-wise W4 weights:

- **W4 + BF16:** actual 16-bit BF16 activation storage; FP32 arithmetic.
- **W4 + A8:** signed INT8 activations; signed unpacked INT4 values; INT32
  accumulation within each weight group, followed by FP32 scaling.

The full build uses **PyTorch → torch-mlir FX → Linalg frontend artifacts**, a
restricted packed-W4 MLIR emitter, a custom **`nm-prepare-executors` MLIR pass**,
and **MLIR → LLVM IR → native object** lowering. The custom pass validates the
executor ABI and generates the numerical-risk guard consumed at runtime.

**Status:** source implementation authored on Windows. No project build, test,
calibration, compiler pass execution, or benchmark was run on the authoring PC.
The commands below perform that validation on your Ubuntu VM. There are no
claimed passing results or performance numbers yet.

## Run using your existing toolchain

This repository reuses the environment layout from your `mlir-quant-linear`
project: `~/tools/torch-mlir`, `mlir_venv`, and `build/bin`. Each `./nm` invocation
loads `env.sh` automatically; you do not need to activate the venv manually.

```bash
git clone https://github.com/uom-cse-fyp-2026/prototype-00.git
cd prototype-00
./nm doctor
./nm all
```

For an existing checkout:

```bash
git pull --ff-only
./nm all
```

Override the old project's toolchain locations if necessary:

```bash
export TORCH_MLIR_SRC="$HOME/tools/torch-mlir"
export TORCH_MLIR_BUILD="$TORCH_MLIR_SRC/build"
./nm doctor
./nm all
```

`all` builds a Release executable and the MLIR pass, runs CTest, fits and freezes
a calibration policy, demonstrates ordinary → outlier → ordinary inputs, runs
held-out benchmarks, and summarizes the actual measurements. It stops on the
first failure. A failed build/test must be fixed before claiming results.

## Short commands

| Command | Purpose |
|---|---|
| `./nm doctor` | Record VM OS, CPU/ISA, RAM, compilers, tools, upstream revisions, optional NVIDIA visibility |
| `./nm build` | Build the full torch-mlir / MLIR / LLVM implementation |
| `./nm test` | Execute automated numerical, runtime, frontend and compiler tests |
| `./nm metadata` | Show compiled formats, configuration version and fallback |
| `./nm calibrate` | Fit scales, guard and measured costs; save `results/policy.nm` |
| `./nm demo` | Show adaptive decisions and separate forced-kernel executions |
| `./nm run --variant fp32` | Run the FP32 reference |
| `./nm run --variant w4-reference` | Isolate the actual W4 representation with FP32 activations |
| `./nm run --variant bf16` | Force BF16 (requires calibration policy) |
| `./nm run --variant a8 --regime outliers` | Force A8 on an outlier input; evaluation only |
| `./nm run --variant adaptive` | Execute the guarded runtime |
| `./nm bench` | Compare five modes on disjoint held-out inputs |
| `./nm summarize results` | Recompute summaries from recorded CSVs |
| `./nm all --sampled` | Separate experiment with sampled features and a separately calibrated policy |
| `./nm sweep` | Independently calibrate/evaluate batches 1, 4, 8, 16 |
| `./nm collect results/<run-directory>` | Package logs, provenance and measurements for review |

Example with different dimensions and more measurement repetitions:

```bash
./nm all --batch 4 --input 512 --output 256 --group 32 --repeats 30
```

Direct executable options include `--samples`, `--warmup`, `--repeats`,
`--seed`, `--policy`, `--results`, `--sampled`, and `--backend mlir|scalar`.
Calibration and evaluation must agree on shape, group size, base seed,
inspection mode, backend, code/compiler identity and generated configuration.
Changing them requires a new policy. Use `./nm all` to manage these together.

## Ubuntu 20.04 and fresh setup

Reuse the already working compiler environment first. Ubuntu 20.04's stock
Python/CMake/compiler may be too old to build a fresh recent torch-mlir tree.
`setup` preserves existing toolchains and does not install system packages.

If Docker is already available on the VM, an optional isolated compiler
environment avoids changing the host distribution:

```bash
./nm setup --container
./nm all --container
```

This builds a CPU toolchain from the torch-mlir revision in `toolchain.lock.json`
and its exact LLVM submodule. It is a substantial source build, not a model
download. Keep ample disk space and limit parallelism to fit VM RAM. See
[Ubuntu 20.04 setup](docs/UBUNTU_20_04.md) for native prerequisites, environment
overrides, container limitations and failure diagnostics.

## Compiler artifacts

The full build retains these files under `build/generated/`:

```text
linear.torch.mlir          torch-mlir import of the restricted PyTorch projection
linear.linalg.mlir         torch-mlir lowering of that operator
frontend.json              checked graph contract and IR hash
nm_config.hpp              validated C++ metadata, defaults and configuration hash
executors.mlir             packed-W4 BF16 / A8 numerical loops
executors.prepared.mlir    custom-pass approvals and generated nm_risk_guard
executors.llvm.mlir        lowered LLVM dialect
executors.ll               LLVM IR
executors.o                native object linked into neurominds_demo
build_manifest.json        configuration + frontend provenance
compiler_manifest.json     actual tool versions, lowering time and object size
```

The frontend deliberately recognizes only the specified bias-free projection.
The quantized loops are emitted from a restricted operator template after that
contract is checked. This is not a general transformation of arbitrary Linalg
graphs. The custom pass generates the guard and validates the executor family;
the activation descriptor reductions remain C++.

## Interpreting the experiment

Input values primarily affect quantization error. A conventional dense kernel
does not become faster just because its inputs contain fewer outliers. The
runtime selects a different implemented numerical path only if both its
calibrated risk predicate and measured total-cost test accept it.

The measured cost comparison includes activation preparation, inspection,
decision/cache costs and dispatch. It requires a configured margin and a p95
cost check. If A8 has no credible advantage, adaptive execution stays on BF16.
The demo still runs both kernels in clearly marked forced mode.

A cheap guard is an empirical predictor, not an accuracy certificate. Sampled
features can miss rare outliers. BF16 fallback still contains W4 weight error.
Held-out false-safe and false-fallback rates must be inspected before extending
the design. A speedup is not required for this milestone.

No CUDA dependency, downloaded LLM, adaptive JIT compilation, KV-cache
adaptation, full MDE hardware controller, or model-quality claim is included.
The NVIDIA GPU can be considered as a separate backend later.

## Project map

```text
compiler/                 custom MLIR driver and preparation/guard-generation pass
configs/                  approved family, quality budget, shape/memory limits
include/neurominds/       typed kernel, descriptor, plan, policy and runtime interfaces
src/quantization/         W4 packing, scales, BF16 and A8 conversion
src/kernels/              independent references and scalar/MLIR dispatch bridge
src/inspector/            full/sampled descriptors and risk/cost predicates
src/executor/             plan validation and guarded execution
src/runtime/              bounded cache, policy persistence and provenance
src/benchmark/            calibration, demonstrations and held-out experiments
tests/                    C++ numerical/runtime and Python compiler/frontend tests
tools/                    frontend, metadata/IR emitter, lowering, CLI and summary tools
scripts/                  toolchain setup and complete validation entry point
docs/                     architecture, numerical contract and research mapping
results/                  VM-produced measurements (ignored except README)
```

Read [architecture](docs/ARCHITECTURE.md), [numerical semantics](docs/NUMERICS.md),
[methodology mapping](docs/METHODOLOGY_MAPPING.md),
[results status](docs/PROTOTYPE_RESULTS.md) and
[supervisor demonstration](docs/SUPERVISOR_DEMO.md).

Upstream integration follows the [torch-mlir development guide](https://github.com/llvm/torch-mlir/blob/main/docs/development.md),
the [MLIR getting-started guide](https://mlir.llvm.org/getting_started/), and the
working environment/lowering conventions in your earlier `mlir-quant-linear` repository.
