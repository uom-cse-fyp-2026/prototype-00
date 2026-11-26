# NeuroMinds Prototype 1

A compiler-directed Inspector–Executor experiment for one CPU linear projection,
`Y = X W^T`. Both paths share real packed group-wise W4 weights:

- **W4 + BF16:** BF16 activation storage with FP32 arithmetic.
- **W4 + A8:** INT8 activations, unpacked signed INT4 weights, INT32 per-group
  accumulation, followed by FP32 scaling.

The full build uses a torch-mlir FX frontend, generated packed-W4 MLIR loops,
a custom `nm-prepare-executors` MLIR pass that generates the runtime risk guard,
and MLIR → LLVM IR → native object lowering. Activation inspection, typed plans,
bounded caching and conservative fallback are implemented in C++.

**Status:** no builds or tests run on the Windows authoring PC. The supplied VM
workflow has now reached benchmark execution and saved its CSV measurements.
It then failed in report formatting; that defect is corrected in source, and
the report can be regenerated from the existing run without repeating the
benchmark. Detailed raw artifacts and full validation output remain to be reviewed.

## Project-local dependencies

This checkout owns its environment. No other project's venv or
`~/tools/torch-mlir` is automatically reused:

```text
.venv/                         prototype Python packages
.deps/host/                    Python 3.11, GCC 13, CMake, Ninja and linker
.deps/torch-mlir/               pinned torch-mlir source + matching LLVM submodule
.deps/torch-mlir/build/         prototype LLVM/MLIR tools and Python bindings
.deps/bootstrap/               local micromamba, exact package lists and hashes
.deps/mamba-root/              local package cache
```

Ubuntu 20.04's stock GCC/Python remain installed. Setup obtains a newer
user-space toolchain through project-local micromamba, creates `.venv`, installs
the pinned upstream Python requirements there, then builds this project's own
LLVM/MLIR/torch-mlir. It does not change shell startup files or system packages.
Build parallelism defaults to four jobs with one linker job.
For a larger VM, `just setup-fast 16 2` starts detached setup with 16 build jobs
and two concurrent link jobs. See the setup guide before changing an active job.

## First setup on the VM

After cloning this repository using your configured GitHub authentication:

```bash
cd ~/prototype-00
git pull --ff-only
sudo apt-get update
sudo apt-get install -y git curl bzip2 ca-certificates tmux
just setup
```

`just setup` opens the installation in a project-specific tmux session. Press
**Ctrl+B, then D**, then close SSH if needed. For an immediate detached start,
use `just setup-bg` instead.

On reconnect:

```bash
cd ~/prototype-00
just status
just attach
```

Wait until setup reports **succeeded, exit status 0**. Detaching is not setup
completion. The first LLVM source build can take substantial time and disk;
later setup attempts reuse completed stages and the incremental build.

## Run the experiment and disconnect safely

After setup succeeds:

```bash
just run
```

This builds the prototype, runs tests, calibrates a frozen policy, demonstrates
ordinary → outlier → ordinary inputs, benchmarks disjoint held-out inputs and
summarizes measured results. It stops at a failed step. It runs in tmux, matching
the workflow in the `quant-sensitivity` reference justfile.

Use `just start` to launch the same workflow detached and return immediately.
Use `just attach` to return, `just status` for the saved exit code and log tail,
or `just logs` to follow output without attaching. Ctrl+C in `just logs` closes
the viewer and leaves the job running. `just stop` stops this project's native
job and its child process group, retaining logs.

tmux protects against SSH disconnection, not VM shutdown/reboot or administrator
process cleanup. The runner uses a dedicated socket per checkout and does not
touch other projects' sessions. Only one managed job is allowed per checkout.

## Commands

| Command | Purpose |
|---|---|
| `just` | List commands |
| `just doctor` | Inspect project dependency status, OS, CPU/ISA, RAM, tools and GPU visibility |
| `just setup` / `just setup-bg` | Set up isolated dependencies, attached/detached |
| `just setup-fast 16 2` | Detached setup using more CPU parallelism |
| `just run` / `just start` | Full experiment, attached/detached |
| `just attach` / `just fg` | Reconnect to the latest job |
| `just status` / `just logs` / `just jobs` | Inspect actual state, output and history |
| `just stop` | Stop this project's job, retaining its logs |
| `just pull` | Update source, refusing while a project job is active |
| `just setup-fg` / `just all` | Foreground setup/full workflow with direct exit codes |
| `just build` / `just test` | Foreground build and automated correctness tests |
| `just calibrate` / `just demo` / `just bench` | Foreground individual experiment stages |
| `just background build` | Run another supported task detached |
| `just background sweep` | Independently calibrate/evaluate batches 1, 4, 8, 16 in the background |
| `just projection --variant fp32` | Run one original FP32 reference projection |
| `just projection --variant w4-reference` | Reference using actual W4 storage and FP32 activations |
| `just projection --variant bf16` | Force BF16 with a calibrated policy |
| `just projection --variant a8 --regime outliers` | Force A8; not evidence of adaptive selection |
| `just projection --variant adaptive` | Run one guarded projection |
| `just metadata` | Show compiled formats/configuration |
| `just summarize results` | Summarize recorded measurements |
| `just collect results/jobs/<job-id>` | Package logs, provenance and results for review |

`just run` now means the full workflow. The previous single-projection command
is `just projection`. The internal environment-aware CLI remains `./nm`.

Arguments are preserved, including paths with spaces:

```bash
just start --batch 4 --input 512 --output 256 --group 32 --repeats 30
```

Full/sampled inspection, shape, backend and base seed must match calibration.
Use a new job/policy when changing these. `just run --sampled` performs a separate
sampled-inspection experiment. Foreground commands remain available for debugging.

If `just` itself is missing on Ubuntu 20.04:

```bash
mkdir -p "$HOME/.local/bin"
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
  | bash -s -- --to "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"
```

Keep that PATH entry in `~/.bashrc` for future sessions. See the
[official installer](https://just.systems/man/en/pre-built-binaries.html).

## Results and compiler artifacts

### Run a small model

The model workflow accepts a chain of up to 16 dense Linear layers, each with
optional bias and ReLU, and at most 1,000,000 parameters. It packs the supplied
weights into W4 and calibrates each layer using activations from the upstream
BF16 model. Evaluation uses fresh synthetic inputs and propagates the actual
chosen outputs between layers. The compiler still prepares the existing linear
kernels ahead of time. The C++ runtime composes those kernels into the model.

On the configured Ubuntu VM, try the included two-layer example:

```bash
just build
just model-run --model examples/tiny_mlp.txt --results results/tiny-mlp --repeats 5
just model-infer --results results/tiny-mlp --input-file examples/tiny_mlp_input.txt
```

Choose a fresh results directory per run. Use `--backend scalar` on
`model-run` for a scalar-only build. Inference loads the saved model snapshot
and layer policies, then takes whitespace-separated FP32 input values in
row-major order. The input file must contain exactly batch × first-layer-width
values. It prints output rows on standard output and layer decisions on
standard error. Build changes require calibrating a fresh run.

To supply your own model, use the helper from your model's Python code at the
repository root, after loading its weights:

```python
from tools.export_model import export_sequential

# model is a torch.nn.Sequential of Linear and optional ReLU modules.
export_sequential(model.eval(), "my-model.txt")
```

The text format starts with `NM_MODEL_V1 <layer-count>`. Each layer starts
with `linear <input-width> <output-width> <relu|identity> <bias|no_bias>`,
followed by the FP32 weight matrix in output-row-major order and, if declared,
one bias per output feature.

The results directory contains `model_benchmark.csv` for complete-forward
latency and relative output error against the original FP32 model,
`model_decisions.csv` for each adaptive layer's choice and reason, and
`layer-*/` for the original calibration artifacts and frozen policies.
The 3% calibration budget applies to the individual linear operation against
its W4 reference. Final output error is measured separately. Synthetic
evaluation characterizes this small model on these inputs. Transformer graphs,
tokenization, and task accuracy are outside this workflow.

Default managed runs save their console log and actual exit code in
`results/jobs/<job-id>/`. Full experiments put measurements in its `experiment/`
subdirectory: raw calibration/benchmark CSVs, frozen policy, guard metrics,
environment and compiler provenance, and `summary.md` / `summary.json`.
Foreground `just all` uses a timestamped results directory. Results are ignored
by Git; no fabricated measurements are committed.

The full build retains `linear.torch.mlir`, `linear.linalg.mlir`, `frontend.json`,
`nm_config.hpp`, `executors.mlir`, `executors.prepared.mlir`,
`executors.llvm.mlir`, optimized LLVM IR and the native object under
`build/generated/`. Exact resolved host packages and Python versions are
retained and copied into successful experiment provenance.

The frontend recognizes only the specified bias-free projection. Quantized
loops are emitted from a restricted template after that contract is checked.
The real preparation pass validates the executor ABI and emits the risk guard;
it does not transform arbitrary Torch/Linalg graphs or generate descriptor
reductions. This is ahead-of-time preparation, not adaptive JIT compilation.

## Research interpretation

The guard estimates activation quantization risk from calibration-only data.
It never reads the current reference-output error. A8 must also have a measured
total-cost advantage, including inspection/decision overhead, before selection.
The adaptive policy may legitimately retain BF16 for every input. Both kernels
are demonstrated separately in forced mode.

Compare numerical error against both actual W4 storage (activation error) and
original FP32 weights (combined error). BF16 fallback retains W4 weight error.
Cheap statistics do not certify per-input quality, and sampling can miss rare
outliers. Report false-safe and false-fallback rates on held-out inputs.

There is no promised speedup, full LLM inference, KV-cache adaptation, global
MDE controller, runtime JIT or model-quality claim. The NVIDIA GPU is detected
for provenance but is not required by this CPU prototype.

Details: [Ubuntu setup and disconnect/reconnect](docs/UBUNTU_20_04.md),
[architecture](docs/ARCHITECTURE.md), [numerical semantics](docs/NUMERICS.md),
[methodology mapping](docs/METHODOLOGY_MAPPING.md),
[results status](docs/PROTOTYPE_RESULTS.md), and
[supervisor demonstration](docs/SUPERVISOR_DEMO.md).
