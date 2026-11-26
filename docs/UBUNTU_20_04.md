# Isolated Ubuntu 20.04 setup and persistent jobs

All setup, builds and tests run on the VM. None are executed on the Windows
authoring PC. This prototype does not auto-detect or reuse another project's
virtual environment or `~/tools/torch-mlir` checkout.

## One-time host prerequisites

The VM needs Git, curl, tar/bzip2, certificates, system Python 3.8 or newer for
the launcher, and tmux for persistent sessions:

```bash
sudo apt-get update
sudo apt-get install -y git curl bzip2 ca-certificates tmux
```

These host utilities are separate from the ML/Python/compiler dependencies.
The setup script does not execute sudo or alter system Python/GCC.

Install `just` if it is missing. This installs a prebuilt binary in your
user directory on Ubuntu 20.04.

```bash
mkdir -p "$HOME/.local/bin"
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh \
  | bash -s -- --to "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"
```

Keep the PATH entry in `~/.bashrc` for future SSH sessions.

## Start setup, disconnect, return

From the already cloned repository:

```bash
git pull --ff-only
just setup
```

`setup` starts the dependency installation and LLVM/torch-mlir source build
inside this checkout's dedicated tmux server. Press **Ctrl+B, then D** to
detach. You can close the SSH connection afterward. To return:

```bash
cd ~/prototype-00
just attach
```

For immediate background execution instead of attaching:

```bash
just setup-bg
```

Check the recorded outcome with `just status`. Detaching or closing the tmux
client is not evidence of successful setup. Wait for **succeeded, exit status
0** before starting the experiment. After an interrupted/failed setup, the
script reuses completed package stages and the incremental LLVM build. It does
not delete the build directory or overwrite a checkout with tracked changes.

`just test-jobs` runs small mocked control-plane tests on the VM without
downloading the compiler stack. These check duplicate-job guards, argument
handling and PID-reuse safety; reconnect behavior still requires a real tmux run.

If an older setup stopped after a successful package transaction with
`The following argument was not expected: --explicit`, update the repository
and restart `just setup-bg`. The corrected exporter uses
`micromamba env export --explicit`, preserves the installed host environment,
and replaces the package lockfile only after a successful, nonempty export.

## Dependency locations

```text
prototype-00/
  .venv/                       Python packages for this prototype only
  .deps/host/                  Python 3.11, GCC 13, CMake, Ninja, linker/sysroot
  .deps/bootstrap/             micromamba binary, package lists and hashes
  .deps/mamba-root/            project-local dependency cache
  .deps/torch-mlir/             pinned source with exact LLVM submodule
  .deps/torch-mlir/build/       project's LLVM/MLIR tools and bindings
  .nm/                        local setup/job control state
  build/                      prototype binary and generated compiler artifacts
  results/jobs/<job-id>/       console log and authoritative exit-status record
```

`configs/host-environment.yml` specifies a conda-forge host environment. A
project-local micromamba downloads the modern compiler/Python tools needed on
Ubuntu 20.04. It does not install a global conda environment, modify `.bashrc`,
or upgrade `/usr/bin/gcc`. Its package cache is local as well. The setup records
exact resolved package URLs/builds in `host-explicit-linux-64.txt` and Python
versions in `python-freeze.txt`.

pip, Torch and general runtime caches also live under this checkout's `.deps`.
The nm subprocess does not load user pip configuration or inherit another
project's pip index/constraint overrides.

`env.sh` loads this project's host environment, then `.venv`, then its MLIR
bindings. It clears inherited Python/compiler-path settings inside the `nm`
child process, without changing the parent SSH shell. Ambient `TORCH_MLIR_SRC`
or `TORCH_MLIR_BUILD` from older instructions are not reused. `NM_ENV_ROOT` is
an explicit alternate root for the optional image; native runs use this checkout.

Fresh setup requires network access and substantial disk space for LLVM. It
uses four compile jobs by default and one linker job. For a smaller machine:

```bash
JOBS=2 just setup
```

### Faster compilation on a larger VM

`setup-bg` changes how setup is launched, not how fast it compiles. Default
setup uses four build jobs and one concurrent linker. On the supplied
80-logical-CPU / 62-GiB machine, start with:

```bash
just setup-fast 16 2
```

This is detached setup with 16 total build jobs and two concurrent link jobs.
If memory remains comfortable, `just setup-fast 24 2` uses more CPU parallelism.
The explicit values are saved in the job record and forwarded to the worker
even if a tmux server already exists. `just status` shows the requested limits.
More jobs do not guarantee proportional speedup; heavy compiler/link processes
also consume RAM and memory bandwidth.

To change limits on an already running setup, use `just stop`, wait until
`just status` shows it has stopped, update the checkout, then launch the faster
setup. Completed object files and installed packages are reused; an in-progress
compilation may need to be repeated. Never run two Ninja builds in the same
build directory concurrently. Only compile/link parallelism is changed; the
Release build, assertions and numerical behavior are retained.

These limits speed up compilation, not network downloads or package solving.
`just status`/`just logs` show whether setup is fetching packages/sources or
compiling C++. The NVIDIA GPU does not accelerate this host C++ toolchain build.

See [LLVM build configuration](https://llvm.org/docs/CMake.html) for its build
parallelism and linker-pool controls.

The torch-mlir revision is fixed in `toolchain.lock.json`, along with the
bootstrap version; LLVM follows that revision's submodule. If a pinned PyTorch
nightly wheel is unavailable, setup fails visibly. The exact toolchain pair
must be resolved explicitly rather than silently substituting another version.

## Run and manage experiments

After setup succeeds:

```bash
just run
```

This is the complete build/test/calibration/demo/benchmark workflow in tmux,
matching the interaction pattern in `quant-sensitivity`. For a detached start:

```bash
just start
```

| Command | Behavior |
|---|---|
| `just attach` / `just fg` | Return to the latest job's terminal |
| `just status` | Show actual job state, exit status and the last log lines |
| `just logs` | Follow the latest log; Ctrl+C closes only the viewer |
| `just jobs` | List recorded jobs for this checkout |
| `just stop` | Stop this project's active native process group; retain logs |
| `just background build` | Start another supported task detached |
| `just background sweep` | Run the batch sweep detached |
| `just pull` | Update source, refusing while a project job is running |
| `just setup-fg` | Foreground dependency setup with its actual exit code |
| `just all` | Foreground complete experiment with its actual exit code |

Only one managed job runs per checkout. The runner uses a tmux socket derived
from this checkout's absolute path and tmux's default key bindings (no shared
tmux config); other projects' tmux sessions are untouched. Job records from
another checkout are rejected before sending process-control signals.
Replacement jobs are created before old completed sessions are removed, so
tmux is not restarted while its last session is being deleted. A retained
placeholder pane is configured before the worker starts. Launch errors and
their exit status are saved in the job log and shown by `just status`.
Native children run in a dedicated process group so stopping a job can stop
compiler/kernel subprocesses too. Exit codes and failure logs persist after
the pane closes. Reboots, host shutdowns, and administrator process cleanup
still terminate jobs; tmux protects against SSH disconnection, not reboot.

`just run` now means the full experiment. The previous single-projection
command is `just projection --variant a8`, or use the internal `./nm run` CLI.

## Results and troubleshooting

A default tmux experiment writes measurements under
`results/jobs/<job-id>/experiment/`, next to `console.log` and `state.json`.
`just status` always identifies the latest log path. Use
`just collect results/jobs/<job-id>` to package the job and results for review.

If `doctor` reports system Python and missing torch/MLIR before setup, that is
an uninitialized project environment. Run `just setup`; installing torch into
system Python or sourcing another project's venv is not the fix.

Native foreground commands reject conflicting active jobs. Build/test failures
stop the complete workflow and are recorded as failed rather than completed.
For an optional container, use `just background setup --container`, then
`just background all --container` after setup succeeds. Docker must already be
installed. The image has its own prototype stack, and uses `build-container/`
to avoid native CMake-cache conflicts.

`just build --scalar-only` isolates C++ numerical problems from MLIR issues;
it is not evidence that the full compiler prototype passed. Sanitizers cover
the C++ runtime, not the separately emitted LLVM object. VM execution and
disconnect/reconnect verification are still required for this source update.

References: [micromamba installation](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html),
[tmux getting started](https://github.com/tmux/tmux/wiki/Getting-Started).
