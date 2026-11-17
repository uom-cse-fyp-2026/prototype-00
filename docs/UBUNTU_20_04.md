# Ubuntu 20.04 VM setup

All commands in this document are intended to be run by you on the VM. The
Windows authoring session does not invoke these builds or tests.

## Reuse the existing environment first

The earlier `mlir-quant-linear` project uses:

```text
~/tools/torch-mlir/
  mlir_venv/bin/python
  build/bin/mlir-opt
  build/bin/mlir-translate
  build/bin/llc
  build/lib/cmake/mlir/MLIRConfig.cmake
  build/tools/torch-mlir/python_packages/torch_mlir/torch_mlir/
```

The new `env.sh` supports that package layout and the alternative hyphenated
package directory used in upstream documentation. It preserves an existing
`PYTHONPATH` and uses the LLVM libraries from the same build. It does not modify
the old repository or compiler checkout.

```bash
just doctor
just all
```

Do not mix a distribution MLIR with torch-mlir built against another LLVM
revision. C++ APIs, dialect syntax, pass names and Python bindings can differ.
The optional new setup pins torch-mlir and follows its exact LLVM submodule;
an existing working build is reused and its actual revisions are recorded.

## Existing environment overrides

```bash
export TORCH_MLIR_SRC=/path/to/torch-mlir
export TORCH_MLIR_BUILD=/path/to/torch-mlir/build
just doctor
just build
just test
```

The C++ prototype uses C++17. Root CMake needs 3.20 or newer and Ninja. Existing
torch-mlir determines the required Python/compiler versions for its own bindings
and headers. Unless you set `CC`/`CXX`, the CLI reuses compiler paths from the
upstream build's `CMakeCache.txt` when available. The project's standard-library-only scripts support Python 3.8;
the pinned fresh upstream toolchain needs newer Python and C++ tooling.

## Optional isolated compiler image

Ubuntu 20.04 stock packages are not sufficient for a fresh recent LLVM build.
If Docker is already installed and usable by your VM account:

```bash
just setup --container
just doctor --container
just all --container
```

The image uses Ubuntu 24.04 userspace, Python development headers and a modern
compiler, then builds the pinned torch-mlir/LLVM pair from source. The host
remains Ubuntu 20.04. The project directory is mounted and results are written
with your VM UID/GID. No GPU passthrough is needed. The compiler build uses two
parallel jobs by default; LLVM source/build storage can be large. Inspect VM
free RAM/disk before beginning a fresh toolchain build.

Container builds use `build-container/` so their CMake cache does not conflict
with a native `build/` directory. Compiler inspection commands should use that
directory when running the container workflow.

Docker installation is a separate host administration action, not hidden in
`just setup`. If it is absent, follow your VM administrator's installation
process. Do not add random package repositories just to make a stock focal
compiler appear newer.

## Fresh native setup

If you already have a suitable modern native environment (Python >=3.10, a
GCC-compatible C++ compiler satisfying the pinned LLVM requirements, CMake
>=3.20, Ninja, Git, LLD, development headers and certificates):

```bash
export TORCH_MLIR_SRC="$PWD/.deps/torch-mlir"
export TORCH_MLIR_BUILD="$TORCH_MLIR_SRC/build"
JOBS=2 just setup
just all
```

The setup does not execute `sudo` or install host packages. It refuses to
overwrite a partial/existing checkout. Requirements come from the pinned
upstream commit. If a pinned nightly PyTorch wheel is no longer available,
setup fails visibly. Resolve the toolchain pair explicitly; do not silently
substitute another torch/MLIR version and claim the same configuration.

## Diagnosing failures

- Missing `torch_mlir`: check the venv and Python package directory in `env.sh`
  against the old project's working environment.
- Missing `mlir-opt`, `llc`, or MLIR CMake package: set `TORCH_MLIR_BUILD` to the
  matching source build, not an unrelated `/usr/bin` installation.
- Unknown pass or C++ API errors: keep the error and toolchain revisions from
  `doctor`; the source integration has not yet been compiled on your VM.
- Linker OOM: use `JOBS=1 just build` and ensure RAM/swap are adequate. The
  custom MLIR driver links upstream dialect/conversion libraries.
- Missing MLIR runner or runner utilities during frontend tests: build them in
  that same upstream toolchain. The old `mlir-quant-linear` runner path uses
  these libraries too.
- Stale policy/code/compiler mismatch: recalibrate, preferably with `just all`.
- Failed numerical tests: stop benchmarking, keep CTest output and share it.

`just build --scalar-only` followed by `just test` can isolate C++ numerical
problems from toolchain issues. Scalar-only mode is explicitly labeled and is
not evidence that the MLIR/LLVM prototype passed. Rebuild full mode with
`just build` afterward.

`just build --sanitizers` instruments the C++ runtime with ASan/UBSan. The
separately emitted LLVM kernel object is not sanitizer-instrumented; numerical
reference comparisons and ABI tests remain necessary.

For complete reports, use `just collect results/<run-directory>` after a run.
Build failures before a run directory exists should be shared with their full
terminal output and `results/environment.json`.
