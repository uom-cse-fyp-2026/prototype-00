#!/usr/bin/env bash
# This script is for the Ubuntu VM only. Existing toolchains are reused untouched.
set -euo pipefail
NM_SETUP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NM_TOOLCHAIN_SOURCE="${TORCH_MLIR_SRC:-$NM_SETUP_ROOT/.deps/torch-mlir}"
NM_TOOLCHAIN_BUILD="${TORCH_MLIR_BUILD:-$NM_TOOLCHAIN_SOURCE/build}"
NM_SETUP_JOBS="${JOBS:-2}"
if [[ $# -ne 0 ]]; then
  echo "setup takes no arguments; use JOBS=N and TORCH_MLIR_SRC/TORCH_MLIR_BUILD environment overrides" >&2
  exit 2
fi
if [[ -x "$NM_TOOLCHAIN_BUILD/bin/mlir-opt" && -f "$NM_TOOLCHAIN_BUILD/lib/cmake/mlir/MLIRConfig.cmake" ]]; then
  echo "Reusing existing toolchain: $NM_TOOLCHAIN_BUILD"
  echo "Source env.sh, then run ./nm doctor and ./nm all. No upstream checkout is changed."
  exit 0
fi
if [[ -e "$NM_TOOLCHAIN_SOURCE" ]]; then
  echo "An incomplete toolchain already exists at $NM_TOOLCHAIN_SOURCE."
  echo "Finish that build or set TORCH_MLIR_SRC to a new empty path. Existing files are preserved." >&2
  exit 1
fi
for NM_REQUIRED in git cmake ninja python3 c++ lld; do
  command -v "$NM_REQUIRED" >/dev/null || { echo "Missing $NM_REQUIRED; see docs/UBUNTU_20_04.md" >&2; exit 1; }
done
python3 -c 'import sys; assert sys.version_info >= (3,10), "Fresh torch-mlir setup needs Python >=3.10. Reuse the old toolchain or use ./nm setup --container on Ubuntu 20.04."'
NM_COMPILER_MAJOR="$(c++ -dumpversion | cut -d. -f1)"
if [[ "$NM_COMPILER_MAJOR" -lt 12 ]]; then
  echo "Fresh pinned LLVM needs a newer C++ compiler. Reuse your existing toolchain or use ./nm setup --container." >&2
  exit 1
fi
NM_UPSTREAM_REV="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch_mlir_revision"])' "$NM_SETUP_ROOT/toolchain.lock.json")"
mkdir -p "$(dirname "$NM_TOOLCHAIN_SOURCE")"
git clone --filter=blob:none https://github.com/llvm/torch-mlir.git "$NM_TOOLCHAIN_SOURCE"
git -C "$NM_TOOLCHAIN_SOURCE" checkout --detach "$NM_UPSTREAM_REV"
git -C "$NM_TOOLCHAIN_SOURCE" submodule update --init --depth=1 --progress
python3 -m venv "$NM_TOOLCHAIN_SOURCE/mlir_venv"
source "$NM_TOOLCHAIN_SOURCE/mlir_venv/bin/activate"
python -m pip install --upgrade pip
# Requirements are taken from the pinned upstream commit, never main at install time.
python -m pip install -r "$NM_TOOLCHAIN_SOURCE/requirements.txt"
cmake -S "$NM_TOOLCHAIN_SOURCE/externals/llvm-project/llvm" -B "$NM_TOOLCHAIN_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=torch-mlir -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$NM_TOOLCHAIN_SOURCE" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython3_EXECUTABLE="$VIRTUAL_ENV/bin/python" -DPython_EXECUTABLE="$VIRTUAL_ENV/bin/python" \
  -DTORCH_MLIR_ENABLE_STABLEHLO=OFF -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF \
  -DLLVM_ENABLE_LLD=OFF -DLLVM_USE_LINKER=lld
cmake --build "$NM_TOOLCHAIN_BUILD" --parallel "$NM_SETUP_JOBS"
echo "torch-mlir: $(git -C "$NM_TOOLCHAIN_SOURCE" rev-parse HEAD)"
echo "LLVM: $(git -C "$NM_TOOLCHAIN_SOURCE/externals/llvm-project" rev-parse HEAD)"
echo "Toolchain built. Run ./nm doctor, then ./nm all."
