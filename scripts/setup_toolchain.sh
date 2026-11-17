#!/usr/bin/env bash
# Ubuntu VM only. Dependencies are installed inside this project, not globally.
set -euo pipefail
NM_SETUP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NM_ENV_ROOT="${NM_ENV_ROOT:-$NM_SETUP_ROOT}"
NM_SETUP_JOBS="${JOBS:-4}"
if [[ $# -ne 0 || ! "$NM_SETUP_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "setup takes no arguments; set JOBS=N to control build parallelism" >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This pinned host environment targets x86-64 Linux." >&2
  exit 1
fi
for NM_REQUIRED in git curl tar bzip2 /usr/bin/python3; do
  command -v "$NM_REQUIRED" >/dev/null || {
    echo "Missing $NM_REQUIRED. On the VM: sudo apt-get install git curl bzip2 ca-certificates tmux" >&2
    exit 1
  }
done
NM_BOOTSTRAP="$NM_ENV_ROOT/.deps/bootstrap"
NM_MICROMAMBA="$NM_BOOTSTRAP/bin/micromamba"
export MAMBA_ROOT_PREFIX="$NM_ENV_ROOT/.deps/mamba-root"
NM_HOST_PREFIX="$NM_ENV_ROOT/.deps/host"
NM_SOURCE="$NM_ENV_ROOT/.deps/torch-mlir"
NM_BUILD="$NM_SOURCE/build"
NM_TARGET_VENV="$NM_ENV_ROOT/.venv"
mkdir -p "$NM_BOOTSTRAP/bin" "$NM_ENV_ROOT/.nm"
rm -f "$NM_ENV_ROOT/.nm/setup-complete.json"
NM_UPSTREAM_REV="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch_mlir_revision"])' "$NM_SETUP_ROOT/toolchain.lock.json")"
NM_MAMBA_VERSION="$(/usr/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["micromamba_version"])' "$NM_SETUP_ROOT/toolchain.lock.json")"

if [[ ! -x "$NM_MICROMAMBA" ]]; then
  echo "Downloading project-local micromamba $NM_MAMBA_VERSION"
  curl --proto '=https' --tlsv1.2 --fail --location --retry 3 \
    "https://micro.mamba.pm/api/micromamba/linux-64/$NM_MAMBA_VERSION" \
    -o "$NM_BOOTSTRAP/micromamba.tar.bz2"
  tar -xjf "$NM_BOOTSTRAP/micromamba.tar.bz2" -C "$NM_BOOTSTRAP" bin/micromamba
  sha256sum "$NM_BOOTSTRAP/micromamba.tar.bz2" > "$NM_BOOTSTRAP/micromamba.sha256"
fi
NM_HOST_SPEC_HASH="$(sha256sum "$NM_SETUP_ROOT/configs/host-environment.yml" | cut -d' ' -f1)"
if [[ ! -f "$NM_HOST_PREFIX/conda-meta/history" || ! -x "$NM_HOST_PREFIX/bin/python" || ! -f "$NM_BOOTSTRAP/host-spec.sha256" || "$(cat "$NM_BOOTSTRAP/host-spec.sha256")" != "$NM_HOST_SPEC_HASH" ]]; then
  echo "Installing project Python 3.11 and GCC 13 (system GCC is untouched)"
  NM_MAMBA_ACTION=create
  [[ ! -f "$NM_HOST_PREFIX/conda-meta/history" ]] || NM_MAMBA_ACTION=install
  NM_HOST_SPEC="$NM_SETUP_ROOT/configs/host-environment.yml"
  if [[ -f "$NM_BOOTSTRAP/host-spec.sha256" && "$(cat "$NM_BOOTSTRAP/host-spec.sha256")" == "$NM_HOST_SPEC_HASH" && -s "$NM_BOOTSTRAP/host-explicit-linux-64.txt" ]]; then
    NM_HOST_SPEC="$NM_BOOTSTRAP/host-explicit-linux-64.txt"
  fi
  "$NM_MICROMAMBA" --no-rc "$NM_MAMBA_ACTION" --yes --prefix "$NM_HOST_PREFIX" \
    --override-channels --channel conda-forge --strict-channel-priority \
    --file "$NM_HOST_SPEC"
  # Package installation succeeded even if a later provenance export fails.
  printf '%s\n' "$NM_HOST_SPEC_HASH" > "$NM_BOOTSTRAP/host-spec.sha256.tmp"
  mv "$NM_BOOTSTRAP/host-spec.sha256.tmp" "$NM_BOOTSTRAP/host-spec.sha256"
fi
# Micromamba's explicit format belongs to `env export`, not `list`.
# Export on retries too, and never truncate a previous usable lockfile on failure.
"$NM_MICROMAMBA" --no-rc env export --prefix "$NM_HOST_PREFIX" --explicit > "$NM_BOOTSTRAP/host-explicit-linux-64.txt.tmp"
/usr/bin/python3 - "$NM_BOOTSTRAP/host-explicit-linux-64.txt.tmp" <<'PY'
from pathlib import Path
import sys
lines = [line.strip() for line in Path(sys.argv[1]).read_text().splitlines()
         if line.strip() and not line.lstrip().startswith('#')]
if len(lines) < 2 or lines[0] != '@EXPLICIT':
    raise SystemExit('Micromamba did not export a nonempty explicit package list; existing lockfile preserved.')
PY
mv "$NM_BOOTSTRAP/host-explicit-linux-64.txt.tmp" "$NM_BOOTSTRAP/host-explicit-linux-64.txt"
source "$NM_SETUP_ROOT/env.sh"

if [[ ! -d "$NM_SOURCE" ]]; then
  git clone --filter=blob:none https://github.com/llvm/torch-mlir.git "$NM_SOURCE"
  git -C "$NM_SOURCE" checkout --detach "$NM_UPSTREAM_REV"
elif [[ ! -d "$NM_SOURCE/.git" ]]; then
  echo "Not a Git checkout: $NM_SOURCE. Existing files are preserved." >&2
  exit 1
elif [[ "$(git -C "$NM_SOURCE" rev-parse HEAD)" != "$NM_UPSTREAM_REV" ]]; then
  echo "Project toolchain is at another revision; refusing an automatic reset: $NM_SOURCE" >&2
  exit 1
fi
if [[ -n "$(git -C "$NM_SOURCE" status --porcelain --untracked-files=no)" ]]; then
  echo "The local upstream checkout has tracked changes; preserving them and stopping." >&2
  exit 1
fi
git -C "$NM_SOURCE" submodule update --init --depth=1 --progress
NM_NEW_VENV=0
if [[ ! -f "$NM_TARGET_VENV/bin/activate" ]]; then
  "$NM_HOST_PREFIX/bin/python" -m venv "$NM_TARGET_VENV"
  NM_NEW_VENV=1
fi
source "$NM_SETUP_ROOT/env.sh"
NM_REQUIREMENTS_HASH="$(sha256sum "$NM_SOURCE/requirements.txt" "$NM_SOURCE/pytorch-requirements.txt" "$NM_SOURCE/build-requirements.txt" "$NM_SOURCE/test-requirements.txt" | sha256sum | cut -d' ' -f1)"
if [[ "$NM_NEW_VENV" == 1 || ! -f "$NM_BOOTSTRAP/python-spec.sha256" || "$(cat "$NM_BOOTSTRAP/python-spec.sha256")" != "$NM_REQUIREMENTS_HASH" ]]; then
  "$NM_PYTHON" -m pip install --upgrade pip
  "$NM_PYTHON" -m pip install -r "$NM_SOURCE/requirements.txt"
  "$NM_PYTHON" -m pip freeze > "$NM_BOOTSTRAP/python-freeze.txt"
  printf '%s\n' "$NM_REQUIREMENTS_HASH" > "$NM_BOOTSTRAP/python-spec.sha256"
fi
cmake -S "$NM_SOURCE/externals/llvm-project/llvm" -B "$NM_BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DLLVM_TARGETS_TO_BUILD=host -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_EXTERNAL_PROJECTS=torch-mlir -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="$NM_SOURCE" \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_FIND_VIRTUALENV=ONLY \
  -DPython3_EXECUTABLE="$NM_PYTHON" -DPython_EXECUTABLE="$NM_PYTHON" \
  -DTORCH_MLIR_ENABLE_STABLEHLO=OFF -DTORCH_MLIR_ENABLE_PYTORCH_EXTENSIONS=OFF \
  -DLLVM_PARALLEL_LINK_JOBS=1 -DLLVM_ENABLE_LLD=OFF -DLLVM_USE_LINKER=lld
cmake --build "$NM_BUILD" --parallel "$NM_SETUP_JOBS"
source "$NM_SETUP_ROOT/env.sh"
"$NM_PYTHON" -c 'import torch, torch_mlir; print("torch:", torch.__version__); print("torch_mlir:", torch_mlir.__file__)'
"$NM_PYTHON" - "$NM_ENV_ROOT/.nm/setup-complete.json" "$NM_SOURCE" <<'PY'
import datetime, json, subprocess, sys
from pathlib import Path
target, source = map(Path, sys.argv[1:])
target.write_text(json.dumps({
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "torch_mlir_revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
    "llvm_revision": subprocess.check_output(["git", "-C", str(source / "externals/llvm-project"), "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.executable,
}, indent=2) + "\n")
PY
echo "Project-local setup completed. Next: just run (tmux) or just all (foreground)."
