# Project-owned dependencies only. NM_ENV_ROOT is an explicit override used by
# the optional container image; no other project's environment is auto-detected.
NM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export NM_ENV_ROOT="${NM_ENV_ROOT:-$NM_PROJECT_ROOT}"
export TORCH_MLIR_SRC="$NM_ENV_ROOT/.deps/torch-mlir"
export TORCH_MLIR_BUILD="$TORCH_MLIR_SRC/build"
export NM_HOST_PREFIX="$NM_ENV_ROOT/.deps/host"
export NM_VENV="$NM_ENV_ROOT/.venv"
export MAMBA_ROOT_PREFIX="$NM_ENV_ROOT/.deps/mamba-root"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$NM_PROJECT_ROOT/.deps/pip-cache"
export XDG_CACHE_HOME="$NM_PROJECT_ROOT/.deps/cache"
export TORCH_HOME="$NM_PROJECT_ROOT/.deps/torch-cache"
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_CONSTRAINT

# These environment changes apply only to the nm child shell, not the SSH shell.
unset PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CMAKE_PREFIX_PATH
export CONDA_SHLVL=0
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin:$HOME/bin"
export PYTHONPATH=""
export LD_LIBRARY_PATH=""
export NM_PYTHON=/usr/bin/python3
NM_MICROMAMBA="$NM_ENV_ROOT/.deps/bootstrap/bin/micromamba"
if [[ -x "$NM_MICROMAMBA" && -f "$NM_HOST_PREFIX/conda-meta/history" ]]; then
  eval "$("$NM_MICROMAMBA" shell hook --shell bash)"
  micromamba activate "$NM_HOST_PREFIX"
fi
if [[ -f "$NM_VENV/bin/activate" ]]; then
  source "$NM_VENV/bin/activate"
  export NM_PYTHON="$NM_VENV/bin/python"
fi
export PATH="$TORCH_MLIR_BUILD/bin:$PATH"
for NM_PYTHON_PACKAGE_DIR in \
  "$TORCH_MLIR_BUILD/tools/torch-mlir/python_packages/torch_mlir" \
  "$TORCH_MLIR_BUILD/tools/torch-mlir/python_packages/torch-mlir" \
  "$TORCH_MLIR_BUILD/python_packages/torch_mlir"; do
  if [[ -d "$NM_PYTHON_PACKAGE_DIR/torch_mlir" ]]; then
    export PYTHONPATH="$NM_PYTHON_PACKAGE_DIR"
    break
  fi
done
export LD_LIBRARY_PATH="$TORCH_MLIR_BUILD/lib:$NM_HOST_PREFIX/lib"
