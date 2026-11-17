# Reuse the same layout as the earlier mlir-quant-linear project.
NM_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${TORCH_MLIR_SRC:-}" ]]; then
  if [[ -d "$HOME/tools/torch-mlir" ]]; then
    export TORCH_MLIR_SRC="$HOME/tools/torch-mlir"
  else
    export TORCH_MLIR_SRC="$NM_PROJECT_ROOT/.deps/torch-mlir"
  fi
fi
export TORCH_MLIR_BUILD="${TORCH_MLIR_BUILD:-$TORCH_MLIR_SRC/build}"
if [[ -f "$TORCH_MLIR_SRC/mlir_venv/bin/activate" ]]; then
  source "$TORCH_MLIR_SRC/mlir_venv/bin/activate"
fi
export PATH="$TORCH_MLIR_BUILD/bin:$PATH"
for NM_PYTHON_PACKAGE_DIR in \
  "$TORCH_MLIR_BUILD/tools/torch-mlir/python_packages/torch_mlir" \
  "$TORCH_MLIR_BUILD/tools/torch-mlir/python_packages/torch-mlir" \
  "$TORCH_MLIR_BUILD/python_packages/torch_mlir"; do
  if [[ -d "$NM_PYTHON_PACKAGE_DIR/torch_mlir" ]]; then
    export PYTHONPATH="$NM_PYTHON_PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"
    break
  fi
done
export LD_LIBRARY_PATH="$TORCH_MLIR_BUILD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

