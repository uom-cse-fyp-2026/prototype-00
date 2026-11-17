"""Execute the torch-mlir FP32 export through MLIR's runner and compare with PyTorch."""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np


def check_frontend(output):
    values = json.loads((output / "frontend_values.json").read_text())
    text = (output / "linear.linalg.mlir").read_text()
    def dense(rows):
        return "[" + ", ".join("[" + ", ".join("%.9e" % x for x in row) + "]" for row in rows) + "]"
    main = """
  func.func private @printMemrefF32(tensor<*xf32>) attributes {llvm.emit_c_interface}
  func.func @main() {
    %x = arith.constant dense<X_VALUES> : tensor<2x16xf32>
    %w = arith.constant dense<W_VALUES> : tensor<8x16xf32>
    %y = call @linear(%x, %w) : (tensor<2x16xf32>, tensor<8x16xf32>) -> tensor<2x8xf32>
    %u = tensor.cast %y : tensor<2x8xf32> to tensor<*xf32>
    call @printMemrefF32(%u) : (tensor<*xf32>) -> ()
    return
  }
""".replace("X_VALUES", dense(values["x"])).replace("W_VALUES", dense(values["w"]))
    cut = text.find("{-#")
    head, tail = (text, "") if cut == -1 else (text[:cut], text[cut:])
    end = head.rstrip().rfind("}")
    source = output / "frontend.main.mlir"
    source.write_text(head[:end] + main + head[end:] + tail)
    opt = shutil.which("mlir-opt")
    runner = shutil.which("mlir-runner") or shutil.which("mlir-cpu-runner")
    if not opt or not runner:
        raise RuntimeError("matching mlir-opt and mlir-runner/mlir-cpu-runner must be on PATH")
    prefix = Path(os.environ.get("TORCH_MLIR_BUILD", str(Path(opt).resolve().parent.parent)))
    libs = [prefix / "lib" / "libmlir_runner_utils.so", prefix / "lib" / "libmlir_c_runner_utils.so"]
    if not all(lib.exists() for lib in libs):
        raise RuntimeError("Build MLIR runner utilities in the same toolchain: " + str(prefix))
    lowered = output / "frontend.llvm.mlir"
    passes = ["--canonicalize", "--one-shot-bufferize=bufferize-function-boundaries", "--convert-linalg-to-loops",
              "--lower-affine", "--convert-scf-to-cf", "--expand-strided-metadata", "--finalize-memref-to-llvm",
              "--convert-math-to-llvm", "--convert-arith-to-llvm", "--convert-index-to-llvm",
              "--convert-cf-to-llvm", "--convert-func-to-llvm", "--reconcile-unrealized-casts"]
    subprocess.run([opt, str(source), *passes, "-o", str(lowered)], check=True)
    result = subprocess.check_output([runner, str(lowered), "-e", "main", "-entry-point-result=void",
                                      "-shared-libs=" + ",".join(map(str, libs))], text=True)
    (output / "frontend.runner.txt").write_text(result)
    match = re.search(r"data\s*=\s*(\[.*\])", result, flags=re.S)
    if not match:
        raise RuntimeError("Cannot parse runner output; see " + str(output / "frontend.runner.txt"))
    actual = np.asarray(json.loads(match.group(1)), dtype=np.float32)
    expected = np.asarray(values["expected"], dtype=np.float32)
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=2e-5, atol=2e-5):
        raise AssertionError("torch-mlir -> MLIR -> LLVM execution disagrees with PyTorch")
    print("PASS: torch-mlir FP32 frontend executed through LLVM and agrees with PyTorch")
