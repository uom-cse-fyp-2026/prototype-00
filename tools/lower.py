#!/usr/bin/env python3
"""Run the custom preparation pass, MLIR lowering, LLVM optimization and object emission."""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def lower(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    versions = {name: subprocess.check_output([str(tool), "--version"], text=True) for name, tool in
                (("mlir-opt", args.mlir_opt), ("mlir-translate", args.mlir_translate), ("llc", args.llc), ("opt", args.llvm_opt))}
    majors = [re.search(r"LLVM version\s+(\d+)", value) for value in versions.values()]
    if all(majors) and len({match.group(1) for match in majors}) != 1:
        raise RuntimeError("MLIR/LLVM tools have different major versions; use the matching torch-mlir build/bin")
    run([args.nm_opt, args.input, "--nm-prepare-executors", "-o", out / "executors.prepared.mlir"])
    passes = ["--canonicalize", "--cse", "--convert-scf-to-cf", "--expand-strided-metadata",
              "--finalize-memref-to-llvm", "--convert-arith-to-llvm", "--convert-index-to-llvm",
              "--convert-func-to-llvm", "--convert-cf-to-llvm", "--reconcile-unrealized-casts"]
    run([args.mlir_opt, out / "executors.prepared.mlir", *passes, "-o", out / "executors.llvm.mlir"])
    run([args.mlir_translate, "--mlir-to-llvmir", out / "executors.llvm.mlir", "-o", out / "executors.unoptimized.ll"])
    run([args.llvm_opt, "-passes=default<O3>", "-S", out / "executors.unoptimized.ll", "-o", out / "executors.ll"])
    # Generic x86-64 target by default. No -mcpu=native, CUDA, or assumed BF16 ISA.
    run([args.llc, "-O=3", "-filetype=obj", "-relocation-model=pic", out / "executors.ll", "-o", out / "executors.o"])
    (out / "compiler_manifest.json").write_text(json.dumps({"tools": versions, "lowering_passes": passes,
        "llvm_ir_optimization": "default<O3>", "llvm_object_optimization": "O3", "target_features": "generic target defaults",
        "compiler_elapsed_seconds": time.monotonic() - started, "object_bytes": (out / "executors.o").stat().st_size}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for option in ("input", "output", "nm-opt", "mlir-opt", "mlir-translate", "llc", "llvm-opt"):
        p.add_argument("--" + option, required=True)
    lower(p.parse_args())
