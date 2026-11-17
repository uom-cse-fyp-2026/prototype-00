#!/usr/bin/env python3
"""Import the restricted PyTorch operator through torch-mlir's FX frontend."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import torch
from torch_mlir import fx


class LinearProjection(torch.nn.Module):
    def forward(self, x, w):
        return torch.mm(x, w.t())


def export(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = LinearProjection().eval()
    # Small representative graph; no checkpoint download or hidden model state.
    x = (torch.arange(32, dtype=torch.float32).reshape(2, 16) - 16) / 11
    w = (torch.arange(128, dtype=torch.float32).reshape(8, 16) - 64) / 71
    program = torch.export.export(model, (x, w), strict=True)
    calls = [str(node.target) for node in program.graph.nodes if node.op == "call_function"]
    if calls != ["aten.t.default", "aten.mm.default"]:
        raise RuntimeError("Unsupported exported graph: " + repr(calls))
    torch_ir = fx.export_and_import(program, output_type="torch", func_name="linear")
    torch_ir.operation.verify()
    linalg_ir = fx.export_and_import(program, output_type="linalg-on-tensors", func_name="linear")
    linalg_ir.operation.verify()
    text = str(linalg_ir)
    if "linalg.matmul" not in text:
        raise RuntimeError("Expected linalg.matmul from the linear frontend; inspect exported graph and toolchain version")
    (output / "linear.torch.mlir").write_text(str(torch_ir) + "\n", encoding="utf-8")
    (output / "linear.linalg.mlir").write_text(text + "\n", encoding="utf-8")
    contract = {"kind": "torch_mlir.fx", "operator": "linear_without_bias", "call_targets": calls,
                "linalg_sha256": hashlib.sha256(text.encode()).hexdigest(), "torch_version": torch.__version__,
                "example_shape": {"batch": 2, "input": 16, "output": 8},
                "transformation": "restricted projection template generalized to positive bounded runtime dimensions"}
    (output / "frontend.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    (output / "frontend_values.json").write_text(json.dumps({"x": x.tolist(), "w": w.tolist(),
        "expected": model(x, w).tolist()}, indent=2) + "\n", encoding="utf-8")
    return text, x, w


def check(output):
    # This command is only invoked by the VM CTest suite, never during Windows authoring.
    from check_frontend import check_frontend
    check_frontend(Path(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    export(args.output)
    if args.check:
        check(args.output)
