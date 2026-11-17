"""Check custom-pass rejection and generation using the VM-built compiler."""
import argparse
import subprocess
import tempfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nm-opt", required=True)
    p.add_argument("--generated", required=True)
    a = p.parse_args()
    source = (Path(a.generated) / "executors.mlir").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for label, text, success in (
            ("valid", source, True),
            ("wrong_variant", source.replace("nm.variant_id = 1", "nm.variant_id = 9"), False),
            ("wrong_format", source.replace('nm.weight_format = "packed_w4"', 'nm.weight_format = "unpacked"'), False),
            ("missing_executor", source.replace("@nm_a8(", "@missing_a8("), False),
        ):
            path = root / (label + ".mlir")
            path.write_text(text)
            result = subprocess.run([a.nm_opt, str(path), "--nm-prepare-executors"], text=True, capture_output=True)
            if (result.returncode == 0) != success:
                raise AssertionError(label + ": unexpected compiler status\n" + result.stderr)
            if success and ("@nm_risk_guard" not in result.stdout or "nm.approved" not in result.stdout):
                raise AssertionError("preparation did not generate the guard/approval attributes")
            print("PASS compiler contract: " + label)


if __name__ == "__main__":
    main()
