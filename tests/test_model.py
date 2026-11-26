"""Exercise model loading, calibration, evaluation and inference through the CLI."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile


def run(binary, *args, error=None):
    result = subprocess.run(
        [str(binary), *map(str, args)], check=False, capture_output=True, text=True
    )
    if error is None:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0 and error in result.stderr, result.stdout + result.stderr
    return result


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def check_workflow(binary, examples, work):
    model = examples / "tiny_mlp.txt"
    inputs = examples / "tiny_mlp_input.txt"
    results = work / "model-run"
    run(binary, "model-run", "--backend", "scalar", "--model", model, "--results", results,
        "--samples", 16, "--evaluation-samples", 4, "--repeats", 2, "--warmup", 0)

    benchmark = read_csv(results / "model_benchmark.csv")
    assert len(benchmark) == 4 * 2 * 4
    assert {row["mode"] for row in benchmark} == {"fp32", "static_bf16", "static_a8", "adaptive"}
    assert all(float(row["error_fp32"]) == 0 for row in benchmark if row["mode"] == "fp32")
    decisions = read_csv(results / "model_decisions.csv")
    assert len(decisions) == 4 * 2 * 2
    for row in benchmark:
        if row["mode"] == "adaptive":
            calls = [call for call in decisions if (call["sample"], call["repeat"]) ==
                     (row["sample"], row["repeat"])]
            assert {call["layer"] for call in calls} == {"0", "1"}
            assert sum(call["selected"] == "w4_a8" for call in calls) == int(row["a8_layers"])
    for layer in (0, 1):
        assert len(read_csv(results / f"layer-{layer}" / "calibration.csv")) == 16

    result = run(binary, "model-infer", "--results", results, "--input-file", inputs)
    actual = list(map(float, result.stdout.split()))
    # Hand-computed first layer after ReLU is [2, 3, 0]. The final affine
    # layer then gives [2.5, -6.5], allowing for W4 and chosen activation rounding.
    assert len(actual) == 2 and all(abs(a - b) < 0.25 for a, b in zip(actual, [2.5, -6.5])), actual
    assert "layer 0 w4_" in result.stderr and "layer 1 w4_" in result.stderr

    wrong_input = work / "wrong.txt"
    wrong_input.write_text("2 1 0 1 5\n", encoding="ascii")
    run(binary, "model-infer", "--results", results, "--input-file", wrong_input,
        error="unexpected values after complete model input")

    first_policy = results / "layer-0" / "policy.nm"
    original_policy = first_policy.read_bytes()
    first_policy.write_bytes((results / "layer-1" / "policy.nm").read_bytes())
    run(binary, "model-infer", "--results", results, "--input-file", inputs,
        error="saved policy does not match model layer")
    first_policy.write_bytes(original_policy)

    snapshot = results / "model.txt"
    snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    run(binary, "model-infer", "--results", results, "--input-file", inputs,
        error="does not match model snapshot")

    broken_model = work / "broken.txt"
    broken_model.write_text("NM_MODEL_V1 2\nlinear 1 1 identity no_bias\n1\n"
                            "linear 2 1 identity no_bias\n1 1\n", encoding="ascii")
    bad_run = work / "bad-run"
    run(binary, "model-run", "--backend", "scalar", "--model", broken_model,
        "--results", bad_run, error="adjacent model layer dimensions do not match")
    assert not bad_run.exists()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="nm-model-test-") as directory:
        check_workflow(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(directory))
    print("Model workflow passed")
