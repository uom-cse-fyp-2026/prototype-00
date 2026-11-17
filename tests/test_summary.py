"""Synthetic report fixtures only; these values are not benchmark measurements."""
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from summarize import summarize


class SummaryTests(unittest.TestCase):
    def fixture(self, directory, samples=2):
        manifest = {"backend": "synthetic_test_fixture", "shape": {"batch": 1, "input": 4, "output": 2, "group": 2}}
        (directory / "benchmark_manifest.json").write_text(json.dumps(manifest))
        (directory / "guard_metrics.json").write_text(json.dumps({"false_safe": 0, "actual_unsafe": 0,
            "false_fallback": 0, "actual_safe": samples, "accepted": samples}))
        rows = []
        for sample in range(samples):
            for mode, latency in (("fp32", 30), ("static_bf16", 10), ("static_a8", 5), ("adaptive", 20), ("guard_disabled", 8)):
                row = {"sample": sample, "mode": mode, "total_us": latency * (sample + 1),
                    "selected": "w4_bf16", "cache_hit": 0, "cache_miss": 0, "invalidations": 0, "fallback": 0,
                    "resident_bytes": 1000, "peak_resident_bytes": 2000, "weight_bytes": 16,
                    "weight_capacity_bytes": 16, "metadata_bytes": 32}
                for metric in ("preparation_us", "kernel_us", "inspector_us", "validity_us", "decision_us", "dispatch_us", "error_w4", "error_fp32"):
                    row[metric] = 0
                rows.append(row)
        path = directory / "benchmark.csv"
        with path.open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_paired_report_writes_json_and_markdown_without_changing_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            csv_path = self.fixture(directory)
            original = csv_path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(csv_path)
            result = json.loads((directory / "summary.json").read_text())
            self.assertEqual(result["modes"]["static_bf16"]["mean_us"], 15)
            self.assertEqual(result["modes"]["adaptive"]["mean_us"], 30)
            self.assertEqual(result["adaptive_speedup_vs_static_bf16_paired_bootstrap_95_interval"], [.5, .5])
            markdown = (directory / "summary.md").read_text()
            self.assertIn("95% interval", markdown)
            self.assertIn("**0.500 to 0.500**", markdown)
            self.assertIn("| static_bf16 |", markdown)
            self.assertIn("| adaptive |", markdown)
            self.assertEqual(csv_path.read_bytes(), original)

    def test_single_input_reports_absent_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            csv_path = self.fixture(directory, samples=1)
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(csv_path)
            result = json.loads((directory / "summary.json").read_text())
            self.assertIsNone(result["adaptive_speedup_vs_static_bf16_paired_bootstrap_95_interval"])
            self.assertNotIn("95% interval", (directory / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
