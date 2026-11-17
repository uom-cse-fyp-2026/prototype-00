"""Configuration rejection tests run by CTest on the Ubuntu VM."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from generate import validate, generate


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/prototype.json").read_text())

    def test_valid_family(self):
        validate(self.config)

    def test_unsupported_variants_and_fallback(self):
        for edit in (lambda c: c.update(fallback_variant=2),
                     lambda c: c["variants"][0].update(approved=False),
                     lambda c: c["variants"][1].update(accumulator="f32"),
                     lambda c: c.update(weight_format="unpacked_int8")):
            c = copy.deepcopy(self.config)
            edit(c)
            with self.assertRaises(ValueError):
                validate(c)

    def test_invalid_numeric_fields(self):
        for key, value in (("sample_size", 1), ("cache_capacity", 0), ("error_budget", float("nan")),
                           ("relative_l2_epsilon", 0), ("configuration_version", True)):
            c = copy.deepcopy(self.config)
            c[key] = value
            with self.assertRaises(ValueError):
                validate(c)

    def test_unknown_fields_and_bad_shapes(self):
        for edit in (lambda c: c.update(typo=True), lambda c: c["defaults"].update(input=0),
                     lambda c: c["shape_limits"].update(input=3000000)):
            c = copy.deepcopy(self.config)
            edit(c)
            with self.assertRaises(ValueError):
                validate(c)

    def test_frontend_contract_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                generate(ROOT / "configs/prototype.json", tmp)
            bad = Path(tmp) / "frontend.json"
            bad.write_text(json.dumps({"operator": "arbitrary_graph", "call_targets": []}))
            with self.assertRaises(ValueError):
                generate(ROOT / "configs/prototype.json", tmp, bad)

    def test_version_changes_generated_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text(json.dumps(self.config))
            generate(config, root / "one", scalar_only=True)
            self.config["configuration_version"] += 1
            config.write_text(json.dumps(self.config))
            generate(config, root / "two", scalar_only=True)
            one = json.loads((root / "one/build_manifest.json").read_text())
            two = json.loads((root / "two/build_manifest.json").read_text())
            self.assertNotEqual(one["configuration_hash"], two["configuration_hash"])


if __name__ == "__main__":
    unittest.main()
