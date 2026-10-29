"""Offline acceptance tests for immutable reference configuration."""

from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vortex.executor.reference_config import ReferenceConfig, validate_reference_config


PIN = "c1899de289a04d12100db370d81485cdf75e47ca"


def valid_config(**changes: object) -> ReferenceConfig:
    """Build an intended CPU configuration without qualifying an environment."""
    fields = dict(model_id="Qwen/Qwen3-0.6B", revision=PIN, tokenizer_revision=PIN,
                  profile_id="windows-cpu", device="cpu", dtype="float32")
    fields.update(changes)
    return ReferenceConfig(**fields)


class ReferenceConfigTests(unittest.TestCase):
    def test_valid_pinned_configuration_is_immutable(self):
        config = valid_config()
        self.assertIsNone(validate_reference_config(config))
        self.assertEqual(config.max_new_tokens, 32)
        with self.assertRaises(FrozenInstanceError):
            config.revision = "main"

    def test_reject_mutable_revision(self):
        for revision in ("main", "latest", "v1.0", PIN[:12], PIN.upper(), " " + PIN,
                         "0" * 40, None, 12, True):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                valid_config(revision=revision)

    def test_model_tokenizer_revision_match(self):
        for revision in ("main", "0" * 40, None):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                valid_config(tokenizer_revision=revision)

    def test_model_repository_is_exact(self):
        for model in ("Qwen3-0.6B", "Qwen/Qwen3-0.6B ", "other/model", None):
            with self.subTest(model=model), self.assertRaises(ValueError):
                valid_config(model_id=model)

    def test_prompt_output_bounds(self):
        for prompt, output in ((1, 1), (32, 2), (512, 64)):
            with self.subTest(prompt=prompt, output=output):
                self.assertIsNone(validate_reference_config(valid_config(
                    max_prompt_tokens=prompt, max_new_tokens=output)))
        for name, bad in (("max_prompt_tokens", 513), ("max_new_tokens", 65),
                          ("max_prompt_tokens", 0), ("max_new_tokens", 0),
                          ("max_new_tokens", -1), ("max_prompt_tokens", 512.0),
                          ("max_new_tokens", True), ("max_new_tokens", "32")):
            with self.subTest(name=name, bad=bad), self.assertRaises(ValueError):
                valid_config(**{name: bad})

    def test_profile_dtype_device_contract(self):
        self.assertIsNone(validate_reference_config(valid_config(
            profile_id="t4-fp16", device="cuda:0", dtype="float16")))
        for changes in ({"profile_id": "auto"}, {"device": "cuda:0"},
                        {"dtype": "float16"}, {"device": "auto"},
                        {"profile_id": []}, {"device": None},
                        {"profile_id": "t4-fp16", "device": "cuda:1", "dtype": "float16"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                valid_config(**changes)

    def test_seed_is_explicit_reference_seed(self):
        for seed in (1, -1, True, 0.0, "0", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                valid_config(seed=seed)

    def test_validator_rejects_non_config(self):
        for value in (None, asdict(valid_config()), "config"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_reference_config(value)

    def test_import_without_inference_packages(self):
        # A fresh no-site interpreter proves validation needs neither installed
        # inference packages nor a network connection, even when they exist here.
        script = f'''
import importlib.abc
import sys
sys.path.insert(0, {str(ROOT / "src")!r})
class NoInference(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {{"torch", "transformers", "numpy", "huggingface_hub"}}:
            raise AssertionError("inference import attempted")
sys.meta_path.insert(0, NoInference())
def audit(event, args):
    if event.startswith("socket.") or event in {{"subprocess.Popen", "os.system"}}:
        raise AssertionError("external side effect attempted")
sys.addaudithook(audit)
from vortex.executor.reference_config import ReferenceConfig, validate_reference_config
from vortex.executor.reference_manifest import canonical_digest, validate_manifest
c = ReferenceConfig("Qwen/Qwen3-0.6B", {PIN!r}, {PIN!r}, "windows-cpu", "cpu", "float32")
validate_reference_config(c)
assert len(canonical_digest({{"schema_version": 1}})) == 64
sys.path.insert(0, {str(ROOT / "tests" / "unit")!r})
from test_reference_manifest import manifest
validate_manifest(manifest())
validate_manifest(manifest(with_numerical=True))
print("offline-ok")
'''
        result = subprocess.run([sys.executable, "-S", "-c", script], capture_output=True,
                                text=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "offline-ok\n")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
