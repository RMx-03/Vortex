"""Synthetic evidence only; no pretrained tokens or qualified environments."""

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vortex.executor.reference_manifest import canonical_digest, validate_manifest
from test_reference_config import valid_config


def config_payload():
    """Independent literal contract for the approved non-thinking greedy mode."""
    return {
        "schema_version": 1,
        "config": asdict(valid_config()),
        "chat_template": {"tokenize": True, "add_generation_prompt": True,
                          "enable_thinking": False},
        "generation": {
            "do_sample": False, "num_beams": 1, "num_beam_groups": 1,
            "num_return_sequences": 1, "max_new_tokens": 32, "min_new_tokens": 0,
            "use_cache": True, "eos_token_id": [151645, 151643],
            "pad_token_id": 151643, "bos_token_id": 151643,
            "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "length_penalty": 1.0,
            "return_dict_in_generate": True, "output_scores": False, "output_logits": False,
            "output_attentions": False, "output_hidden_states": False,
            "temperature": 1.0, "top_p": 1.0, "top_k": 50,
            "use_model_defaults": False, "forward_kwargs": {"logits_to_keep": 1},
            "cache_strategy": "fresh_dynamic", "custom_logits_processors": [],
            "custom_stopping_criteria": [], "streamer": None, "assistant_model": None,
            "custom_generate": None, "max_time": None,
        },
    }


def resign(record):
    """Recompute hashes so malformed-field tests reach semantic validation."""
    payload = {key: record[key] for key in
               ("schema_version", "config", "chat_template", "generation")}
    record["config_sha256"] = canonical_digest(payload)
    record["comparison_key_sha256"] = None if record["numerical"] is None else canonical_digest(
        {"config_sha256": record["config_sha256"], "numerical": record["numerical"]})
    return record


def manifest(*, with_numerical=False):
    """Create a draft or a synthetic numerical record, never real execution evidence."""
    record = config_payload()
    record.update(
        numerical=None, comparison_key_sha256=None, outputs=None,
        complete=False, outcome="configured",
        provenance={
            "source_sha": "a" * 40, "dirty": False, "patch_sha256": None,
            "adapter_version": "reference-schema-v1",
            "run_id": "00000000-0000-4000-8000-000000000001",
            "session_id": "00000000-0000-4000-8000-000000000002",
            "timestamp": "2026-10-29T12:00:00Z",
        })
    if with_numerical:
        controls = {"use_model_defaults", "forward_kwargs", "cache_strategy",
                    "custom_logits_processors", "custom_stopping_criteria", "streamer",
                    "assistant_model", "custom_generate"}
        effective = {key: deepcopy(value) for key, value in record["generation"].items()
                     if key not in controls}
        effective.update(max_length=20, transformers_version="4.53.3")
        record["numerical"] = {
            "effective_generation_config": effective,
            "assets": {
                "model_config_sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
                "generation_config_sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
                "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
                "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
                "tokenizer_assets_sha256": "b" * 64,
                "weights_sha256": "c" * 64,
            },
            "prompt": {
                "messages": [{"role": "user", "content": "Reply with the single word vortex."}],
                "messages_sha256": canonical_digest({"messages": [
                    {"role": "user", "content": "Reply with the single word vortex."}]}),
                "rendered_prompt": "synthetic prompt, not a Qwen rendering",
                "rendered_sha256": hashlib.sha256(
                    b"synthetic prompt, not a Qwen rendering").hexdigest(),
                "input_ids": [1, 2, 3],
            },
            "environment": {
                "python_version": "3.11.5", "python_build": "synthetic-build",
                "platform": "Windows-synthetic", "machine": "AMD64",
                "packages": {
                    "torch": {"version": "2.10.0+cpu", "wheel_sha256": "d" * 64},
                    "transformers": {"version": "4.53.3", "wheel_sha256": "e" * 64},
                },
                "hardware": {"device": "cpu", "gpu_name": None, "compute_capability": None,
                             "driver_version": None, "cuda_runtime": None},
            },
            "determinism": {
                "seed": 0, "pythonhashseed": "0", "cublas_workspace_config": ":4096:8",
                "cudnn_benchmark": False, "allow_tf32_matmul": False,
                "allow_tf32_cudnn": False, "deterministic_algorithms": True,
            },
        }
    return resign(record)


class CanonicalDigestTests(unittest.TestCase):
    def test_stable_canonical_serialization(self):
        # The expected byte string is independent of the serializer under test.
        expected_bytes = b'{"a":1,"b":["\xc3\xa9",true,null,1.0]}'
        expected = hashlib.sha256(expected_bytes).hexdigest()
        self.assertEqual(canonical_digest({"b": ["é", True, None, 1.0], "a": 1}), expected)
        self.assertEqual(canonical_digest({"a": 1, "b": ["é", True, None, 1.0]}), expected)
        self.assertNotEqual(canonical_digest({"a": [1, 2]}), canonical_digest({"a": [2, 1]}))

    def test_reject_noncanonical_or_malformed_values(self):
        cycle = {}
        cycle["cycle"] = cycle
        for value in (None, [], {1: "bad"}, {"a": (1, 2)}, {"a": {1, 2}},
                      {"a": float("nan")}, {"a": float("inf")}, {"a": object()},
                      {"a": "\ud800"}, {"a": "x" * 1048577}, cycle):
            with self.subTest(kind=type(value).__name__), self.assertRaises(ValueError):
                canonical_digest(value)

    def test_stable_digest_across_fresh_process(self):
        script = f'''
import sys
sys.path.insert(0, {str(ROOT / "src")!r})
from vortex.executor.reference_manifest import canonical_digest
print(canonical_digest({{"b": [2, 3], "a": 1}}))
'''
        result = subprocess.run([sys.executable, "-S", "-c", script], capture_output=True,
                                text=True, timeout=15, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), canonical_digest({"a": 1, "b": [2, 3]}))

    def test_oversized_shared_structure_is_rejected_during_snapshot(self):
        class CountedLeaf(Mapping):
            visits = 0

            def __iter__(self):
                self.visits += 1
                if self.visits > 64:
                    raise AssertionError("oversized structure expanded before rejection")
                return iter(("text",))

            def __len__(self):
                return 1

            def __getitem__(self, key):
                return "x" * 65536

        leaf = CountedLeaf()
        shared = leaf
        for _ in range(8):
            shared = [shared, shared]
        with self.assertRaises(ValueError):
            canonical_digest({"shared": shared})

    def test_lazy_mapping_keys_are_budgeted_before_full_iteration(self):
        class LazyMapping(Mapping):
            def __iter__(self):
                for index in range(1000):
                    if index > 64:
                        raise AssertionError("unbudgeted key iteration")
                    yield str(index)

            def __len__(self):
                return 1000

            def __getitem__(self, key):
                return "x" * 1048576

        with self.assertRaises(ValueError):
            canonical_digest(LazyMapping())

    def test_large_integer_budget_is_enforced_before_serialization(self):
        # Instrument the allocation boundary: no huge decimal string should be
        # built merely to discover that the encoded-size ceiling was exceeded.
        with patch("vortex.executor.reference_manifest.json.dumps",
                   side_effect=AssertionError("oversized serialization attempted")):
            with self.assertRaises(ValueError):
                canonical_digest({"items": [10 ** 3999] * 1000})


class ReferenceManifestTests(unittest.TestCase):
    def test_valid_draft_and_synthetic_numerical_manifest(self):
        for record in (manifest(), manifest(with_numerical=True)):
            before = deepcopy(record)
            self.assertIsNone(validate_manifest(record))
            self.assertEqual(record, before)

    def test_manifest_schema_and_missing_fields(self):
        valid = manifest(with_numerical=True)
        for key in valid:
            record = deepcopy(valid)
            del record[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_manifest(record)
        for version in (0, 2, "1", True, None):
            record = manifest()
            record["schema_version"] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                validate_manifest(record)

    def test_unknown_fields_are_rejected(self):
        for path in ((), ("config",), ("generation",), ("chat_template",),
                     ("provenance",), ("numerical",), ("numerical", "environment"),
                     ("numerical", "assets"), ("numerical", "prompt"),
                     ("numerical", "determinism")):
            record = manifest(with_numerical=True)
            target = record
            for key in path:
                target = target[key]
            target["unexpected"] = "unsupported"
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_manifest(resign(record))

    def test_malformed_manifest_fields(self):
        for key in ("config", "generation", "chat_template", "provenance", "numerical"):
            for bad in ([], "bad", 1, True):
                record = manifest()
                record[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    validate_manifest(record)
        for bad in (None, [], "bad"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_manifest(bad)

    def test_reject_sampling_inheritance_and_unsupported_generation(self):
        for key, bad in (("do_sample", True), ("temperature", .6), ("top_p", .95),
                         ("top_k", 20), ("num_beams", 2), ("num_beams", True),
                         ("use_model_defaults", True), ("use_cache", False),
                         ("forward_kwargs", {"logits_to_keep": 0}),
                         ("custom_logits_processors", ["processor"]),
                         ("custom_stopping_criteria", ["stop"]),
                         ("max_time", 10), ("max_new_tokens", 64),
                         ("cache_strategy", "static"), ("custom_generate", "remote/code")):
            record = manifest()
            record["generation"][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                validate_manifest(resign(record))

    def test_stop_and_non_thinking_policy(self):
        for key, bad in (("eos_token_id", [151645]), ("eos_token_id", [151643]),
                         ("eos_token_id", [151643, 151645]), ("pad_token_id", 0),
                         ("bos_token_id", None)):
            record = manifest()
            record["generation"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(resign(record))
        for key, bad in (("enable_thinking", True), ("tokenize", False),
                         ("add_generation_prompt", False)):
            record = manifest()
            record["chat_template"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(resign(record))

    def test_stable_configuration_identity(self):
        original = manifest()
        reordered = json.loads(json.dumps(original, sort_keys=True))
        self.assertIsNone(validate_manifest(reordered))
        self.assertEqual(original["config_sha256"], reordered["config_sha256"])
        changed = deepcopy(original)
        changed["config"]["max_new_tokens"] = 64
        changed["generation"]["max_new_tokens"] = 64
        resign(changed)
        self.assertIsNone(validate_manifest(changed))
        self.assertNotEqual(original["config_sha256"], changed["config_sha256"])

    def test_run_provenance_is_not_numerical_identity(self):
        original = manifest(with_numerical=True)
        record = deepcopy(original)
        record["provenance"].update(
            source_sha="f" * 40, dirty=True, patch_sha256="a" * 64,
            adapter_version="next-adapter", timestamp="2026-10-29T13:00:00Z",
            run_id="00000000-0000-4000-8000-000000000003",
            session_id="00000000-0000-4000-8000-000000000004")
        self.assertIsNone(validate_manifest(record))
        self.assertEqual(original["config_sha256"], record["config_sha256"])
        self.assertEqual(original["comparison_key_sha256"], record["comparison_key_sha256"])

    def test_numerical_changes_require_new_key(self):
        original = manifest(with_numerical=True)
        for path, value in ((("prompt", "input_ids"), [1, 2, 4]),
                            (("assets", "weights_sha256"), "d" * 64),
                            (("environment", "python_build"), "different-build")):
            record = deepcopy(original)
            record["numerical"][path[0]][path[1]] = value
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_manifest(record)
                resign(record)
                self.assertIsNone(validate_manifest(record))
                self.assertNotEqual(original["comparison_key_sha256"],
                                    record["comparison_key_sha256"])

    def test_full_effective_generation_config_is_recorded_in_numerical_identity(self):
        original = manifest(with_numerical=True)
        self.assertIsNone(validate_manifest(original))
        changed = deepcopy(original)
        changed["numerical"]["effective_generation_config"]["max_length"] = 21
        with self.assertRaises(ValueError):
            validate_manifest(changed)
        resign(changed)
        self.assertIsNone(validate_manifest(changed))
        self.assertEqual(original["config_sha256"], changed["config_sha256"])
        self.assertNotEqual(original["comparison_key_sha256"], changed["comparison_key_sha256"])
        for key, bad in (("do_sample", True), ("top_k", 20), ("temperature", .6),
                         ("eos_token_id", [151645]), ("transformers_version", "unknown")):
            record = deepcopy(original)
            record["numerical"]["effective_generation_config"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(resign(record))
        missing = deepcopy(original)
        del missing["numerical"]["effective_generation_config"]["do_sample"]
        with self.assertRaises(ValueError):
            validate_manifest(resign(missing))

    def test_digest_tampering_is_rejected(self):
        for key in ("config_sha256", "comparison_key_sha256"):
            for bad in ("0" * 64, "BAD", None, 1):
                record = manifest(with_numerical=True)
                record[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    validate_manifest(record)

    def test_provenance_is_required_and_typed(self):
        for key, bad in (("source_sha", "main"), ("dirty", 0), ("patch_sha256", "a" * 64),
                         ("timestamp", "2026-02-30T12:00:00Z"),
                         ("timestamp", "2026-10-29T12:00:00"),
                         ("run_id", "account-name"), ("session_id", ""),
                         ("adapter_version", "")):
            record = manifest()
            record["provenance"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(record)
        record = manifest()
        record["provenance"]["dirty"] = True
        with self.assertRaises(ValueError):
            validate_manifest(record)

    def test_invalid_hashes_and_known_asset_drift(self):
        for key in manifest(with_numerical=True)["numerical"]["assets"]:
            for bad in ("bad", "F" * 64, None):
                record = manifest(with_numerical=True)
                record["numerical"]["assets"][key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    validate_manifest(resign(record))
        record = manifest(with_numerical=True)
        record["numerical"]["assets"]["chat_template_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_manifest(resign(record))

    def test_prompt_ids_messages_and_rendered_hashes(self):
        for key, bad in (("input_ids", []), ("input_ids", [True]), ("input_ids", [-1]),
                         ("input_ids", [1.0]), ("input_ids", [1] * 513),
                         ("messages", []), ("messages", [{"role": "system", "content": "x"}]),
                         ("messages", [{"role": "user", "content": " "}]),
                         ("messages_sha256", "0" * 64), ("rendered_sha256", "0" * 64),
                         ("rendered_prompt", "")):
            record = manifest(with_numerical=True)
            record["numerical"]["prompt"][key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                validate_manifest(resign(record))
        record = manifest(with_numerical=True)
        record["config"]["max_prompt_tokens"] = 2
        with self.assertRaises(ValueError):
            validate_manifest(resign(record))

    def test_environment_and_determinism_fields(self):
        for key, bad in (("python_version", "3.11"), ("python_build", ""),
                         ("packages", {}), ("hardware", {"device": "auto"})):
            record = manifest(with_numerical=True)
            record["numerical"]["environment"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(resign(record))
        for key, bad in (("seed", True), ("pythonhashseed", "random"),
                         ("allow_tf32_matmul", True), ("deterministic_algorithms", False)):
            record = manifest(with_numerical=True)
            record["numerical"]["determinism"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(resign(record))

    def test_synthetic_gpu_evidence_matches_intended_profile(self):
        record = manifest(with_numerical=True)
        record["config"].update(profile_id="t4-fp16", device="cuda:0", dtype="float16")
        environment = record["numerical"]["environment"]
        environment["packages"]["torch"]["version"] = "2.10.0+cu126"
        environment["hardware"].update(
            device="cuda:0", gpu_name="synthetic T4", compute_capability=[7, 5],
            driver_version="synthetic-driver", cuda_runtime="12.6")
        self.assertIsNone(validate_manifest(resign(record)))
        for key, bad in (("device", "cpu"), ("gpu_name", None),
                         ("compute_capability", [True, 5]), ("compute_capability", [7]),
                         ("driver_version", ""), ("cuda_runtime", None)):
            changed = deepcopy(record)
            changed["numerical"]["environment"]["hardware"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(resign(changed))
    def test_draft_cannot_claim_success_or_comparison_identity(self):
        for changes in ({"complete": True, "outcome": "success"},
                        {"complete": 0}, {"outcome": "success"},
                        {"comparison_key_sha256": "a" * 64}, {"outputs": {}}):
            record = manifest()
            record.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_manifest(record)

    def test_synthetic_completed_outputs_are_bounded_and_consistent(self):
        record = manifest(with_numerical=True)
        record.update(complete=True, outcome="success", outputs={
            "prompt_ids": [1, 2, 3], "sequence_ids": [1, 2, 3, 151645],
            "generated_ids": [151645], "raw_text": "synthetic", "display_text": "",
            "stop_reason": "eos", "reached_limit": False})
        self.assertIsNone(validate_manifest(record))
        for key, bad in (("prompt_ids", [1]), ("sequence_ids", [151645]),
                         ("prompt_ids", [True, 2, 3]),
                         ("generated_ids", [True]), ("generated_ids", [151645, 4]),
                         ("stop_reason", "limit"), ("reached_limit", True),
                         ("raw_text", None), ("display_text", 1)):
            changed = deepcopy(record)
            changed["outputs"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_synthetic_eos_at_limit_and_non_eos_limit(self):
        for cap in (1, 64):
            for terminal, reason in ((7, "limit"), (151645, "eos"), (151643, "eos")):
                record = manifest(with_numerical=True)
                record["config"]["max_new_tokens"] = cap
                record["generation"]["max_new_tokens"] = cap
                record["numerical"]["effective_generation_config"]["max_new_tokens"] = cap
                generated = [7] * (cap - 1) + [terminal]
                record.update(complete=True, outcome="success", outputs={
                    "prompt_ids": [1, 2, 3], "sequence_ids": [1, 2, 3] + generated,
                    "generated_ids": generated, "raw_text": "synthetic", "display_text": "",
                    "stop_reason": reason, "reached_limit": True})
                with self.subTest(cap=cap, reason=reason):
                    self.assertIsNone(validate_manifest(resign(record)))
        record = manifest(with_numerical=True)
        record.update(complete=True, outcome="success", outputs={
            "prompt_ids": [1, 2, 3], "sequence_ids": [1, 2, 3, 7], "generated_ids": [7],
            "raw_text": "synthetic", "display_text": "", "stop_reason": "limit",
            "reached_limit": False})
        with self.assertRaises(ValueError):
            validate_manifest(record)

    def test_validation_errors_do_not_echo_untrusted_values(self):
        marker = "sensitive-input-never-echo"
        record = manifest()
        record["config"]["model_id"] = marker
        with self.assertRaises(ValueError) as caught:
            validate_manifest(record)
        self.assertNotIn(marker, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
