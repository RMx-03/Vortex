"""Offline synthetic comparison evidence, never qualified pretrained goldens."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from test_reference_manifest import manifest, resign
from vortex.executor.reference_manifest import validate_manifest


def completed(ids=(10, 20, 30, 151645), cap=4):
    """Controlled schema-1 data; software/asset evidence is synthetic and unqualified."""
    record = manifest(with_numerical=True)
    record["config"]["max_new_tokens"] = cap
    record["generation"]["max_new_tokens"] = cap
    effective = record["numerical"]["effective_generation_config"]
    effective.update(max_new_tokens=cap, max_length=3 + cap, min_length=3)
    tokens = list(ids)
    record.update(complete=True, outcome="success", outputs={
        "prompt_ids": [1, 2, 3], "sequence_ids": [1, 2, 3] + tokens,
        "generated_ids": tokens, "raw_text": "synthetic raw", "display_text": "synthetic",
        "stop_reason": "eos" if tokens and tokens[-1] in (151645, 151643) else "limit",
        "reached_limit": len(tokens) == cap})
    return resign(record)


class ReferenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((ROOT / "scripts/compare_reference.py").is_file(), "offline comparator is not implemented")
        self.compare = importlib.import_module("compare_reference")

    def check(self, expected, actual, status, index=None, reason=None):
        result = self.compare.compare_reference(expected, actual)
        self.assertEqual(result.status, status)
        self.assertEqual(result.first_mismatch, index)
        if reason is not None:
            self.assertEqual(result.reason, reason)
        self.assertEqual(set(asdict(result)), {"status", "first_mismatch", "reason"})
        return result

    def test_equal_ids_match_even_when_decoded_display_differs(self):
        expected, actual = completed(), completed()
        actual["outputs"].update(raw_text="different raw text", display_text="different display")
        validate_manifest(expected)
        validate_manifest(actual)
        self.check(expected, actual, "MATCH", reason="identical_tokens")

    def test_first_middle_and_last_divergence_positions(self):
        expected = completed()
        for ids, index in (((11, 20, 30, 151645), 0), ((10, 21, 30, 151645), 1),
                           ((10, 20, 30, 151643), 3)):
            actual = completed(ids)
            self.assertEqual(expected["config_sha256"], actual["config_sha256"])
            self.assertEqual(expected["comparison_key_sha256"], actual["comparison_key_sha256"])
            self.check(expected, actual, "MISMATCH", index, "token_mismatch")

    def test_different_lengths_report_first_divergence_and_count_mismatch(self):
        self.check(completed((10, 151645)), completed((10, 20, 151645)),
                   "MISMATCH", 1, "output_length_mismatch")
        self.check(completed((10, 20, 151645)), completed((10, 151645)),
                   "MISMATCH", 1, "output_length_mismatch")

    def test_early_eos_and_eos_at_limit_preserve_terminal_ids(self):
        for cap in (1, 4, 64):
            for eos in (151645, 151643):
                for ids in ((eos,), tuple([10] * (cap - 1) + [eos])):
                    with self.subTest(cap=cap, eos=eos, length=len(ids)):
                        record = completed(ids, cap)
                        validate_manifest(record)
                        self.check(record, deepcopy(record), "MATCH")

    def test_maximum_non_eos_output_count_and_vocabulary_boundary(self):
        record = completed([151935] * 64, 64)
        self.check(record, deepcopy(record), "MATCH")

    def test_zero_output_is_not_a_successful_schema1_sequence(self):
        self.check(completed(()), completed(), "INCOMPATIBLE", reason="invalid_reference")
        self.check(completed(), completed(()), "INCOMPATIBLE", reason="invalid_candidate")
        self.check(completed((), 0), completed((), 0), "INCOMPATIBLE", reason="invalid_reference")

    def test_post_eos_missing_terminal_and_inconsistent_stop_metadata_are_invalid(self):
        expected = completed((10, 151645))
        for ids in ((10,), (10, 151645, 20), (151643, 151645)):
            self.check(expected, completed(ids), "INCOMPATIBLE", reason="invalid_candidate")
        for field, value in (("stop_reason", "limit"), ("reached_limit", True),
                             ("sequence_ids", [1, 2, 3, 10]), ("prompt_ids", [1, 2])):
            actual = completed((10, 151645))
            actual["outputs"][field] = value
            self.check(expected, actual, "INCOMPATIBLE", reason="invalid_candidate")

    def test_invalid_token_types_ranges_and_counts_are_rejected(self):
        for token in (True, -1, 1.0, "10", None, 151936, 10 ** 5000):
            actual = completed((10, 20, token, 151645))
            self.check(completed(), actual, "INCOMPATIBLE", reason="invalid_candidate")
        self.check(completed(), completed([10] * 64 + [151645], 64),
                   "INCOMPATIBLE", reason="invalid_candidate")

    def test_prompt_vocabulary_is_validated_beyond_manifest_hashes(self):
        actual = completed()
        actual["numerical"]["prompt"]["input_ids"] = [151936, 2, 3]
        actual["outputs"]["prompt_ids"] = [151936, 2, 3]
        actual["outputs"]["sequence_ids"] = [151936, 2, 3, 10, 20, 30, 151645]
        resign(actual)
        validate_manifest(actual)  # Schema validates nonnegative IDs, not pinned vocabulary size.
        self.check(completed(), actual, "INCOMPATIBLE", reason="invalid_candidate")

    def test_run_source_and_timestamp_provenance_do_not_change_comparability(self):
        actual = completed()
        actual["provenance"].update(source_sha="f" * 40, dirty=True, patch_sha256="e" * 64,
            adapter_version="candidate-adapter", timestamp="2026-11-01T00:00:00Z",
            run_id="00000000-0000-4000-8000-000000000003",
            session_id="00000000-0000-4000-8000-000000000004")
        self.check(completed(), actual, "MATCH")

    def test_mismatched_environment_and_asset_identity_are_incompatible(self):
        for field in ("python_build", "platform", "machine"):
            actual = completed()
            actual["numerical"]["environment"][field] = "different synthetic identity"
            resign(actual)
            self.assertEqual(actual["config_sha256"], completed()["config_sha256"])
            self.check(completed(), actual, "INCOMPATIBLE", reason="numerical_identity_mismatch")
        actual = completed()
        actual["numerical"]["assets"]["weights_sha256"] = "f" * 64
        self.check(completed(), resign(actual), "INCOMPATIBLE", reason="numerical_identity_mismatch")

    def test_wrong_profile_is_incompatible_not_token_mismatch(self):
        actual = completed()
        actual["config"].update(profile_id="t4-fp16", device="cuda:0", dtype="float16")
        environment = actual["numerical"]["environment"]
        environment["packages"]["torch"]["version"] = "2.10.0+cu126"
        environment["hardware"].update(device="cuda:0", gpu_name="synthetic T4",
            compute_capability=[7, 5], driver_version="synthetic-driver", cuda_runtime="12.6")
        validate_manifest(resign(actual))
        self.check(completed(), actual, "INCOMPATIBLE", reason="numerical_identity_mismatch")

    def test_input_and_effective_generation_drift_change_numerical_identity(self):
        actual = completed()
        actual["numerical"]["prompt"]["input_ids"] = [1, 2, 4]
        actual["outputs"]["prompt_ids"] = [1, 2, 4]
        actual["outputs"]["sequence_ids"] = [1, 2, 4, 10, 20, 30, 151645]
        self.check(completed(), resign(actual), "INCOMPATIBLE", reason="numerical_identity_mismatch")
        actual = completed()
        actual["numerical"]["effective_generation_config"]["min_length"] = 4
        self.check(completed(), resign(actual), "INCOMPATIBLE", reason="numerical_identity_mismatch")
        self.check(completed(), completed((10, 151645), 32),
                   "INCOMPATIBLE", reason="numerical_identity_mismatch")

    def test_template_hash_drift_and_unsupported_generation_fail_validation(self):
        for group, key, value in (("assets", "chat_template_sha256", "f" * 64),
                                  ("effective_generation_config", "do_sample", True)):
            actual = completed()
            actual["numerical"][group][key] = value
            self.check(completed(), resign(actual), "INCOMPATIBLE", reason="invalid_candidate")

    def test_invalid_manifest_schema_fields_and_stale_identity_are_rejected(self):
        for key, value in (("schema_version", 2), ("comparison_key_sha256", "0" * 64),
                            ("config_sha256", "0" * 64), ("provenance", {})):
            actual = completed()
            actual[key] = value
            self.check(completed(), actual, "INCOMPATIBLE", reason="invalid_candidate")
        actual = completed()
        actual["qualified"] = True  # An invented trusted flag must not bypass schema validation.
        self.check(completed(), actual, "INCOMPATIBLE", reason="invalid_candidate")
        actual = completed()
        del actual["numerical"]
        self.check(completed(), actual, "INCOMPATIBLE", reason="invalid_candidate")

    def test_incomplete_drafts_failures_and_numerical_records_cannot_match(self):
        for numerical in (False, True):
            record = manifest(with_numerical=numerical)
            for outcome in ("configured", "failed", "interrupted"):
                record["outcome"] = outcome
                self.check(record, completed(), "INCOMPATIBLE", reason="incomplete_reference")
                self.check(completed(), record, "INCOMPATIBLE", reason="incomplete_candidate")

    def test_malformed_oversized_and_cyclic_inputs_are_isolated(self):
        cycle = {}
        cycle["cycle"] = cycle
        for record in (None, [], "sensitive", {"sensitive": "x" * 1048577}, cycle):
            self.check(record, completed(), "INCOMPATIBLE", reason="invalid_reference")
        class Hostile(Mapping):
            def __len__(self): return 1
            def __iter__(self): raise RuntimeError("sensitive mapping diagnostic")
            def __getitem__(self, key): raise AssertionError("must not read")
        self.check(completed(), Hostile(), "INCOMPATIBLE", reason="invalid_candidate")

    def test_lazy_mapping_work_is_bounded_by_snapshot_budget(self):
        visits = []
        class Large(Mapping):
            def __len__(self): return 10000000
            def __iter__(self):
                while True:
                    visits.append(1)
                    yield "x" * 65536
            def __getitem__(self, key): return 0
        self.check(Large(), completed(), "INCOMPATIBLE", reason="invalid_reference")
        self.assertLessEqual(len(visits), 17)

    def test_snapshot_does_not_revisit_caller_mapping_or_mutate_records(self):
        record, actual = completed(), completed()
        before = deepcopy((record, actual))
        reads = {}
        class Once(Mapping):
            def __len__(self): return len(record)
            def __iter__(self): return iter(record)
            def __getitem__(self, key):
                reads[key] = reads.get(key, 0) + 1
                if reads[key] != 1:
                    raise RuntimeError("caller reread after validation")
                return record[key]
        self.check(Once(), actual, "MATCH")
        self.assertEqual((record, actual), before)
        self.assertEqual(set(reads.values()), {1})

    def test_invalid_reference_does_not_inspect_candidate(self):
        class Untouched(Mapping):
            def __len__(self): raise AssertionError("candidate touched")
            def __iter__(self): raise AssertionError("candidate touched")
            def __getitem__(self, key): raise AssertionError("candidate touched")
        self.check(None, Untouched(), "INCOMPATIBLE", reason="invalid_reference")

    def test_interruptions_are_not_converted_to_comparison_outcomes(self):
        class Interrupt(Mapping):
            def __len__(self): return 1
            def __iter__(self): raise KeyboardInterrupt()
            def __getitem__(self, key): return None
        with self.assertRaises(KeyboardInterrupt):
            self.compare.compare_reference(Interrupt(), completed())

    def test_reports_are_deterministic_and_exclude_raw_input(self):
        expected, actual = completed(), completed((11, 20, 30, 151645))
        marker = "sensitive-display-do-not-echo"
        actual["outputs"]["raw_text"] = marker
        result = self.check(expected, actual, "MISMATCH", 0)
        encoded = json.dumps(asdict(result), sort_keys=True, separators=(",", ":"))
        self.assertEqual(encoded, '{"first_mismatch":0,"reason":"token_mismatch","status":"MISMATCH"}')
        self.assertNotIn(marker, encoded)
        self.assertEqual(result, self.compare.compare_reference(expected, actual))

    def cli(self, expected, actual):
        return subprocess.run([sys.executable, "-B", "-S", str(ROOT / "scripts/compare_reference.py"),
            "--expected", str(expected), "--actual", str(actual)], capture_output=True, text=True,
            encoding="utf-8", timeout=15, check=False)

    def test_cli_replay_exit_codes_and_no_automatic_golden_rewrite(self):
        with tempfile.TemporaryDirectory(prefix="vortex comparison ") as folder:
            expected, actual = Path(folder) / "expected.json", Path(folder) / "actual.json"
            for ids, code, status in (((10, 20, 30, 151645), 0, "MATCH"),
                                     ((11, 20, 30, 151645), 1, "MISMATCH"), ((), 2, "INCOMPATIBLE")):
                expected.write_text(json.dumps(completed()), encoding="utf-8")
                actual.write_text(json.dumps(completed(ids)), encoding="utf-8")
                before = (expected.read_bytes(), actual.read_bytes())
                first, second = self.cli(expected, actual), self.cli(expected, actual)
                self.assertEqual(first.returncode, code, first.stderr)
                self.assertEqual(json.loads(first.stdout)["status"], status)
                self.assertEqual(first.stdout, second.stdout)
                self.assertEqual(first.stderr, "")
                self.assertEqual((expected.read_bytes(), actual.read_bytes()), before)

    def test_cli_missing_malformed_duplicate_nonfinite_and_oversized_files(self):
        with tempfile.TemporaryDirectory(prefix="vortex comparison ") as folder:
            expected, actual = Path(folder) / "expected.json", Path(folder) / "actual.json"
            expected.write_text(json.dumps(completed()), encoding="utf-8")
            for body in ('{"schema_version":1,"schema_version":1}', '{"value":NaN}',
                         '{"sensitive":"broken', "x" * 1048577):
                actual.write_text(body, encoding="utf-8")
                run = self.cli(expected, actual)
                self.assertEqual(run.returncode, 2, run.stderr)
                self.assertEqual(json.loads(run.stdout), {"status": "INCOMPATIBLE",
                    "first_mismatch": None, "reason": "invalid_candidate"})
                self.assertEqual(run.stderr, "")
                self.assertNotIn("sensitive", run.stdout)
            run = self.cli(Path(folder) / "sensitive-missing.json", expected)
            self.assertEqual(json.loads(run.stdout)["reason"], "invalid_reference")
            self.assertEqual(run.returncode, 2)

    def test_cli_incomplete_reference_does_not_read_candidate(self):
        calls = []
        def read(filename):
            calls.append(filename)
            if filename != "reference.json":
                raise AssertionError("incomplete reference must short-circuit candidate I/O")
            return manifest(with_numerical=True)
        out = io.StringIO()
        with patch.object(self.compare, "_read_record", side_effect=read), redirect_stdout(out):
            code = self.compare.main(["--expected", "reference.json", "--actual", "candidate.json"])
        self.assertEqual(code, 2)
        self.assertEqual(calls, ["reference.json"])
        self.assertEqual(json.loads(out.getvalue())["reason"], "incomplete_reference")

    def test_cli_requires_utf8_input(self):
        with tempfile.TemporaryDirectory(prefix="vortex comparison ") as folder:
            expected, actual = Path(folder) / "expected.json", Path(folder) / "actual.json"
            expected.write_text(json.dumps(completed()), encoding="utf-8")
            actual.write_bytes(json.dumps(completed()).encode("utf-16"))
            run = self.cli(expected, actual)
            self.assertEqual(run.returncode, 2)
            self.assertEqual(json.loads(run.stdout)["reason"], "invalid_candidate")

    def test_unc_separator_variants_are_rejected_before_filesystem_access(self):
        from pathlib import PureWindowsPath
        for filename in ("\\\\example.invalid/share/record.json", "//example.invalid/share/record.json",
                         "\\/example.invalid/share/record.json", "/\\example.invalid/share/record.json"):
            self.assertTrue(PureWindowsPath(filename).drive.startswith("\\\\"))
            with patch.object(self.compare, "Path") as filesystem:
                self.assertIsNone(self.compare._read_record(filename))
                filesystem.assert_not_called()

    def test_import_and_comparison_without_inference_packages_or_network(self):
        source = f"""
import sys, socket
sys.path.insert(0, {str(ROOT / 'scripts')!r})
sys.path.insert(0, {str(ROOT / 'tests/unit')!r})
class Block:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','transformers','huggingface_hub','numpy'):
   raise AssertionError('unexpected inference import')
sys.meta_path.insert(0,Block())
socket.socket.connect=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('unexpected network'))
socket.create_connection=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('unexpected network'))
import compare_reference
from test_reference_comparison import completed
assert compare_reference.compare_reference(completed(),completed()).status == 'MATCH'
print('offline-ok')
"""
        run = subprocess.run([sys.executable, "-B", "-S", "-c", source], capture_output=True,
                             text=True, timeout=15, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout, "offline-ok\n")


if __name__ == "__main__":
    unittest.main()
