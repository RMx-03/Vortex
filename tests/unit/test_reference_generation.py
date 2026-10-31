"""Bounded generation contracts with small CPU doubles; no inference packages."""

from contextlib import contextmanager
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import gc
import weakref
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vortex.executor.reference_config import MODEL_ID, REVISION, ReferenceConfig
from vortex.executor.reference_loading import LoadedReference, PreparedPrompt

try:
    from vortex.executor import reference_generation as generation
except ImportError:
    generation = None


def config(cap=2):
    return ReferenceConfig(MODEL_ID, REVISION, REVISION, "windows-cpu", "cpu", "float32",
                           max_new_tokens=cap)


class Tensor:
    def __init__(self, rows, dtype="int64", device="cpu"):
        self.rows, self.dtype, self.device = rows, dtype, device
        self.shape = (len(rows), len(rows[0]) if rows else 0)
        self.ndim = 2
        self.reads = 0

    def tolist(self):
        self.reads += 1
        return [list(row) for row in self.rows]


class Torch:
    long = "int64"
    is_tensor = staticmethod(lambda x: isinstance(x, Tensor))
    tensor = staticmethod(lambda rows, **kw: Tensor(rows, **kw))
    equal = staticmethod(lambda a, b: a.rows == b.rows and a.dtype == b.dtype)
    ones_like = staticmethod(lambda x: Tensor([[1] * x.shape[1]], x.dtype, x.device))


class Cache:
    def __init__(self):
        self.length = 0

    def get_seq_length(self):
        return self.length


class GenerationConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return dict(self.__dict__, transformers_version="4.53.3", forced_eos_token_id=None,
                    bad_words_ids=None, cache_implementation=None)


class Boundary:
    def __init__(self, cap=2):
        self.cfg = config(cap)
        self.active = False
        self.calls = []
        self.outputs = [21, 22][:cap]
        self.failure = None
        self.transform = lambda result: result
        self.decoder_calls = []
        self.torch = Torch()
        self.model = SimpleNamespace(training=False, config=SimpleNamespace(
            model_type="qwen3", vocab_size=151936, _attn_implementation="eager"),
            generation_config=SimpleNamespace(do_sample=True, temperature=.6, top_p=.95, top_k=20),
            generate=self.generate)
        self.tokenizer = SimpleNamespace(decode=self.decode)
        self.loaded = LoadedReference(self.model, self.tokenizer, "cpu", "float32", self.cfg,
            {}, {"profile_id": "windows-cpu", "status": "IMPORTS_VERIFIED", "qualified": False},
            self.torch, self.inference)
        self.prompt = PreparedPrompt(Tensor([[11, 12, 13]]), Tensor([[1, 1, 1]]), (11, 12, 13),
            hashlib.sha256(b"synthetic prompt").hexdigest(), "synthetic prompt", "a" * 64)

    @contextmanager
    def inference(self):
        if self.active:
            raise AssertionError("nested fixture context")
        self.active = True
        try:
            yield
        finally:
            self.active = False

    def generate(self, **kwargs):
        if not self.active:
            raise AssertionError("model called outside inference mode")
        self.calls.append(kwargs)
        if self.failure:
            raise self.failure
        cache = kwargs["past_key_values"]
        cache.length = 3 + len(self.outputs) - 1
        return self.transform(SimpleNamespace(sequences=Tensor([[11, 12, 13] + self.outputs]),
                                               past_key_values=cache))

    def decode(self, ids, **kwargs):
        self.decoder_calls.append((tuple(ids), kwargs))
        return "display" if kwargs["skip_special_tokens"] else "raw <control>"

    def installed(self):
        return patch.object(generation, "_dependencies", return_value=SimpleNamespace(
            GenerationConfig=GenerationConfig, DynamicCache=Cache))

    def run(self):
        return generation.generate_reference(self.loaded, self.prompt, self.cfg)


class ReferenceGenerationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(generation, "reference generation adapter is not implemented")
        self.b = Boundary()
        self.context = self.b.installed()
        self.context.start()
        self.addCleanup(self.context.stop)

    def assert_code(self, code, operation):
        with self.assertRaises(generation.ReferenceGenerationError) as caught:
            operation()
        self.assertEqual(caught.exception.code, code)
        self.assertIsNone(caught.exception.__context__)
        return caught.exception

    def test_saved_sampling_defaults_cannot_override_greedy_contract(self):
        result = self.b.run()
        call = self.b.calls[0]
        self.assertIs(call["use_model_defaults"], False)
        self.assertEqual(call["logits_to_keep"], 1)
        effective = result.effective_generation_config
        expected = {"do_sample": False, "num_beams": 1, "num_beam_groups": 1,
            "num_return_sequences": 1, "max_new_tokens": 2, "min_new_tokens": 0,
            "use_cache": True, "eos_token_id": [151645, 151643], "pad_token_id": 151643,
            "bos_token_id": 151643, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0,
            "length_penalty": 1.0, "return_dict_in_generate": True, "output_scores": False,
            "output_logits": False, "output_attentions": False, "output_hidden_states": False,
            "temperature": 1.0, "top_p": 1.0, "top_k": 50, "max_time": None,
            "max_length": 5, "min_length": 3, "transformers_version": "4.53.3"}
        for key, value in expected.items():
            self.assertEqual(effective[key], value, key)
            self.assertIs(type(effective[key]), type(value), key)
        self.assertEqual(effective["forced_eos_token_id"], None)
        self.assertIs(self.b.model.generation_config.do_sample, True)
        self.assertIsNone(call["custom_generate"])
        self.assertIsNone(call["assistant_model"])
        self.assertIsNone(call["streamer"])
        self.assertEqual(call["logits_processor"], [])
        self.assertEqual(call["stopping_criteria"], [])
        self.assertFalse(self.b.active)

    def test_raw_boundary_does_not_materialize_or_decode_and_owns_fresh_cache(self):
        with patch.object(generation, "_export", side_effect=AssertionError("serialization inside raw interval")):
            raw = generation.generate_raw_reference(self.b.loaded, self.b.prompt, self.b.cfg)
        self.assertEqual(raw.sequences.reads, 0)
        self.assertEqual(self.b.prompt.input_ids.reads, 0)
        self.assertEqual(self.b.decoder_calls, [])
        other = generation.generate_raw_reference(self.b.loaded, self.b.prompt, self.b.cfg)
        self.assertIsNot(raw.past_key_values, other.past_key_values)
        self.assertIsNot(self.b.calls[0]["generation_config"], self.b.calls[1]["generation_config"])
        result = generation.materialize_generation(raw, self.b.loaded, self.b.prompt, self.b.cfg)
        self.assertEqual(result.prompt_ids, (11, 12, 13))
        self.assertEqual(result.sequence_ids, (11, 12, 13, 21, 22))
        self.assertEqual(result.generated_ids, (21, 22))
        self.assertEqual((result.stop_reason, result.reached_limit), ("limit", True))

    def test_raw_ids_survive_decoding_with_separate_display_text(self):
        self.b.outputs = [21, 151645]
        result = self.b.run()
        self.assertEqual(result.generated_ids, (21, 151645))
        self.assertEqual((result.raw_text, result.display_text), ("raw <control>", "display"))
        self.assertEqual(self.b.decoder_calls, [((21, 151645), {"skip_special_tokens": False,
            "clean_up_tokenization_spaces": False}), ((21, 151645), {"skip_special_tokens": True,
            "clean_up_tokenization_spaces": False})])

    def test_dual_eos_first_and_eos_at_limit(self):
        for eos in (151645, 151643):
            for outputs, reached in (([eos], False), ([21, eos], True)):
                with self.subTest(eos=eos, outputs=outputs):
                    self.b.outputs = outputs
                    result = self.b.run()
                    self.assertEqual(result.generated_ids, tuple(outputs))
                    self.assertEqual((result.stop_reason, result.reached_limit), ("eos", reached))

    def test_limit_one_and_hard_output_ceiling(self):
        for cap in (1, 64):
            b = Boundary(cap)
            b.outputs = [21] * cap
            with b.installed():
                result = b.run()
            self.assertEqual(len(result.generated_ids), cap)
            self.assertEqual((result.stop_reason, result.reached_limit), ("limit", True))

    def test_invalid_config_and_unsafe_options_fail_before_model(self):
        for cap in (0, 65, True):
            cfg = config()
            object.__setattr__(cfg, "max_new_tokens", cap)
            self.assert_code("INVALID_CONFIG", lambda: generation.generate_reference(
                self.b.loaded, self.b.prompt, cfg))
        for kwargs in ({"do_sample": True}, {"max_time": 10}, {"custom_generate": "unsafe"}):
            with self.assertRaises(TypeError):
                generation.generate_reference(self.b.loaded, self.b.prompt, self.b.cfg, **kwargs)
        self.assertEqual(self.b.calls, [])

    def test_profile_and_mutated_model_mismatch_fail_before_model(self):
        for loaded in (replace(self.b.loaded, device="cuda:0"),
                       replace(self.b.loaded, dtype="float16"),
                       replace(self.b.loaded, eos_token_ids=(151645,)),
                       replace(self.b.loaded, environment={"profile_id": "t4-fp16"})):
            self.assert_code("INVALID_CONFIG", lambda: generation.generate_reference(
                loaded, self.b.prompt, self.b.cfg))
        self.b.model.training = True
        self.assert_code("INVALID_CONFIG", self.b.run)
        self.assertEqual(self.b.calls, [])

    def test_malformed_inputs_rejected_before_generation(self):
        for tensor in (Tensor([[]]), Tensor([list(range(513))]), Tensor([[11, 12, 13]], "float32"),
                       Tensor([[11, 12, 13]], device="cuda:0"), Tensor([[11], [12]]),
                       Tensor([[11, 12, 99]])):
            prompt = replace(self.b.prompt, input_ids=tensor)
            self.assert_code("INVALID_PROMPT", lambda: generation.generate_reference(
                self.b.loaded, prompt, self.b.cfg))
        for ids in ((True,), (-1,), (151936,), (), (11,) * 513):
            prompt = replace(self.b.prompt, prompt_token_ids=ids)
            self.assert_code("INVALID_PROMPT", lambda: generation.generate_reference(
                self.b.loaded, prompt, self.b.cfg))
        for mask in (Tensor([[1, 0, 1]]), Tensor([[1, 1]])):
            prompt = replace(self.b.prompt, attention_mask=mask)
            self.assert_code("INVALID_PROMPT", lambda: generation.generate_reference(
                self.b.loaded, prompt, self.b.cfg))
        self.assertEqual(self.b.calls, [])

    def test_exact_prompt_ceiling_is_not_truncated(self):
        ids = tuple(range(512))
        self.b.prompt = replace(self.b.prompt, input_ids=Tensor([list(ids)]),
            attention_mask=Tensor([[1] * 512]), prompt_token_ids=ids)
        def boundary(**kwargs):
            self.b.calls.append(kwargs)
            kwargs["past_key_values"].length = 513
            return SimpleNamespace(sequences=Tensor([list(ids) + [21, 22]]),
                                   past_key_values=kwargs["past_key_values"])
        self.b.model.generate = boundary
        self.assertEqual(self.b.run().prompt_ids, ids)

    def test_malformed_output_shape_dtype_device_and_lengths_fail(self):
        invalid = (None, Tensor([[]]), Tensor([[11, 12, 13]]), Tensor([[11, 12, 13, 21, 22, 23]]),
                   Tensor([[11, 12, 13, 21]], "float32"),
                   Tensor([[11, 12, 13, 21]], device="cuda:0"), Tensor([[11], [12]]))
        for tensor in invalid:
            self.b.transform = lambda output: SimpleNamespace(sequences=tensor,
                                                              past_key_values=output.past_key_values)
            self.assert_code("INVALID_OUTPUT", self.b.run)

    def test_invalid_generated_ids_prefix_and_stopping_are_rejected(self):
        for outputs in ([True, 22], [-1, 22], [151936, 22], [21], [151645, 22], [151643, 151645]):
            self.b.outputs = outputs
            self.assert_code("INVALID_OUTPUT", self.b.run)
        self.b.outputs = [21, 22]
        self.b.transform = lambda output: SimpleNamespace(sequences=Tensor([[99, 12, 13, 21, 22]]),
                                                          past_key_values=output.past_key_values)
        self.assert_code("INVALID_OUTPUT", self.b.run)

    def test_missing_wrong_or_inconsistent_cache_fails(self):
        for cache in (None, (), Cache()):
            self.b.transform = lambda output: SimpleNamespace(sequences=output.sequences,
                                                              past_key_values=cache)
            self.assert_code("INVALID_OUTPUT", self.b.run)
        def stale(output):
            output.past_key_values.length = 3
            return output
        self.b.transform = stale
        self.assert_code("INVALID_OUTPUT", self.b.run)

    def test_missing_output_fields_are_invalid_output(self):
        for output in (None, SimpleNamespace(), SimpleNamespace(sequences=Tensor([[11, 12, 13, 21, 22]]))):
            self.b.transform = lambda result: output
            self.assert_code("INVALID_OUTPUT", self.b.run)

    def test_failed_request_cache_unwinds_without_native_exception_retention(self):
        refs = []
        def fail(**kwargs):
            refs.append(weakref.ref(kwargs["past_key_values"]))
            raise RuntimeError("sensitive request")
        self.b.model.generate = fail
        exception = self.assert_code("GENERATION_FAILED", self.b.run)
        gc.collect()
        self.assertIsNone(refs[0]())
        self.assertIsNone(exception.__context__)

    def test_materialization_failure_does_not_retain_cache_in_error_traceback(self):
        refs = []
        def observe(output):
            refs.append(weakref.ref(output.past_key_values))
            return output
        self.b.transform = observe
        def fail(*args, **kwargs):
            raise RuntimeError("sensitive decoding diagnostic")
        for invalid_ids in (False, True):
            if invalid_ids:
                self.b.outputs = [21, 151936]
            else:
                self.b.tokenizer.decode = fail
            retained = None
            try:
                self.b.run()
            except generation.ReferenceGenerationError as error:
                retained = error
            self.assertIsNotNone(retained)
            self.assertIsNotNone(retained.__traceback__)
            self.b.calls.clear()  # Remove only the test boundary's intentional ownership.
            gc.collect()
            self.assertIsNone(refs[-1](), "retained sanitized error must not own request cache")
            self.assertIsNone(retained.__context__)

    def test_ordinary_native_exception_types_are_sanitized(self):
        class NativeFailure(Exception):
            pass
        for error in (IndexError("sensitive"), OverflowError("sensitive"),
                      AssertionError("sensitive"), NativeFailure("sensitive")):
            self.b.failure = error
            exception = self.assert_code("GENERATION_FAILED", self.b.run)
            self.assertNotIn("sensitive", str(exception))
            self.assertFalse(self.b.active)

    def test_limit_one_eos_is_both_eos_and_at_limit(self):
        b = Boundary(1)
        for eos in (151645, 151643):
            b.outputs = [eos]
            with b.installed():
                result = b.run()
            self.assertEqual(result.generated_ids, (eos,))
            self.assertEqual((result.stop_reason, result.reached_limit), ("eos", True))

    def test_execution_errors_are_sanitized_and_inference_mode_restored(self):
        for error, code in ((MemoryError("sensitive"), "OUT_OF_MEMORY"),
                            (RuntimeError("sensitive"), "GENERATION_FAILED"),
                            (ValueError("sensitive"), "GENERATION_FAILED")):
            self.b.failure = error
            exception = self.assert_code(code, self.b.run)
            self.assertNotIn("sensitive", str(exception))
            self.assertFalse(self.b.active)
        self.b.failure = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.b.run()
        self.assertFalse(self.b.active)

    def test_materialization_rejects_wrong_request_and_decode_failure(self):
        raw = generation.generate_raw_reference(self.b.loaded, self.b.prompt, self.b.cfg)
        other = replace(self.b.prompt, prompt_token_ids=(11, 12, 99))
        self.assert_code("INVALID_PROMPT", lambda: generation.materialize_generation(
            raw, self.b.loaded, other, self.b.cfg))
        self.b.tokenizer.decode = lambda *a, **kw: None
        self.assert_code("INVALID_OUTPUT", lambda: generation.materialize_generation(
            raw, self.b.loaded, self.b.prompt, self.b.cfg))

    def test_repeat_mock_results_are_deterministic_without_shared_cache(self):
        first, second = self.b.run(), self.b.run()
        self.assertEqual(first, second)
        self.assertIsNot(self.b.calls[0]["past_key_values"], self.b.calls[1]["past_key_values"])

    def test_import_requires_no_inference_packages(self):
        source = f"""
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
class Block:
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in ('torch', 'transformers', 'numpy', 'huggingface_hub'):
   raise AssertionError('unexpected inference import')
sys.meta_path.insert(0, Block())
import vortex.executor.reference_generation
print('offline-ok')
"""
        run = subprocess.run([sys.executable, "-S", "-c", source], capture_output=True,
                             text=True, timeout=10, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout, "offline-ok\n")


class ReferenceCLITests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "benchmarks/run_baseline.py"
        self.assertTrue(path.is_file(), "bounded baseline CLI is not implemented")
        spec = importlib.util.spec_from_file_location("reference_baseline_cli", path)
        self.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.cli)
        self.args = ["--profile", "windows-cpu", "--source-sha", "1" * 40, "--max-new-tokens", "2"]

    def invoke(self, extra=()):
        out, error = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(error):
            code = self.cli.main(self.args + list(extra))
        records = [json.loads(line) for line in out.getvalue().splitlines()]
        return code, records, error.getvalue()

    def test_configuration_only_cli_emits_three_valid_drafts_without_loading(self):
        from vortex.executor.reference_manifest import validate_manifest
        with patch.object(self.cli, "load_reference", side_effect=AssertionError("loading from draft mode")):
            code, records, error = self.invoke()
        self.assertEqual(code, 0, error)
        self.assertEqual(len(records), 3)
        for record in records:
            validate_manifest(record)
            self.assertEqual((record["complete"], record["outcome"], record["outputs"]),
                             (False, "configured", None))
        workload = self.cli.read_workloads()
        self.assertEqual([item["messages"][0]["content"] for item in workload], [
            "Reply with the single word vortex.", "What is 2 + 2? Answer briefly.",
            "Explain a key-value cache in one short sentence."])

    def test_cli_execution_builds_schema_valid_output_from_external_boundaries(self):
        from vortex.executor import reference_manifest as manifest
        b = Boundary()
        b.outputs = [21, 151643]
        observed = {"python_version": "3.11.18", "python_build": ["fixture", "date"],
            "implementation": "CPython", "platform_build": "synthetic Windows", "machine": "AMD64",
            "packages": {"torch": "2.10.0+cpu", "transformers": "4.53.3"},
            "wheel_hashes": {"torch": "1" * 64, "transformers": "2" * 64},
            "torch_git_version": "fixture", "glibc_version": None}
        b.loaded = replace(b.loaded, assets=dict(manifest._KNOWN_ASSETS,
            tokenizer_assets_sha256="3" * 64, weights_sha256="4" * 64),
            environment=dict(b.loaded.environment, observation=observed))
        def prepare(loaded, messages, cfg):
            self.assertEqual(messages, [{"role": "user", "content": "Reply with the single word vortex."}])
            return replace(b.prompt, messages_sha256=manifest.canonical_digest({"messages": messages}))
        determinism = {"seed": 0, "pythonhashseed": "0", "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False, "allow_tf32_matmul": False,
            "allow_tf32_cudnn": False, "deterministic_algorithms": True}
        with b.installed(), patch.object(self.cli, "load_reference", return_value=b.loaded), \
                patch.object(self.cli, "prepare_prompt", side_effect=prepare), \
                patch.object(self.cli, "configure_determinism", return_value=determinism), \
                patch.dict(self.cli.os.environ, {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            code, records, error = self.invoke(["--execute", "--offline", "--workload", "A"])
        self.assertEqual(code, 0, error)
        self.assertEqual(len(records), 1)
        manifest.validate_manifest(records[0])
        self.assertEqual(records[0]["outputs"]["generated_ids"], [21, 151643])
        self.assertEqual((records[0]["complete"], records[0]["outcome"]), (True, "success"))

    def test_cli_failed_execution_is_incomplete_and_sanitized(self):
        from vortex.executor.reference_manifest import validate_manifest
        from vortex.executor.reference_loading import ReferenceLoadingError
        with patch.object(self.cli, "load_reference", side_effect=ReferenceLoadingError("ASSET_INVALID")), \
                patch.dict(self.cli.os.environ, {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            code, records, error = self.invoke(["--execute", "--offline", "--workload", "B"])
        self.assertEqual(code, 3)
        self.assertEqual(len(records), 1)
        validate_manifest(records[0])
        self.assertEqual((records[0]["complete"], records[0]["outcome"], records[0]["outputs"]),
                         (False, "failed", None))
        self.assertIn("ASSET_INVALID", error)

    def test_cli_rejects_bad_limits_and_missing_process_settings_without_execution(self):
        with patch.object(self.cli, "load_reference", side_effect=AssertionError("invalid preflight reached loading")):
            for args in (["--max-new-tokens", "65"], ["--source-sha", "main"]):
                code, records, error = self.invoke(args)
                self.assertEqual((code, records), (2, []))
                self.assertIn("INVALID_CONFIG", error)
            with patch.dict(self.cli.os.environ, {}, clear=True):
                code, records, error = self.invoke(["--execute"])
                self.assertEqual((code, records), (2, []))
                self.assertIn("DETERMINISM_REQUIRED", error)

    def test_cli_help_is_offline(self):
        run = subprocess.run([sys.executable, "-S", str(ROOT / "benchmarks/run_baseline.py"), "--help"],
                             capture_output=True, text=True, timeout=10, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("--execute", run.stdout)

    def test_cli_interruption_returns_only_incomplete_record(self):
        from vortex.executor.reference_manifest import validate_manifest
        with patch.object(self.cli, "load_reference", side_effect=KeyboardInterrupt()), \
                patch.dict(self.cli.os.environ, {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            code, records, error = self.invoke(["--execute", "--workload", "A"])
        self.assertEqual(code, 130)
        validate_manifest(records[0])
        self.assertEqual(records[0]["outcome"], "interrupted")
        self.assertIsNone(records[0]["outputs"])
        self.assertIn("INTERRUPTED", error)

    def test_cli_unexpected_native_failure_emits_incomplete_record_without_traceback(self):
        from vortex.executor.reference_manifest import validate_manifest
        with patch.object(self.cli, "load_reference", side_effect=IndexError("sensitive native diagnostic")), \
                patch.dict(self.cli.os.environ, {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            code, records, error = self.invoke(["--execute", "--workload", "A"])
        self.assertEqual(code, 3)
        validate_manifest(records[0])
        self.assertEqual(records[0]["outcome"], "failed")
        self.assertIsNone(records[0]["outputs"])
        self.assertEqual(json.loads(error)["error_code"], "REQUEST_FAILED")
        self.assertNotIn("sensitive", error)

    def test_workload_drift_duplicate_fields_and_oversize_fail_closed(self):
        for body in (b'{"schema_version":1,"schema_version":1,"workloads":[]}',
                     b'{}', b'x' * 16385):
            with patch.object(Path, "open", return_value=io.BytesIO(body)), self.assertRaises(ValueError):
                self.cli.read_workloads()

    def test_cpu_determinism_settings_are_applied_without_cuda_seed(self):
        calls = []
        torch = SimpleNamespace(manual_seed=lambda seed: calls.append(("torch", seed)),
            cuda=SimpleNamespace(manual_seed_all=lambda seed: self.fail("CPU requested CUDA seed")),
            backends=SimpleNamespace(cudnn=SimpleNamespace(benchmark=True, allow_tf32=True),
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
            use_deterministic_algorithms=lambda flag: calls.append(("algorithms", flag)),
            are_deterministic_algorithms_enabled=lambda: True)
        numpy = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: calls.append(("numpy", seed))))
        with patch.dict(sys.modules, {"numpy": numpy}), \
                patch.dict(self.cli.os.environ, {"PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            result = self.cli.configure_determinism(SimpleNamespace(_torch=torch, device="cpu"))
        self.assertEqual(calls, [("numpy", 0), ("torch", 0), ("algorithms", True)])
        self.assertFalse(result["allow_tf32_matmul"])
        self.assertFalse(result["allow_tf32_cudnn"])
        self.assertFalse(result["cudnn_benchmark"])
        self.assertTrue(result["deterministic_algorithms"])


if __name__ == "__main__":
    unittest.main()
