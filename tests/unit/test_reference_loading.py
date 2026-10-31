"""CPU-only reference loading contracts using synthetic assets and external factories."""

from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vortex.executor.reference_config import MODEL_ID, REVISION, ReferenceConfig
from vortex.executor.reference_manifest import canonical_digest
from vortex.executor import reference_loading as loading


def config(profile="windows-cpu"):
    return ReferenceConfig(MODEL_ID, REVISION, REVISION, profile,
                           "cpu" if profile == "windows-cpu" else "cuda:0",
                           "float32" if profile == "windows-cpu" else "float16")


class Tensor:
    """Small owned integer tensor fixture; tensor execution is an external boundary."""

    def __init__(self, values, device="cpu", dtype="int64"):
        self.values, self.device, self.dtype = deepcopy(values), device, dtype
        self.shape = (len(values), len(values[0]) if values else 0)
        self.ndim = 2

    def tolist(self):
        return deepcopy(self.values)

    def to(self, *, device):
        return Tensor(self.values, device, self.dtype)


class BoundaryFixture:
    def __init__(self, directory):
        self.cache = Path(directory)
        self.snapshot = self.cache / "models--Qwen--Qwen3-0.6B" / "snapshots" / REVISION
        self.snapshot.mkdir(parents=True)
        self.template = "synthetic template with explicit enable_thinking"
        self.raw_config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                           "hidden_size": 1024, "intermediate_size": 3072,
                           "num_hidden_layers": 28, "num_attention_heads": 16,
                           "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
                           "bos_token_id": 151643, "eos_token_id": 151645}
        self.generation = {"eos_token_id": [151645, 151643], "bos_token_id": 151643,
                           "pad_token_id": 151643, "do_sample": True}
        self.payloads = {"config.json": json.dumps(self.raw_config).encode(),
                         "generation_config.json": json.dumps(self.generation).encode(),
                         "tokenizer_config.json": json.dumps({"chat_template": self.template}).encode(),
                         "tokenizer.json": b"synthetic tokenizer", "vocab.json": b"{}",
                         "merges.txt": b"synthetic merges", "model.safetensors": b"synthetic weights"}
        self.identities = {}
        for name, data in self.payloads.items():
            (self.snapshot / name).write_bytes(data)
            self.identities[name] = (len(data), "sha256", hashlib.sha256(data).hexdigest())
        self.specials = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}
        self.tokenizer = Mock(is_fast=True, eos_token_id=151645, pad_token_id=151643,
                              chat_template=self.template)
        self.tokenizer.convert_tokens_to_ids.side_effect = self.specials.__getitem__
        self.ids = [[11, 12, 13]]
        self.render = "synthetic rendered prompt <think>\n\n</think>\n\n"
        self.tokenizer.apply_chat_template.side_effect = self.templating
        self.tokenizer.side_effect = lambda *a, **kw: self.tokens()
        self.model = Mock(training=False)
        self.model.config = SimpleNamespace(**self.raw_config, _attn_implementation="eager")
        self.model.generation_config = SimpleNamespace(**self.generation)
        self.model.parameters.return_value = iter([SimpleNamespace(device="cpu", dtype="float32")])
        self.model.to.return_value = self.model
        self.active_inference = False
        self.context_calls = 0
        self.torch = SimpleNamespace(float32="float32", float16="float16", long="int64",
                                     is_tensor=lambda t: isinstance(t, Tensor),
                                     inference_mode=self.inference, device=self.cpu_context)
        self.backend = SimpleNamespace(torch=self.torch,
            AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=self.tokenizer)),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(side_effect=self.make_model)),
            Qwen3Config=SimpleNamespace(from_dict=Mock(side_effect=lambda d: SimpleNamespace(**d))),
            GenerationConfig=SimpleNamespace(from_dict=Mock(side_effect=lambda d: SimpleNamespace(**d))),
            hf_hub_download=Mock(side_effect=lambda **kw: str(self.snapshot / kw["filename"])),
            cache=self.cache)

    def tokens(self):
        return {"input_ids": Tensor(self.ids), "attention_mask": Tensor([[1] * len(self.ids[0])])}

    def templating(self, messages, **kwargs):
        return self.tokens() if kwargs["tokenize"] else self.render

    @contextmanager
    def inference(self):
        self.context_calls += 1
        self.active_inference = True
        try:
            yield
        finally:
            self.active_inference = False

    @contextmanager
    def cpu_context(self, device):
        assert device == "cpu"
        yield

    def make_model(self, *args, **kwargs):
        assert self.active_inference, "model construction must be scoped to inference mode"
        return self.model

    @contextmanager
    def installed(self):
        # Only external receipts/factories/artifact identities are synthetic.
        with patch.object(loading, "_environment", return_value={"profile_id": "windows-cpu"}), \
                patch.object(loading, "_dependencies", return_value=self.backend), \
                patch.object(loading, "_ASSETS", self.identities), \
                patch.object(loading, "_TEMPLATE_SHA256", hashlib.sha256(self.template.encode()).hexdigest()):
            yield


class ReferenceLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.boundary = BoundaryFixture(self.directory.name)

    def load(self, **kwargs):
        with self.boundary.installed():
            return loading.load_reference(config(), **kwargs)

    def test_loader_pins_both_assets_and_uses_only_closed_native_snapshot(self):
        b = self.boundary
        loaded = self.load(offline=True)
        self.assertIs(loaded.model, b.model)
        self.assertEqual(loaded.config, config())
        self.assertEqual(loaded.eos_token_ids, (151645, 151643))
        self.assertEqual(loaded.assets["weights_sha256"], canonical_digest({
            "model.safetensors": hashlib.sha256(b.payloads["model.safetensors"]).hexdigest()}))
        self.assertEqual(set(loaded.assets), {"model_config_sha256", "generation_config_sha256",
                         "tokenizer_config_sha256", "chat_template_sha256",
                         "tokenizer_assets_sha256", "weights_sha256"})
        calls = b.backend.hf_hub_download.call_args_list
        self.assertEqual({c.kwargs["filename"] for c in calls}, set(b.payloads))
        self.assertEqual(len(calls), 7)
        for call in calls:
            self.assertEqual(call.kwargs["repo_id"], MODEL_ID)
            self.assertEqual(call.kwargs["revision"], REVISION)
            self.assertIs(call.kwargs["local_files_only"], True)
            self.assertIs(call.kwargs["token"], False)
        for factory in (b.backend.AutoTokenizer, b.backend.AutoModelForCausalLM):
            args, kw = factory.from_pretrained.call_args
            self.assertEqual(Path(args[0]), b.snapshot)
            self.assertEqual(kw["revision"], REVISION)
            self.assertIs(kw["trust_remote_code"], False)
            self.assertIs(kw["local_files_only"], True)
        kw = b.backend.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertIs(kw["use_safetensors"], True)
        self.assertEqual(kw["torch_dtype"], "float32")
        self.assertEqual(kw["attn_implementation"], "eager")
        self.assertIn("generation_config", kw)
        self.assertNotIn("device_map", kw)
        b.model.to.assert_called_once_with(device="cpu", dtype="float32")
        b.model.eval.assert_called_once_with()
        self.assertFalse(b.active_inference)
        with loaded.inference_context():
            self.assertTrue(b.active_inference)
        with self.assertRaises(TypeError):
            loaded.assets["weights_sha256"] = "bad"

    def test_invalid_contract_before_environment_imports_or_network(self):
        bad = config()
        object.__setattr__(bad, "revision", "main")
        for cfg, kwargs in ((bad, {}), (config(), {"offline": 1})):
            with patch.object(loading, "_environment") as env, self.assertRaises(loading.ReferenceLoadingError):
                loading.load_reference(cfg, **kwargs)
            env.assert_not_called()
        with self.assertRaises(loading.ReferenceLoadingError) as caught:
            loading.load_reference(bad, offline=True)
        self.assertEqual(caught.exception.code, "INVALID_CONFIG")
        for kwargs in ({"trust_remote_code": True}, {"device_map": "auto"}, {"dtype": "bfloat16"}):
            with self.assertRaises(TypeError):
                loading.load_reference(config(), **kwargs)

    def test_dependency_profile_mismatch_stops_before_acquisition(self):
        with patch.object(loading.subprocess, "run", return_value=SimpleNamespace(returncode=0,
                stdout=json.dumps({"status": "IMPORTS_VERIFIED", "profile_id": "t4-fp16", "qualified": False}), stderr="")), \
                patch.object(loading, "_dependencies") as deps, self.assertRaises(loading.ReferenceLoadingError):
            loading.load_reference(config(), offline=True)
        deps.assert_not_called()

    def test_missing_offline_assets_never_retry_online(self):
        b = self.boundary
        b.backend.hf_hub_download.side_effect = OSError("secret cache path")
        with b.installed(), self.assertRaises(loading.ReferenceLoadingError) as caught:
            loading.load_reference(config(), offline=True)
        self.assertEqual(b.backend.hf_hub_download.call_count, 1)
        self.assertNotIn("secret", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        b.backend.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_hash_size_and_sidecar_fail_before_native_factories(self):
        b = self.boundary
        for name, data in (("config.json", b"bad"), ("model.safetensors", b"x" * 17)):
            path = b.snapshot / name
            old = path.read_bytes()
            path.write_bytes(data)
            with b.installed(), self.assertRaises(loading.ReferenceLoadingError):
                loading.load_reference(config(), offline=True)
            path.write_bytes(old)
        for name in ("model.safetensors.index.json", "pytorch_model.bin", "chat_template.jinja", "custom_generate"):
            path = b.snapshot / name
            path.write_text("unsafe")
            with b.installed(), self.assertRaises(loading.ReferenceLoadingError):
                loading.load_reference(config(), offline=True)
            path.unlink()
        b.backend.AutoModelForCausalLM.from_pretrained.assert_not_called()
        b.backend.AutoTokenizer.from_pretrained.assert_not_called()

    def test_model_tokenizer_and_template_mismatch(self):
        b = self.boundary
        for obj, field, value in ((b.tokenizer, "eos_token_id", 151643),
                                  (b.tokenizer, "chat_template", "drift"),
                                  (b.model.config, "head_dim", 64),
                                  (b.model.config, "_attn_implementation", "sdpa"),
                                  (b.model, "training", True)):
            old = getattr(obj, field)
            setattr(obj, field, value)
            with b.installed(), self.assertRaises(loading.ReferenceLoadingError):
                loading.load_reference(config(), offline=True)
            setattr(obj, field, old)
            b.model.parameters.return_value = iter([SimpleNamespace(device="cpu", dtype="float32")])

    def test_load_oom_sanitized_and_inference_mode_restored(self):
        b = self.boundary
        b.model.to.side_effect = MemoryError("private diagnostic")
        with b.installed(), self.assertRaises(loading.ReferenceLoadingError) as caught:
            loading.load_reference(config(), offline=True)
        self.assertEqual(caught.exception.code, "OUT_OF_MEMORY")
        self.assertNotIn("private", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        self.assertFalse(b.active_inference)

    def test_no_auto_device_or_dtype_fallback(self):
        b = self.boundary
        b.model.parameters.return_value = iter([SimpleNamespace(device="cpu", dtype="float16")])
        with b.installed(), self.assertRaises(loading.ReferenceLoadingError):
            loading.load_reference(config(), offline=True)
        self.assertEqual(b.model.to.call_count, 1)

    def test_non_thinking_template_integer_ids_and_stable_hashes(self):
        b = self.boundary
        with b.installed():
            loaded = loading.load_reference(config(), offline=True)
            for content in loading.SYNTHETIC_PROMPTS:
                messages = [{"role": "user", "content": content}]
                prompt = loading.prepare_prompt(loaded, messages, config())
                self.assertEqual(prompt.prompt_token_ids, (11, 12, 13))
                self.assertEqual(prompt.input_ids.shape, (1, 3))
                self.assertEqual(prompt.attention_mask.tolist(), [[1, 1, 1]])
                self.assertEqual(prompt.input_ids.dtype, "int64")
                self.assertEqual(prompt.input_ids.device, "cpu")
                self.assertEqual(prompt.rendered_sha256, hashlib.sha256(b.render.encode()).hexdigest())
                self.assertEqual(prompt.rendered_sha256, loading.prepare_prompt(loaded, messages, config()).rendered_sha256)
                self.assertEqual(messages, [{"role": "user", "content": content}])
        for call in b.tokenizer.apply_chat_template.call_args_list:
            self.assertIs(call.kwargs["enable_thinking"], False)
            self.assertIs(call.kwargs["add_generation_prompt"], True)
        token_call = [c for c in b.tokenizer.apply_chat_template.call_args_list if c.kwargs["tokenize"]][0]
        self.assertEqual(token_call.kwargs["return_tensors"], "pt")
        self.assertIs(token_call.kwargs["return_dict"], True)
        self.assertIs(token_call.kwargs["padding"], False)
        self.assertIs(token_call.kwargs["truncation"], False)
        self.assertIs(b.tokenizer.call_args.kwargs["add_special_tokens"], False)
        b.model.generate.assert_not_called()

    def test_prompt_limits_and_malformed_messages(self):
        b = self.boundary
        with b.installed():
            loaded = loading.load_reference(config(), offline=True)
            invalid = [None, [], [{"role": "assistant", "content": "x"}],
                       [{"role": "user", "content": ""}], [{"role": "user", "content": True}],
                       [{"role": "user", "content": "x", "tools": []}],
                       [{"role": "user", "content": "x"}] * 2,
                       [{"role": "user", "content": "x" * 16385}]]
            for messages in invalid:
                with self.subTest(messages=type(messages)), self.assertRaises(loading.ReferenceLoadingError):
                    loading.prepare_prompt(loaded, messages, config())
            for length in (0, 513):
                b.ids = [list(range(length))]
                with self.assertRaises(loading.ReferenceLoadingError):
                    loading.prepare_prompt(loaded, [{"role": "user", "content": "x"}], config())
            b.ids = [list(range(512))]
            self.assertEqual(len(loading.prepare_prompt(loaded, [{"role": "user", "content": "x"}], config()).prompt_token_ids), 512)

    def test_module_import_without_torch_transformers_or_network(self):
        code = f"""
import sys
sys.path.insert(0,{str(ROOT / 'src')!r})
class Block:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','transformers','huggingface_hub','numpy'):
   raise AssertionError('heavy import attempted')
sys.meta_path.insert(0,Block())
import vortex.executor.reference_loading
print('offline-ok')
"""
        run = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True,
                             timeout=10, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout, "offline-ok\n")

    def test_git_blob_identity_and_same_size_hash_drift(self):
        b = self.boundary
        data = b.payloads["vocab.json"]
        git_hash = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        b.identities["vocab.json"] = (len(data), "git-sha1", git_hash)
        self.load(offline=True)
        b.model.parameters.return_value = iter([SimpleNamespace(device="cpu", dtype="float32")])
        (b.snapshot / "vocab.json").write_bytes(b"[]")
        with b.installed(), self.assertRaises(loading.ReferenceLoadingError):
            loading.load_reference(config(), offline=True)

    def test_private_config_entrypoints_and_unsafe_returned_paths(self):
        b = self.boundary
        for key in ("_attn_implementation_internal", "auto_map", "quantization_config", "custom_generate"):
            raw = dict(b.raw_config, **{key: "unsafe"})
            data = json.dumps(raw).encode()
            (b.snapshot / "config.json").write_bytes(data)
            b.identities["config.json"] = (len(data), "sha256", hashlib.sha256(data).hexdigest())
            with b.installed(), self.assertRaises(loading.ReferenceLoadingError):
                loading.load_reference(config(), offline=True)
        b.backend.AutoModelForCausalLM.from_pretrained.assert_not_called()
        b.backend.hf_hub_download.side_effect = lambda **kw: str(b.cache / kw["filename"])
        with b.installed(), self.assertRaises(loading.ReferenceLoadingError):
            loading.load_reference(config(), offline=True)

    def test_invalid_tensor_shapes_masks_ids_and_replay(self):
        b = self.boundary
        with b.installed():
            loaded = loading.load_reference(config(), offline=True)
            messages = [{"role": "user", "content": "x"}]
            for ids, mask in ((Tensor([[1.0]], dtype="float32"), Tensor([[1]])),
                              (Tensor([[True]]), Tensor([[1]])),
                              (Tensor([[-1]]), Tensor([[1]])),
                              (Tensor([[151936]]), Tensor([[1]])),
                              (Tensor([[1], [2]]), Tensor([[1], [1]])),
                              (Tensor([[1, 2]]), Tensor([[1]])),
                              (Tensor([[1]]), Tensor([[0]]))):
                b.tokenizer.apply_chat_template.side_effect = lambda m, **kw: (
                    {"input_ids": ids, "attention_mask": mask} if kw["tokenize"] else b.render)
                with self.assertRaises(loading.ReferenceLoadingError):
                    loading.prepare_prompt(loaded, messages, config())
            b.tokenizer.apply_chat_template.side_effect = b.templating
            b.tokenizer.side_effect = lambda *a, **kw: {"input_ids": Tensor([[99]]),
                                                      "attention_mask": Tensor([[1]])}
            with self.assertRaises(loading.ReferenceLoadingError):
                loading.prepare_prompt(loaded, messages, config())

    def test_failed_partial_model_is_not_kept_by_sanitized_exception(self):
        b = self.boundary
        observed = []

        class PartialModel:
            def __init__(self):
                observed.append(weakref.ref(self))

            def to(self, **kwargs):
                raise MemoryError("sensitive partial object diagnostic")

        b.backend.AutoModelForCausalLM.from_pretrained.side_effect = lambda *a, **kw: PartialModel()
        with b.installed(), self.assertRaises(loading.ReferenceLoadingError) as caught:
            loading.load_reference(config(), offline=True)
        self.assertIsNone(caught.exception.__context__)
        gc.collect()
        self.assertIsNone(observed[0]())

    def test_cache_symlink_escape_is_rejected_before_acquisition(self):
        b = self.boundary
        # Patch the filesystem's link observation boundary: Windows symlink
        # creation itself depends on host privileges, not the loader contract.
        original = Path.is_symlink
        with b.installed(), patch.object(Path, "is_symlink", autospec=True,
                side_effect=lambda p: p == b.cache or original(p)), \
                self.assertRaises(loading.ReferenceLoadingError):
            loading.load_reference(config(), offline=True)
        b.backend.hf_hub_download.assert_not_called()

    def test_hub_write_directories_reject_links_before_download(self):
        b = self.boundary
        original = Path.is_symlink
        for target in (b.cache / ".locks", b.cache / ".locks/models--Qwen--Qwen3-0.6B",
                       b.snapshot.parents[1] / ".no_exist"):
            with b.installed(), patch.object(Path, "is_symlink", autospec=True,
                    side_effect=lambda p: p == target or original(p)), \
                    self.assertRaises(loading.ReferenceLoadingError):
                loading.load_reference(config(), offline=True)
            b.backend.hf_hub_download.assert_not_called()

    def test_parent_import_origins_are_bound_before_native_imports(self):
        import builtins
        import importlib.machinery
        original = builtins.__import__

        def no_native(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'transformers', 'huggingface_hub', 'tokenizers', 'safetensors'):
                raise AssertionError("unreviewed native import reached")
            return original(name, *args, **kwargs)

        for origin in (None, str(Path(self.directory.name) / 'unreviewed/torch/__init__.py')):
            spec = importlib.machinery.ModuleSpec('torch', None, origin=origin)
            with patch.object(importlib.util, 'find_spec', return_value=spec), \
                    patch('builtins.__import__', side_effect=no_native), \
                    self.assertRaises(loading.ReferenceLoadingError):
                loading._dependencies()

    def test_preimported_shadow_module_is_rejected(self):
        import builtins
        from types import ModuleType
        module = ModuleType('torch')
        module.__file__ = str(Path(self.directory.name) / 'shadow.py')
        module.__spec__ = None
        original = builtins.__import__

        def no_native(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'transformers', 'huggingface_hub'):
                raise AssertionError('shadow module reached import')
            return original(name, *args, **kwargs)

        with patch.dict(sys.modules, {'torch': module}), \
                patch('builtins.__import__', side_effect=no_native), \
                self.assertRaises(loading.ReferenceLoadingError):
            loading._dependencies()

    def test_lazy_mapping_keys_are_bounded_before_snapshot(self):
        from collections.abc import Mapping
        visited = []

        class Messages(Mapping):
            def __len__(self):
                return 3

            def __iter__(self):
                for index in range(100):
                    visited.append(index)
                    yield 'extra' + str(index)

            def __getitem__(self, key):
                raise AssertionError('invalid mapping must not be snapshotted')

        b = self.boundary
        with b.installed():
            loaded = loading.load_reference(config(), offline=True)
            with self.assertRaises(loading.ReferenceLoadingError):
                loading.prepare_prompt(loaded, [Messages()], config())
        self.assertLessEqual(len(visited), 3)

    def test_prepare_invalid_configuration_is_actionable(self):
        b = self.boundary
        with b.installed():
            loaded = loading.load_reference(config(), offline=True)
            bad = config()
            object.__setattr__(bad, 'max_prompt_tokens', 0)
            with self.assertRaises(loading.ReferenceLoadingError) as caught:
                loading.prepare_prompt(loaded, [{'role': 'user', 'content': 'x'}], bad)
        self.assertEqual(caught.exception.code, 'INVALID_CONFIG')

    def test_incomplete_blob_symlink_is_rejected_before_download(self):
        b = self.boundary
        blob = b.snapshot.parents[1] / 'blobs'
        blob.mkdir()
        leaf = blob / 'unreviewed.incomplete'
        leaf.write_bytes(b'partial')
        original = Path.is_symlink
        with b.installed(), patch.object(Path, 'is_symlink', autospec=True,
                side_effect=lambda p: p == leaf or original(p)), \
                self.assertRaises(loading.ReferenceLoadingError):
            loading.load_reference(config(), offline=True)
        b.backend.hf_hub_download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
