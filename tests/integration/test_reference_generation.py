"""Opt-in pinned API checks: native config/cache/tensors with a stub generate call.

No Qwen object, model weights, network, forward pass or GPU operation is used.
"""

import inspect
import importlib.metadata
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vortex.executor.reference_config import MODEL_ID, REVISION, ReferenceConfig
from vortex.executor.reference_loading import LoadedReference, PreparedPrompt
from vortex.executor import reference_generation as generation


@unittest.skipUnless(os.environ.get("VORTEX_REFERENCE_GENERATION_API") == "1",
                     "NOT VERIFIED: opt in inside the existing locked CPU profile")
class InstalledGenerationAPITests(unittest.TestCase):
    def test_pinned_defaults_isolation_cache_and_materialization_with_stub_execution(self):
        self.assertEqual(importlib.metadata.version("transformers"), "4.53.3")
        self.assertEqual(importlib.metadata.version("torch"), "2.10.0+cpu")
        import torch
        from transformers import DynamicCache, GenerationConfig
        from transformers.generation.utils import GenerationMixin
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        self.assertIn("logits_to_keep", inspect.signature(Qwen3ForCausalLM.forward).parameters)
        cfg = ReferenceConfig(MODEL_ID, REVISION, REVISION, "windows-cpu", "cpu", "float32", max_new_tokens=2)
        saved = GenerationConfig(do_sample=True, temperature=.6, top_p=.95, top_k=20)
        model = SimpleNamespace(training=False, config=SimpleNamespace(model_type="qwen3",
            vocab_size=151936, _attn_implementation="eager"), generation_config=saved)
        calls = []
        def stub(**kwargs):
            inspect.signature(GenerationMixin.generate).bind(model, **kwargs)
            self.assertTrue(torch.is_inference_mode_enabled())
            settings, unused = GenerationMixin._prepare_generation_config(model,
                kwargs["generation_config"], use_model_defaults=kwargs["use_model_defaults"],
                decoder_start_token_id=kwargs["decoder_start_token_id"])
            self.assertFalse(settings.do_sample)
            self.assertEqual((settings.temperature, settings.top_p, settings.top_k), (1.0, 1.0, 50))
            self.assertEqual(settings.eos_token_id, [151645, 151643])
            cache = kwargs["past_key_values"]
            self.assertIs(type(cache), DynamicCache)
            self.assertEqual(cache.get_seq_length(), 0)
            # Synthetic cache metadata only: 4 positions, one tiny test layer.
            cache.update(torch.zeros((1, 1, 4, 1)), torch.zeros((1, 1, 4, 1)), 0)
            calls.append(cache)
            return SimpleNamespace(sequences=torch.tensor([[11, 12, 13, 21, 151643]]), past_key_values=cache)
        model.generate = stub
        tokenizer = SimpleNamespace(decode=lambda ids, **kw: "fixture" if kw["skip_special_tokens"] else "fixture EOS")
        loaded = LoadedReference(model, tokenizer, "cpu", "float32", cfg, {},
            {"profile_id": "windows-cpu", "status": "IMPORTS_VERIFIED", "qualified": False}, torch, torch.inference_mode)
        prompt = PreparedPrompt(torch.tensor([[11, 12, 13]]), torch.ones((1, 3), dtype=torch.int64),
            (11, 12, 13), hashlib.sha256(b"synthetic").hexdigest(), "synthetic", "a" * 64)
        with patch.object(generation, "_dependencies", return_value=SimpleNamespace(
                DynamicCache=DynamicCache, GenerationConfig=GenerationConfig)):
            first = generation.generate_reference(loaded, prompt, cfg)
            second = generation.generate_reference(loaded, prompt, cfg)
        self.assertEqual(first, second)
        self.assertEqual(first.generated_ids, (21, 151643))
        self.assertEqual((first.stop_reason, first.reached_limit), ("eos", True))
        self.assertIsNot(calls[0], calls[1])
        self.assertTrue(saved.do_sample)
        self.assertEqual(first.effective_generation_config["max_length"], 5)
        self.assertIn("forced_eos_token_id", first.effective_generation_config)
        self.assertFalse(torch.is_inference_mode_enabled())


if __name__ == "__main__":
    unittest.main()
