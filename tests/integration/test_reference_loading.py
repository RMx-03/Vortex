"""Opt-in real tokenizer API checks with a tiny synthetic vocabulary, never a model."""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vortex.executor.reference_config import MODEL_ID, REVISION, ReferenceConfig
from vortex.executor import reference_loading as loading


@unittest.skipUnless(os.environ.get("VORTEX_REFERENCE_TOKENIZER_API") == "1",
                     "NOT VERIFIED: opt in inside the existing locked CPU profile")
class InstalledTokenizerAPITests(unittest.TestCase):
    def test_real_chat_template_api_with_synthetic_artifact_identities(self):
        self.assertEqual(importlib.metadata.version("transformers"), "4.53.3")
        self.assertEqual(importlib.metadata.version("torch"), "2.10.0+cpu")
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast

        vocabulary = {"[UNK]": 0, "<|endoftext|>": 1, "<|im_start|>": 2, "<|im_end|>": 3,
                      "<think>": 4, "</think>": 5, "user": 6, "assistant": 7}
        native = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        native.pre_tokenizer = WhitespaceSplit()
        template = ("{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
                    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant "
                    "{% if enable_thinking is defined and enable_thinking is false %}"
                    "<think>\n\n</think>\n\n{% endif %}{% endif %}")
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=native, unk_token="[UNK]",
            eos_token="<|im_end|>", pad_token="<|endoftext|>", chat_template=template)
        cfg = ReferenceConfig(MODEL_ID, REVISION, REVISION, "windows-cpu", "cpu", "float32")
        loaded = loading.LoadedReference(SimpleNamespace(), tokenizer, "cpu", "float32", cfg,
            MappingProxyType({}), MappingProxyType({}), torch, torch.inference_mode)
        # Fixture metadata substitutes for Qwen artifact identity only. Actual
        # templating, tokenization, Tensor creation, replay and transfer are real.
        with patch.object(loading, "_TEMPLATE_SHA256", hashlib.sha256(template.encode()).hexdigest()), \
                patch.object(loading, "_SPECIAL_IDS", {token: vocabulary[token] for token in
                    ("<|endoftext|>", "<|im_start|>", "<|im_end|>")}, create=True):
            for content in loading.SYNTHETIC_PROMPTS:
                messages = [{"role": "user", "content": content}]
                result = loading.prepare_prompt(loaded, messages, cfg)
                replay = loading.prepare_prompt(loaded, messages, cfg)
                self.assertEqual(result.prompt_token_ids, replay.prompt_token_ids)
                self.assertEqual(result.rendered_sha256, replay.rendered_sha256)
                self.assertEqual(tuple(result.input_ids.shape), (1, len(result.prompt_token_ids)))
                self.assertEqual(result.input_ids.dtype, torch.int64)
                self.assertEqual(result.attention_mask.dtype, torch.int64)
                self.assertEqual(str(result.input_ids.device), "cpu")
                self.assertEqual(result.attention_mask.tolist(), [[1] * len(result.prompt_token_ids)])
                self.assertTrue(result.rendered_prompt.endswith("<think>\n\n</think>\n\n"))


if __name__ == "__main__":
    unittest.main()
