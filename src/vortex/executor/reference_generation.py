"""Bounded single-request HF reference generation, separate from host decoding.

Imports are standard-library only. Execution requires an already loaded pinned
reference on an independently authorized host. This module acquires no assets,
sets no process-wide determinism flags and implements no decoding loop or timer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import re
from types import SimpleNamespace
from typing import TYPE_CHECKING, TypeVar

from vortex.executor.reference_config import ReferenceConfig, validate_reference_config
from vortex.executor.reference_loading import LoadedReference, PreparedPrompt, _import_origins
from vortex.executor.reference_manifest import _generation, canonical_digest

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import DynamicCache, GenerationConfig

_T = TypeVar("_T")
_ERRORS = {
    "INVALID_CONFIG": "Use the matching pinned configuration, eval model and explicit profile.",
    "INVALID_PROMPT": "Prepare unpadded integer inputs within the configured limit before generation.",
    "ENVIRONMENT_MISMATCH": "Use the complete locked Transformers 4.53.3 environment used for loading.",
    "INVALID_OUTPUT": "Generation returned inconsistent tokens, bounds or cache; do not accept this attempt.",
    "GENERATION_FAILED": "Generation failed; stop this attempt and inspect the authorized host without changing settings.",
    "OUT_OF_MEMORY": "Generation exhausted memory; release this attempt and use an authorized host with sufficient capacity.",
}


class ReferenceGenerationError(ValueError):
    """Finite sanitized reason code; never retain a native exception or its data."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(_ERRORS[code])


@dataclass(frozen=True)
class RawGenerationOutput:
    """Own device sequences and the request's mutable cache until released.

    No CPU token lists or decoded text are produced by raw generation. The caller
    must drop this result/cache before another measured trial; allocator-reserved
    memory reclamation is not promised. The request-owned native generation
    configuration is serialized only during materialization, after timing.
    """

    sequences: Tensor
    past_key_values: DynamicCache
    generation_config: GenerationConfig
    _config: ReferenceConfig = field(repr=False)
    _prompt_ids: tuple[int, ...] = field(repr=False)
    _model_id: int = field(repr=False)
    _cache_type: type = field(repr=False)


@dataclass(frozen=True)
class GenerationResult:
    """Host output, with terminal IDs retained and prompt IDs excluded from new IDs.

    Tuple fields map to schema-1 output lists. Decoded strings are display evidence,
    not the numerical oracle. Effective configuration is a complete JSON export;
    invocation controls remain in the manifest's separate generation envelope.
    This is successful adapter output, never a claim of GPU or golden qualification.
    """

    prompt_ids: tuple[int, ...]
    sequence_ids: tuple[int, ...]
    generated_ids: tuple[int, ...]
    raw_text: str
    display_text: str
    stop_reason: str
    reached_limit: bool
    effective_generation_config: dict[str, object]


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ReferenceGenerationError(code)


def _checked(operation: Callable[..., _T], *args: object) -> _T:
    try:
        return operation(*args)
    except ReferenceGenerationError as error:
        code = error.code
    except MemoryError:
        code = "OUT_OF_MEMORY"
    except RuntimeError as error:
        code = "OUT_OF_MEMORY" if type(error).__name__ == "OutOfMemoryError" else "GENERATION_FAILED"
    except Exception:
        # Native backends expose an open set of exception classes. Translate
        # ordinary failures once, while BaseException interruption still escapes.
        code = "GENERATION_FAILED"
    # Outside except: native frames and partially populated request caches unwind.
    args = ()
    raise ReferenceGenerationError(code)


def _dependencies(loaded: LoadedReference) -> SimpleNamespace:
    try:
        _import_origins(loaded.environment)
        import torch
        import transformers
        from transformers import DynamicCache, GenerationConfig
        _require(torch is loaded._torch and transformers.__version__ == "4.53.3",
                 "ENVIRONMENT_MISMATCH")
        return SimpleNamespace(DynamicCache=DynamicCache, GenerationConfig=GenerationConfig)
    except (ImportError, ValueError, KeyError, TypeError, AttributeError, OSError):
        pass
    raise ReferenceGenerationError("ENVIRONMENT_MISMATCH")


def _tensor(tensor: Tensor, loaded: LoadedReference, minimum: int, maximum: int,
            code: str) -> None:
    torch = loaded._torch
    _require(torch.is_tensor(tensor) and tensor.dtype == torch.long and tensor.ndim == 2
             and tensor.shape[0] == 1 and minimum <= tensor.shape[1] <= maximum
             and str(tensor.device) == loaded.device, code)


def _validate(loaded: LoadedReference, prompt: PreparedPrompt, config: ReferenceConfig) -> None:
    try:
        validate_reference_config(config)
    except ValueError:
        raise ReferenceGenerationError("INVALID_CONFIG") from None
    _require(isinstance(loaded, LoadedReference) and loaded.config == config
             and (loaded.device, loaded.dtype) == (config.device, config.dtype)
             and loaded.eos_token_ids == (151645, 151643)
             and loaded.environment.get("profile_id") == config.profile_id
             and loaded.environment.get("status") == "IMPORTS_VERIFIED"
             and loaded.model.training is False
             and loaded.model.config.model_type == "qwen3"
             and loaded.model.config.vocab_size == 151936
             and loaded.model.config._attn_implementation == "eager", "INVALID_CONFIG")
    _require(isinstance(prompt, PreparedPrompt), "INVALID_PROMPT")
    ids = prompt.prompt_token_ids
    _require(type(ids) is tuple and 1 <= len(ids) <= config.max_prompt_tokens
             and all(type(v) is int and 0 <= v < 151936 for v in ids), "INVALID_PROMPT")
    _tensor(prompt.input_ids, loaded, len(ids), len(ids), "INVALID_PROMPT")
    _tensor(prompt.attention_mask, loaded, len(ids), len(ids), "INVALID_PROMPT")
    torch = loaded._torch
    # Prepared tensors remain mutable. Compare against bounded CPU identity before
    # executing, without .cpu(), .tolist(), output slicing or token decoding.
    expected = torch.tensor([list(ids)], dtype=torch.long, device=loaded.device)
    _require(torch.equal(prompt.input_ids, expected)
             and torch.equal(prompt.attention_mask, torch.ones_like(prompt.attention_mask)), "INVALID_PROMPT")
    text = prompt.rendered_prompt
    _require(type(text) is str and 0 < len(text) <= 65536
             and len(text.encode("utf-8")) <= 65536
             and hashlib.sha256(text.encode("utf-8")).hexdigest() == prompt.rendered_sha256
             and type(prompt.messages_sha256) is str
             and re.fullmatch(r"[0-9a-f]{64}", prompt.messages_sha256) is not None, "INVALID_PROMPT")


def _settings(config: ReferenceConfig, length: int) -> dict[str, object]:
    # Reuse the schema's single policy source, separating invocation metadata
    # from supported GenerationConfig fields. Never forward the envelope whole.
    settings = _generation(config)
    for key in ("use_model_defaults", "forward_kwargs", "cache_strategy", "custom_logits_processors",
                "custom_stopping_criteria", "streamer", "assistant_model", "custom_generate"):
        del settings[key]
    settings.update(max_length=length + config.max_new_tokens, min_length=length)
    return settings


def _export(settings: object) -> dict[str, object]:
    data = settings.to_dict()
    canonical_digest(data)  # bounded JSON validation before retaining any evidence
    return json.loads(json.dumps(data))


def _raw(loaded: LoadedReference, prompt: PreparedPrompt, config: ReferenceConfig) -> RawGenerationOutput:
    _validate(loaded, prompt, config)
    backend = _dependencies(loaded)
    settings = backend.GenerationConfig(**_settings(config, len(prompt.prompt_token_ids)))
    cache = backend.DynamicCache()
    _require(type(cache.get_seq_length()) is int and cache.get_seq_length() == 0, "INVALID_OUTPUT")
    with loaded.inference_context():
        output = loaded.model.generate(input_ids=prompt.input_ids, attention_mask=prompt.attention_mask,
            generation_config=settings, use_model_defaults=False, past_key_values=cache,
            logits_to_keep=1, logits_processor=[], stopping_criteria=[], streamer=None,
            assistant_model=None, custom_generate=None, synced_gpus=False,
            decoder_start_token_id=None)
    # The decoder-only model does not use decoder_start_token_id; explicitly
    # forwarding None also prevents the library's legacy special-ID fallback.
    sequences = getattr(output, "sequences", None)
    _tensor(sequences, loaded, len(prompt.prompt_token_ids) + 1,
            len(prompt.prompt_token_ids) + config.max_new_tokens, "INVALID_OUTPUT")
    _require(getattr(output, "past_key_values", None) is cache and type(cache) is backend.DynamicCache
             and type(cache.get_seq_length()) is int
             and cache.get_seq_length() == sequences.shape[1] - 1, "INVALID_OUTPUT")
    return RawGenerationOutput(sequences, cache, settings, config,
                               prompt.prompt_token_ids, id(loaded.model), backend.DynamicCache)


def generate_raw_reference(loaded: LoadedReference, prompt: PreparedPrompt,
                           config: ReferenceConfig) -> RawGenerationOutput:
    """Validate and call HF generate once with fresh greedy settings/cache.

    Bounds are token caps, not a wall-time/memory guarantee. Synchronous native
    failures are sanitized without retries or fallback; interruption propagates.
    No host token materialization, serialization or decoding occurs. A future raw-operation
    interval includes input comparisons, configuration validation and fresh cache
    construction; CUDA synchronization/timing remain the caller's responsibility.
    The caller must serialize use of the mutable loaded model and input tensors.
    """
    return _checked(_raw, loaded, prompt, config)


def _materialize(raw: RawGenerationOutput, loaded: LoadedReference, prompt: PreparedPrompt,
                 config: ReferenceConfig) -> GenerationResult:
    _validate(loaded, prompt, config)
    _require(isinstance(raw, RawGenerationOutput) and raw._config == config
             and raw._prompt_ids == prompt.prompt_token_ids and raw._model_id == id(loaded.model), "INVALID_OUTPUT")
    p = len(prompt.prompt_token_ids)
    _tensor(raw.sequences, loaded, p + 1, p + config.max_new_tokens, "INVALID_OUTPUT")
    _require(type(raw.past_key_values) is raw._cache_type
             and raw.past_key_values.get_seq_length() == raw.sequences.shape[1] - 1, "INVALID_OUTPUT")
    rows = raw.sequences.tolist()
    _require(type(rows) is list and len(rows) == 1 and type(rows[0]) is list
             and len(rows[0]) == raw.sequences.shape[1]
             and all(type(v) is int and 0 <= v < 151936 for v in rows[0]), "INVALID_OUTPUT")
    sequence = tuple(rows[0])
    generated = sequence[p:]
    _require(sequence[:p] == prompt.prompt_token_ids
             and not any(token in loaded.eos_token_ids for token in generated[:-1]), "INVALID_OUTPUT")
    reached = len(generated) == config.max_new_tokens
    reason = "eos" if generated[-1] in loaded.eos_token_ids else "limit"
    _require(reason == "eos" or reached, "INVALID_OUTPUT")
    raw_text = loaded.tokenizer.decode(list(generated), skip_special_tokens=False,
                                       clean_up_tokenization_spaces=False)
    display_text = loaded.tokenizer.decode(list(generated), skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
    _require(all(type(text) is str and len(text) <= 65536 and len(text.encode("utf-8")) <= 65536
                 for text in (raw_text, display_text)), "INVALID_OUTPUT")
    evidence = _export(raw.generation_config)
    expected = _settings(config, p)
    expected["transformers_version"] = "4.53.3"
    _require(all(key in evidence and type(evidence[key]) is type(value)
                 and evidence[key] == value for key, value in expected.items()), "INVALID_OUTPUT")
    return GenerationResult(prompt.prompt_token_ids, sequence, generated, raw_text,
                            display_text, reason, reached, evidence)


def materialize_generation(raw: RawGenerationOutput, loaded: LoadedReference, prompt: PreparedPrompt,
                           config: ReferenceConfig) -> GenerationResult:
    """After timing, validate/slice raw IDs and decode bounded raw/display strings.

    Includes the terminal EOS in raw generated IDs. EOS wins at the output cap;
    reached_limit is independently true at that cap. Reject incomplete non-EOS
    output and every token after an EOS instead of silently trimming. This can
    synchronize device tensors. It does not release the caller-owned raw cache.
    """
    try:
        return _checked(_materialize, raw, loaded, prompt, config)
    finally:
        # An outward exception retains this wrapper frame. Do not let that frame
        # extend the lifetime of caller-owned device sequences/cache.
        del raw


def generate_reference(loaded: LoadedReference, prompt: PreparedPrompt,
                       config: ReferenceConfig) -> GenerationResult:
    """Untimed correctness convenience; release local raw/cache references on exit.

    For future generation timing use generate_raw_reference, end the interval,
    then materialize_generation. No successful result is returned on failure.
    """
    raw = generate_raw_reference(loaded, prompt, config)
    try:
        return materialize_generation(raw, loaded, prompt, config)
    finally:
        del raw
