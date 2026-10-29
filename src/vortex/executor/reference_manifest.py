"""Offline version-one configuration and evidence validation.

Only JSON-compatible data crosses this boundary. These functions neither load
assets nor qualify environments; supplied evidence must be verified separately.
"""

from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime
import hashlib
import json
import math
import re
from uuid import UUID

from vortex.executor.reference_config import ReferenceConfig

_MAX_BYTES = 1_048_576
_CONFIG_KEYS = frozenset(field.name for field in fields(ReferenceConfig))
_PAYLOAD_KEYS = ("schema_version", "config", "chat_template", "generation")
_MANIFEST_KEYS = frozenset((*_PAYLOAD_KEYS, "config_sha256", "numerical",
                            "comparison_key_sha256", "provenance", "outputs",
                            "complete", "outcome"))
_KNOWN_ASSETS = {
    "model_config_sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config_sha256": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "tokenizer_config_sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
}


def _json_value(value: object, depth: int, budget: list[int]) -> object:
    if depth > 32:
        raise ValueError("JSON nesting exceeds limit")
    # Bound work while expanding Mapping inputs, including repeated aliases;
    # checking encoded length alone happens after the expensive snapshot.
    cost = 1
    if type(value) is str:
        cost += len(value)
    elif type(value) is int:
        # An upper bound on decimal digits avoids converting a huge integer
        # just to measure it. log10(2) < 30103/100000.
        cost += value.bit_length() * 30103 // 100000 + 1 + (value < 0)
    elif type(value) is float:
        cost += 24
    budget[0] -= cost
    if budget[0] < 0:
        raise ValueError("JSON snapshot exceeds limit")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        if len(value) > _MAX_BYTES:
            raise ValueError("JSON string exceeds limit")
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        result = {}
        for key in value:
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            budget[0] -= 1 + len(key)
            if budget[0] < 0:
                raise ValueError("JSON snapshot exceeds limit")
            result[key] = _json_value(value[key], depth + 1, budget)
        return result
    if type(value) is list:
        return [_json_value(item, depth + 1, budget) for item in value]
    raise ValueError("unsupported JSON value")


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object")
    try:
        data = json.dumps(_json_value(value, 0, [_MAX_BYTES]), sort_keys=True, ensure_ascii=False,
                          allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (UnicodeError, RecursionError, ValueError) as exc:
        raise ValueError("invalid or oversized canonical JSON") from exc
    if len(data) > _MAX_BYTES:
        raise ValueError("canonical JSON exceeds byte limit")
    return data


def canonical_digest(value: Mapping[str, object]) -> str:
    """Return SHA256 of compact, key-sorted JSON encoded as UTF-8, without BOM.

    Arrays keep their order; Unicode codepoints and numeric types are preserved
    (1 and 1.0 differ). Only finite JSON values are accepted, with a 1 MiB encoded
    ceiling and 32 nesting levels. Raise ValueError without echoing input on
    malformed/oversized data. No mutation, I/O or inference imports occur.
    """
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: object, keys: frozenset[str] | set[str], name: str) -> dict:
    if not isinstance(value, dict) or value.keys() != keys:
        raise ValueError(f"invalid {name} fields")
    return value


def _text(value: object, name: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"invalid {name}")


def _hash(value: object, name: str, length: int = 64) -> None:
    if type(value) is not str or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ValueError(f"invalid {name} digest")


def _equal(value: object, expected: object, name: str) -> None:
    # Python's True == 1 must not turn malformed settings into valid integers.
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"unsupported {name}")


def _generation(config: ReferenceConfig) -> dict[str, object]:
    return {
        "do_sample": False, "num_beams": 1, "num_beam_groups": 1,
        "num_return_sequences": 1, "max_new_tokens": config.max_new_tokens,
        "min_new_tokens": 0, "use_cache": True, "eos_token_id": [151645, 151643],
        "pad_token_id": 151643, "bos_token_id": 151643,
        "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "length_penalty": 1.0,
        "return_dict_in_generate": True, "output_scores": False, "output_logits": False,
        "output_attentions": False, "output_hidden_states": False,
        "temperature": 1.0, "top_p": 1.0, "top_k": 50,
        "use_model_defaults": False, "forward_kwargs": {"logits_to_keep": 1},
        "cache_strategy": "fresh_dynamic", "custom_logits_processors": [],
        "custom_stopping_criteria": [], "streamer": None, "assistant_model": None,
        "custom_generate": None, "max_time": None,
    }


def _fixed_fields(value: object, expected: dict[str, object], name: str) -> None:
    record = _object(value, set(expected), name)
    for key, setting in expected.items():
        # Compare canonical forms as well to preserve nested integer/bool types.
        _equal(record[key], setting, name)
        if canonical_digest({"v": record[key]}) != canonical_digest({"v": setting}):
            raise ValueError(f"unsupported {name} setting")


def _provenance(value: object) -> None:
    record = _object(value, {"source_sha", "dirty", "patch_sha256", "adapter_version",
                             "run_id", "session_id", "timestamp"}, "provenance")
    _hash(record["source_sha"], "source", 40)
    if type(record["dirty"]) is not bool:
        raise ValueError("dirty must be boolean")
    if record["dirty"]:
        _hash(record["patch_sha256"], "patch")
    elif record["patch_sha256"] is not None:
        raise ValueError("clean source cannot have a patch digest")
    _text(record["adapter_version"], "adapter version")
    for key in ("run_id", "session_id"):
        value = record[key]
        if type(value) is not str:
            raise ValueError("invalid provenance identifier")
        try:
            if str(UUID(value)) != value:
                raise ValueError("noncanonical UUID")
        except ValueError:
            raise ValueError("invalid provenance identifier") from None
    timestamp = record["timestamp"]
    if (type(timestamp) is not str or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", timestamp) is None):
        raise ValueError("timestamp must be UTC ISO8601")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("invalid provenance timestamp") from None


def _ids(value: object, maximum: int, name: str) -> list[int]:
    if (type(value) is not list or not 1 <= len(value) <= maximum
            or any(type(token) is not int or token < 0 for token in value)):
        raise ValueError(f"invalid {name} token IDs")
    return value


def _environment(value: object, config: ReferenceConfig) -> None:
    record = _object(value, {"python_version", "python_build", "platform", "machine",
                             "packages", "hardware"}, "environment")
    for name in ("python_version", "python_build", "platform", "machine"):
        _text(record[name], "environment version/build")
    if re.fullmatch(r"3\.11\.[0-9]+", record["python_version"]) is None:
        raise ValueError("reference requires an exact Python 3.11 patch")
    packages = record["packages"]
    if not isinstance(packages, dict) or not {"torch", "transformers"} <= packages.keys():
        raise ValueError("missing reference package evidence")
    for name, package in packages.items():
        if re.fullmatch(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", name) is None:
            raise ValueError("invalid package name")
        info = _object(package, {"version", "wheel_sha256"}, "package")
        _text(info["version"], "package version")
        _hash(info["wheel_sha256"], "wheel")
    torch_version = "2.10.0+cpu" if config.device == "cpu" else "2.10.0+cu126"
    _equal(packages["torch"]["version"], torch_version, "torch version")
    _equal(packages["transformers"]["version"], "4.53.3", "transformers version")
    hardware = _object(record["hardware"], {"device", "gpu_name", "compute_capability",
                                           "driver_version", "cuda_runtime"}, "hardware")
    _equal(hardware["device"], config.device, "hardware device")
    if config.device == "cpu":
        if any(hardware[key] is not None for key in hardware if key != "device"):
            raise ValueError("CPU profile cannot claim GPU hardware")
    else:
        for key in ("gpu_name", "driver_version", "cuda_runtime"):
            _text(hardware[key], "GPU evidence")
        capability = hardware["compute_capability"]
        if (type(capability) is not list or len(capability) != 2
                or any(type(item) is not int or item < 0 for item in capability)):
            raise ValueError("invalid compute capability")


def _numerical(value: object, config: ReferenceConfig) -> dict:
    record = _object(value, {"assets", "prompt", "environment", "determinism",
                             "effective_generation_config"}, "numerical")
    assets = _object(record["assets"], set(_KNOWN_ASSETS) | {
        "tokenizer_assets_sha256", "weights_sha256"}, "assets")
    for name, digest in assets.items():
        _hash(digest, "asset")
        if name in _KNOWN_ASSETS:
            _equal(digest, _KNOWN_ASSETS[name], "pinned asset hash")
    prompt = _object(record["prompt"], {"messages", "messages_sha256", "rendered_prompt",
                                        "rendered_sha256", "input_ids"}, "prompt")
    messages = prompt["messages"]
    if type(messages) is not list or len(messages) != 1:
        raise ValueError("reference requires one user message")
    message = _object(messages[0], {"role", "content"}, "message")
    _equal(message["role"], "user", "message role")
    _text(message["content"], "message content")
    _equal(prompt["messages_sha256"], canonical_digest({"messages": messages}), "messages hash")
    _text(prompt["rendered_prompt"], "rendered prompt")
    rendered_digest = hashlib.sha256(prompt["rendered_prompt"].encode("utf-8")).hexdigest()
    _equal(prompt["rendered_sha256"], rendered_digest, "rendered hash")
    _ids(prompt["input_ids"], config.max_prompt_tokens, "prompt")
    _environment(record["environment"], config)
    # The envelope is a strict input contract; this separate object retains
    # the library's full serialized evidence, including inactive/default fields.
    # Extra evidence is hashed, never forwarded as execution kwargs here.
    controls = {"use_model_defaults", "forward_kwargs", "cache_strategy",
                "custom_logits_processors", "custom_stopping_criteria", "streamer",
                "assistant_model", "custom_generate"}
    expected = {key: setting for key, setting in _generation(config).items()
                if key not in controls}
    expected["transformers_version"] = "4.53.3"
    effective = record["effective_generation_config"]
    if not isinstance(effective, dict) or not expected.keys() <= effective.keys():
        raise ValueError("missing effective generation configuration")
    _fixed_fields({key: effective[key] for key in expected}, expected, "effective generation")
    _fixed_fields(record["determinism"], {
        "seed": config.seed, "pythonhashseed": "0", "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False, "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False, "deterministic_algorithms": True,
    }, "determinism")
    return record


def _outputs(value: object, numerical: dict, config: ReferenceConfig) -> None:
    record = _object(value, {"prompt_ids", "sequence_ids", "generated_ids", "raw_text",
                             "display_text", "stop_reason", "reached_limit"}, "outputs")
    prompt = numerical["prompt"]["input_ids"]
    _ids(record["prompt_ids"], config.max_prompt_tokens, "output prompt")
    _equal(record["prompt_ids"], prompt, "output prompt IDs")
    generated = _ids(record["generated_ids"], config.max_new_tokens, "generated")
    sequence = _ids(record["sequence_ids"], 576, "sequence")
    if sequence != prompt + generated or any(token in (151645, 151643) for token in generated[:-1]):
        raise ValueError("inconsistent output sequence")
    reached_limit = len(generated) == config.max_new_tokens
    reason = "eos" if generated[-1] in (151645, 151643) else "limit"
    if reason == "limit" and not reached_limit:
        raise ValueError("output stopped before EOS or limit")
    _equal(record["stop_reason"], reason, "stop reason")
    _equal(record["reached_limit"], reached_limit, "reached limit")
    for name in ("raw_text", "display_text"):
        if type(record[name]) is not str:
            raise ValueError("decoded output must be text")


def validate_manifest(record: Mapping[str, object]) -> None:
    """Raise ValueError on a malformed, incompatible or internally stale record.

    Schema 1 rejects missing/extra fields, including unsupported generation
    settings. Digests cover configuration and numerical evidence separately;
    provenance and outputs are excluded from numerical compatibility. Drafts
    without numerical evidence have no comparison key and cannot be complete.
    Success means structural consistency only, never model/environment
    qualification or token parity. Input mappings are copied, not mutated.
    """
    # Roundtrip gives validation an owned JSON snapshot and applies the same
    # finite-value, depth and size boundary used by canonical_digest.
    snapshot = json.loads(_canonical_bytes(record))
    data = _object(snapshot, _MANIFEST_KEYS, "manifest")
    _equal(data["schema_version"], 1, "schema version")
    config = ReferenceConfig(**_object(data["config"], _CONFIG_KEYS, "config"))
    _fixed_fields(data["chat_template"], {
        "tokenize": True, "add_generation_prompt": True, "enable_thinking": False,
    }, "chat template")
    _fixed_fields(data["generation"], _generation(config), "generation")
    payload = {key: data[key] for key in _PAYLOAD_KEYS}
    _equal(data["config_sha256"], canonical_digest(payload), "configuration digest")
    _provenance(data["provenance"])
    numerical = None
    if data["numerical"] is None:
        if data["comparison_key_sha256"] is not None:
            raise ValueError("draft has no numerical comparison key")
    else:
        numerical = _numerical(data["numerical"], config)
        key = canonical_digest({"config_sha256": data["config_sha256"], "numerical": numerical})
        _equal(data["comparison_key_sha256"], key, "numerical comparison key")
    if type(data["complete"]) is not bool:
        raise ValueError("complete must be boolean")
    if data["complete"]:
        if numerical is None:
            raise ValueError("complete record requires numerical evidence")
        _equal(data["outcome"], "success", "complete outcome")
        _outputs(data["outputs"], numerical, config)
    elif (data["outcome"] not in ("configured", "failed", "interrupted")
          or data["outputs"] is not None):
        raise ValueError("incomplete record requires explicit outcome and null outputs")
