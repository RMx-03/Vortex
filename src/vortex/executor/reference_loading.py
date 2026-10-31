"""Pinned native loading and bounded prompt preparation; no generation calls.

Importing this module needs only the standard library. Calling load_reference
may acquire the seven fixed assets unless offline=True; actual model loading
requires a separately authorized execution host. Use a private, non-shared Hub
cache outside the checkout. Neither API qualifies pretrained outputs or GPUs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import importlib.util
from itertools import islice
import json
from pathlib import Path
import stat
import re
import subprocess
import sys
import sysconfig
from types import MappingProxyType, SimpleNamespace
from typing import TYPE_CHECKING

from vortex.executor.reference_config import MODEL_ID, REVISION, ReferenceConfig, validate_reference_config
from vortex.executor.reference_manifest import canonical_digest

if TYPE_CHECKING:
    from torch import Tensor
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

_ROOT = Path(__file__).resolve().parents[3]
_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
_SPECIAL_IDS = {"<|endoftext|>": 151643, "<|im_start|>": 151644, "<|im_end|>": 151645}
# Exact pinned Hub tree identities. git-sha1 includes the Git blob header;
# LFS SHA256 values identify actual bodies, never Git pointers or Xet hashes.
_ASSETS = {
    "config.json": (726, "sha256", "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
    "generation_config.json": (239, "sha256", "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
    "tokenizer_config.json": (9732, "sha256", "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
    "tokenizer.json": (11422654, "sha256", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "vocab.json": (2776833, "git-sha1", "4783fe10ac3adce15ac8f358ef5462739852c569"),
    "merges.txt": (1671853, "git-sha1", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
    "model.safetensors": (1503300328, "sha256", "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
}
_DIMENSIONS = {"model_type": "qwen3", "hidden_size": 1024, "intermediate_size": 3072,
               "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
               "head_dim": 128, "vocab_size": 151936, "bos_token_id": 151643, "eos_token_id": 151645}
SYNTHETIC_PROMPTS = ("Reply with the single word vortex.", "What is 2 + 2? Answer briefly.",
                     "Explain a key-value cache in one short sentence.")
_MESSAGES = {
    "INVALID_CONFIG": "Use the immutable pinned configuration and a boolean offline flag.",
    "ENVIRONMENT_MISMATCH": "Verify the complete locked profile with the checkout environment utility.",
    "ASSET_INVALID": "Use the exact reviewed seven-file snapshot in a private cache; check identities and missing files.",
    "MODEL_MISMATCH": "The native model or tokenizer does not match the pinned configuration and template.",
    "LOAD_FAILED": "Loading failed; inspect the authorized host and reviewed cached assets without changing the profile.",
    "OUT_OF_MEMORY": "Loading exhausted memory; release this attempt and use an authorized host with sufficient capacity.",
    "INVALID_PROMPT": "Use one bounded nonempty user message and unpadded integer inputs within the prompt limit.",
}


class ReferenceLoadingError(ValueError):
    """A finite public reason code; raw dependency diagnostics are not retained."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(_MESSAGES[code])


@dataclass(frozen=True)
class LoadedReference:
    """Own mutable native objects; scalar/config/asset evidence is immutable.

    The caller owns their lifetime. Drop all references after use/failure; this
    does not promise allocator-reserved memory reclamation or resumable loading.
    Future execution must separately enter inference_context(); loading's mode
    is scoped and does not disable gradients globally for later callers.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str
    dtype: str
    config: ReferenceConfig
    assets: Mapping[str, str]
    environment: Mapping[str, object]
    _torch: object = field(repr=False, compare=False)
    _inference: Callable[[], AbstractContextManager[object]] = field(repr=False, compare=False)
    eos_token_ids: tuple[int, int] = (151645, 151643)

    def inference_context(self) -> AbstractContextManager[object]:
        """Return a fresh inference-mode context; never execute a model here."""
        return self._inference()


@dataclass(frozen=True)
class PreparedPrompt:
    """Own integer [1,P] tensors and immutable CPU IDs; tensors remain mutable.

    P counts the entire template. Output/generation settings are not tokenization
    arguments. The rendered text/hash and message digest support manifest input
    identity; they are not evidence of pretrained correctness.
    """

    input_ids: Tensor
    attention_mask: Tensor
    prompt_token_ids: tuple[int, ...]
    rendered_sha256: str
    rendered_prompt: str
    messages_sha256: str


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ReferenceLoadingError(code)


def _json(data: bytes) -> dict[str, object]:
    _require(len(data) <= 65536, "ASSET_INVALID")

    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "ASSET_INVALID")
            result[key] = value
        return result

    def invalid(_):
        raise ReferenceLoadingError("ASSET_INVALID")

    result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    _require(type(result) is dict, "ASSET_INVALID")
    canonical_digest(result)
    return result


def _environment(config: ReferenceConfig) -> dict[str, object]:
    # Reuse the complete 0.1b receipt/profile validator; this adapter requires
    # the reviewed checkout, not merely an installed namespace-only wheel.
    script = _ROOT / "scripts/verify_reference_environment.py"
    _require(script.is_file(), "ENVIRONMENT_MISMATCH")
    run = subprocess.run([sys.executable, "-I", "-B", str(script), "--profile", config.profile_id],
                         capture_output=True, timeout=150, check=False, stdin=subprocess.DEVNULL)
    _require(run.returncode == 0 and len(run.stdout) <= 65536 and not run.stderr,
             "ENVIRONMENT_MISMATCH")
    report = _json(run.stdout)
    _require(report.get("status") == "IMPORTS_VERIFIED" and report.get("profile_id") == config.profile_id
             and report.get("qualified") is False and report.get("forward_smoke") == "NOT_VERIFIED"
             and type(report.get("observation")) is dict
             and report["observation"].get("api_contract") is True
             and report["observation"].get("pip_check") is True, "ENVIRONMENT_MISMATCH")
    return report


def _import_origins(environment: Mapping[str, object] | None) -> None:
    roots = {Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")}
    names = {"torch", "transformers", "huggingface_hub", "tokenizers", "safetensors"}
    if environment is not None:
        expected = environment["observation"]["packages"]
        found = set()
        for distribution in importlib.metadata.distributions(path=[str(p) for p in roots]):
            name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
            _require(name in expected and distribution.version == expected[name]
                     and distribution.files is not None, "ENVIRONMENT_MISMATCH")
            found.add(name)
            for entry in distribution.files:
                first = entry.parts[0]
                if len(entry.parts) == 1:
                    first = first.split(".")[0] if str(entry).endswith((".py", ".pyd", ".so")) else ""
                if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", first) and first != "__pycache__":
                    names.add(first)
        _require(found == set(expected), "ENVIRONMENT_MISMATCH")
    for name in sorted(names):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            raise ReferenceLoadingError("ENVIRONMENT_MISMATCH") from None
        _require(spec is not None, "ENVIRONMENT_MISMATCH")
        if spec.origin is not None:
            origin = Path(spec.origin).resolve()
            allowed = {root / name / "__init__.py" for root in roots} | {root / (name + ".py") for root in roots}
            # Standalone extension filenames carry interpreter/platform suffixes.
            _require(origin in allowed or (origin.parent in roots and origin.name.split(".")[0] == name
                     and origin.suffix in (".pyd", ".so")), "ENVIRONMENT_MISMATCH")
        else:
            locations = list(islice(spec.submodule_search_locations or (), 5))
            _require(1 <= len(locations) <= 4 and all(Path(p).resolve() in
                     {root / name for root in roots} for p in locations), "ENVIRONMENT_MISMATCH")
        present = sys.modules.get(name)
        if present is not None:
            _require(getattr(present, "__file__", None) == spec.origin, "ENVIRONMENT_MISMATCH")
            if hasattr(present, "__path__"):
                paths = list(islice(present.__path__, 5))
                _require(1 <= len(paths) <= 4 and all(Path(p).resolve() in
                         {root / name for root in roots} for p in paths), "ENVIRONMENT_MISMATCH")


def _dependencies(environment: Mapping[str, object] | None = None) -> SimpleNamespace:
    # A qualified -I worker does not bind this process's sys.path/sys.modules.
    # Check every locked distribution's import roots before native imports.
    _import_origins(environment)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, Qwen3Config
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE

    return SimpleNamespace(torch=torch, AutoModelForCausalLM=AutoModelForCausalLM,
                           AutoTokenizer=AutoTokenizer, GenerationConfig=GenerationConfig,
                           Qwen3Config=Qwen3Config, hf_hub_download=hf_hub_download, cache=Path(HF_HUB_CACHE))


def _no_reparse(path: Path) -> None:
    for item in (path, *path.parents):
        _require(not item.is_symlink(), "ASSET_INVALID")
        if item.exists():
            _require(not getattr(item.lstat(), "st_file_attributes", 0)
                     & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0), "ASSET_INVALID")


def _asset_path(path: Path, snapshot: Path, repo: Path) -> None:
    _require(path.parent == snapshot and path.name in _ASSETS, "ASSET_INVALID")
    if path.is_symlink():
        _require(path.resolve().is_relative_to(repo / "blobs"), "ASSET_INVALID")
    else:
        _no_reparse(path)
    _require(path.resolve().is_relative_to(repo.resolve()), "ASSET_INVALID")


def _hash_asset(path: Path, identity: tuple[int, str, str]) -> str:
    size, algorithm, expected = identity
    _require(path.is_file() and path.stat().st_size == size, "ASSET_INVALID")
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(usedforsecurity=False)
    git_blob.update(b"blob " + str(size).encode("ascii") + b"\0")
    read = 0
    with path.open("rb") as source:
        while chunk := source.read(min(1048576, size - read + 1)):
            read += len(chunk)
            _require(read <= size, "ASSET_INVALID")
            sha256.update(chunk)
            git_blob.update(chunk)
    actual = sha256.hexdigest() if algorithm == "sha256" else git_blob.hexdigest()
    _require(read == size and actual == expected, "ASSET_INVALID")
    return sha256.hexdigest()


def _entries(path: Path, maximum: int) -> list[Path]:
    entries = list(islice(path.iterdir(), maximum + 1)) if path.exists() else []
    _require(len(entries) <= maximum, "ASSET_INVALID")
    return entries


def _write_paths(cache: Path, repo: Path) -> None:
    locks = cache / ".locks" / repo.name
    for directory in (locks, repo / "blobs", repo / ".no_exist"):
        _no_reparse(directory)
        for entry in _entries(directory, 128):
            _no_reparse(entry)
            if directory.name == ".no_exist":
                _require(entry.name == REVISION and entry.is_dir(), "ASSET_INVALID")
                for leaf in _entries(entry, 7):
                    _no_reparse(leaf)
                    _require(leaf.name in _ASSETS and leaf.is_file(), "ASSET_INVALID")
            else:
                _require(entry.is_file(), "ASSET_INVALID")


def _assets(backend: SimpleNamespace, offline: bool) -> tuple[Path, dict, dict[str, str]]:
    cache = Path(backend.cache)
    _require(cache.is_absolute() and not cache.resolve().is_relative_to(_ROOT), "ASSET_INVALID")
    repo = cache / "models--Qwen--Qwen3-0.6B"
    snapshot = repo / "snapshots" / REVISION
    for directory in (cache, snapshot, repo / "blobs"):
        _no_reparse(directory)
    _write_paths(cache, repo)
    if snapshot.exists():
        _require({p.name for p in _entries(snapshot, 7)} <= _ASSETS.keys(), "ASSET_INVALID")
    hashes, parsed = {}, {}
    for name, identity in _ASSETS.items():
        path = snapshot / name
        if path.exists() or path.is_symlink():
            _asset_path(path, snapshot, repo)
        returned = Path(backend.hf_hub_download(repo_id=MODEL_ID, filename=name, revision=REVISION,
            cache_dir=str(cache), local_files_only=offline, token=False, etag_timeout=10))
        _require(returned == path, "ASSET_INVALID")
        _asset_path(path, snapshot, repo)
        hashes[name] = _hash_asset(path, identity)
        if name in ("config.json", "generation_config.json", "tokenizer_config.json"):
            parsed[name] = _json(path.read_bytes())
            _require(not any(key.startswith(("_attn", "attn_implementation", "auto_map",
                         "quantization_config", "adapter", "custom_generate")) for key in parsed[name]),
                     "ASSET_INVALID")
    _require({p.name for p in _entries(snapshot, 7)} == _ASSETS.keys(), "ASSET_INVALID")
    template = parsed["tokenizer_config.json"].get("chat_template")
    _require(type(template) is str and hashlib.sha256(template.encode()).hexdigest() == _TEMPLATE_SHA256,
             "MODEL_MISMATCH")
    raw = parsed["config.json"]
    _require(all(type(raw.get(k)) is type(v) and raw[k] == v for k, v in _DIMENSIONS.items())
             and raw.get("architectures") == ["Qwen3ForCausalLM"], "MODEL_MISMATCH")
    generation = parsed["generation_config.json"]
    _require(generation.get("eos_token_id") == [151645, 151643]
             and generation.get("pad_token_id") == 151643
             and generation.get("bos_token_id") == 151643, "MODEL_MISMATCH")
    evidence = {"model_config_sha256": hashes["config.json"],
                "generation_config_sha256": hashes["generation_config.json"],
                "tokenizer_config_sha256": hashes["tokenizer_config.json"],
                "chat_template_sha256": _TEMPLATE_SHA256,
                "weights_sha256": canonical_digest({"model.safetensors": hashes["model.safetensors"]}),
                "tokenizer_assets_sha256": canonical_digest({name: hashes[name] for name in
                    ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt")})}
    return snapshot, parsed, evidence


def _tokenizer_matches(tokenizer: PreTrainedTokenizerBase) -> None:
    _require(tokenizer.is_fast is True and type(tokenizer.eos_token_id) is int
             and tokenizer.eos_token_id == _SPECIAL_IDS["<|im_end|>"]
             and type(tokenizer.pad_token_id) is int
             and tokenizer.pad_token_id == _SPECIAL_IDS["<|endoftext|>"] and type(tokenizer.chat_template) is str
             and hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == _TEMPLATE_SHA256
             and all(type(tokenizer.convert_tokens_to_ids(token)) is int
                     and tokenizer.convert_tokens_to_ids(token) == value for token, value in _SPECIAL_IDS.items()),
             "MODEL_MISMATCH")


def _load(config: ReferenceConfig, offline: bool) -> LoadedReference:
    try:
        validate_reference_config(config)
    except ValueError:
        raise ReferenceLoadingError("INVALID_CONFIG") from None
    _require(type(offline) is bool, "INVALID_CONFIG")
    environment = _environment(config)
    backend = _dependencies(environment)
    snapshot, parsed, assets = _assets(backend, offline)
    native_config = backend.Qwen3Config.from_dict(parsed["config.json"])
    generation = backend.GenerationConfig.from_dict(parsed["generation_config.json"])
    dtype = getattr(backend.torch, config.dtype)
    tokenizer = backend.AutoTokenizer.from_pretrained(str(snapshot), revision=REVISION,
        local_files_only=True, trust_remote_code=False, use_fast=True, token=False)
    _tokenizer_matches(tokenizer)
    with backend.torch.inference_mode(), backend.torch.device("cpu"):
        model = backend.AutoModelForCausalLM.from_pretrained(str(snapshot), revision=REVISION,
            local_files_only=True, trust_remote_code=False, use_safetensors=True, token=False,
            config=native_config, generation_config=generation, torch_dtype=dtype,
            attn_implementation="eager")
        model.to(device=config.device, dtype=dtype)
        model.eval()
    _require(model.training is False and model.config._attn_implementation == "eager"
             and all(type(getattr(model.config, k)) is type(v) and getattr(model.config, k) == v
                     for k, v in _DIMENSIONS.items())
             and model.generation_config.eos_token_id == [151645, 151643], "MODEL_MISMATCH")
    count = 0
    for parameter in model.parameters():
        count += 1
        _require(count <= 1024 and str(parameter.device) == config.device and parameter.dtype == dtype,
                 "MODEL_MISMATCH")
    _require(count > 0, "MODEL_MISMATCH")
    return LoadedReference(model, tokenizer, config.device, config.dtype, config,
                           MappingProxyType(assets), MappingProxyType(environment), backend.torch,
                           backend.torch.inference_mode)


def load_reference(config: ReferenceConfig, *, offline: bool = False) -> LoadedReference:
    """Validate environment/assets, load on CPU, then explicitly move and evaluate.

    No generation occurs. No caller-supplied HF options, local paths, aliases,
    remote code, pickle, shard indexes or fallback profiles are accepted. Hub
    handles fixed-file I/O with an ETag timeout; no retry loop is added here.
    Native construction/transfer is synchronous, without a hard wall-time or
    memory guarantee. Failed local references unwind; cached files are retained,
    never deleted. Errors are bounded ReferenceLoadingError codes without raw
    dependency causes/tracebacks. KeyboardInterrupt propagates normally.
    """
    try:
        return _load(config, offline)
    except ReferenceLoadingError as error:
        code = error.code
    except MemoryError:
        code = "OUT_OF_MEMORY"
    except RuntimeError as error:
        code = "OUT_OF_MEMORY" if type(error).__name__ == "OutOfMemoryError" else "LOAD_FAILED"
    except (ImportError, OSError, ValueError, TypeError, AttributeError, KeyError,
            RecursionError, subprocess.SubprocessError):
        code = "LOAD_FAILED"
    # Raising outside the except suite avoids retaining raw exception frames
    # (and their partially loaded model references) through __context__.
    raise ReferenceLoadingError(code)


def _tensor_ids(tensor: Tensor, torch: object, maximum: int, device: str) -> tuple[int, ...]:
    _require(torch.is_tensor(tensor) and tensor.dtype == torch.long and tensor.ndim == 2
             and tensor.shape[0] == 1 and 1 <= tensor.shape[1] <= maximum
             and str(tensor.device) == device, "INVALID_PROMPT")
    values = tensor.tolist()[0]
    _require(len(values) == tensor.shape[1] and all(type(v) is int and 0 <= v < 151936 for v in values),
             "INVALID_PROMPT")
    return tuple(values)


def _prepare(loaded: LoadedReference, messages: Sequence[Mapping[str, str]],
             config: ReferenceConfig) -> PreparedPrompt:
    try:
        validate_reference_config(config)
    except ValueError:
        raise ReferenceLoadingError("INVALID_CONFIG") from None
    _require(isinstance(loaded, LoadedReference) and loaded.config == config
             and (loaded.device, loaded.dtype) == (config.device, config.dtype), "INVALID_CONFIG")
    _require(type(messages) in (list, tuple) and len(messages) == 1
             and isinstance(messages[0], Mapping),
             "INVALID_PROMPT")
    keys = list(islice(iter(messages[0]), 3))
    _require(len(keys) == 2 and all(type(k) is str for k in keys)
             and set(keys) == {"role", "content"}, "INVALID_PROMPT")
    message = {key: messages[0][key] for key in keys}
    content = message["content"]
    _require(type(message["role"]) is str and message["role"] == "user"
             and type(content) is str and content.strip() != ""
             and len(content) <= 16384 and len(content.encode("utf-8")) <= 65536, "INVALID_PROMPT")
    owned = [message]
    tokenizer, torch = loaded.tokenizer, loaded._torch
    _tokenizer_matches(tokenizer)
    rendered = tokenizer.apply_chat_template(owned, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    _require(type(rendered) is str and 0 < len(rendered.encode("utf-8")) <= 65536, "INVALID_PROMPT")
    batch = tokenizer.apply_chat_template(owned, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_tensors="pt", return_dict=True, padding=False, truncation=False,
        tokenizer_kwargs={"return_attention_mask": True})
    _require(isinstance(batch, Mapping) and {"input_ids", "attention_mask"} <= batch.keys(), "INVALID_PROMPT")
    ids = _tensor_ids(batch["input_ids"], torch, config.max_prompt_tokens, "cpu")
    mask = _tensor_ids(batch["attention_mask"], torch, config.max_prompt_tokens, "cpu")
    _require(len(mask) == len(ids) and all(v == 1 for v in mask), "INVALID_PROMPT")
    replay = tokenizer(rendered, add_special_tokens=False, return_tensors="pt", return_attention_mask=True,
                       padding=False, truncation=False)
    _require(_tensor_ids(replay["input_ids"], torch, config.max_prompt_tokens, "cpu") == ids,
             "INVALID_PROMPT")
    inputs = batch["input_ids"].to(device=config.device)
    attention = batch["attention_mask"].to(device=config.device)
    _require(_tensor_ids(inputs, torch, config.max_prompt_tokens, config.device) == ids
             and _tensor_ids(attention, torch, config.max_prompt_tokens, config.device) == mask, "INVALID_PROMPT")
    return PreparedPrompt(inputs, attention, ids, hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                          rendered, canonical_digest({"messages": owned}))


def prepare_prompt(loaded: LoadedReference, messages: Sequence[Mapping[str, str]],
                   config: ReferenceConfig) -> PreparedPrompt:
    """Prepare one unpadded non-thinking user prompt, rejecting rather than truncating.

    Input text is bounded to 16,384 codepoints/64 KiB before templating; rendered
    text to 64 KiB; the final prompt to config.max_prompt_tokens <=512. Validate
    CPU integer tensors before transfer and preserve dtype on the explicit target.
    Check rendered/tokenized replay without adding special tokens twice. No
    generation settings, network, model call or logging of input text occurs.
    """
    try:
        return _prepare(loaded, messages, config)
    except ReferenceLoadingError as error:
        code = error.code
    except MemoryError:
        code = "OUT_OF_MEMORY"
    except (ValueError, TypeError, RuntimeError, OSError, AttributeError, KeyError, RecursionError):
        code = "INVALID_PROMPT"
    raise ReferenceLoadingError(code)
