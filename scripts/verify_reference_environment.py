#!/usr/bin/env python3
"""Read-only profile checks and hash-lock extraction; never load a model.

Exit 0 means the requested static/import checks passed, not inference qualification.
Exit 2 means missing, incompatible or incomplete evidence. Heavy imports execute
only in a bounded isolated worker; offline unit tests use supplied observations.
"""

from collections.abc import Mapping
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vortex.executor.reference_manifest import canonical_digest

PROFILES = {
    "windows-cpu": ("Windows", "AMD64", "2.10.0+cpu", "cpu"),
    "t4-fp16": ("Linux", "x86_64", "2.10.0+cu126", "cu126"),
}
LOCK_FIELDS = {"schema_version", "profile_id", "python_version", "implementation", "system",
               "machine", "index_url", "torch_index_url", "roots", "packages"}
PACKAGE_FIELDS = {"name", "version", "url", "sha256", "requires"}
FORBIDDEN = {"kernels", "liger-kernel", "flash-attn", "bitsandbytes", "accelerate",
             "peft", "torchvision", "torchaudio", "tensorflow", "flax"}
REMEDIATIONS = {
    "MISSING_DEPENDENCY": "Install the complete reviewed profile lock in an isolated environment.",
    "INTERPRETER_MISMATCH": "Use the exact locked Python build and platform.",
    "IMPORT_FAILED": "Check pinned native wheels and platform runtime prerequisites; do not substitute builds.",
    "CUDA_UNAVAILABLE": "Use the approved Linux GPU profile with a compatible device and driver; no CPU fallback.",
    "DEPENDENCY_CHECK_FAILED": "Recreate the isolated environment from the complete hash lock and run pip check.",
    "INSPECTION_TIMEOUT": "Inspect the isolated environment for a stalled native import or device query.",
    "LOCK_INVALID": "Check the reviewed lock schema, wheel origins, hashes and dependency closure.",
}


class EnvironmentFailure(ValueError):
    """A bounded reason code, never raw native diagnostics or supplied values."""

    def __init__(self, code: str) -> None:
        self.code = code if code in REMEDIATIONS else "LOCK_INVALID"
        super().__init__(REMEDIATIONS[self.code])


def _name(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is not None


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def validate_lock(lock: Mapping[str, object]) -> None:
    """Validate profile identity, official wheel origins, hashes and named closure.

    Pure standard-library validation. This does not establish that supplied wheel
    metadata is truthful or that native imports work; resolver and import evidence
    must accompany a lock. Malformed data raises ValueError without echoing it.
    """
    canonical_digest(lock)
    _require(isinstance(lock, Mapping) and set(lock) == LOCK_FIELDS, "invalid lock fields")
    _require(type(lock["schema_version"]) is int and lock["schema_version"] == 1,
             "unsupported lock schema")
    profile = lock["profile_id"]
    _require(type(profile) is str and profile in PROFILES, "unsupported profile")
    system, machine, torch_version, backend = PROFILES[profile]
    _require(lock["system"] == system and lock["machine"] == machine
             and lock["implementation"] == "CPython", "invalid platform profile")
    _require(lock["python_version"] == "3.11.18", "reviewed Python 3.11.18 patch required")
    _require(lock["index_url"] == "https://pypi.org/simple" and lock["torch_index_url"] ==
             "https://download.pytorch.org/whl/" + backend, "incorrect wheel indexes")
    packages, roots = lock["packages"], lock["roots"]
    _require(type(packages) is list and 2 <= len(packages) <= 128, "incomplete package lock")
    _require(type(roots) is list and all(_name(item) for item in roots)
             and len(set(roots)) == len(roots) and {"torch", "transformers"} <= set(roots),
             "invalid root packages")
    by_name = {}
    for package in packages:
        _require(type(package) is dict and set(package) == PACKAGE_FIELDS, "invalid package fields")
        name = package["name"]
        _require(_name(name) and name not in by_name and name not in FORBIDDEN,
                 "duplicate or unsupported package")
        _require(type(package["version"]) is str and re.fullmatch(
            r"[0-9][A-Za-z0-9.+!-]*", package["version"]) is not None, "invalid package version")
        _require(type(package["sha256"]) is str and re.fullmatch(
            r"[a-f0-9]{64}", package["sha256"]) is not None, "invalid wheel hash")
        dependencies = package["requires"]
        _require(type(dependencies) is list and all(_name(item) for item in dependencies)
                 and len(dependencies) == len(set(dependencies)), "invalid package dependencies")
        _require(type(package["url"]) is str, "invalid wheel URL")
        _require(re.fullmatch(r"[!-~]+", package["url"]) is not None,
                 "whitespace or control character in wheel URL")
        try:
            url = urlsplit(package["url"])
            port = url.port
        except ValueError:
            raise ValueError("invalid wheel URL") from None
        hosts = {"download.pytorch.org", "download-r2.pytorch.org"} if name == "torch" else {
            "files.pythonhosted.org"}
        _require(url.scheme == "https" and url.hostname in hosts and port is None
                 and url.username is None and url.password is None
                 and not url.query and not url.fragment and url.path.endswith(".whl"),
                 "incorrect wheel origin")
        if name == "torch":
            suffix = "win_amd64.whl" if profile == "windows-cpu" else "manylinux_2_28_x86_64.whl"
            _require(unquote(url.path) == "/whl/" + backend + "/torch-" + torch_version
                     + "-cp311-cp311-" + suffix, "incorrect platform torch wheel")
        by_name[name] = package
    _require({"torch", "transformers"} <= by_name.keys(), "missing reference packages")
    _require(by_name["torch"]["version"] == torch_version
             and by_name["transformers"]["version"] == "4.53.3", "reference version mismatch")
    pending, reachable = list(roots), set()
    while pending:
        name = pending.pop()
        _require(name in by_name, "missing transitive package")
        if name not in reachable:
            reachable.add(name)
            pending.extend(by_name[name]["requires"])
    _require(reachable == by_name.keys(), "orphan or incomplete package lock")


def _strict_json(text: str) -> object:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def constant(_):
        raise ValueError("nonfinite JSON value")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except RecursionError:
        raise ValueError("JSON nesting exceeds limit") from None


def read_lock(document: Path, profile_id: str) -> dict[str, object]:
    """Read one reviewed JSON lock fence; reject duplicates and ambiguous blocks."""
    _require(profile_id in PROFILES, "unsupported profile")
    _require(document.stat().st_size <= 1_048_576, "lock document exceeds limit")
    pattern = r"<!-- vortex-lock:" + re.escape(profile_id) + r" -->\r?\n```json\r?\n(.*?)\r?\n```"
    blocks = re.findall(pattern, document.read_text(encoding="utf-8"), re.S)
    _require(len(blocks) == 1, "missing or ambiguous profile lock")
    lock = _strict_json(blocks[0])
    validate_lock(lock)
    _require(lock["profile_id"] == profile_id, "profile identity mismatch")
    return lock


def requirements_text(lock: Mapping[str, object], *, bootstrap_only: bool = False) -> str:
    """Return a pip hash lock using only reviewed direct wheel URLs, without indexes."""
    validate_lock(lock)
    _require(type(bootstrap_only) is bool, "invalid bootstrap option")
    packages = lock["packages"]
    if bootstrap_only:
        packages = [item for item in packages if item["name"] == "pip"]
        _require(len(packages) == 1, "missing locked pip bootstrap")
    rows = ["--no-index", "--only-binary=:all:"]
    if not bootstrap_only:
        rows.append("--require-hashes")
    rows.extend(f"{p['name']} @ {p['url']} --hash=sha256:{p['sha256']}"
                for p in sorted(packages, key=lambda item: item["name"]))
    return "\n".join(rows) + "\n"


def _version(value: object) -> tuple[int, ...]:
    _require(type(value) is str and re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", value)
             is not None, "invalid platform version")
    return tuple(int(part) for part in value.split("."))


def validate_observation(lock: Mapping[str, object], observed: Mapping[str, object]) -> None:
    """Check supplied installed metadata against its exact intended profile.

    Imports/API compatibility and pip check must be observed separately. Even a
    successful result is not a model-forward or numerical qualification claim.
    """
    validate_lock(lock)
    canonical_digest(observed)
    _require(isinstance(observed, Mapping), "invalid observation")
    for key in ("python_version", "implementation", "system", "machine"):
        _require(observed.get(key) == lock[key], "interpreter or platform mismatch")
    versions = observed.get("packages")
    wheel_hashes = observed.get("wheel_hashes")
    origins = observed.get("wheel_origins")
    _require(type(versions) is dict, "missing package inventory")
    _require(type(wheel_hashes) is dict, "missing installed wheel evidence")
    _require(type(origins) is dict, "missing installed wheel origins")
    for package in lock["packages"]:
        _require(versions.get(package["name"]) == package["version"], "package missing or version mismatch")
        _require(wheel_hashes.get(package["name"]) == package["sha256"], "installed wheel hash mismatch")
        _require(origins.get(package["name"]) == package["url"], "installed wheel origin mismatch")
    _require(observed.get("pip_check") is True, "dependency resolution check failed")
    _require(observed.get("api_contract") is True, "reference API contract failed")
    _require(observed.get("forbidden_packages") == [], "unsupported optional packages installed")
    _require(type(observed.get("cuda_available")) is bool
             and type(observed.get("device_count")) is int
             and 0 <= observed["device_count"] <= 64, "invalid CUDA observation")
    if lock["profile_id"] == "windows-cpu":
        _require(observed.get("cuda_runtime") is None and observed["cuda_available"] is False
                 and observed["device_count"] == 0, "CPU profile cannot use CUDA build")
    else:
        _require(observed.get("cuda_runtime") == "12.6" and observed["cuda_available"] is True
                 and observed["device_count"] >= 1, "requested CUDA is unavailable or incompatible")
        capability, architectures = observed.get("compute_capability"), observed.get("architectures")
        _require(type(capability) is list and len(capability) == 2
                 and all(type(item) is int for item in capability) and capability == [7, 5]
                 and type(architectures) is list and 1 <= len(architectures) <= 256
                 and all(type(item) is str and re.fullmatch(r"(?:sm|compute)_[0-9]+", item)
                         for item in architectures) and "sm_75" in architectures,
                 "unsupported GPU architecture")
        _require(_version(observed.get("driver_version")) >= (560, 35, 5), "unsupported CUDA driver")
        _require(_version(observed.get("glibc_version")) >= (2, 28), "unsupported Linux glibc")


def _inspect(lock: dict[str, object]) -> dict[str, object]:
    """Worker only: import pinned libraries and inspect APIs without model execution."""
    import inspect
    import os

    def no_network(event: str, args: tuple) -> None:
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise EnvironmentFailure("IMPORT_FAILED")

    sys.addaudithook(no_network)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    for key, actual in (("python_version", platform.python_version()),
                        ("system", platform.system()), ("machine", platform.machine()),
                        ("implementation", platform.python_implementation())):
        if actual != lock[key]:
            raise EnvironmentFailure("INTERPRETER_MISMATCH")
    versions, hashes, origins = {}, {}, {}
    for package in lock["packages"]:
        try:
            distribution = importlib.metadata.distribution(package["name"])
            versions[package["name"]] = distribution.version
            origin = distribution.read_text("direct_url.json")
            if origin is None:
                raise EnvironmentFailure("DEPENDENCY_CHECK_FAILED")
            origin_data = _strict_json(origin)
            origins[package["name"]] = origin_data.get("url")
            archive = origin_data.get("archive_info", {})
            hashes[package["name"]] = archive.get("hashes", {}).get("sha256")
            if hashes[package["name"]] is None and archive.get("hash", "").startswith("sha256="):
                hashes[package["name"]] = archive["hash"][7:]
        except importlib.metadata.PackageNotFoundError:
            raise EnvironmentFailure("MISSING_DEPENDENCY") from None
    _require(all(versions[p["name"]] == p["version"] for p in lock["packages"]),
             "installed package version mismatch")
    _require(all(hashes[p["name"]] == p["sha256"] and origins[p["name"]] == p["url"]
                 for p in lock["packages"]), "installed wheel hash or origin mismatch")
    present = {re.sub(r"[-_.]+", "-", item.metadata["Name"]).lower()
               for item in importlib.metadata.distributions()}
    forbidden = sorted(present & FORBIDDEN)
    _require(not forbidden, "unsupported optional packages installed")
    _require(present == set(versions), "unlocked packages are installed")
    import torch
    import transformers
    from transformers import DynamicCache, GenerationConfig, Qwen3Config, Qwen3ForCausalLM
    from transformers.generation.utils import GenerationMixin
    sys.path.insert(0, str(ROOT / "scripts"))
    import gpu_capabilities
    from vortex.executor.reference_config import ReferenceConfig, REVISION

    ReferenceConfig("Qwen/Qwen3-0.6B", REVISION, REVISION, lock["profile_id"],
                    "cpu" if lock["profile_id"] == "windows-cpu" else "cuda:0",
                    "float32" if lock["profile_id"] == "windows-cpu" else "float16")
    probe = gpu_capabilities.capability()
    _require(probe["status"] in ("cuda_available", "cuda_unavailable"), "torch capability probe failed")
    forward = inspect.signature(Qwen3ForCausalLM.forward).parameters
    generate = inspect.signature(GenerationMixin.generate).parameters
    update = inspect.signature(DynamicCache.update).parameters
    api = ({"past_key_values", "cache_position", "logits_to_keep"} <= forward.keys()
           and "use_model_defaults" in generate
           and {"key_states", "value_states", "layer_idx"} <= update.keys()
           and DynamicCache().get_seq_length() == 0
           and Qwen3Config().model_type == "qwen3"
           and GenerationConfig(do_sample=False).do_sample is False)
    check = subprocess.run([sys.executable, "-I", "-m", "pip", "check"], capture_output=True,
                           timeout=60, check=False)
    observed = {"python_version": platform.python_version(), "implementation": platform.python_implementation(),
                "system": platform.system(), "machine": platform.machine(), "packages": versions,
                "wheel_hashes": hashes,
                "wheel_origins": origins,
                "python_build": list(platform.python_build()), "platform_build": platform.platform(),
                "torch_git_version": torch.version.git_version, "transformers_version": transformers.__version__,
                "pip_check": check.returncode == 0, "api_contract": bool(api),
                "cuda_runtime": torch.version.cuda, "cuda_available": probe["cuda_available"],
                "device_count": probe["device_count"], "compute_capability": None,
                "architectures": [], "driver_version": None, "glibc_version": None,
                "forbidden_packages": forbidden}
    if lock["profile_id"] == "t4-fp16" and probe["cuda_available"]:
        observed.update(compute_capability=probe["devices"][0]["compute_capability"],
                        architectures=torch.cuda.get_arch_list(), glibc_version=platform.libc_ver()[1])
        driver = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=driver_version",
                                 "--format=csv,noheader"], capture_output=True, timeout=10, check=True)
        observed["driver_version"] = driver.stdout.decode("ascii").strip()
    validate_observation(lock, observed)
    return observed


def inspect_environment(lock: dict[str, object]) -> dict[str, object]:
    """Run one bounded quiet worker; return supplied import evidence or raise ValueError."""
    validate_lock(lock)
    with tempfile.TemporaryDirectory(prefix="vortex-profile-") as directory:
        import os

        source, target = Path(directory) / "lock.json", Path(directory) / "result.json"
        source.write_text(json.dumps(lock), encoding="utf-8")
        child_env = os.environ.copy()
        for name in ("TRANSFORMERS_CACHE", "HF_HUB_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                     "PYTORCH_PRETRAINED_BERT_CACHE"):
            child_env.pop(name, None)
        child_env.update(HF_HOME=str(Path(directory) / "hf"), HF_HUB_OFFLINE="1",
                         TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        try:
            result = subprocess.run([sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                                     "--_worker", str(source), str(target)],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, timeout=120, check=False, env=child_env)
        except subprocess.TimeoutExpired:
            raise EnvironmentFailure("INSPECTION_TIMEOUT") from None
        if not target.is_file() or target.stat().st_size > 1_048_576:
            raise EnvironmentFailure("IMPORT_FAILED")
        data = _strict_json(target.read_text(encoding="utf-8"))
        _require(type(data) is dict, "invalid environment result")
        if result.returncode:
            raise EnvironmentFailure(data.get("error_code", "IMPORT_FAILED"))
        validate_observation(lock, data)
        return data


def main(argv: list[str] | None = None) -> int:
    """Validate/extract a lock or verify installed imports; print one sanitized JSON record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--lock-document", type=Path, default=ROOT / "docs/reference-environment.md")
    parser.add_argument("--emit-requirements", type=Path)
    parser.add_argument("--bootstrap-only", action="store_true",
                        help="Emit only the locked pip wheel; requires --emit-requirements")
    args = parser.parse_args(argv)
    try:
        lock = read_lock(args.lock_document, args.profile)
        _require(not args.bootstrap_only or args.emit_requirements is not None,
                 "bootstrap extraction requires an output path")
        if args.emit_requirements:
            target = args.emit_requirements.resolve()
            _require(not target.is_relative_to(ROOT), "write requirements outside the public checkout")
            with target.open("x", encoding="utf-8", newline="\n") as output:
                output.write(requirements_text(lock, bootstrap_only=args.bootstrap_only))
            result = {"status": "LOCK_VALIDATED", "profile_id": args.profile,
                      "lock_sha256": canonical_digest(lock), "qualified": False}
        else:
            observed = inspect_environment(lock)
            result = {"status": "IMPORTS_VERIFIED", "profile_id": args.profile,
                      "lock_sha256": canonical_digest(lock), "qualified": False,
                      "forward_smoke": "NOT_VERIFIED", "observation": observed}
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        code = error.code if isinstance(error, EnvironmentFailure) else "LOCK_INVALID"
        print(json.dumps({"status": "NOT_VERIFIED", "profile_id": args.profile,
                          "qualified": False, "error_code": code, "error": REMEDIATIONS[code]}))
        return 2


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--_worker":
        try:
            value = _inspect(_strict_json(Path(sys.argv[2]).read_text(encoding="utf-8")))
            code = 0
        except (ImportError, OSError, ValueError, RuntimeError, TypeError, AttributeError,
                subprocess.SubprocessError) as error:
            reason = error.code if isinstance(error, EnvironmentFailure) else "IMPORT_FAILED"
            value, code = {"error_code": reason, "error": REMEDIATIONS[reason]}, 2
        Path(sys.argv[3]).write_text(json.dumps(value), encoding="utf-8")
        raise SystemExit(code)
    raise SystemExit(main())
