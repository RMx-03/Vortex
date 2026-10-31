#!/usr/bin/env python3
"""Emit schema-1 reference drafts; explicitly opt in to execution on an authorized host.

This is an untimed single-request HF reference CLI, not a performance benchmark,
golden qualifier, export service or custom Vortex decoding engine.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import re
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vortex.executor.reference_config import MODEL_ID, REVISION, ReferenceConfig
from vortex.executor.reference_generation import ReferenceGenerationError, generate_reference
from vortex.executor.reference_loading import (
    SYNTHETIC_PROMPTS, LoadedReference, ReferenceLoadingError, load_reference, prepare_prompt,
)
from vortex.executor.reference_manifest import _generation, canonical_digest, validate_manifest


def read_workloads() -> list[dict[str, object]]:
    """Read only the three fixed synthetic messages; reject file/schema drift."""
    path = ROOT / "benchmarks/workloads/phase0-prompts.json"
    with path.open("rb") as stream:
        data = stream.read(16385)
    if len(data) > 16384:
        raise ValueError("workload exceeds limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate workload field")
            result[key] = value
        return result
    parsed = json.loads(data, object_pairs_hook=pairs)
    expected = {"schema_version": 1, "workloads": [
        {"id": name, "messages": [{"role": "user", "content": content}], "max_new_tokens": 32}
        for name, content in zip(("A", "B", "C"), SYNTHETIC_PROMPTS)]}
    if canonical_digest(parsed) != canonical_digest(expected):
        raise ValueError("unsupported workload")
    return parsed["workloads"]


def configure_determinism(loaded: LoadedReference) -> dict[str, object]:
    """Set strict process-local settings after early environment preflight.

    This intentionally affects the CLI process. Library callers own their own
    process initialization and must record actual settings before comparisons.
    These settings alone do not qualify token parity across devices.
    """
    import numpy
    torch = loaded._torch
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    if loaded.device != "cpu":
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {"seed": 0, "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _environment(loaded: LoadedReference) -> dict[str, object]:
    observed = loaded.environment["observation"]
    hardware = {"device": loaded.device, "gpu_name": None, "compute_capability": None,
                "driver_version": None, "cuda_runtime": None}
    if loaded.device != "cpu":
        hardware.update(gpu_name=loaded._torch.cuda.get_device_name(0),
            compute_capability=observed["compute_capability"],
            driver_version=observed["driver_version"], cuda_runtime=observed["cuda_runtime"])
    return {"python_version": observed["python_version"],
        "python_build": json.dumps({key: observed[key] for key in
            ("python_build", "implementation", "torch_git_version", "glibc_version")}, sort_keys=True),
        "platform": observed["platform_build"], "machine": observed["machine"],
        "packages": {name: {"version": version, "wheel_sha256": observed["wheel_hashes"][name]}
                     for name, version in observed["packages"].items()}, "hardware": hardware}


def _draft(config: ReferenceConfig, source: str, dirty: bool, patch: str | None,
           session: str) -> dict[str, object]:
    payload = {"schema_version": 1, "config": asdict(config), "chat_template": {
        "tokenize": True, "add_generation_prompt": True, "enable_thinking": False},
        "generation": _generation(config)}
    return dict(payload, config_sha256=canonical_digest(payload), numerical=None,
        comparison_key_sha256=None, provenance={"source_sha": source, "dirty": dirty,
            "patch_sha256": patch, "adapter_version": "reference-generation-1",
            "run_id": str(uuid4()), "session_id": session,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")},
        outputs=None, complete=False, outcome="configured")


def _execute(record: dict[str, object], loaded: LoadedReference, messages: list[dict[str, str]],
             config: ReferenceConfig, determinism: dict[str, object]) -> None:
    prompt = prepare_prompt(loaded, messages, config)
    result = generate_reference(loaded, prompt, config)
    numerical = {"assets": dict(loaded.assets), "prompt": {"messages": messages,
        "messages_sha256": prompt.messages_sha256, "rendered_prompt": prompt.rendered_prompt,
        "rendered_sha256": prompt.rendered_sha256, "input_ids": list(prompt.prompt_token_ids)},
        "environment": _environment(loaded), "determinism": determinism,
        "effective_generation_config": result.effective_generation_config}
    outputs = {"prompt_ids": list(result.prompt_ids), "sequence_ids": list(result.sequence_ids),
        "generated_ids": list(result.generated_ids), "raw_text": result.raw_text,
        "display_text": result.display_text, "stop_reason": result.stop_reason,
        "reached_limit": result.reached_limit}
    candidate = dict(record, numerical=numerical, outputs=outputs, complete=True, outcome="success",
        comparison_key_sha256=canonical_digest({"config_sha256": record["config_sha256"], "numerical": numerical}))
    validate_manifest(candidate)
    record.update(candidate)


def _emit(record: dict[str, object]) -> None:
    validate_manifest(record)
    print(json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    """Print one schema-valid JSON line per request; diagnostics use finite codes.

    Exit 0: configured drafts or successful adapter requests; 2: invalid preflight;
    3: failed request; 130: interruption. Success never implies GPU qualification.
    Stdout may contain synthetic prompt/output evidence: keep captures outside Git.
    No destination files, measurements or golden comparisons are implemented here.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("windows-cpu", "t4-fp16"), required=True)
    parser.add_argument("--source-sha", required=True, help="Operator-supplied reviewed source commit")
    parser.add_argument("--dirty", action="store_true")
    parser.add_argument("--patch-sha256")
    parser.add_argument("--workload", choices=("A", "B", "C", "all"), default="all")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--offline", action="store_true", help="Never acquire missing assets")
    parser.add_argument("--execute", action="store_true", help="Load/generate only on a separately authorized host")
    args = parser.parse_args(argv)
    try:
        if (re.fullmatch(r"[0-9a-f]{40}", args.source_sha) is None
                or (args.dirty and (args.patch_sha256 is None
                    or re.fullmatch(r"[0-9a-f]{64}", args.patch_sha256) is None))
                or (not args.dirty and args.patch_sha256 is not None)):
            raise ValueError("invalid source provenance")
        device, dtype = ("cpu", "float32") if args.profile == "windows-cpu" else ("cuda:0", "float16")
        config = ReferenceConfig(MODEL_ID, REVISION, REVISION, args.profile, device, dtype,
                                 max_new_tokens=args.max_new_tokens)
        workloads = read_workloads()
        selected = [w for w in workloads if args.workload in ("all", w["id"])]
        if args.execute and (os.environ.get("PYTHONHASHSEED") != "0"
                             or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
            print(json.dumps({"error_code": "DETERMINISM_REQUIRED", "complete": False}), file=sys.stderr)
            return 2
    except (ValueError, OSError, TypeError, RecursionError):
        print(json.dumps({"error_code": "INVALID_CONFIG", "complete": False}), file=sys.stderr)
        return 2
    session = str(uuid4())
    loaded = None
    record = None
    try:
        for workload in selected:
            record = _draft(config, args.source_sha, args.dirty, args.patch_sha256, session)
            if args.execute:
                if loaded is None:
                    loaded = load_reference(config, offline=args.offline)
                    determinism = configure_determinism(loaded)
                _execute(record, loaded, workload["messages"], config, determinism)
            _emit(record)
        return 0
    except KeyboardInterrupt:
        code, status, exit_code = "INTERRUPTED", "interrupted", 130
    except (ReferenceLoadingError, ReferenceGenerationError) as error:
        code, status, exit_code = error.code, "failed", 3
    except Exception:
        # Native/import/determinism failures have no closed exception hierarchy.
        # Preserve failure evidence without exposing backend diagnostics.
        code, status, exit_code = "REQUEST_FAILED", "failed", 3
    finally:
        loaded = None
    if record is not None:
        record.update(complete=False, outcome=status, outputs=None, numerical=None, comparison_key_sha256=None)
        _emit(record)
    print(json.dumps({"error_code": code, "complete": False}), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
