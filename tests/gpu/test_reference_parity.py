"""Opt-in GPU qualification and CPU-only controller regression fixtures.

Import/discovery uses only the standard library and offline Vortex contracts.
Synthetic fixtures are never pretrained evidence or human-approved goldens.
"""

from __future__ import annotations

from copy import deepcopy
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import stat
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parents[2]
for _directory in (ROOT / "src", ROOT / "scripts", ROOT / "benchmarks"):
    sys.path.insert(0, str(_directory))

from vortex.executor.reference_manifest import canonical_digest, validate_manifest
from compare_reference import compare_reference

LIMIT = 1_048_576
TOTAL_LIMIT = 32 * LIMIT
CAPTURES = ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "A4", "B4", "C4")
PAIRS = tuple(f"{prompt}1-{prompt}{repeat}" for prompt in "ABC" for repeat in (2, 3, 4))
NATIVE_FAILURES = frozenset(("OUT_OF_MEMORY", "GENERATION_FAILED", "INVALID_OUTPUT", "LOAD_FAILED",
    "ASSET_INVALID", "ENVIRONMENT_MISMATCH", "MODEL_MISMATCH", "INVALID_PROMPT", "INVALID_CONFIG",
    "SMOKE_FAILED", "OFFLINE_NETWORK_ATTEMPT", "OFFLINE_NEGATIVE_FAILED", "WORKER_FAILED", "INTERRUPTED"))


class Blocked(ValueError):
    """Finite controller failure; never stringify native exceptions or paths."""


def require(condition, code):
    if not condition:
        raise Blocked(code)


def private_path(name):
    require(type(name) is str and name and not name.replace("\\", "/").startswith("//")
            and ".." not in name.replace("\\", "/").split("/"), "PRIVATE_PATH")
    path = Path(name)
    require(path.is_absolute(), "PRIVATE_PATH")
    for item in (path, *path.parents):
        require(not item.is_symlink(), "PRIVATE_PATH")
        if item.exists():
            require(not getattr(item.lstat(), "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0), "PRIVATE_PATH")
    require(not path.resolve().is_relative_to(ROOT.resolve()), "PRIVATE_PATH")
    return path


def encode_json(value):
    canonical_digest(value)  # existing depth, finite-value and work budgets
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    require(len(data) <= LIMIT, "JSON_LIMIT")
    return data


def parse_json(data):
    require(len(data) <= LIMIT, "JSON_LIMIT")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "JSON_INVALID")
            result[key] = value
        return result
    def constant(_):
        raise Blocked("JSON_INVALID")
    try:
        result = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
        encode_json(result)
        return result
    except (ValueError, UnicodeError, RecursionError, TypeError):
        raise Blocked("JSON_INVALID") from None


def read_bytes(path, maximum=LIMIT):
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= maximum,
            "FILE_INVALID")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    require(len(data) <= maximum, "FILE_LIMIT")
    return data


def read_json(path):
    return parse_json(read_bytes(path))


def atomic_bytes(path, data):
    """Owner-only temp sibling, fsync, replace, directory fsync and readback."""
    private_path(str(path))
    require(len(data) <= LIMIT, "WRITE_LIMIT")
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(shutil.disk_usage(path.parent).free >= len(data) + LIMIT, "WRITE_SPACE")
        descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        require(read_bytes(path) == data, "WRITE_READBACK")
        return hashlib.sha256(data).hexdigest()
    except (OSError, ValueError):
        raise Blocked("WRITE_FAILED") from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    return atomic_bytes(path, encode_json(value))


def source_identity(root, approved_sha):
    require(type(approved_sha) is str and re.fullmatch(r"[0-9a-f]{40}", approved_sha), "SOURCE_SHA")
    require(not any(name in os.environ for name in
        ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE")), "SOURCE_OVERRIDE")
    env = dict(os.environ, GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0",
               GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    def git(*args):
        result = subprocess.run(["git", "--no-replace-objects", "-C", str(root), *args],
            capture_output=True, stdin=subprocess.DEVNULL, timeout=15, check=False, env=env)
        require(result.returncode == 0 and not result.stderr and len(result.stdout) <= LIMIT,
                "SOURCE_INSPECTION")
        return result.stdout
    require(git("rev-parse", "HEAD").decode("ascii").strip() == approved_sha, "SOURCE_SHA")
    require(not git("status", "--porcelain=v1", "-z", "--untracked-files=all"), "SOURCE_DIRTY")
    tree = {}
    for entry in git("ls-tree", "-rz", "--full-tree", approved_sha).split(b"\0")[:-1]:
        metadata, name = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        require(mode in ("100644", "100755") and kind == "blob", "SOURCE_MODE")
        tree[name.decode("utf-8")] = oid
    listing = git("ls-files", "-z").decode("utf-8").split("\0")[:-1]
    require(1 <= len(listing) <= 128, "SOURCE_INVENTORY")
    require(set(listing) == set(tree), "SOURCE_INVENTORY")
    hashes = {}
    for name in listing:
        require(not name.startswith("/") and ".." not in name.split("/"), "SOURCE_PATH")
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), "SOURCE_PATH")
        data = read_bytes(path)
        blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data,
                            usedforsecurity=False).hexdigest()
        require(blob == tree[name], "SOURCE_BYTES")
        hashes[name] = hashlib.sha256(data).hexdigest()
    return {"source_sha": approved_sha, "dirty": False, "files": hashes}


def require_execution(environment):
    require(environment.get("VORTEX_GPU_EXECUTE") == "1"
            and environment.get("VORTEX_GPU_PROFILE") == "t4-fp16"
            and platform.system() == "Linux", "BLOCKED_EXECUTION")
    require(all(environment.get(key) == value for key, value in {
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1"}.items()),
        "BLOCKED_DETERMINISM")
    require(re.fullmatch(r"[0-9a-f]{40}", environment.get("VORTEX_GPU_SOURCE_SHA", ""))
            and re.fullmatch(r"[0-9a-f]{64}", environment.get("VORTEX_GPU_APPROVAL_SHA256", "")),
            "BLOCKED_APPROVAL")
    return {key: private_path(environment.get(variable, "")) for key, variable in {
        "approval": "VORTEX_GPU_APPROVAL", "root": "VORTEX_GPU_PRIVATE_ROOT",
        "cache": "VORTEX_GPU_CACHE", "uv": "VORTEX_GPU_UV"}.items()}


def verify_asset(path, identity):
    from vortex.executor.reference_loading import _hash_asset
    try:
        return _hash_asset(path, identity)
    except (ValueError, OSError):
        raise Blocked("ASSET_INVALID") from None


def validate_captures(records, attempts, source, session):
    """Structural evidence/parity only; no native execution or human attestation."""
    from run_baseline import read_workloads
    require(set(records) == set(CAPTURES) and len(attempts) == 12
            and tuple(item["capture"] for item in attempts) == CAPTURES, "CAPTURE_COUNT")
    primary = {(a["pid"], a["instance"]) for a in attempts[:9]}
    replay = {(a["pid"], a["instance"]) for a in attempts[9:]}
    # PID may be reused after primary exits; process-instance UUID plus supervisor
    # launch records establishes a boundary even if the OS recycles the PID.
    require(len(primary) == len(replay) == 1 and primary != replay
            and all(a["state"] == "COMPLETE" and a["process"] ==
                    ("primary" if index < 9 else "replay") for index, a in enumerate(attempts)),
            "CAPTURE_PROCESS")
    messages = {w["id"]: w["messages"] for w in read_workloads()}
    run_ids = set()
    for label, record in records.items():
        validate_manifest(record)
        provenance = record["provenance"]
        config = record["config"]
        require(record["complete"] and provenance["source_sha"] == source
                and not provenance["dirty"] and provenance["session_id"] == session
                and provenance["run_id"] not in run_ids and config["profile_id"] == "t4-fp16"
                and config["max_new_tokens"] == 32 and config["max_prompt_tokens"] == 512
                and record["numerical"]["prompt"]["messages"] == messages[label[0]], "CAPTURE_INVALID")
        run_ids.add(provenance["run_id"])
        require(compare_reference(record, record).status == "MATCH", "CAPTURE_INVALID")
    reports = {}
    for pair in PAIRS:
        left, right = pair.split("-")
        report = asdict(compare_reference(records[left], records[right]))
        reports[pair] = report
        require(report["status"] == "MATCH", report["status"])
    return reports


def terminate_tree(child):
    """Kill the entire isolated Linux group, then reap its direct child.

    Windows is only for disposable CPU fixtures; taskkill /T handles their tree.
    Linux group cleanup runs even after the leader exits (inherited pipe handles).
    """
    if os.name == "posix":
        for action in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(child.pid, action)
            except ProcessLookupError:
                break
            if action == signal.SIGTERM:
                time.sleep(.1)
    elif child.poll() is None:
        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=5, check=False)
    if child.poll() is None:
        child.kill()
    child.wait(timeout=5)


def supervise(command, environment, stages, on_event, process_seconds, session_deadline):
    """Bounded event pipe with parent ACK only after durable callback.

    Each expected stage has one START/DONE; duplicates cannot reset a deadline.
    A timeout fails the process; it never truncates a valid generation or retries.
    Raw stderr is counted/discarded, never copied into durable diagnostics.
    """
    require(0 < process_seconds <= 2400 and len(stages) <= 16
            and all(0 < seconds <= 1200 for _, seconds in stages), "BOUNDS_INVALID")
    end = min(time.monotonic() + process_seconds, session_deadline)
    require(end > time.monotonic(), "TIMEOUT")
    child = subprocess.Popen(command, env=environment, cwd=ROOT, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=os.name == "posix",
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0)
    chunks, stop = queue.Queue(maxsize=16), threading.Event()
    def drain(stream, channel):
        try:
            while not stop.is_set():
                data = stream.read1(4096)
                while not stop.is_set():
                    try:
                        chunks.put((channel, data), timeout=.05)
                        break
                    except queue.Full:
                        pass
                if not data:
                    return
        except (OSError, ValueError):
            return
    readers = [threading.Thread(target=drain, args=(stream, name), daemon=True)
               for stream, name in ((child.stdout, "stdout"), (child.stderr, "stderr"))]
    for reader in readers:
        reader.start()
    pending, closed, index, active = b"", set(), 0, False
    acquisition_deadline = None
    counts = {"stdout": 0, "stderr": 0}
    deadline = min(end, time.monotonic() + 180)  # worker startup is bounded too
    try:
        while len(closed) < 2:
            require(time.monotonic() < deadline, "TIMEOUT")
            try:
                channel, data = chunks.get(timeout=min(.05, max(.001, deadline - time.monotonic())))
            except queue.Empty:
                # Clean leader exit cannot hide a descendant holding pipes open.
                if child.poll() is not None and chunks.empty():
                    terminate_tree(child)
                continue
            if not data:
                closed.add(channel)
                continue
            counts[channel] += len(data)
            require(counts[channel] <= (TOTAL_LIMIT if channel == "stdout" else LIMIT), "OUTPUT_LIMIT")
            if channel == "stderr":
                continue
            pending += data
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                require(len(line) <= LIMIT and index < len(stages), "OUTPUT_LIMIT" if len(line) > LIMIT else "PROTOCOL")
                event = parse_json(line)
                expected_name, seconds = stages[index]
                required = {"event", "stage", "data"} if active else {"event", "stage"}
                if active and event.keys() == required and event["stage"] == expected_name and event["event"] == "failed":
                    require(type(event["data"]) is dict and event["data"].keys() == {"code"}
                            and event["data"]["code"] in NATIVE_FAILURES, "PROTOCOL_FAILURE")
                    raise Blocked(event["data"]["code"])
                require(event.keys() == required and event["stage"] == expected_name
                        and event["event"] == ("done" if active else "start"), "PROTOCOL")
                on_event(event, child.pid)
                if active:
                    index += 1
                    active = False
                    deadline = min(end, time.monotonic() + 10)  # next START/exit
                else:
                    active = True
                    if expected_name == "assets":
                        acquisition_deadline = time.monotonic() + seconds
                    deadline = min(end, time.monotonic() + seconds,
                        acquisition_deadline if expected_name in ("assets", "load")
                        and acquisition_deadline is not None else end)
                if acquisition_deadline is not None and expected_name in ("assets", "load"):
                    deadline = min(deadline, acquisition_deadline)
                require(time.monotonic() < deadline, "TIMEOUT")
                try:
                    child.stdin.write(b"GO\n")
                    child.stdin.flush()
                except BrokenPipeError:
                    raise Blocked("CHILD_TERMINATED") from None
            require(len(pending) <= LIMIT, "OUTPUT_LIMIT")
        remaining = min(5, max(.001, deadline - time.monotonic()))
        try:
            returncode = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            raise Blocked("TIMEOUT") from None
        require(returncode == 0, "CHILD_FAILED")
        require(not pending and not active and index == len(stages), "PROTOCOL_INCOMPLETE")
        return {"pid": child.pid, "returncode": returncode, "stdout_bytes": counts["stdout"],
                "stderr_bytes": counts["stderr"]}
    finally:
        stop.set()
        terminate_tree(child)
        for stream in (child.stdin, child.stdout, child.stderr):
            stream.close()
        for reader in readers:
            reader.join(timeout=1)


_STATIC_MEMBERS = frozenset(("session.json", "source.json", "attempts.json", "capture.COMPLETE",
    "environment/lock.json", "environment/requirements.txt", "environment/receipts.json",
    "environment/imports.json", "environment/qualification.json", "environment/replay.json",
    "environment/offline-negative.json", "assets/inventory.json", "assets/replay.json"))


def capture_member(name):
    return (name in _STATIC_MEMBERS or name in {f"manifests/{label}.json" for label in CAPTURES}
        or name in {f"manifests/{label}.capture.COMPLETE" for label in CAPTURES}
        or name in {f"comparisons/{pair}.json" for pair in PAIRS}
        or re.fullmatch(r"failures/(?:primary|replay|negative|session|export)\.json", name) is not None)


def bundle_files(bundle):
    """Closed bounded tree; no arbitrary recursive source/cache export."""
    files = {}
    pending = [bundle]
    seen = 0
    while pending:
        directory = pending.pop()
        for path in directory.iterdir():
            seen += 1
            require(seen <= 128 and not path.is_symlink(), "MEMBER_INVALID")
            name = path.relative_to(bundle).as_posix()
            private_path(str(path))  # Windows junction/reparse guard, not just symlink
            require(path.resolve().is_relative_to(bundle.resolve()), "MEMBER_ESCAPE")
            if path.is_dir():
                require(name in {"environment", "assets", "manifests", "comparisons", "failures"}, "MEMBER_INVALID")
                pending.append(path)
            else:
                require(capture_member(name) or name in {"bundle-index.json", "bundle-index.sha256"}, "MEMBER_INVALID")
                files[name] = path
    return files


def seal_bundle(bundle):
    private_path(str(bundle))
    require(not (bundle / "bundle-index.json").exists()
            and not (bundle / "bundle-index.sha256").exists(), "SEALED_EXISTS")
    members, total = [], 0
    for name, path in sorted(bundle_files(bundle).items()):
        data = read_bytes(path)
        if name.endswith((".json", ".COMPLETE")):
            parse_json(data)
        total += len(data)
        require(total <= TOTAL_LIMIT, "MEMBER_LIMIT")
        members.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    require(members and len(members) <= 126, "MEMBER_LIMIT")
    digest = atomic_json(bundle / "bundle-index.json", {"members": members})
    atomic_bytes(bundle / "bundle-index.sha256", digest.encode("ascii"))
    return digest


def checked_members(contents, expected_index=None):
    require({"bundle-index.json", "bundle-index.sha256"} <= contents.keys(), "CHECKSUM_INDEX")
    index_bytes = contents["bundle-index.json"]
    digest = hashlib.sha256(index_bytes).hexdigest()
    require(contents["bundle-index.sha256"] == digest.encode("ascii")
            and (expected_index is None or digest == expected_index), "CHECKSUM_INDEX")
    index = parse_json(index_bytes)
    require(index.keys() == {"members"} and type(index["members"]) is list
            and 1 <= len(index["members"]) <= 126, "MEMBER_INDEX")
    expected, total = set(), 0
    for entry in index["members"]:
        require(type(entry) is dict and entry.keys() == {"path", "size", "sha256"}, "MEMBER_INDEX")
        name = entry["path"]
        require(type(name) is str and capture_member(name) and name not in expected, "MEMBER_INDEX")
        data = contents.get(name)
        require(type(data) is bytes and type(entry["size"]) is int and len(data) == entry["size"]
                and len(data) <= LIMIT and hashlib.sha256(data).hexdigest() == entry["sha256"], "CHECKSUM_MEMBER")
        expected.add(name)
        total += len(data)
        require(total <= TOTAL_LIMIT, "MEMBER_LIMIT")
        if name.endswith((".json", ".COMPLETE")):
            value = parse_json(data)
            if name.startswith("manifests/") and name.endswith(".json"):
                validate_manifest(value)
    require(set(contents) == expected | {"bundle-index.json", "bundle-index.sha256"}, "MEMBER_EXTRA")
    return digest, len(expected)


def export_bundle(bundle, archive):
    """Create one closed qualification archive; no transport or cloud API calls."""
    private_path(str(archive))
    require(not archive.exists(), "EXPORT_EXISTS")
    contents = {name: read_bytes(path) for name, path in bundle_files(bundle).items()}
    checked_members(contents)
    archive.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(shutil.disk_usage(archive.parent).free >= TOTAL_LIMIT + LIMIT, "EXPORT_SPACE")
    descriptor, name = tempfile.mkstemp(prefix=".archive-", dir=archive.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as target:
                for member, data in sorted(contents.items()):
                    info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                    info.external_attr = (stat.S_IFREG | 0o600) << 16
                    target.writestr(info, data)
            stream.flush()
            os.fsync(stream.fileno())
        require(temporary.stat().st_size <= TOTAL_LIMIT + LIMIT, "EXPORT_LIMIT")
        os.replace(temporary, archive)
        return hashlib.sha256(read_bytes(archive, TOTAL_LIMIT + LIMIT)).hexdigest()
    except OSError:
        raise Blocked("EXPORT_FAILED") from None
    finally:
        if temporary.exists():
            temporary.unlink()


_CAPTURE_COMPLETE = {"captures": 12, "comparisons": 9, "execution": "VERIFIED", "evidence": "CAPTURED",
    "numerical_comparability": "VERIFIED", "token_parity": "MATCH", "independent_review": "NOT_VERIFIED",
    "human_approval": "NOT_APPROVED"}
_SMOKE_COMPLETE = {"status": "VERIFIED", "device": "cuda:0", "dtype": "float16", "attention": "eager",
                   "prefill_length": 2, "decode_length": 3, "finite": True}
_NEGATIVE_COMPLETE = {"status": "VERIFIED", "offline": True, "network_attempts": 0,
                      "model_loaded": False, "empty_cache": True}
_LOAD_COMPLETE = {"status": "VERIFIED", "offline": True, "dtype": "float16", "attention": "eager"}


def numerical_environment(preflight):
    observed, hardware = preflight["imports"]["observation"], preflight["hardware"]
    return {"python_version": observed["python_version"],
        "python_build": json.dumps({key: observed[key] for key in
            ("python_build", "implementation", "torch_git_version", "glibc_version")}, sort_keys=True),
        "platform": observed["platform_build"], "machine": observed["machine"],
        "packages": {name: {"version": version, "wheel_sha256": observed["wheel_hashes"][name]}
            for name, version in observed["packages"].items()},
        "hardware": {"device": "cuda:0", "gpu_name": hardware["gpu_name"],
            "compute_capability": observed["compute_capability"], "driver_version": hardware["driver_version"],
            "cuda_runtime": observed["cuda_runtime"]}}


def inventory_identities(data):
    from vortex.executor.reference_config import MODEL_ID, REVISION
    from vortex.executor.reference_loading import _ASSETS, _TEMPLATE_SHA256
    require(data.keys() == {"model_id", "revision", "inventory", "assets"}
            and data["model_id"] == MODEL_ID and data["revision"] == REVISION
            and set(data["inventory"]) == set(_ASSETS), "ASSET_INVENTORY")
    hashes = {}
    for name, (size, algorithm, expected) in _ASSETS.items():
        entry = data["inventory"][name]
        require(entry.keys() == {"size", "algorithm", "expected", "raw_sha256"}
            and type(entry["size"]) is int and entry["size"] == size and entry["algorithm"] == algorithm
            and entry["expected"] == expected and type(entry["raw_sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", entry["raw_sha256"])
            and (algorithm != "sha256" or entry["raw_sha256"] == expected), "ASSET_INVENTORY")
        hashes[name] = entry["raw_sha256"]
    assets = {"model_config_sha256": hashes["config.json"], "generation_config_sha256": hashes["generation_config.json"],
        "tokenizer_config_sha256": hashes["tokenizer_config.json"], "chat_template_sha256": _TEMPLATE_SHA256,
        "weights_sha256": canonical_digest({"model.safetensors": hashes["model.safetensors"]}),
        "tokenizer_assets_sha256": canonical_digest({name: hashes[name] for name in
            ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt")})}
    require(data["assets"] == assets, "ASSET_IDENTITY")
    return assets


def validate_execution_contents(contents):
    """All required recorded execution evidence, beyond transport/token equality.

    This is not a cryptographic attestation of the host or independent review.
    Human audit still checks provenance and actual execution before approval.
    """
    from verify_reference_environment import read_lock, requirements_text
    required = {"session.json", "source.json", "attempts.json", "capture.COMPLETE", "assets/inventory.json",
        "assets/replay.json", "environment/lock.json", "environment/requirements.txt", "environment/receipts.json",
        "environment/imports.json", "environment/qualification.json", "environment/replay.json",
        "environment/offline-negative.json"}
    required |= {f"manifests/{label}{suffix}" for label in CAPTURES for suffix in (".json", ".capture.COMPLETE")}
    required |= {f"comparisons/{pair}.json" for pair in PAIRS}
    require(required <= contents.keys() and not any(name.startswith("failures/") for name in contents),
            "QUALIFICATION_INCOMPLETE")
    try:
        get = lambda name: parse_json(contents[name])
        require(encode_json(get("capture.COMPLETE")) == encode_json(_CAPTURE_COMPLETE), "QUALIFICATION_MARKER")
        session, source, ledger = get("session.json"), get("source.json"), get("attempts.json")
        require(session["profile_id"] == "t4-fp16" and session["provider"] in ("colab-free", "kaggle-free")
            and session["candidate"] == "CAPTURED_UNREVIEWED" and session["human_approval"] == "NOT_APPROVED"
            and source["dirty"] is False and source["source_sha"] == session["source_sha"]
            and 1 <= len(source["files"]) <= 128
            and all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) for v in source["files"].values()),
            "QUALIFICATION_SOURCE")
        lock = get("environment/lock.json")
        require(lock == read_lock(ROOT / "docs/reference-environment.md", "t4-fp16")
            and contents["environment/requirements.txt"] == requirements_text(lock).encode("utf-8")
            and encode_json(get("environment/offline-negative.json")) == encode_json(_NEGATIVE_COMPLETE),
            "QUALIFICATION_ENVIRONMENT")
        primary, replay = get("environment/qualification.json"), get("environment/replay.json")
        captured_at = datetime.fromisoformat(session["started_utc"].replace("Z", "+00:00"))
        for group in (primary, replay):
            preflight = group["preflight"]
            approval = preflight["approval"]
            validate_approval(approval, session["source_sha"], at=captured_at)
            validate_profile(lock, preflight["imports"], preflight["hardware"], approval)
            require(preflight["source"] == source and approval["provider"] == session["provider"]
                and preflight["python_sys_version"] == approval["tools"]["python_sys_version"]
                and all(preflight[key] == approval["tools"][key] for key in
                    ("python_executable_sha256", "uv_executable_sha256", "python_archive_sha256", "uv_archive_sha256"))
                and encode_json(group["smoke"]) == encode_json(_SMOKE_COMPLETE)
                and encode_json(group["load"]) == encode_json(_LOAD_COMPLETE), "QUALIFICATION_NATIVE")
        require(primary["preflight"]["imports"] == get("environment/imports.json"), "QUALIFICATION_IMPORTS")
        observed = get("environment/imports.json")["observation"]
        require(get("environment/receipts.json") == {key: observed[key]
            for key in ("packages", "wheel_hashes", "wheel_origins")}, "QUALIFICATION_RECEIPTS")
        assets = inventory_identities(get("assets/inventory.json"))
        require(get("assets/inventory.json") == get("assets/replay.json"), "QUALIFICATION_ASSETS")
        processes = ledger["processes"]
        require(set(processes) == {"negative", "primary", "replay"}
            and all(p["state"] == "COMPLETE" and p["returncode"] == 0 and type(p["pid"]) is int
                    and p["pid"] > 0 for p in processes.values())
            and len({p["instance"] for p in processes.values()}) == 3, "QUALIFICATION_PROCESS")
        records = {label: get(f"manifests/{label}.json") for label in CAPTURES}
        reports = validate_captures(records, ledger["attempts"], session["source_sha"], session["session_id"])
        for attempt in ledger["attempts"]:
            label = attempt["capture"]
            digest = hashlib.sha256(contents[f"manifests/{label}.json"]).hexdigest()
            record = records[label]
            preflight = (primary if label[1] != "4" else replay)["preflight"]
            require(attempt["sha256"] == digest and get(f"manifests/{label}.capture.COMPLETE") == {"sha256": digest}
                and (attempt["pid"], attempt["instance"]) ==
                    (processes[attempt["process"]]["pid"], processes[attempt["process"]]["instance"])
                and record["numerical"]["environment"] == numerical_environment(preflight)
                and record["numerical"]["assets"] == assets
                and record["numerical"]["determinism"] == preflight["determinism"], "QUALIFICATION_IDENTITY")
        for pair, report in reports.items():
            require(get(f"comparisons/{pair}.json") == report, "COMPARISON_READBACK")
    except (KeyError, TypeError, AttributeError, ValueError):
        raise Blocked("QUALIFICATION_INVALID") from None


def reopen_bundle(archive, destination, expected_archive, expected_index):
    """Read the actual downloaded file, validate before extracting; never overwrite."""
    private_path(str(archive))
    private_path(str(destination))
    require(not destination.exists(), "DESTINATION_EXISTS")
    require(hashlib.sha256(read_bytes(archive, TOTAL_LIMIT + LIMIT)).hexdigest() == expected_archive,
            "CHECKSUM_ARCHIVE")
    contents, names, total = {}, set(), 0
    with zipfile.ZipFile(archive) as source:
        require(1 <= len(source.infolist()) <= 128, "MEMBER_LIMIT")
        for member in source.infolist():
            name = member.filename
            require(name.casefold() not in names and not member.is_dir()
                and (capture_member(name) or name in {"bundle-index.json", "bundle-index.sha256"})
                and stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG)
                and member.compress_type == zipfile.ZIP_STORED and not member.flag_bits & 1
                and member.file_size <= LIMIT, "MEMBER_INVALID")
            names.add(name.casefold())
            total += member.file_size
            require(total <= TOTAL_LIMIT + LIMIT, "MEMBER_LIMIT")
            with source.open(member) as stream:
                contents[name] = stream.read(LIMIT + 1)
    index, count = checked_members(contents, expected_index)
    destination.mkdir(mode=0o700)
    for name, data in contents.items():
        atomic_bytes(destination / "capture" / name, data)
    actual = {name: read_bytes(path) for name, path in bundle_files(destination / "capture").items()}
    checked_members(actual, expected_index)
    complete = "capture.COMPLETE" in contents
    if complete:
        validate_execution_contents(actual)
    receipt = {"capture_index_sha256": index, "archive_sha256": expected_archive,
        "files_readback": count, "qualification_complete": complete,
        "execution": "VERIFIED" if complete else "NOT_VERIFIED", "evidence": "EXPORTED_REOPENED",
        "supporting_hashes": {"archive": expected_archive, "index": index},
        "independent_review": "NOT_VERIFIED", "human_approval": "NOT_APPROVED"}
    receipt_hash = atomic_json(destination / "review/export-reopen.json", receipt)
    atomic_json(destination / "review" / ("qualification.COMPLETE" if complete else "qualification.INCOMPLETE"),
                dict(receipt, supporting_hashes=dict(receipt["supporting_hashes"], readback=receipt_hash)))
    return receipt


def validate_profile(lock, report, hardware, approval):
    """Static/observed compatibility only. Native smoke is a separate stage."""
    from verify_reference_environment import validate_observation
    try:
        require(report["status"] == "IMPORTS_VERIFIED" and report["profile_id"] == "t4-fp16"
            and report["qualified"] is False and report["forward_smoke"] == "NOT_VERIFIED", "ENVIRONMENT_REPORT")
        observed = report["observation"]
        validate_observation(lock, observed)
        names = {p["name"] for p in lock["packages"]}
        require(set(observed["packages"]) == names and set(observed["wheel_hashes"]) == names
            and set(observed["wheel_origins"]) == names and observed["device_count"] == 1,
            "ENVIRONMENT_PACKAGES")
        require(hardware["gpu_name"] in ("Tesla T4", "NVIDIA T4") and hardware["device"] == "cuda:0"
            and approval["security_reviewed"] is True
            and hardware["driver_version"] == observed["driver_version"] == approval["driver_version"],
            "ENVIRONMENT_HARDWARE")
        for key, minimum in (("total_vram_bytes", 8 * 2**30), ("free_vram_bytes", 8 * 2**30),
                             ("host_available_bytes", 8 * 2**30), ("disk_free_bytes", 4 * 2**30)):
            require(type(hardware[key]) is int and hardware[key] >= minimum, "ENVIRONMENT_RESOURCE")
    except (KeyError, TypeError, ValueError):
        raise Blocked("ENVIRONMENT_INCOMPATIBLE") from None


def validate_approval(approval, source, at=None):
    fields = {"execution_authorized", "source_sha", "profile_id", "provider", "tools",
              "security_reviewed", "driver_version", "reviewed_utc", "remaining_free_seconds"}
    tools_fields = {"python_archive_sha256", "python_executable_sha256", "python_sys_version",
        "uv_version", "uv_archive_sha256", "uv_executable_sha256", "uv_identity_reviewed", "uv_release"}
    require(approval.keys() == fields and approval["execution_authorized"] is True
        and approval["source_sha"] == source and approval["profile_id"] == "t4-fp16"
        and approval["provider"] in ("colab-free", "kaggle-free")
        and approval["security_reviewed"] is True
        and type(approval["remaining_free_seconds"]) is int
        and 0 < approval["remaining_free_seconds"] <= 5400, "APPROVAL_INVALID")
    tools = approval["tools"]
    require(type(tools) is dict and tools.keys() == tools_fields
        and tools["python_archive_sha256"] == "17c4978a36c496dc53c08899ef8dad1ad2734f6325f08d6033a1e66419e0af26"
        and tools["uv_version"] == "0.10.7" and tools["uv_identity_reviewed"] is True
        and tools["uv_release"] == "https://github.com/astral-sh/uv/releases/tag/0.10.7"
        and type(tools["python_sys_version"]) is str and tools["python_sys_version"], "TOOL_IDENTITY_BLOCKED")
    for key in ("python_executable_sha256", "uv_archive_sha256", "uv_executable_sha256"):
        require(type(tools[key]) is str and re.fullmatch(r"[0-9a-f]{64}", tools[key]), "TOOL_IDENTITY_BLOCKED")
    require(type(approval["driver_version"]) is str
            and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", approval["driver_version"]), "APPROVAL_DRIVER")
    try:
        reviewed = datetime.fromisoformat(approval["reviewed_utc"].replace("Z", "+00:00"))
        # Execution-specific advisory review must be current, not the preparation review.
        age = ((datetime.now(timezone.utc) if at is None else at) - reviewed).total_seconds()
        require(reviewed.utcoffset().total_seconds() == 0 and 0 <= age <= 86400, "APPROVAL_STALE")
    except (ValueError, TypeError, AttributeError):
        raise Blocked("APPROVAL_STALE") from None


def capture_request(loaded, messages, source, session, determinism):
    from run_baseline import _draft, _execute
    from vortex.executor.reference_config import MODEL_ID, REVISION, ReferenceConfig
    config = ReferenceConfig(MODEL_ID, REVISION, REVISION, "t4-fp16", "cuda:0", "float16")
    record = _draft(config, source, False, None, session)
    _execute(record, loaded, messages, config, determinism)
    validate_manifest(record)
    require(record["complete"] and record["outcome"] == "success", "CAPTURE_INCOMPLETE")
    return record


def tool_identity(path, expected):
    # uv-created venv interpreters normally symlink to the approved standalone
    # executable. Bind the resolved regular executable, not the link's spelling.
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256(read_bytes(resolved, 64 * LIMIT)).hexdigest()
    require(digest == expected, "TOOL_CHECKSUM")
    return digest


def checked_controls(environment):
    paths = require_execution(environment)
    approval_bytes = read_bytes(paths["approval"])
    require(hashlib.sha256(approval_bytes).hexdigest() == environment["VORTEX_GPU_APPROVAL_SHA256"],
            "APPROVAL_CHECKSUM")
    approval = parse_json(approval_bytes)
    validate_approval(approval, environment["VORTEX_GPU_SOURCE_SHA"])
    require(platform.python_version() == "3.11.18" and platform.machine() == "x86_64"
            and sys.version == approval["tools"]["python_sys_version"], "TOOL_PYTHON")
    for path, key in ((Path(sys.executable), "python_executable_sha256"),
                      (paths["uv"], "uv_executable_sha256")):
        tool_identity(path, approval["tools"][key])
    require(not paths["root"].is_relative_to(paths["cache"])
            and not paths["cache"].is_relative_to(paths["root"]), "PRIVATE_PATH_OVERLAP")
    return paths, approval


def native_preflight(paths, approval, source, state):
    """Authorized Linux worker only; driver security gate precedes native imports."""
    from verify_reference_environment import read_lock
    from vortex.executor.reference_config import MODEL_ID, REVISION, ReferenceConfig
    from vortex.executor.reference_loading import _environment, _dependencies
    from run_baseline import configure_determinism
    actual_source = source_identity(ROOT, source)
    probe = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=name,driver_version,memory.total,memory.free",
        "--format=csv,noheader,nounits"], capture_output=True, stdin=subprocess.DEVNULL, timeout=10, check=False)
    require(probe.returncode == 0 and len(probe.stdout) <= 1024 and not probe.stderr, "ENVIRONMENT_GPU")
    fields = probe.stdout.decode("ascii").strip().split(",")
    require(len(fields) == 4 and "\n" not in probe.stdout.decode("ascii").strip(), "ENVIRONMENT_GPU")
    name, driver, total, free = (field.strip() for field in fields)
    require(approval["security_reviewed"] is True and driver == approval["driver_version"]
        and name in ("Tesla T4", "NVIDIA T4") and tuple(map(int, driver.split("."))) >= (560, 35, 5),
        "ENVIRONMENT_DRIVER_SECURITY")
    memory = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            memory["available"] = int(line.split()[1]) * 1024
    hardware = {"gpu_name": name, "device": "cuda:0", "driver_version": driver,
        "total_vram_bytes": int(total) * 2**20, "free_vram_bytes": int(free) * 2**20,
        "host_available_bytes": memory.get("available", 0), "disk_free_bytes": shutil.disk_usage(paths["root"]).free}
    config = ReferenceConfig(MODEL_ID, REVISION, REVISION, "t4-fp16", "cuda:0", "float16")
    # Existing verifier preserves its original, deliberately unqualified report.
    report = _environment(config)
    lock = read_lock(ROOT / "docs/reference-environment.md", "t4-fp16")
    validate_profile(lock, report, hardware, approval)
    backend = _dependencies(report)
    torch = backend.torch
    require(torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0
            and torch.cuda.get_device_name(0) == name and list(torch.cuda.get_device_capability(0)) == [7, 5],
            "ENVIRONMENT_DEVICE")
    determinism = configure_determinism(SimpleNamespace(_torch=torch, device="cuda:0"))
    state.update(backend=backend, torch=torch, config=config, report=report, determinism=determinism)
    return {"source": actual_source, "imports": report, "hardware": hardware,
        "approval": approval, "python_sys_version": sys.version,
        "determinism": determinism, "python_executable_sha256": approval["tools"]["python_executable_sha256"],
        "uv_executable_sha256": approval["tools"]["uv_executable_sha256"],
        "python_archive_sha256": approval["tools"]["python_archive_sha256"],
        "uv_archive_sha256": approval["tools"]["uv_archive_sha256"]}


def tiny_smoke(state):
    """Two tiny random forwards on the approved GPU; never called by CPU discovery."""
    from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM
    torch = state["torch"]
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=32,
        bos_token_id=1, eos_token_id=2, pad_token_id=0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    model = cache = output = None
    try:
        with torch.inference_mode():
            model = Qwen3ForCausalLM(config).to(device="cuda:0", dtype=torch.float16).eval()
            cache = DynamicCache()
            for ids, length in (([1, 3], 2), ([4], 3)):
                start = length - len(ids)
                output = model(input_ids=torch.tensor([ids], device="cuda:0", dtype=torch.long),
                    attention_mask=torch.ones((1, length), device="cuda:0", dtype=torch.long),
                    position_ids=torch.arange(start, length, device="cuda:0").unsqueeze(0),
                    cache_position=torch.arange(start, length, device="cuda:0"),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                torch.cuda.synchronize(0)
                require(tuple(output.logits.shape) == (1, 1, 32)
                    and output.logits.dtype == torch.float16 and str(output.logits.device) == "cuda:0"
                    and bool(torch.isfinite(output.logits).all()) and output.past_key_values is cache
                    and cache.get_seq_length() == length, "SMOKE_FAILED")
            require(all(p.dtype == torch.float16 and str(p.device) == "cuda:0" for p in model.parameters())
                    and all(t.dtype == torch.float16 and str(t.device) == "cuda:0"
                            for tensors in (cache.key_cache, cache.value_cache) for t in tensors), "SMOKE_FAILED")
        return {"status": "VERIFIED", "device": "cuda:0", "dtype": "float16", "attention": "eager",
                "prefill_length": 2, "decode_length": 3, "finite": True}
    finally:
        output = cache = model = None


def asset_inventory(state, offline):
    from vortex.executor.reference_config import MODEL_ID, REVISION
    from vortex.executor.reference_loading import _ASSETS, _assets
    snapshot, _, identities = _assets(state["backend"], offline)
    entries = {}
    for name, identity in _ASSETS.items():
        raw = verify_asset(snapshot / name, identity)
        entries[name] = {"size": identity[0], "algorithm": identity[1], "expected": identity[2], "raw_sha256": raw}
    state["assets"] = dict(identities)
    return {"model_id": MODEL_ID, "revision": REVISION, "inventory": entries, "assets": dict(identities)}


def deny_network():
    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            raise Blocked("OFFLINE_NETWORK_ATTEMPT")
    sys.addaudithook(audit)


def offline_negative(state):
    from vortex.executor.reference_loading import _assets
    # Cache path was empty and selected BEFORE this fresh interpreter's imports.
    cache = Path(os.environ["HF_HUB_CACHE"])
    require(not cache.exists() or not any(cache.iterdir()), "OFFLINE_PROBE_NOT_EMPTY")
    attempted = []
    def audit(event, args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            attempted.append(True)
            raise Blocked("OFFLINE_NETWORK_ATTEMPT")
    sys.addaudithook(audit)
    try:
        _assets(state["backend"], True)
    except Exception as error:
        # Expected closed-cache failure, before any tokenizer/model construction.
        require(not attempted and (getattr(error, "code", None) == "ASSET_INVALID"
                or type(error).__name__ == "LocalEntryNotFoundError"), "OFFLINE_NEGATIVE_FAILED")
    else:
        raise Blocked("OFFLINE_NEGATIVE_FAILED")
    return {"status": "VERIFIED", "offline": True, "network_attempts": 0, "model_loaded": False,
            "empty_cache": True}


def guarded_load(state, *, offline):
    from vortex.executor.reference_loading import load_reference
    require(offline is True, "OFFLINE_REQUIRED")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    deny_network()
    loaded = load_reference(state["config"], offline=True)
    require(dict(loaded.assets) == state["assets"] and dict(loaded.environment) == state["report"],
            "LOAD_IDENTITY_DRIFT")
    return loaded


def worker_protocol(role, paths, approval, source, session, emit, state=None):
    require(role in ("negative", "primary", "replay"), "PROTOCOL_ROLE")
    state = {} if state is None else state
    def stage(name, operation):
        emit({"event": "start", "stage": name})
        try:
            data = operation()
        except BaseException as error:
            code = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else getattr(error, "code", None)
            if isinstance(error, Blocked):
                code = str(error)
            if type(error).__name__ == "OutOfMemoryError" or isinstance(error, MemoryError):
                code = "OUT_OF_MEMORY"
            emit({"event": "failed", "stage": name,
                  "data": {"code": code if code in NATIVE_FAILURES else "WORKER_FAILED"}})
            raise Blocked("WORKER_FAILED") from None
        emit({"event": "done", "stage": name, "data": data})
        return data
    loaded = None
    try:
        stage("environment", lambda: native_preflight(paths, approval, source, state))
        if role == "negative":
            stage("negative", lambda: offline_negative(state))
            return
        stage("smoke", lambda: tiny_smoke(state))
        stage("assets", lambda: asset_inventory(state, role == "replay"))
        def load():
            nonlocal loaded
            loaded = guarded_load(state, offline=True)
            return {"status": "VERIFIED", "offline": True, "dtype": "float16", "attention": "eager"}
        stage("load", load)
        from run_baseline import read_workloads
        workloads = {w["id"]: w["messages"] for w in read_workloads()}
        labels = CAPTURES[:9] if role == "primary" else CAPTURES[9:]
        for label in labels:
            stage(label, lambda label=label: capture_request(loaded, workloads[label[0]], source,
                                                           session, state.get("determinism", {})))
        source_identity(ROOT, source)
    finally:
        loaded = None
        state.clear()


def worker_entry(role, session):
    paths, approval = checked_controls(os.environ)
    def emit(event):
        sys.stdout.buffer.write(encode_json(event) + b"\n")
        sys.stdout.buffer.flush()
        require(sys.stdin.buffer.readline(4) == b"GO\n", "PROTOCOL_ACK")
    worker_protocol(role, paths, approval, os.environ["VORTEX_GPU_SOURCE_SHA"], session, emit)


def launch_worker(role, environment, callback, deadline):
    stages = (("environment", 180), ("negative", 120)) if role == "negative" else (
        ("environment", 180), ("smoke", 120), ("assets", 1200), ("load", 1200),
        *((label, 120) for label in (CAPTURES[:9] if role == "primary" else CAPTURES[9:])))
    return supervise([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--_worker", role,
        environment["VORTEX_GPU_SESSION"]], environment, stages, callback,
        2400 if role == "primary" else 1200, deadline)


def capture_session(root, source, environment, approval, *, session=None):
    """One attempt, no retries. Return private paths only to the local operator."""
    from verify_reference_environment import read_lock, requirements_text
    session = str(uuid4()) if session is None else session
    directory = private_path(str(root)) / session
    directory.mkdir(mode=0o700)  # exclusive attempt identity
    bundle, review = directory / "capture", directory / "review"
    bundle.mkdir(mode=0o700)
    review.mkdir(mode=0o700)
    atomic_json(review / "qualification.INCOMPLETE", {"capture_index_sha256": None, "supporting_hashes": {},
        "status": "INCOMPLETE", "human_approval": "NOT_APPROVED"})
    ledger = [{"capture": label, "state": "PLANNED", "process": "primary" if index < 9 else "replay",
               "pid": None, "instance": None} for index, label in enumerate(CAPTURES)]
    records, processes = {}, {}
    atomic_json(bundle / "attempts.json", {"attempts": ledger, "processes": processes})
    atomic_json(bundle / "session.json", {"session_id": session, "source_sha": source,
        "profile_id": "t4-fp16", "provider": approval.get("provider", "synthetic-test"),
        "started_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "candidate": "CAPTURED_UNREVIEWED", "human_approval": "NOT_APPROVED"})
    actual_source = source_identity(ROOT, source)
    atomic_json(bundle / "source.json", actual_source)
    lock = read_lock(ROOT / "docs/reference-environment.md", "t4-fp16")
    atomic_json(bundle / "environment/lock.json", lock)
    atomic_bytes(bundle / "environment/requirements.txt", requirements_text(lock).encode("utf-8"))
    deadline = time.monotonic() + min(5400, approval.get("remaining_free_seconds", 5400))
    failure, active, role = None, None, "negative"
    def persist():
        atomic_json(bundle / "attempts.json", {"attempts": ledger, "processes": processes})
    def checkpoint():
        snapshot = directory / "checkpoints/primary"
        snapshot.mkdir(parents=True, mode=0o700)
        for name, path in bundle_files(bundle).items():
            atomic_bytes(snapshot / name, read_bytes(path))
        index = seal_bundle(snapshot)
        digest = export_bundle(snapshot, directory / "exports/primary.zip")
        atomic_json(review / "primary-export.json", {"capture_index_sha256": index,
            "archive_sha256": digest, "supporting_hashes": {"archive": digest},
            "status": "EXPORT_PENDING_DESTINATION_READBACK"})
    try:
        for role in ("negative", "primary", "replay"):
            instance = str(uuid4())
            processes[role] = {"instance": instance, "state": "RUNNING", "pid": None}
            persist()
            def on_event(event, pid):
                nonlocal active
                stage = event["stage"]
                processes[role]["pid"] = pid
                if event["event"] == "start":
                    if stage in CAPTURES:
                        active = ledger[CAPTURES.index(stage)]
                        active.update(state="RUNNING", pid=pid, instance=instance)
                    persist()  # worker waits for this durable START before any work
                    return
                data = event["data"]
                if stage == "environment":
                    require(data["source"] == actual_source, "SOURCE_DRIFT")
                    validate_profile(lock, data["imports"], data["hardware"], approval)
                    name = "replay" if role == "replay" else "qualification"
                    atomic_json(bundle / f"environment/{name}.json", {"preflight": data})
                    if role == "primary":
                        atomic_json(bundle / "environment/imports.json", data["imports"])
                        atomic_json(bundle / "environment/receipts.json", {
                            "packages": data["imports"]["observation"]["packages"],
                            "wheel_hashes": data["imports"]["observation"]["wheel_hashes"],
                            "wheel_origins": data["imports"]["observation"]["wheel_origins"]})
                elif stage == "negative":
                    atomic_json(bundle / "environment/offline-negative.json", data)
                elif stage == "smoke":
                    name = "replay" if role == "replay" else "qualification"
                    qualification = read_json(bundle / f"environment/{name}.json")
                    qualification["smoke"] = data
                    atomic_json(bundle / f"environment/{name}.json", qualification)
                elif stage == "assets":
                    atomic_json(bundle / ("assets/inventory.json" if role == "primary" else "assets/replay.json"), data)
                elif stage == "load":
                    require(data.get("status") == "VERIFIED", "LOAD_FAILED")
                    name = "replay" if role == "replay" else "qualification"
                    qualification = read_json(bundle / f"environment/{name}.json")
                    qualification["load"] = data
                    atomic_json(bundle / f"environment/{name}.json", qualification)
                elif stage in CAPTURES:
                    validate_manifest(data)
                    require(data["complete"] and data["provenance"]["source_sha"] == source
                            and data["provenance"]["session_id"] == session, "CAPTURE_INVALID")
                    digest = atomic_json(bundle / f"manifests/{stage}.json", data)
                    atomic_json(bundle / f"manifests/{stage}.capture.COMPLETE", {"sha256": digest})
                    records[stage] = data
                    active.update(state="COMPLETE", sha256=digest)
                    persist()
                    active = None
                    if stage[1] != "1":
                        pair = f"{stage[0]}1-{stage}"
                        report = asdict(compare_reference(records[stage[0] + "1"], data))
                        atomic_json(bundle / f"comparisons/{pair}.json", report)
                        require(report["status"] == "MATCH", report["status"])
            env = dict(environment, VORTEX_GPU_SESSION=session)
            env.update(HF_HUB_OFFLINE="0" if role == "primary" else "1",
                       TRANSFORMERS_OFFLINE="0" if role == "primary" else "1")
            if role == "negative":
                probe = directory / "runtime/negative-cache"
                probe.mkdir(parents=True, mode=0o700)
                env.update(HF_HOME=str(probe / "home"), HF_HUB_CACHE=str(probe / "hub"))
            processes[role].update(launch_worker(role, env, on_event, deadline))
            processes[role]["state"] = "COMPLETE"
            persist()
            if role == "primary":
                checkpoint()  # distinct immutable partial snapshot, never amended
        reports = validate_captures(records, ledger, source, session)
        for pair, report in reports.items():
            atomic_json(bundle / f"comparisons/{pair}.json", report)
        require(source_identity(ROOT, source) == actual_source, "SOURCE_DRIFT")
        contents = {name: read_bytes(path) for name, path in bundle_files(bundle).items()}
        contents["capture.COMPLETE"] = encode_json(_CAPTURE_COMPLETE)
        validate_execution_contents(contents)
        atomic_json(bundle / "capture.COMPLETE", _CAPTURE_COMPLETE)
    except BaseException as error:
        failure = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else (
            str(error) if isinstance(error, Blocked) else "EXPERIMENT_FAILED")
        if active is not None:
            active["state"] = "FAILED"
        if role in processes:
            processes[role].update(state="FAILED", failure_code=failure)
        persist()
        atomic_json(bundle / f"failures/{role}.json", {"code": failure, "complete": False})
    # Freeze even incomplete attempts. Export failures after sealing are external;
    # the capture ledger/index/archive never gains mutable export statuses.
    index = seal_bundle(bundle)
    archive = directory / "exports/capture.zip"
    archive_digest = None
    try:
        archive_digest = export_bundle(bundle, archive)
    except (ValueError, OSError):
        failure = failure or "EXPORT_FAILED"
    receipt = {"capture_index_sha256": index, "archive_sha256": archive_digest,
        "supporting_hashes": {"index": index, **({"archive": archive_digest} if archive_digest else {})},
        "status": "INCOMPLETE" if failure else "CAPTURED_UNREVIEWED_EXPORT_PENDING",
        "failure_code": failure, "human_approval": "NOT_APPROVED"}
    atomic_json(review / "qualification.INCOMPLETE", receipt)
    return dict(receipt, capture_directory=str(bundle), archive=str(archive))


def execute_gpu():
    paths, approval = checked_controls(os.environ)
    source = os.environ["VORTEX_GPU_SOURCE_SHA"]
    source_identity(ROOT, source)
    paths["root"].mkdir(parents=True, exist_ok=True, mode=0o700)
    require(not paths["cache"].exists(), "CACHE_MUST_BE_FRESH")
    paths["cache"].mkdir(parents=True, mode=0o700)
    for directory in (paths["root"], paths["cache"]):
        require(directory.stat().st_uid == os.getuid() and stat.S_IMODE(directory.stat().st_mode) == 0o700,
                "PRIVATE_PERMISSIONS")
    # Whitelist environment instead of inheriting account credentials/native hooks.
    environment = {key: value for key, value in os.environ.items() if key in {"PATH", "LANG", "LC_ALL",
        "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "PYTHONUNBUFFERED"}
        or key.startswith("VORTEX_GPU_")}
    environment.update(HOME=str(paths["root"] / "private-home"), HF_HOME=str(paths["cache"] / "home"),
        HF_HUB_CACHE=str(paths["cache"] / "hub"), HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1", DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false")
    return capture_session(paths["root"], source, environment, approval)


def _fixture_records():
    """Hand-controlled host records, never a claim of native qualification."""
    sys.path.insert(0, str(ROOT / "tests/unit"))
    from test_reference_comparison import completed
    from test_reference_manifest import resign
    from vortex.executor.reference_loading import SYNTHETIC_PROMPTS
    from vortex.executor.reference_manifest import canonical_digest
    session = str(uuid4())
    records, attempts = {}, []
    for label in ("A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "A4", "B4", "C4"):
        record = completed((10, 151645), 32)
        record["config"].update(profile_id="t4-fp16", device="cuda:0", dtype="float16")
        numerical = record["numerical"]
        numerical["environment"]["packages"]["torch"]["version"] = "2.10.0+cu126"
        numerical["environment"]["hardware"].update(device="cuda:0", gpu_name="synthetic T4",
            compute_capability=[7, 5], driver_version="synthetic-driver", cuda_runtime="12.6")
        messages = [{"role": "user", "content": SYNTHETIC_PROMPTS[ord(label[0]) - ord("A")]}]
        numerical["prompt"].update(messages=messages,
            messages_sha256=canonical_digest({"messages": messages}))
        record["provenance"].update(session_id=session, run_id=str(uuid4()))
        records[label] = resign(record)
        attempts.append({"capture": label, "state": "COMPLETE", "process": "replay" if label[1] == "4"
                         else "primary", "pid": 202 if label[1] == "4" else 101,
                         "instance": "replay-instance" if label[1] == "4" else "primary-instance"})
    return records, attempts, session


class ControllerTests(unittest.TestCase):
    def api(self, name):
        self.assertTrue(callable(globals().get(name)), f"qualification controller missing: {name}")
        return globals()[name]

    def test_explicit_opt_in_required_before_any_files_or_native_imports(self):
        gate = self.api("require_execution")
        for environment in ({}, {"VORTEX_GPU_EXECUTE": "yes"}, {"VORTEX_GPU_EXECUTE": "1"}):
            with self.subTest(environment=environment), self.assertRaisesRegex(ValueError, "BLOCKED"):
                gate(environment)

    def test_windows_execution_refused_even_with_explicit_opt_in(self):
        gate = self.api("require_execution")
        with patch("platform.system", return_value="Windows"):
            with self.assertRaisesRegex(ValueError, "BLOCKED"):
                gate({"VORTEX_GPU_EXECUTE": "1", "VORTEX_GPU_PROFILE": "t4-fp16"})

    def test_source_sha_and_dirty_checkout_are_independent_gates(self):
        source = self.api("source_identity")
        with patch("subprocess.run") as run:
            run.return_value = subprocess.CompletedProcess([], 0, b"b" * 40 + b"\n", b"")
            with self.assertRaisesRegex(ValueError, "SOURCE"):
                source(ROOT, "a" * 40)
        with patch("subprocess.run") as run:
            run.side_effect = [subprocess.CompletedProcess([], 0, b"a" * 40 + b"\n", b""),
                               subprocess.CompletedProcess([], 0, b" M docs/example.md\0", b"")]
            with self.assertRaisesRegex(ValueError, "SOURCE"):
                source(ROOT, "a" * 40)

    def test_atomic_write_readback_and_no_partial_destination(self):
        write = self.api("atomic_json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            digest = write(path, {"complete": False, "code": "INTERRUPTED"})
            self.assertEqual(json.loads(path.read_bytes()), {"complete": False, "code": "INTERRUPTED"})
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            before = path.read_bytes()
            with patch("os.replace", side_effect=OSError("sensitive native detail")):
                with self.assertRaisesRegex(ValueError, "WRITE"):
                    write(path, {"complete": True})
            self.assertEqual(path.read_bytes(), before)

    def test_oversized_nonfinite_duplicate_and_deep_json_rejected(self):
        read, write = self.api("read_json"), self.api("atomic_json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            for data in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":' + b'[' * 40 + b'0' + b']' * 40 + b'}',
                         b" " * 1048577):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    read(path)
            for value in ({"x": float("inf")}, {"x": "x" * 1048577}):
                with self.assertRaises(ValueError):
                    write(path, value)

    def test_unsafe_private_paths_refused_lexically(self):
        check = self.api("private_path")
        for name in ("//server/share/evidence", "\\\\server\\share", "\\/server/share", "../escape",
                     str(ROOT / "results/evidence")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                check(name)

    def test_asset_size_and_both_digest_algorithms_checked_without_loading(self):
        verify = self.api("verify_asset")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            path.write_bytes(b"abc")
            for algorithm, expected in (("sha256", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
                                        ("git-sha1", "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f")):
                self.assertEqual(verify(path, (3, algorithm, expected)),
                                 "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
                with self.assertRaisesRegex(ValueError, "ASSET"):
                    verify(path, (4, algorithm, expected))
                with self.assertRaisesRegex(ValueError, "ASSET"):
                    verify(path, (3, algorithm, "0" * len(expected)))

    def test_twelve_captures_produce_nine_exact_comparisons_without_approval(self):
        validate = self.api("validate_captures")
        records, attempts, session = _fixture_records()
        result = validate(records, attempts, "a" * 40, session)
        self.assertEqual(list(result), ["A1-A2", "A1-A3", "A1-A4", "B1-B2", "B1-B3", "B1-B4",
                                       "C1-C2", "C1-C3", "C1-C4"])
        self.assertTrue(all(item["status"] == "MATCH" for item in result.values()))
        self.assertNotIn("golden_approved", json.dumps(result))

    def test_invalid_incomplete_and_missing_capture_rejected(self):
        validate = self.api("validate_captures")
        for mode in ("missing", "incomplete", "invalid", "source", "run-id", "prompt"):
            records, attempts, session = _fixture_records()
            if mode == "missing":
                records.pop("C4")
            elif mode == "incomplete":
                records["A1"]["complete"] = False
            elif mode == "invalid":
                records["A1"]["comparison_key_sha256"] = "0" * 64
            elif mode == "source":
                records["A4"]["provenance"]["source_sha"] = "b" * 40
            elif mode == "run-id":
                records["A2"]["provenance"]["run_id"] = records["A1"]["provenance"]["run_id"]
            else:
                records["B1"] = deepcopy(records["A1"])
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate(records, attempts, "a" * 40, session)

    def test_replay_must_cross_process_boundary_and_primary_must_share_instance(self):
        validate = self.api("validate_captures")
        for mode in ("same-process", "split-primary", "uncompleted", "order"):
            records, attempts, session = _fixture_records()
            if mode == "same-process":
                for attempt in attempts:
                    attempt.update(pid=101, instance="primary-instance")
            elif mode == "split-primary":
                attempts[1]["instance"] = "another-instance"
            elif mode == "uncompleted":
                attempts[-1]["state"] = "RUNNING"
            else:
                attempts[0], attempts[1] = attempts[1], attempts[0]
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "CAPTURE"):
                validate(records, attempts, "a" * 40, session)

    def test_token_mismatch_and_numerical_drift_stop_qualification(self):
        validate = self.api("validate_captures")
        from vortex.executor.reference_manifest import canonical_digest
        for mode, code in (("tokens", "MISMATCH"), ("identity", "INCOMPATIBLE")):
            records, attempts, session = _fixture_records()
            changed = records["A2"]
            if mode == "tokens":
                changed["outputs"]["generated_ids"][0] = 11
                changed["outputs"]["sequence_ids"][3] = 11
            else:
                changed["numerical"]["environment"]["python_build"] = "other synthetic build"
                changed["comparison_key_sha256"] = canonical_digest({
                    "config_sha256": changed["config_sha256"], "numerical": changed["numerical"]})
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, code):
                validate(records, attempts, "a" * 40, session)


class SupervisionTests(unittest.TestCase):
    api = ControllerTests.api
    def run_child(self, code, stages=(), callback=lambda event, pid: None, seconds=2):
        return self.api("supervise")([sys.executable, "-I", "-B", "-c", code],
            dict(os.environ), stages, callback, seconds, time.monotonic() + 5)

    def test_child_timeout_stops_and_reaps_without_waiting_for_native_return(self):
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, "TIMEOUT"):
            self.run_child("import time; time.sleep(30)", seconds=.15)
        self.assertLess(time.monotonic() - started, 3)

    def test_native_crash_and_unexpected_clean_exit_are_not_success(self):
        for code, stages in (("import os; os._exit(23)", ()), ("pass", (("load", 1),))):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "CHILD|PROTOCOL"):
                self.run_child(code, stages)

    def test_stdout_and_stderr_size_limits_stop_child(self):
        for target in ("stdout", "stderr"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "OUTPUT_LIMIT"):
                self.run_child(f"import sys; sys.{target}.buffer.write(b'x' * 1048578); sys.{target}.flush()")

    def test_ordered_start_is_acknowledged_only_after_durable_callback(self):
        seen = []
        code = ("import json,sys; "
                "print(json.dumps({'event':'start','stage':'A1'}),flush=True); sys.stdin.readline(); "
                "print(json.dumps({'event':'done','stage':'A1','data':{'complete':True}}),flush=True); "
                "sys.stdin.readline()")
        result = self.run_child(code, (("A1", 1),), lambda event, pid: seen.append((event, pid)))
        self.assertEqual([e[0]["event"] for e in seen], ["start", "done"])
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(len({e[1] for e in seen}), 1)

    def test_repeated_wrong_and_malformed_events_do_not_extend_deadlines(self):
        for event in ({"event": "done", "stage": "load", "data": {}},
                      {"event": "start", "stage": "wrong"}, {"unexpected": "secret"}):
            code = f"import sys; print({json.dumps(json.dumps(event))},flush=True); sys.stdin.readline()"
            with self.subTest(event=event), self.assertRaisesRegex(ValueError, "PROTOCOL"):
                self.run_child(code, (("load", 1),))

    def test_interrupt_during_callback_terminates_child_and_propagates(self):
        def interrupt(event, pid):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_child("import sys,time; print('{\"event\":\"start\",\"stage\":\"load\"}',flush=True); time.sleep(30)",
                           (("load", 1),), interrupt)

    def test_request_deadline_and_session_deadline_are_nested(self):
        code = "import sys,time; print('{\"event\":\"start\",\"stage\":\"load\"}',flush=True); sys.stdin.readline(); time.sleep(30)"
        with self.assertRaisesRegex(ValueError, "TIMEOUT"):
            self.run_child(code, (("load", .1),), seconds=2)
        with self.assertRaisesRegex(ValueError, "TIMEOUT"):
            self.api("supervise")([sys.executable, "-B", "-c", "import time;time.sleep(30)"],
                dict(os.environ), (), lambda e, p: None, 2, time.monotonic() + .1)


class EvidenceTests(unittest.TestCase):
    api = ControllerTests.api

    def small_bundle(self, root):
        bundle = root / "capture"
        bundle.mkdir()
        (bundle / "session.json").write_text('{"complete":false,"synthetic":true}', encoding="utf-8")
        (bundle / "attempts.json").write_text('{"attempts":[]}', encoding="utf-8")
        return bundle

    def test_immutable_index_and_archive_exclude_all_external_status(self):
        seal, export = self.api("seal_bundle"), self.api("export_bundle")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.small_bundle(root)
            (root / "qualification.INCOMPLETE").write_text("external")
            digest = seal(bundle)
            archive = root / "capture.zip"
            export(bundle, archive)
            before = archive.read_bytes()
            (root / "qualification.INCOMPLETE").unlink()
            (root / "qualification.COMPLETE").write_text("external updated")
            with zipfile.ZipFile(archive) as source:
                self.assertEqual(set(source.namelist()), {"session.json", "attempts.json", "bundle-index.json",
                                                        "bundle-index.sha256"})
            self.assertEqual(digest, hashlib.sha256((bundle / "bundle-index.json").read_bytes()).hexdigest())
            self.assertEqual(archive.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "SEALED|EXISTS"):
                seal(bundle)

    def test_unexpected_status_or_weight_member_cannot_enter_bundle(self):
        seal = self.api("seal_bundle")
        for name in ("qualification.COMPLETE", "golden-approval.json", "model.safetensors", "export-reopen.json"):
            with tempfile.TemporaryDirectory() as directory:
                bundle = self.small_bundle(Path(directory))
                (bundle / name).write_text("sensitive")
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "MEMBER"):
                    seal(bundle)

    def test_actual_archive_reopen_and_every_hash_are_verified(self):
        seal, export, reopen = self.api("seal_bundle"), self.api("export_bundle"), self.api("reopen_bundle")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.small_bundle(root)
            index = seal(bundle)
            archive = root / "capture.zip"
            digest = export(bundle, archive)
            receipt = reopen(archive, root / "downloaded", digest, index)
            self.assertEqual(receipt["files_readback"], 2)
            self.assertFalse(receipt["qualification_complete"])
            self.assertEqual(receipt["human_approval"], "NOT_APPROVED")
            self.assertEqual((root / "downloaded/capture/session.json").read_bytes(),
                             (bundle / "session.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "CHECKSUM"):
                reopen(archive, root / "wrong-hash", "0" * 64, index)

    def test_corrupted_index_member_and_destination_readback_fail_closed(self):
        seal, export, reopen = self.api("seal_bundle"), self.api("export_bundle"), self.api("reopen_bundle")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self.small_bundle(root)
            index = seal(bundle)
            (bundle / "session.json").write_text('{"complete":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CHECKSUM"):
                export(bundle, root / "corrupt.zip")
            # Rebuild a distinct disposable attempt, never repair a sealed attempt.
            bundle2 = root / "other"
            bundle2.mkdir()
            (bundle2 / "session.json").write_text('{"complete":false}')
            index = seal(bundle2)
            archive = root / "good.zip"
            digest = export(bundle2, archive)
            with patch.dict(globals(), {"atomic_bytes": lambda *args: (_ for _ in ()).throw(Blocked("WRITE_READBACK"))}):
                with self.assertRaisesRegex(ValueError, "READBACK|WRITE"):
                    reopen(archive, root / "failed-readback", digest, index)
            self.assertFalse((root / "failed-readback/review/qualification.COMPLETE").exists())

    def test_archive_traversal_duplicates_symlinks_and_extra_members_refused(self):
        reopen = self.api("reopen_bundle")
        for mode in ("traversal", "duplicate", "symlink", "extra"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / "unsafe.zip"
                with zipfile.ZipFile(archive, "w") as target:
                    if mode == "symlink":
                        member = zipfile.ZipInfo("session.json")
                        member.create_system = 3
                        member.external_attr = (stat.S_IFLNK | 0o777) << 16
                        target.writestr(member, "outside")
                    elif mode == "duplicate":
                        target.writestr("session.json", "{}")
                        target.writestr("SESSION.json", "{}")
                    else:
                        target.writestr("../outside" if mode == "traversal" else "unexpected.json", "{}")
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    reopen(archive, root / "destination", digest, "0" * 64)
                self.assertFalse((root / "outside").exists())


def _profile_fixture():
    from verify_reference_environment import read_lock
    lock = read_lock(ROOT / "docs/reference-environment.md", "t4-fp16")
    observed = {key: lock[key] for key in ("python_version", "implementation", "system", "machine")}
    observed.update(packages={p["name"]: p["version"] for p in lock["packages"]},
        wheel_hashes={p["name"]: p["sha256"] for p in lock["packages"]},
        wheel_origins={p["name"]: p["url"] for p in lock["packages"]},
        pip_check=True, api_contract=True, forbidden_packages=[], cuda_available=True,
        device_count=1, cuda_runtime="12.6", compute_capability=[7, 5], architectures=["sm_75"],
        driver_version="560.35.05", glibc_version="2.28", python_build=["synthetic", "synthetic"],
        platform_build="Linux-synthetic", torch_git_version="synthetic", transformers_version="4.53.3")
    report = {"status": "IMPORTS_VERIFIED", "profile_id": "t4-fp16", "qualified": False,
              "forward_smoke": "NOT_VERIFIED", "observation": observed}
    hardware = {"gpu_name": "Tesla T4", "device": "cuda:0", "total_vram_bytes": 16 * 2**30,
        "free_vram_bytes": 12 * 2**30, "host_available_bytes": 12 * 2**30, "disk_free_bytes": 8 * 2**30,
        "driver_version": "560.35.05"}
    approval = {"driver_version": "560.35.05", "security_reviewed": True, "execution_authorized": True,
        "source_sha": "a" * 40, "profile_id": "t4-fp16", "provider": "colab-free",
        "reviewed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "remaining_free_seconds": 5400, "tools": {"python_archive_sha256":
            "17c4978a36c496dc53c08899ef8dad1ad2734f6325f08d6033a1e66419e0af26",
            "python_executable_sha256": "1" * 64, "python_sys_version": "synthetic Python",
            "uv_archive_sha256": "2" * 64, "uv_executable_sha256": "3" * 64,
            "uv_version": "0.10.7", "uv_identity_reviewed": True,
            "uv_release": "https://github.com/astral-sh/uv/releases/tag/0.10.7"}}
    return lock, report, hardware, approval


class NativeBoundaryTests(unittest.TestCase):
    api = ControllerTests.api

    def test_complete_profile_metadata_is_still_not_gpu_qualification(self):
        validate = self.api("validate_profile")
        lock, report, hardware, approval = _profile_fixture()
        before = deepcopy(report)
        validate(lock, report, hardware, approval)
        self.assertEqual(report, before)
        self.assertFalse(report["qualified"])

    def test_missing_or_incompatible_environment_and_hardware_refused(self):
        validate = self.api("validate_profile")
        for key, value in (("python_version", "3.11.5"), ("cuda_runtime", "12.5"), ("device_count", 2),
                           ("compute_capability", [8, 0]), ("architectures", ["sm_80"]),
                           ("pip_check", False), ("api_contract", False), ("packages", {})):
            lock, report, hardware, approval = _profile_fixture()
            report["observation"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "ENVIRONMENT"):
                validate(lock, report, hardware, approval)
        for key, value in (("gpu_name", "NVIDIA A100"), ("device", "cpu"), ("free_vram_bytes", 1),
                           ("host_available_bytes", 1), ("disk_free_bytes", 1)):
            lock, report, hardware, approval = _profile_fixture()
            hardware[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "ENVIRONMENT"):
                validate(lock, report, hardware, approval)

    def test_driver_security_and_import_report_cannot_be_self_qualified(self):
        validate = self.api("validate_profile")
        for mode in ("unreviewed", "driver", "qualified", "smoke"):
            lock, report, hardware, approval = _profile_fixture()
            if mode == "unreviewed":
                approval["security_reviewed"] = False
            elif mode == "driver":
                approval["driver_version"] = "different"
            elif mode == "qualified":
                report["qualified"] = True
            else:
                report["forward_smoke"] = "VERIFIED"
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate(lock, report, hardware, approval)

    def test_uv_unresolved_identity_gate_refuses_before_worker_launch(self):
        check = self.api("validate_approval")
        approval = {"execution_authorized": True, "source_sha": "a" * 40,
                    "profile_id": "t4-fp16", "provider": "colab-free"}
        with self.assertRaisesRegex(ValueError, "APPROVAL|TOOL"):
            check(approval, "a" * 40)

    def test_capture_adapter_preserves_eos_at_cap_and_non_eos_limit(self):
        capture = self.api("capture_request")
        sys.path.insert(0, str(ROOT / "tests/unit"))
        from test_reference_comparison import completed
        for ids, reason in (((151645,), "eos"), (tuple([10] * 31 + [151643]), "eos"),
                            (tuple([10] * 32), "limit")):
            reference = completed(ids, 32)
            def execute(record, loaded, messages, config, determinism):
                record.update(deepcopy(reference))
            with patch("run_baseline._execute", side_effect=execute):
                record = capture(object(), [{"role": "user", "content": "synthetic"}], "a" * 40,
                                 str(uuid4()), {})
            self.assertEqual(record["outputs"]["generated_ids"], list(ids))
            self.assertEqual(record["outputs"]["stop_reason"], reason)
            self.assertEqual(record["outputs"]["reached_limit"], len(ids) == 32)

    def test_discovery_never_imports_native_ml_or_contacts_network(self):
        code = ("import sys,runpy; "
            "sys.addaudithook(lambda event,args: (_ for _ in ()).throw(RuntimeError('forbidden')) "
            "if event in ('socket.connect','socket.getaddrinfo') or "
            "(event=='import' and args[0].split('.')[0] in ('torch','transformers','numpy','huggingface_hub')) else None); "
            f"runpy.run_path({str(Path(__file__).resolve())!r},run_name='safe_discovery')")
        run = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, timeout=10)
        self.assertEqual(run.returncode, 0, run.stderr.decode("utf-8", "replace"))


class CaptureSessionTests(unittest.TestCase):
    api = ControllerTests.api

    def simulate(self, root, failure=None):
        run = self.api("capture_session")
        records, attempts, session = _fixture_records()
        lock, report, hardware, approval = _profile_fixture()
        source = {"source_sha": "a" * 40, "dirty": False, "files": {"example.py": "0" * 64}}
        preflight = {"source": source, "imports": report, "hardware": hardware, "approval": approval,
            "python_sys_version": "synthetic Python", "determinism": records["A1"]["numerical"]["determinism"],
            **{key: approval["tools"][key] for key in ("python_executable_sha256", "uv_executable_sha256",
                "python_archive_sha256", "uv_archive_sha256")}}
        from vortex.executor.reference_loading import _ASSETS, _TEMPLATE_SHA256
        from vortex.executor.reference_config import MODEL_ID, REVISION
        inventory = {name: {"size": size, "algorithm": algorithm, "expected": expected,
            "raw_sha256": expected if algorithm == "sha256" else "4" * 64}
            for name, (size, algorithm, expected) in _ASSETS.items()}
        assets = {"model_config_sha256": inventory["config.json"]["raw_sha256"],
            "generation_config_sha256": inventory["generation_config.json"]["raw_sha256"],
            "tokenizer_config_sha256": inventory["tokenizer_config.json"]["raw_sha256"],
            "chat_template_sha256": _TEMPLATE_SHA256,
            "weights_sha256": canonical_digest({"model.safetensors": inventory["model.safetensors"]["raw_sha256"]}),
            "tokenizer_assets_sha256": canonical_digest({name: inventory[name]["raw_sha256"] for name in
                ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt")})}
        asset_data = {"model_id": MODEL_ID, "revision": REVISION, "inventory": inventory, "assets": assets}
        for record in records.values():
            record["numerical"].update(environment=numerical_environment(preflight), assets=assets)
            record["comparison_key_sha256"] = canonical_digest({"config_sha256": record["config_sha256"],
                                                                 "numerical": record["numerical"]})
        def launch(role, environment, callback, deadline):
            labels = CAPTURES[:9] if role == "primary" else CAPTURES[9:]
            stages = [("environment", preflight), ("negative", _NEGATIVE_COMPLETE)] if role == "negative" else [
                ("environment", preflight), ("smoke", _SMOKE_COMPLETE), ("assets", asset_data),
                ("load", _LOAD_COMPLETE)] + [(label, records[label]) for label in labels]
            for stage, data in stages:
                callback({"event": "start", "stage": stage}, 202 if role == "replay" else 101)
                if failure == stage:
                    raise Blocked("TIMEOUT")
                if failure == "interrupt" and stage == "B1":
                    raise KeyboardInterrupt
                callback({"event": "done", "stage": stage, "data": data}, 202 if role == "replay" else 101)
            return {"pid": 202 if role == "replay" else 101, "returncode": 0}
        with patch.dict(globals(), {"launch_worker": launch, "source_identity": lambda *args: source}):
            result = run(root, "a" * 40, {}, approval, session=session)
        return result, records

    def test_successful_synthetic_session_preserves_twelve_and_nine_unapproved(self):
        with tempfile.TemporaryDirectory() as directory:
            result, records = self.simulate(Path(directory))
            self.assertEqual(result["status"], "CAPTURED_UNREVIEWED_EXPORT_PENDING")
            bundle = Path(result["capture_directory"])
            self.assertEqual(len(list((bundle / "manifests").glob("*.json"))), 12)
            self.assertEqual(len(list((bundle / "comparisons").glob("*.json"))), 9)
            self.assertEqual(read_json(bundle / "manifests/A1.json")["outputs"]["generated_ids"], [10, 151645])
            self.assertFalse((bundle.parent / "review/qualification.COMPLETE").exists())
            self.assertTrue((bundle.parent / "review/qualification.INCOMPLETE").exists())
            self.assertTrue((bundle.parent / "exports/primary.zip").is_file())
            self.assertEqual(result["human_approval"], "NOT_APPROVED")

    def test_hung_capture_and_interruption_keep_incomplete_attempts_and_prior_results(self):
        for failure, code in (("B1", "TIMEOUT"), ("interrupt", "INTERRUPTED")):
            with tempfile.TemporaryDirectory() as directory:
                result, _ = self.simulate(Path(directory), failure)
                self.assertEqual(result["status"], "INCOMPLETE")
                self.assertEqual(result["failure_code"], code)
                bundle = Path(result["capture_directory"])
                ledger = read_json(bundle / "attempts.json")["attempts"]
                self.assertEqual([a["state"] for a in ledger[:4]], ["COMPLETE", "COMPLETE", "COMPLETE", "FAILED"])
                self.assertTrue((bundle / "manifests/A1.json").is_file())
                self.assertFalse((bundle / "manifests/B1.json").exists())
                self.assertFalse((bundle / "capture.COMPLETE").exists())

    def test_incomplete_export_cannot_qualify_successful_captures(self):
        self.api("capture_session")
        with tempfile.TemporaryDirectory() as directory:
            original = globals()["export_bundle"]
            def fail_final(bundle, archive):
                if archive.name == "capture.zip":
                    raise Blocked("EXPORT_FAILED")
                return original(bundle, archive)
            with patch.dict(globals(), {"export_bundle": fail_final}):
                result, _ = self.simulate(Path(directory))
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual(result["failure_code"], "EXPORT_FAILED")
            self.assertFalse((Path(result["capture_directory"]).parent / "review/qualification.COMPLETE").exists())

    def test_mock_worker_runs_primary_repeats_then_replay_without_loading_locally(self):
        protocol = self.api("worker_protocol")
        records, _, session = _fixture_records()
        events, loads = [], []
        def fake_load(*args, **kwargs):
            loads.append(kwargs)
            return object()
        def capture(loaded, messages, source, supplied_session, determinism):
            from vortex.executor.reference_loading import SYNTHETIC_PROMPTS
            letter = "ABC"[SYNTHETIC_PROMPTS.index(messages[0]["content"])]
            return records[letter + "1"]
        with patch.dict(globals(), {"native_preflight": lambda *a: {}, "tiny_smoke": lambda *a: {},
             "asset_inventory": lambda *a: {}, "offline_negative": lambda *a: {},
             "guarded_load": fake_load, "capture_request": capture, "source_identity": lambda *a: {}}):
            protocol("primary", {}, {}, "a" * 40, session,
                     lambda event: events.append(event), {"observed": {}})
            protocol("replay", {}, {}, "a" * 40, session,
                     lambda event: events.append(event), {"observed": {}})
        self.assertEqual([e["stage"] for e in events if e["event"] == "done" and e["stage"] in CAPTURES], list(CAPTURES))
        self.assertEqual(len(loads), 2)
        self.assertTrue(all(item["offline"] is True for item in loads))


class ReviewRegressionTests(unittest.TestCase):
    api = ControllerTests.api

    def test_acquisition_and_load_share_the_same_absolute_budget(self):
        code = """import sys,json,time
for stage,seconds in [('assets',.09),('load',.2)]:
 print(json.dumps({'event':'start','stage':stage}),flush=True);sys.stdin.readline()
 time.sleep(seconds)
 print(json.dumps({'event':'done','stage':stage,'data':{}}),flush=True);sys.stdin.readline()
"""
        with self.assertRaisesRegex(ValueError, "TIMEOUT"):
            supervise([sys.executable, "-B", "-c", code], dict(os.environ),
                (("assets", .2), ("load", .5)), lambda e, p: None, 2, time.monotonic() + 3)

    def test_symlinked_venv_python_is_bound_to_approved_resolved_executable(self):
        verify = self.api("tool_identity")
        with tempfile.TemporaryDirectory() as directory:
            target, link = Path(directory) / "python-real", Path(directory) / "python"
            target.write_bytes(b"synthetic executable")
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("Windows symlink fixture permission unavailable; Linux path NOT VERIFIED")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            self.assertEqual(verify(link, digest), digest)
            with self.assertRaisesRegex(ValueError, "TOOL"):
                verify(link, "0" * 64)

    def test_repeat_mismatch_stops_before_next_capture_and_retains_report(self):
        fixture = CaptureSessionTests()
        original = globals()["capture_session"]
        def changed(*args, **kwargs):
            real_compare = globals()["compare_reference"]
            def compare(left, right):
                result = real_compare(left, right)
                return type(result)("MISMATCH", 0, "token_mismatch")
            with patch.dict(globals(), {"compare_reference": compare}):
                return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(globals(), {"capture_session": changed}):
                result, _ = fixture.simulate(Path(directory))
            bundle = Path(result["capture_directory"])
            self.assertEqual(result["failure_code"], "MISMATCH")
            self.assertFalse((bundle / "manifests/A3.json").exists())
            self.assertEqual(read_json(bundle / "comparisons/A1-A2.json")["status"], "MISMATCH")

    def test_capture_marker_alone_cannot_certify_incomplete_qualification_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = EvidenceTests().small_bundle(root)
            atomic_json(bundle / "capture.COMPLETE", {"captures": 12, "comparisons": 9})
            index = seal_bundle(bundle)
            archive = root / "forged.zip"
            digest = export_bundle(bundle, archive)
            with self.assertRaises(ValueError):
                reopen_bundle(archive, root / "reopened", digest, index)
            self.assertFalse((root / "reopened/review/qualification.COMPLETE").exists())

    def test_direct_worker_invocation_requires_authorization_before_native_import(self):
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--_worker", "primary", str(uuid4())]
        environment = {key: value for key, value in os.environ.items() if not key.startswith("VORTEX_GPU_")}
        result = subprocess.run(command, env=environment, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout), {"status": "BLOCKED", "code": "BLOCKED_EXECUTION"})
        self.assertEqual(result.stderr, b"")

    def test_assume_unchanged_cannot_hide_unapproved_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.run(["git", "-C", str(root), *args], check=True,
                    capture_output=True, timeout=10).stdout
            git("init", "-q")
            git("config", "core.autocrlf", "false")
            (root / "source.py").write_bytes(b"approved\n")
            git("add", "source.py")
            git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
            sha = git("rev-parse", "HEAD").decode().strip()
            git("update-index", "--assume-unchanged", "source.py")
            (root / "source.py").write_bytes(b"unreviewed\n")
            self.assertEqual(git("status", "--porcelain"), b"")
            with self.assertRaisesRegex(ValueError, "SOURCE"):
                source_identity(root, sha)

    def test_native_oom_is_a_finite_failed_stage_and_never_acknowledged_as_done(self):
        code = """import sys,json
print(json.dumps({'event':'start','stage':'load'}),flush=True);sys.stdin.readline()
print(json.dumps({'event':'failed','stage':'load','data':{'code':'OUT_OF_MEMORY'}}),flush=True)
sys.stdin.readline()
"""
        with self.assertRaisesRegex(ValueError, "OUT_OF_MEMORY"):
            supervise([sys.executable, "-B", "-c", code], dict(os.environ), (("load", 1),),
                lambda e, p: None, 2, time.monotonic() + 3)

    def test_successful_synthetic_full_bundle_reopens_and_tampered_smoke_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, _ = CaptureSessionTests().simulate(root)
            receipt = reopen_bundle(Path(result["archive"]), root / "actual-download",
                result["archive_sha256"], result["capture_index_sha256"])
            self.assertTrue(receipt["qualification_complete"])
            self.assertEqual(receipt["independent_review"], "NOT_VERIFIED")
            bundle = Path(result["capture_directory"])
            contents = {name: read_bytes(path) for name, path in bundle_files(bundle).items()}
            qualification = parse_json(contents["environment/qualification.json"])
            qualification["smoke"]["finite"] = False
            contents["environment/qualification.json"] = encode_json(qualification)
            with self.assertRaisesRegex(ValueError, "QUALIFICATION"):
                validate_execution_contents(contents)


class GPUCaptureTests(unittest.TestCase):
    def test_gpu_capture_unreviewed(self):
        if "VORTEX_GPU_EXECUTE" not in os.environ:
            self.skipTest("NOT VERIFIED: GPU capture requires explicit approved Linux execution controls")
        result = execute_gpu()
        self.assertEqual(result["status"], "CAPTURED_UNREVIEWED_EXPORT_PENDING",
                         result.get("failure_code", "CAPTURE_INCOMPLETE"))
        self.assertEqual(result["human_approval"], "NOT_APPROVED")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--_worker"]:
        try:
            require(len(argv) == 3, "BLOCKED_WORKER")
            worker_entry(argv[1], argv[2])
            return 0
        except BaseException as error:
            code = str(error) if isinstance(error, Blocked) else "WORKER_FAILED"
            print(json.dumps({"status": "BLOCKED", "code": code}, sort_keys=True))
            return 2
    if argv[:1] == ["--reopen"]:
        parser = argparse.ArgumentParser(description="Verify an actually downloaded private capture archive")
        parser.add_argument("--reopen", type=Path, required=True)
        parser.add_argument("--destination", type=Path, required=True)
        parser.add_argument("--archive-sha256", required=True)
        parser.add_argument("--index-sha256", required=True)
        args = parser.parse_args(argv)
        try:
            receipt = reopen_bundle(args.reopen, args.destination, args.archive_sha256, args.index_sha256)
            print(json.dumps(receipt, sort_keys=True))
            return 0 if receipt["qualification_complete"] else 2
        except (ValueError, OSError, zipfile.BadZipFile, KeyError):
            print('{"status":"INCOMPLETE","code":"READBACK_FAILED"}')
            return 2
    unittest.main(argv=[sys.argv[0], *argv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
