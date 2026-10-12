#!/usr/bin/env python3
"""Query optional CUDA metadata; the CLI isolates one worker, without model work."""

import importlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Literal, TypedDict

PROBE_TIMEOUT_SECONDS = 30
MAX_RESULT_BYTES = 64 * 1024
MAX_DEVICES = 64
MAX_TEXT_LENGTH = 512
STAGES = {"nvidia_smi_lookup", "torch_lookup", "torch_import", "cuda_availability",
          "device_count", "device_properties", "probe_process", "probe_result"}


class Device(TypedDict):
    index: int
    name: str
    total_bytes: int
    compute_capability: list[int]


class ProbeError(TypedDict):
    stage: str
    type: str


class Capability(TypedDict):
    platform: str
    python: str
    status: Literal["torch_unavailable", "cuda_unavailable", "cuda_available", "probe_error"]
    torch_installed: bool | None
    torch_available: bool
    torch_version: str | None
    torch_cuda_version: str | None
    nvidia_smi_available: bool | None
    cuda_available: bool | None
    device_count: int | None
    devices: list[Device]
    probe_error: ProbeError | None


def _empty_report() -> Capability:
    return {
        "platform": platform.system(), "python": sys.version.split()[0],
        "status": "torch_unavailable", "torch_installed": None,
        "torch_available": False, "torch_version": None, "torch_cuda_version": None,
        "nvidia_smi_available": None, "cuda_available": None, "device_count": None,
        "devices": [], "probe_error": None,
    }


def _text(value: object) -> bool:
    return (isinstance(value, str) and 0 < len(value) <= MAX_TEXT_LENGTH
            and len(value.encode("utf-8")) <= MAX_TEXT_LENGTH * 4)


def capability() -> Capability:
    """Query in-process; use main() for isolation, timeout and sanitized CLI output.

    torch_available means import succeeded; nvidia_smi_available is PATH presence
    only. Memory is total capacity, not free memory or a workload guarantee.
    """
    out = _empty_report()
    stage = "nvidia_smi_lookup"
    try:
        out["nvidia_smi_available"] = shutil.which("nvidia-smi") is not None
        stage = "torch_lookup"
        out["torch_installed"] = importlib.util.find_spec("torch") is not None
        if not out["torch_installed"]:
            return out
        stage = "torch_import"
        torch = importlib.import_module("torch")
        out["torch_available"] = True
        version, cuda_version = torch.__version__, torch.version.cuda
        if not _text(version) or (cuda_version is not None and not _text(cuda_version)):
            raise TypeError("Unexpected version metadata")
        out["torch_version"], out["torch_cuda_version"] = version, cuda_version
        stage = "cuda_availability"
        available = torch.cuda.is_available()
        if type(available) is not bool:
            raise TypeError("Unexpected CUDA availability")
        out["cuda_available"] = available
        if not available:
            out["status"], out["device_count"] = "cuda_unavailable", 0
            return out
        stage = "device_count"
        count = torch.cuda.device_count()
        if type(count) is not int or not 1 <= count <= MAX_DEVICES:
            raise ValueError("Inconsistent CUDA device count")
        stage = "device_properties"
        devices: list[Device] = []
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            name, memory, major, minor = props.name, props.total_memory, props.major, props.minor
            if not _text(name):
                raise ValueError("Invalid device name")
            if type(memory) is not int or not 1 <= memory <= 2**63 - 1:
                raise ValueError("Invalid total device memory")
            if (type(major) is not int or not 1 <= major <= 255
                    or type(minor) is not int or not 0 <= minor <= 255):
                raise ValueError("Invalid compute capability")
            devices.append({"index": index, "name": name, "total_bytes": memory,
                            "compute_capability": [major, minor]})
        # Publish an inventory only after every visible device was queried successfully.
        out["devices"], out["device_count"], out["status"] = devices, count, "cuda_available"
    except Exception as error:
        out["status"] = "probe_error"
        kind = type(error).__name__
        if not (kind.isascii() and kind.isidentifier() and len(kind) <= 64):
            kind = "Exception"
        out["probe_error"] = {"stage": stage, "type": kind}
    return out


def _failure(stage: str, kind: str) -> Capability:
    report = _empty_report()
    report["status"] = "probe_error"
    report["probe_error"] = {"stage": stage, "type": kind}
    return report


def _validate_report(report: object) -> Capability:
    """Validate transport data without trusting a successful worker exit alone."""
    if not isinstance(report, dict) or set(report) != set(_empty_report()):
        raise ValueError("Invalid report fields")
    if not _text(report["platform"]) or not _text(report["python"]):
        raise ValueError("Invalid environment metadata")
    for name in ("torch_installed", "nvidia_smi_available", "cuda_available"):
        if report[name] is not None and type(report[name]) is not bool:
            raise ValueError("Invalid availability")
    if type(report["torch_available"]) is not bool:
        raise ValueError("Invalid import availability")
    for name in ("torch_version", "torch_cuda_version"):
        if report[name] is not None and not _text(report[name]):
            raise ValueError("Invalid version")
    imported = report["torch_available"]
    if imported and report["torch_installed"] is not True:
        raise ValueError("Inconsistent import")
    if not imported and any(report[k] is not None for k in
                            ("torch_version", "torch_cuda_version", "cuda_available")):
        raise ValueError("Unestablished metadata")
    count, devices = report["device_count"], report["devices"]
    if count is not None and (type(count) is not int or not 0 <= count <= MAX_DEVICES):
        raise ValueError("Invalid device count")
    if not isinstance(devices, list) or len(devices) > MAX_DEVICES:
        raise ValueError("Invalid inventory")
    for index, device in enumerate(devices):
        if not isinstance(device, dict) or set(device) != {
                "index", "name", "total_bytes", "compute_capability"}:
            raise ValueError("Invalid device fields")
        if type(device["index"]) is not int or device["index"] != index or not _text(device["name"]):
            raise ValueError("Invalid device identity")
        memory, cc = device["total_bytes"], device["compute_capability"]
        if type(memory) is not int or not 1 <= memory <= 2**63 - 1:
            raise ValueError("Invalid memory")
        if (not isinstance(cc, list) or len(cc) != 2 or any(type(v) is not int for v in cc)
                or not 1 <= cc[0] <= 255 or not 0 <= cc[1] <= 255):
            raise ValueError("Invalid compute capability")
    status, error = report["status"], report["probe_error"]
    if status == "probe_error":
        if (not isinstance(error, dict) or set(error) != {"stage", "type"}
                or not isinstance(error["stage"], str) or error["stage"] not in STAGES
                or not isinstance(error["type"], str) or not error["type"].isascii()
                or not error["type"].isidentifier() or len(error["type"]) > 64
                or devices or count not in (None, 0)):
            raise ValueError("Invalid error report")
    elif status == "torch_unavailable":
        if error is not None or imported or report["torch_installed"] is not False or count is not None or devices:
            raise ValueError("Inconsistent missing dependency")
    elif status in ("cuda_unavailable", "cuda_available"):
        if error is not None or not imported or report["torch_version"] is None:
            raise ValueError("Unestablished CUDA metadata")
        if status == "cuda_unavailable":
            if report["cuda_available"] is not False or count != 0 or devices:
                raise ValueError("Inconsistent CPU outcome")
        elif (report["cuda_available"] is not True or count is None or count < 1
              or len(devices) != count):
            raise ValueError("Incomplete CUDA inventory")
    else:
        raise ValueError("Unknown outcome")
    return report


def _worker(result_path: str) -> int:
    """Write only bounded structured data; native diagnostics use launch-time DEVNULL."""
    report = _validate_report(capability())
    data = json.dumps(report, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(data) > MAX_RESULT_BYTES:
        data = json.dumps(_failure("probe_result", "OutputLimit")).encode("utf-8")
    # Exclusive creation cannot overwrite an existing file, including on Windows.
    with Path(result_path).open("xb") as output:
        output.write(data)
    return 0  # Transport succeeded; parent maps the report's status to CLI exit0/2.


def _isolated_probe() -> Capability:
    stage = "probe_process"
    try:
        # Restore filesystem import visibility, including user-site and parent -S,
        # without rerunning site initialization in the worker. No parent env mutation.
        paths = json.dumps([str(Path(p or ".").resolve()) for p in sys.path if isinstance(p, str)])
        if len(paths.encode("utf-8")) > MAX_RESULT_BYTES:
            return _failure(stage, "OutputLimit")
        loader = ("import json,runpy,sys; sys.path[:]=json.loads(sys.argv[3]); "
                  "raise SystemExit(runpy.run_path(sys.argv[1])['_worker'](sys.argv[2]))")
        with tempfile.TemporaryDirectory(prefix="vortex-gpu-probe-") as directory:
            result_path = Path(directory) / "report.json"
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", loader,
                 str(Path(__file__).resolve()), str(result_path), paths],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                close_fds=True, timeout=PROBE_TIMEOUT_SECONDS, check=False)
            # No pipes or diagnostic buffers: floods cannot consume parent RAM/disk
            # or block waiting for a drain. run() kills/reaps its worker on timeout.
            if result.returncode:
                return _failure(stage, "ProcessError")
            stage = "probe_result"
            if not stat.S_ISREG(result_path.lstat().st_mode):
                raise ValueError("Result must be a regular file")
            with result_path.open("rb") as source:
                data = source.read(MAX_RESULT_BYTES + 1)
            if len(data) > MAX_RESULT_BYTES:
                return _failure(stage, "OutputLimit")
            def pairs(items: list[tuple[str, object]]) -> dict:
                value = {}
                for key, item in items:
                    if key in value:
                        raise ValueError("Duplicate field")
                    value[key] = item
                return value
            def constant(_: str) -> None:
                raise ValueError("Nonfinite number")
            return _validate_report(json.loads(data.decode("utf-8"), object_pairs_hook=pairs,
                                               parse_constant=constant))
    except subprocess.TimeoutExpired:
        return _failure("probe_process", "TimeoutExpired")
    except KeyboardInterrupt:
        return _failure("probe_process", "KeyboardInterrupt")
    except OSError:
        return _failure(stage, "OSError")
    except (ValueError, TypeError, RecursionError, RuntimeError):
        return _failure(stage, "InvalidResult")


def main() -> int:
    """Print exactly one JSON report; exit0 normal, exit2 when metadata is unreliable."""
    report = _isolated_probe()
    print(json.dumps(report, indent=2, allow_nan=False))
    return 2 if report["status"] == "probe_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
