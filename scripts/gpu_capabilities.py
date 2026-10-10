#!/usr/bin/env python3
"""Query optional CUDA metadata; no tensors, models, installs or external commands."""

from contextlib import redirect_stderr, redirect_stdout
import importlib
import importlib.util
import io
import json
import platform
import shutil
import sys
from typing import Literal, TypedDict


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


def capability() -> Capability:
    """Return visible CUDA metadata, with null for facts that were not established.

    torch_available means import succeeded; nvidia_smi_available is PATH presence
    only. Device memory is total capacity in bytes, not free memory or a workload
    guarantee. Optional-library errors expose stage/type, never raw messages.
    """
    out: Capability = {
        "platform": platform.system(), "python": sys.version.split()[0],
        "status": "torch_unavailable", "torch_installed": None,
        "torch_available": False, "torch_version": None, "torch_cuda_version": None,
        "nvidia_smi_available": None, "cuda_available": None, "device_count": None,
        "devices": [], "probe_error": None,
    }
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
        if not isinstance(version, str) or (cuda_version is not None and not isinstance(cuda_version, str)):
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
        if type(count) is not int or count < 1:
            raise ValueError("Inconsistent CUDA device count")
        stage = "device_properties"
        devices: list[Device] = []
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            name, memory, major, minor = props.name, props.total_memory, props.major, props.minor
            if not isinstance(name, str) or not name:
                raise ValueError("Invalid device name")
            if type(memory) is not int or memory < 1:
                raise ValueError("Invalid total device memory")
            if type(major) is not int or major < 1 or type(minor) is not int or minor < 0:
                raise ValueError("Invalid compute capability")
            devices.append({"index": index, "name": name, "total_bytes": memory,
                            "compute_capability": [major, minor]})
        # Publish an inventory only after every visible device was queried successfully.
        out["devices"], out["device_count"], out["status"] = devices, count, "cuda_available"
    except Exception as error:
        out["status"] = "probe_error"
        out["probe_error"] = {"stage": stage, "type": type(error).__name__}
    return out


def main() -> int:
    """Print one JSON report; exit 0 for normal CPU/CUDA outcomes, 2 for probe errors."""
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        # Dependency chatter must not contaminate the JSON or expose raw diagnostics.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            report = capability()
    finally:
        sys.dont_write_bytecode = previous
    print(json.dumps(report, indent=2, allow_nan=False))
    return 2 if report["status"] == "probe_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
