"""Offline environment-contract tests; all package and device records are synthetic."""

from copy import deepcopy
from contextlib import redirect_stdout
import builtins
import importlib.util
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("reference_environment", ROOT / "scripts/verify_reference_environment.py")
environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(environment)


def lock_fixture(profile="windows-cpu"):
    cpu = profile == "windows-cpu"
    versions = {"torch": "2.10.0+cpu" if cpu else "2.10.0+cu126", "transformers": "4.53.3"}
    packages = []
    for name, version in versions.items():
        url = ("https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp311-cp311-win_amd64.whl"
               if cpu else "https://download.pytorch.org/whl/cu126/torch-2.10.0%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl")
        if name != "torch":
            url = "https://files.pythonhosted.org/packages/synthetic/transformers-4.53.3-py3-none-any.whl"
        packages.append({"name": name, "version": version, "url": url,
                         "sha256": "a" * 64, "requires": []})
    return {"schema_version": 1, "profile_id": profile, "python_version": "3.11.18",
            "implementation": "CPython", "system": "Windows" if cpu else "Linux",
            "machine": "AMD64" if cpu else "x86_64",
            "index_url": "https://pypi.org/simple",
            "torch_index_url": "https://download.pytorch.org/whl/" + ("cpu" if cpu else "cu126"),
            "roots": ["torch", "transformers"], "packages": packages}


def observation_fixture(profile="windows-cpu"):
    lock = lock_fixture(profile)
    cpu = profile == "windows-cpu"
    return {"python_version": "3.11.18", "implementation": "CPython",
            "system": lock["system"], "machine": lock["machine"],
            "packages": {item["name"]: item["version"] for item in lock["packages"]},
            "wheel_hashes": {item["name"]: item["sha256"] for item in lock["packages"]},
            "wheel_origins": {item["name"]: item["url"] for item in lock["packages"]},
            "pip_check": True, "api_contract": True, "cuda_runtime": None if cpu else "12.6",
            "cuda_available": not cpu, "device_count": 0 if cpu else 1,
            "compute_capability": None if cpu else [7, 5], "architectures": [] if cpu else ["sm_75"],
            "driver_version": None if cpu else "560.35.05", "glibc_version": None if cpu else "2.28",
            "forbidden_packages": []}


class EnvironmentContractTests(unittest.TestCase):
    def test_unreviewed_interpreter_patch_is_rejected(self):
        for version in ("3.11.5", "3.11.17", "3.12.0", "3.11", None):
            lock = lock_fixture()
            lock["python_version"] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                environment.validate_lock(lock)

    def test_optional_extras_preserve_dependency_free_base(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        self.assertFalse(project.get("dependencies"))
        self.assertEqual(project["optional-dependencies"]["baseline-cpu"],
                         ["torch==2.10.0+cpu", "transformers==4.53.3"])
        self.assertEqual(project["optional-dependencies"]["baseline-cuda"],
                         ["torch==2.10.0+cu126", "transformers==4.53.3"])

    def test_reviewed_platform_locks_are_complete_and_separate(self):
        cpu = environment.read_lock(ROOT / "docs/reference-environment.md", "windows-cpu")
        gpu = environment.read_lock(ROOT / "docs/reference-environment.md", "t4-fp16")
        cpu_names = {p["name"] for p in cpu["packages"]}
        gpu_names = {p["name"] for p in gpu["packages"]}
        self.assertFalse(any(name.startswith("nvidia-") for name in cpu_names))
        self.assertNotIn("triton", cpu_names)
        self.assertTrue({"nvidia-cuda-runtime-cu12", "triton", "cuda-bindings"} <= gpu_names)
        self.assertEqual(cpu["python_version"], "3.11.18")
        self.assertEqual(gpu["python_version"], "3.11.18")

    def test_valid_cpu_lock_and_observation(self):
        lock, observed = lock_fixture(), observation_fixture()
        self.assertIsNone(environment.validate_lock(lock))
        self.assertIsNone(environment.validate_observation(lock, observed))

    def test_missing_torch_is_actionable(self):
        observed = observation_fixture()
        del observed["packages"]["torch"]
        with self.assertRaisesRegex(ValueError, "package"):
            environment.validate_observation(lock_fixture(), observed)

    def test_version_mismatch_fails(self):
        for name, version in (("torch", "2.1.1"), ("torch", "2.10.0+cu126"),
                              ("transformers", "4.54.0")):
            observed = observation_fixture()
            observed["packages"][name] = version
            with self.subTest(name=name, version=version), self.assertRaises(ValueError):
                environment.validate_observation(lock_fixture(), observed)

    def test_python_and_platform_mismatch(self):
        for key, value in (("python_version", "3.12.0"), ("python_version", "3.11.6"),
                           ("implementation", "PyPy"), ("system", "Linux"), ("machine", "ARM64")):
            observed = observation_fixture()
            observed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                environment.validate_observation(lock_fixture(), observed)

    def test_cpu_build_cannot_claim_cuda(self):
        for key, value in (("cuda_runtime", "12.6"), ("cuda_available", True), ("device_count", 1)):
            observed = observation_fixture()
            observed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                environment.validate_observation(lock_fixture(), observed)

    def test_unavailable_gpu_not_cpu_fallback(self):
        observed = observation_fixture("t4-fp16")
        observed.update(cuda_available=False, device_count=0)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            environment.validate_observation(lock_fixture("t4-fp16"), observed)

    def test_unsupported_driver_arch_profile(self):
        valid = observation_fixture("t4-fp16")
        self.assertIsNone(environment.validate_observation(lock_fixture("t4-fp16"), valid))
        for key, value in (("driver_version", "560.35.04"), ("glibc_version", "2.27"),
                           ("architectures", ["sm_80"]), ("compute_capability", [6, 1]),
                           ("cuda_runtime", "12.8"), ("device_count", 0)):
            observed = deepcopy(valid)
            observed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                environment.validate_observation(lock_fixture("t4-fp16"), observed)

    def test_lock_manifest_complete(self):
        for missing in lock_fixture():
            lock = lock_fixture()
            del lock[missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                environment.validate_lock(lock)
        for bad in ([], {}, None, "bad"):
            lock = lock_fixture()
            lock["packages"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                environment.validate_lock(lock)
        lock = lock_fixture()
        lock["packages"][0]["requires"] = ["missing-transitive"]
        with self.assertRaises(ValueError):
            environment.validate_lock(lock)

    def test_duplicate_or_invalid_package_fields(self):
        for key, value in (("name", "../bad"), ("version", ""), ("sha256", "A" * 64),
                           ("url", "https://example.invalid/torch.whl"), ("requires", [True])):
            lock = lock_fixture()
            lock["packages"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                environment.validate_lock(lock)
        lock = lock_fixture()
        lock["packages"].append(deepcopy(lock["packages"][0]))
        with self.assertRaises(ValueError):
            environment.validate_lock(lock)

    def test_incorrect_wheel_indexes_are_rejected(self):
        for field, value in (("index_url", "https://example.invalid/simple"),
                             ("torch_index_url", "https://download.pytorch.org/whl/cu126")):
            lock = lock_fixture()
            lock[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                environment.validate_lock(lock)
        lock = lock_fixture()
        lock["packages"][0]["url"] = lock["packages"][0]["url"].replace("/cpu/", "/cu126/")
        with self.assertRaises(ValueError):
            environment.validate_lock(lock)

    def test_requirement_injection_in_raw_url_is_rejected(self):
        lock = lock_fixture()
        lock["packages"][1]["url"] += "\n--extra-index-url=https://example.invalid/x.whl"
        with self.assertRaises(ValueError):
            environment.requirements_text(lock)

    def test_gpu_observation_has_strict_types_and_bounds(self):
        for key, bad in (("compute_capability", [7.0, 5.0]), ("compute_capability", [True, 5]),
                         ("architectures", "sm_75"), ("device_count", 999999999)):
            observed = observation_fixture("t4-fp16")
            observed[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                environment.validate_observation(lock_fixture("t4-fp16"), observed)

    def test_installed_wheel_evidence_cannot_be_missing_or_stale(self):
        for bad in ({}, {"torch": "b" * 64}, None):
            observed = observation_fixture()
            observed["wheel_hashes"] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                environment.validate_observation(lock_fixture(), observed)

    def test_deep_json_is_rejected_without_recursion_error(self):
        with self.assertRaises(ValueError):
            environment._strict_json("[" * 2000 + "0" + "]" * 2000)

    def test_cli_missing_dependency_has_actionable_code(self):
        output = io.StringIO()
        with patch.object(environment, "read_lock", return_value=lock_fixture()), \
                patch.object(environment, "inspect_environment", side_effect=
                             environment.EnvironmentFailure("MISSING_DEPENDENCY")), redirect_stdout(output):
            result = environment.main(["--profile", "windows-cpu"])
        self.assertEqual(result, 2)
        record = json.loads(output.getvalue())
        self.assertEqual(record["error_code"], "MISSING_DEPENDENCY")
        self.assertIn("install", record["error"].lower())

    def test_native_crash_without_result_is_import_failure(self):
        output = io.StringIO()
        crashed = subprocess.CompletedProcess([], 139)
        with patch.object(environment, "read_lock", return_value=lock_fixture()), \
                patch.object(environment.subprocess, "run", return_value=crashed), redirect_stdout(output):
            result = environment.main(["--profile", "windows-cpu"])
        self.assertEqual(result, 2)
        self.assertEqual(json.loads(output.getvalue())["error_code"], "IMPORT_FAILED")

    def test_unreviewed_hash_rejected_before_native_import(self):
        lock = lock_fixture()
        packages = {item["name"]: item for item in lock["packages"]}

        def distribution(name):
            return SimpleNamespace(version=packages[name]["version"],
                                   read_text=lambda _: json.dumps({"url": packages[name]["url"],
                                       "archive_info": {"hashes": {"sha256": "b" * 64}}}))

        original_import = builtins.__import__

        def block_native(name, *args, **kwargs):
            if name in ("torch", "transformers"):
                raise AssertionError("unreviewed native import reached")
            return original_import(name, *args, **kwargs)

        records = [SimpleNamespace(metadata={"Name": name}) for name in packages]
        with patch.object(environment.importlib.metadata, "distribution", side_effect=distribution), \
                patch.object(environment.importlib.metadata, "distributions", return_value=records), \
                patch.object(environment.platform, "python_version", return_value="3.11.18"), \
                patch.object(environment.sys, "addaudithook"), \
                patch("builtins.__import__", side_effect=block_native):
            with self.assertRaisesRegex(ValueError, "wheel"):
                environment._inspect(lock)

    def test_wrong_interpreter_rejected_before_package_inspection(self):
        with patch.object(environment.platform, "python_version", return_value="3.12.0"), \
                patch.object(environment.sys, "addaudithook"), \
                patch.object(environment.importlib.metadata, "distribution", side_effect=
                             AssertionError("wrong interpreter reached package inspection")):
            with self.assertRaises(environment.EnvironmentFailure) as caught:
                environment._inspect(lock_fixture())
        self.assertEqual(caught.exception.code, "INTERPRETER_MISMATCH")

    def test_dependency_check_and_optional_kernel_failures(self):
        for key, value in (("pip_check", False), ("api_contract", False),
                           ("forbidden_packages", ["kernels"]), ("cuda_available", 0)):
            observed = observation_fixture()
            observed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                environment.validate_observation(lock_fixture(), observed)

    def test_requirements_preserve_exact_urls_hashes_and_disable_indexes(self):
        text = environment.requirements_text(lock_fixture())
        self.assertTrue(text.startswith("--no-index\n--only-binary=:all:\n--require-hashes\n"))
        self.assertIn("torch @ https://download.pytorch.org/whl/cpu/torch-2.10.0%2Bcpu-cp311-cp311-win_amd64.whl --hash=sha256:" + "a" * 64, text)
        self.assertNotIn("extra-index", text)

    def test_bootstrap_requirements_include_only_locked_pip(self):
        lock = environment.read_lock(ROOT / "docs/reference-environment.md", "windows-cpu")
        text = environment.requirements_text(lock, bootstrap_only=True)
        requirements = [line for line in text.splitlines() if not line.startswith("--")]
        self.assertEqual(len(requirements), 1)
        self.assertTrue(requirements[0].startswith("pip @ https://files.pythonhosted.org/"))
        self.assertIn(" --hash=sha256:", requirements[0])
        self.assertNotIn("torch @", text)

    def test_bootstrap_cli_extracts_without_importing_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "bootstrap.txt"
            output = io.StringIO()
            with patch.object(environment, "inspect_environment", side_effect=AssertionError(
                    "bootstrap must not import packages")), redirect_stdout(output):
                status = environment.main(["--profile", "windows-cpu", "--bootstrap-only",
                                           "--emit-requirements", str(target)])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "LOCK_VALIDATED")
            self.assertIn("pip @", target.read_text())

    def test_document_lock_parser_rejects_duplicates_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment.md"
            marker = "<!-- vortex-lock:windows-cpu -->\n```json\n"
            path.write_text(marker + json.dumps(lock_fixture()) + "\n```\n", encoding="utf-8")
            self.assertEqual(environment.read_lock(path, "windows-cpu"), lock_fixture())
            for content in ('{"schema_version":1,"schema_version":1}', '{"bad":NaN}'):
                path.write_text(marker + content + "\n```\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    environment.read_lock(path, "windows-cpu")

    def test_module_import_does_not_load_inference_libraries(self):
        script = f'''
import importlib.util,sys
class Block:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {{'torch','transformers','numpy'}}:
            raise AssertionError('inference import attempted')
sys.meta_path.insert(0,Block())
spec=importlib.util.spec_from_file_location('probe',{str(ROOT / 'scripts/verify_reference_environment.py')!r})
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.validate_lock({lock_fixture()!r})
print('offline-ok')
'''
        run = subprocess.run([sys.executable, "-S", "-c", script], capture_output=True,
                             text=True, timeout=15, check=False)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout, "offline-ok\n")


if __name__ == "__main__":
    unittest.main()
