"""Offline probe/verifier tests: no real torch, CUDA, downloads or host changes."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gpu = load_script("vortex_test_gpu_probe", ROOT / "scripts/gpu_capabilities.py")
# Importing a tool must never launch checks or change the checkout.
with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as import_launch:
    try:
        verify = load_script("vortex_test_verifier", ROOT / "scripts/verify.py")
    except SystemExit:
        verify = sys.modules["vortex_test_verifier"]
    IMPORT_LAUNCH_COUNT = import_launch.call_count


def fake_torch(available: bool = False) -> SimpleNamespace:
    cuda = Mock()
    cuda.is_available.return_value = available
    cuda.device_count.return_value = 2
    cuda.get_device_properties.side_effect = [
        SimpleNamespace(name="Synthetic A", total_memory=8 * 1024**3, major=8, minor=6),
        SimpleNamespace(name="Synthetic B", total_memory=16 * 1024**3, major=9, minor=0),
    ]
    return SimpleNamespace(__version__="2.5.1", version=SimpleNamespace(cuda="12.1"), cuda=cuda)


class GPUProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.finder = self.enterContext(patch.object(gpu.importlib.util, "find_spec", return_value=object()))
        self.importer = self.enterContext(patch.object(gpu.importlib, "import_module", return_value=fake_torch()))
        self.which = self.enterContext(patch.object(gpu.shutil, "which", return_value=None))
        self.enterContext(patch.dict(sys.modules, {"torch": self.importer.return_value}))

    def test_missing_torch_leaves_cuda_unknown(self) -> None:
        self.finder.return_value = None
        before = dict(os.environ)
        report = gpu.capability()
        self.assertEqual(report["status"], "torch_unavailable")
        self.assertFalse(report["torch_installed"])
        self.assertFalse(report["torch_available"])
        self.assertIsNone(report["cuda_available"])
        self.assertIsNone(report["device_count"])
        self.assertEqual(report["devices"], [])
        self.importer.assert_not_called()
        self.assertEqual(dict(os.environ), before)

    def test_nvidia_smi_is_presence_only_never_executed(self) -> None:
        self.finder.return_value = None
        self.which.return_value = "C:/Synthetic Tools/nvidia-smi.exe"
        with patch.object(subprocess, "run") as launch:
            report = gpu.capability()
        self.assertTrue(report["nvidia_smi_available"])
        self.assertIsNone(report["cuda_available"])
        launch.assert_not_called()

    def test_cpu_torch_and_cuda_build_without_visible_devices(self) -> None:
        for version in (None, "12.1"):
            with self.subTest(version=version):
                torch = fake_torch()
                torch.version.cuda = version
                self.importer.return_value = torch
                report = gpu.capability()
                self.assertEqual(report["status"], "cuda_unavailable")
                self.assertTrue(report["torch_available"])
                self.assertFalse(report["cuda_available"])
                self.assertEqual(report["device_count"], 0)
                self.assertEqual(report["torch_cuda_version"], version)
                torch.cuda.device_count.assert_not_called()

    def test_mocked_cuda_reports_devices_bytes_and_compute_capability(self) -> None:
        self.importer.return_value = fake_torch(True)
        report = gpu.capability()
        self.assertEqual(report["status"], "cuda_available")
        self.assertEqual(report["device_count"], 2)
        self.assertEqual(report["devices"], [
            {"index": 0, "name": "Synthetic A", "total_bytes": 8 * 1024**3, "compute_capability": [8, 6]},
            {"index": 1, "name": "Synthetic B", "total_bytes": 16 * 1024**3, "compute_capability": [9, 0]},
        ])
        json.dumps(report, allow_nan=False)

    def test_lookup_import_cuda_and_path_exceptions_are_sanitized(self) -> None:
        cases = [(self.finder, "torch_lookup", ValueError),
                 (self.importer, "torch_import", OSError),
                 (self.importer.return_value.cuda.is_available, "cuda_availability", RuntimeError),
                 (self.which, "nvidia_smi_lookup", PermissionError)]
        for target, stage, kind in cases:
            with self.subTest(stage=stage):
                target.side_effect = kind("synthetic-detail-do-not-print")
                try:
                    report = gpu.capability()
                    self.assertEqual(report["status"], "probe_error")
                    self.assertEqual(report["probe_error"], {"stage": stage, "type": kind.__name__})
                    self.assertIsNone(report["cuda_available"])
                    self.assertNotIn("synthetic-detail-do-not-print", json.dumps(report))
                    if stage == "torch_import":
                        self.assertTrue(report["torch_installed"])
                        self.assertFalse(report["torch_available"])
                finally:
                    target.side_effect = None

    def test_invalid_count_and_properties_are_rejected(self) -> None:
        for count in (0, -1, True, "2"):
            with self.subTest(count=count):
                torch = fake_torch(True)
                torch.cuda.device_count.return_value = count
                self.importer.return_value = torch
                report = gpu.capability()
                self.assertEqual(report["status"], "probe_error")
                self.assertEqual(report["probe_error"]["stage"], "device_count")
        for memory in (-1, True, "1024"):
            with self.subTest(memory=memory):
                torch = fake_torch(True)
                torch.cuda.device_count.return_value = 1
                torch.cuda.get_device_properties.side_effect = None
                torch.cuda.get_device_properties.return_value = SimpleNamespace(name="GPU", total_memory=memory, major=8, minor=0)
                self.importer.return_value = torch
                self.assertEqual(gpu.capability()["status"], "probe_error")

    def test_partial_inventory_is_not_claimed_complete(self) -> None:
        torch = fake_torch(True)
        torch.cuda.get_device_properties.side_effect = [
            SimpleNamespace(name="GPU", total_memory=1024, major=8, minor=0), RuntimeError("synthetic-detail-do-not-print")]
        self.importer.return_value = torch
        report = gpu.capability()
        self.assertEqual(report["probe_error"]["stage"], "device_properties")
        self.assertTrue(report["cuda_available"])
        self.assertIsNone(report["device_count"])
        self.assertEqual(report["devices"], [])

    def test_cli_missing_torch_success_and_probe_error_exit(self) -> None:
        for missing, expected in ((True, 0), (False, 2)):
            with self.subTest(missing=missing):
                self.finder.return_value = None if missing else object()
                self.importer.side_effect = ImportError("synthetic-detail-do-not-print")
                output = io.StringIO()
                with redirect_stdout(output):
                    code = gpu.main()
                self.assertEqual(code, expected)
                self.assertNotIn("synthetic-detail-do-not-print", output.getvalue())
                self.assertIn(json.loads(output.getvalue())["status"], ("torch_unavailable", "probe_error"))

    def test_cli_dependency_noise_is_suppressed_and_streams_restored(self) -> None:
        def noisy_import(name: str) -> SimpleNamespace:
            print("synthetic dependency chatter")
            print("synthetic dependency warning", file=sys.stderr)
            return fake_torch()

        self.importer.side_effect = noisy_import
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(gpu.main(), 0)
            self.assertIs(sys.stdout, output)
            self.assertIs(sys.stderr, errors)
        self.assertEqual(json.loads(output.getvalue())["status"], "cuda_unavailable")
        self.assertEqual(errors.getvalue(), "")

    def test_keyboard_interrupt_is_not_swallowed(self) -> None:
        self.importer.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            gpu.capability()


class VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="vortex verifier with spaces ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "tests/unit").mkdir(parents=True)
        (self.root / "tests/unit/test_sample.py").write_text('"""Synthetic fixture."""\n')
        (self.root / "scripts").mkdir()
        (self.root / "scripts/publication.py").write_text('print("synthetic publication")\n')
        self.launch = self.enterContext(patch.object(verify.subprocess, "run"))
        self.launch.return_value = subprocess.CompletedProcess([], 0, "", "Ran 1 test in 0.001s\n\nOK\n")
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def invoke(self) -> tuple[int, str]:
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = verify.main(self.root)
        return code, output.getvalue() + errors.getvalue()

    def test_import_is_safe(self) -> None:
        self.assertEqual(IMPORT_LAUNCH_COUNT, 0)
        runpy.run_path(str(ROOT / "scripts/verify.py"), run_name="import_check")
        self.launch.assert_not_called()

    def test_unit_failure_prevents_scan_and_normalizes_exit(self) -> None:
        (self.root / ".git").mkdir()
        for code in (1, -9):
            with self.subTest(code=code):
                self.launch.reset_mock()
                self.launch.return_value.returncode = code
                self.assertEqual(self.invoke()[0], 1)
                self.assertEqual(self.launch.call_count, 1)

    def test_success_uses_current_python_safe_arguments_and_index_scan(self) -> None:
        (self.root / ".git").mkdir()
        before = dict(os.environ)
        code, output = self.invoke()
        self.assertEqual(code, 0)
        calls = self.launch.call_args_list
        self.assertEqual(len(calls), 2)
        unit, scan = [call.args[0] for call in calls]
        self.assertEqual(unit[0], sys.executable)
        self.assertIn("-B", unit)
        self.assertIn("unittest", unit)
        self.assertIn("discover", unit)
        self.assertIn("test_*.py", unit)
        self.assertEqual(scan[0], sys.executable)
        self.assertEqual(scan[-1], "staged")
        self.assertEqual(Path(scan[-2]), self.root / "scripts/publication.py")
        for call in calls:
            self.assertEqual(call.kwargs["cwd"], self.root)
            self.assertTrue(call.kwargs["capture_output"])
            self.assertEqual(call.kwargs["encoding"], "utf-8")
            self.assertGreater(call.kwargs["timeout"], 0)
            self.assertFalse(call.kwargs.get("shell", False))
        self.assertEqual(dict(os.environ), before)
        self.assertIn("PASS", output)

    def test_git_file_and_override_cannot_skip_scan(self) -> None:
        (self.root / ".git").write_text("gitdir: synthetic location\n")
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual(self.launch.call_count, 2)
        (self.root / ".git").unlink()
        self.launch.reset_mock()
        with patch.dict(os.environ, {"GIT_DIR": "synthetic-location"}):
            self.assertEqual(self.invoke()[0], 0)
        self.assertEqual(self.launch.call_count, 2)

    def test_archive_skip_is_explicit(self) -> None:
        code, output = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual(self.launch.call_count, 1)
        self.assertIn("SKIP", output)

    def test_scan_failure_and_missing_script_fail_closed(self) -> None:
        (self.root / ".git").mkdir()
        success = self.launch.return_value
        for child_code, expected in ((7, 1), (2, 2)):
            with self.subTest(child_code=child_code):
                self.launch.side_effect = [success, subprocess.CompletedProcess([], child_code, "", "synthetic failure\n")]
                self.assertEqual(self.invoke()[0], expected)
        self.launch.side_effect = None
        self.launch.reset_mock()
        (self.root / "scripts/publication.py").unlink()
        self.assertEqual(self.invoke()[0], 2)
        self.assertEqual(self.launch.call_count, 1)

    def test_missing_directory_zero_tests_and_missing_summary_are_not_passes(self) -> None:
        for summary in ("Ran 0 tests in 0.000s\n\nOK\n", ""):
            with self.subTest(summary=summary):
                self.launch.return_value.stderr = summary
                self.assertEqual(self.invoke()[0], 2)
        self.launch.reset_mock()
        shutil.rmtree(self.root / "tests")
        self.assertEqual(self.invoke()[0], 2)
        self.launch.assert_not_called()

    def test_launch_and_timeout_errors_are_actionable_without_raw_details(self) -> None:
        for failure in (FileNotFoundError("synthetic-detail-do-not-print"), subprocess.TimeoutExpired(["synthetic"], 1)):
            with self.subTest(failure=type(failure).__name__):
                self.launch.side_effect = failure
                code, output = self.invoke()
                self.assertEqual(code, 2)
                self.assertNotIn("synthetic-detail-do-not-print", output)
                self.assertIn("ERROR", output)


class ToolCLITests(unittest.TestCase):
    def test_probe_without_site_packages_is_json_success(self) -> None:
        run = subprocess.run([sys.executable, "-I", "-S", "-B", str(ROOT / "scripts/gpu_capabilities.py")],
                             capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report["status"], "torch_unavailable")
        self.assertFalse(report["torch_available"])
        self.assertIsNone(report["cuda_available"])

    def test_disposable_archive_success_and_empty_suite(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vortex CLI with spaces ") as folder:
            root = Path(folder)
            (root / "scripts").mkdir()
            shutil.copy2(ROOT / "scripts/verify.py", root / "scripts/verify.py")
            (root / "tests/unit").mkdir(parents=True)
            test = root / "tests/unit/test_sample.py"
            test.write_text("import unittest\nclass Fixture(unittest.TestCase):\n    def test_ok(self): self.assertTrue(True)\n")
            command = [sys.executable, "-I", "-B", str(root / "scripts/verify.py")]
            # The synthetic archive must not inherit the caller's checkout or hook index.
            environment = {key: value for key, value in os.environ.items()
                           if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"}}
            run = subprocess.run(command, cwd=root.parent, capture_output=True, text=True,
                                 timeout=30, env=environment)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("SKIP", run.stdout)
            test.write_text('"""Empty synthetic suite."""\n')
            run = subprocess.run(command, cwd=root.parent, capture_output=True, text=True,
                                 timeout=30, env=environment)
            self.assertEqual(run.returncode, 2, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
