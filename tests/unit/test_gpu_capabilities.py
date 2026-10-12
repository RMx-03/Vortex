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

    def test_keyboard_interrupt_is_not_swallowed(self) -> None:
        self.importer.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            gpu.capability()


class NativeOutputRegressionTests(unittest.TestCase):
    def test_parent_import_noise_cannot_contaminate_cli_json(self) -> None:
        # Removing process isolation lets these native writes bypass Python streams.
        code = '''import importlib.util, os, sys, types
spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
gpu = importlib.util.module_from_spec(spec); spec.loader.exec_module(gpu)
def noisy(name):
    os.write(1, b"SYNTHETIC_NATIVE_STDOUT\\n")
    os.write(2, b"SYNTHETIC_NATIVE_STDERR\\n")
    return types.SimpleNamespace(__version__="synthetic", version=types.SimpleNamespace(cuda=None),
                                cuda=types.SimpleNamespace(is_available=lambda: False))
gpu.importlib.util.find_spec = lambda name: object()
gpu.importlib.import_module = noisy
raise SystemExit(gpu.main())
'''
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", code,
                                 str(ROOT / "scripts/gpu_capabilities.py")],
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.stderr, "", "Native diagnostics escaped the CLI boundary")
        report = json.loads(result.stdout)
        self.assertIn(report["status"], {"torch_unavailable", "cuda_unavailable", "probe_error"})
        self.assertIn(result.returncode, (0, 2))

class IsolatedCLITests(unittest.TestCase):
    def invoke(self, body: str = "", timeout: float | None = None,
               os_streams: bool = False) -> tuple[int, dict]:
        # Exercise the real launcher, worker and result transport; only torch is synthetic.
        with tempfile.TemporaryDirectory(prefix="vortex synthetic worker with spaces ") as directory:
            driver = Path(directory) / "probe driver.py"
            driver.write_text(
                "import importlib.util,os,sys,time,types\n"
                f"spec=importlib.util.spec_from_file_location('real_probe',{str(ROOT/'scripts/gpu_capabilities.py')!r})\n"
                "real=importlib.util.module_from_spec(spec);spec.loader.exec_module(real)\n"
                "props=lambda i:types.SimpleNamespace(name='Synthetic '+str(i),total_memory=(i+1)*1024**3,major=8,minor=6)\n"
                "cuda=types.SimpleNamespace(is_available=lambda:True,device_count=lambda:2,get_device_properties=props)\n"
                "torch=types.SimpleNamespace(__version__='synthetic-worker',version=types.SimpleNamespace(cuda='12.1'),cuda=cuda)\n"
                "real.importlib.util.find_spec=lambda name:object()\n"
                "real.importlib.import_module=lambda name:torch\n"
                "real.shutil.which=lambda name:None\n"
                "_worker=real._worker\n" + body + "\n", encoding="utf-8")
            output, errors = io.StringIO(), io.StringIO()
            launch = subprocess.run
            paths = []
            def record(args, **kwargs):
                paths.append(Path(args[-2]))
                return launch(args, **kwargs)
            duration = gpu.PROBE_TIMEOUT_SECONDS if timeout is None else timeout
            before = dict(os.environ)
            if os_streams:
                # An outer OS process catches leaks that StringIO cannot intercept.
                code = ("import importlib.util,sys; "
                        "s=importlib.util.spec_from_file_location('probe',sys.argv[1]); "
                        "p=importlib.util.module_from_spec(s);s.loader.exec_module(p); "
                        "p.__file__=sys.argv[2];raise SystemExit(p.main())")
                result = launch([sys.executable, "-I", "-S", "-B", "-c", code,
                                 str(ROOT / "scripts/gpu_capabilities.py"), str(driver)],
                                capture_output=True, text=True, encoding="utf-8", timeout=20)
                self.assertEqual(result.stderr, "")
                self.assertNotIn("SYNTHETIC_PRIVATE", result.stdout)
                return result.returncode, json.loads(result.stdout)
            with patch.object(gpu, "__file__", str(driver)), \
                    patch.object(gpu.subprocess, "run", side_effect=record), \
                    patch.object(gpu, "PROBE_TIMEOUT_SECONDS", duration), \
                    redirect_stdout(output), redirect_stderr(errors):
                code = gpu.main()
            self.assertEqual(errors.getvalue(), "")
            self.assertNotIn("SYNTHETIC_PRIVATE", output.getvalue())
            self.assertEqual(dict(os.environ), before)
            self.assertTrue(paths and all(not p.parent.exists() for p in paths), "IPC directory leaked")
            report = json.loads(output.getvalue(), parse_constant=lambda _: self.fail("Nonfinite JSON"))
            return code, report

    def test_worker_native_stdout_is_discarded_with_valid_metadata(self) -> None:
        code, report = self.invoke("os.write(1,b'SYNTHETIC_PRIVATE_STDOUT\\n')")
        self.assertEqual(code, 0)
        self.assertEqual(report["torch_version"], "synthetic-worker")
        self.assertEqual(report["status"], "cuda_available")
        self.assertEqual(report["device_count"], 2)

    def test_worker_native_stderr_is_discarded_with_valid_metadata(self) -> None:
        code, report = self.invoke("os.write(2,b'SYNTHETIC_PRIVATE_STDERR\\n')")
        self.assertEqual(code, 0)
        self.assertEqual(report["devices"][1]["total_bytes"], 2*1024**3)

    def test_complete_os_streams_are_clean_under_worker_noise(self) -> None:
        code, report = self.invoke(
            "os.write(1,b'SYNTHETIC_PRIVATE_STDOUT');os.write(2,b'SYNTHETIC_PRIVATE_STDERR')",
            os_streams=True)
        self.assertEqual((code, report["status"], report["torch_version"]),
                         (0, "cuda_available", "synthetic-worker"))

    @unittest.skipUnless(os.name == "nt", "Win32 standard-handle regression")
    def test_windows_native_standard_handles_are_isolated(self) -> None:
        body = """import ctypes
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.GetStdHandle.argtypes=[ctypes.c_ulong];kernel.GetStdHandle.restype=ctypes.c_void_p
kernel.WriteFile.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong,ctypes.POINTER(ctypes.c_ulong),ctypes.c_void_p]
for stream in (-11,-12):
    message=ctypes.create_string_buffer(b'SYNTHETIC_PRIVATE_WIN32')
    written=ctypes.c_ulong()
    if not kernel.WriteFile(kernel.GetStdHandle(stream & 0xffffffff),message,len(message)-1,ctypes.byref(written),None):
        raise OSError('Synthetic Win32 write failed')
"""
        code, report = self.invoke(body, os_streams=True)
        self.assertEqual((code, report["torch_version"]), (0, "synthetic-worker"))

    def test_worker_diagnostic_flood_does_not_buffer_or_deadlock(self) -> None:
        code, report = self.invoke(
            "for i in range(128):\n    os.write(1,b'x'*65536);os.write(2,b'y'*65536)")
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "cuda_available")

    def test_missing_torch_remains_distinct_from_cpu_cuda(self) -> None:
        for body, wanted in [
                ("real.importlib.util.find_spec=lambda name:None", "torch_unavailable"),
                ("cuda.is_available=lambda:False;torch.version.cuda=None", "cuda_unavailable")]:
            with self.subTest(status=wanted):
                code, report = self.invoke(body)
                self.assertEqual((code, report["status"]), (0, wanted))
                self.assertEqual(report["device_count"], None if wanted=="torch_unavailable" else 0)

    def test_worker_errors_are_sanitized_and_partial_facts_retained(self) -> None:
        code, report = self.invoke(
            "def fail(name): raise OSError('SYNTHETIC_PRIVATE/path')\nreal.importlib.import_module=fail")
        self.assertEqual(code, 2)
        self.assertEqual(report["probe_error"], {"stage":"torch_import","type":"OSError"})
        self.assertTrue(report["torch_installed"])
        code, report = self.invoke(
            "def fail(i): raise RuntimeError('SYNTHETIC_PRIVATE/path')\ncuda.get_device_properties=fail")
        self.assertEqual(code, 2)
        self.assertTrue(report["cuda_available"])
        self.assertIsNone(report["device_count"])
        self.assertEqual(report["devices"], [])

    def test_failed_worker_cannot_claim_success_from_valid_file(self) -> None:
        code, report = self.invoke(
            "def _worker(path):\n    real._worker(path);return 9")
        self.assertEqual(code, 2)
        self.assertEqual(report["probe_error"], {"stage":"probe_process","type":"ProcessError"})
        self.assertIsNone(report["cuda_available"])

    def test_worker_crash_without_result_is_sanitized(self) -> None:
        code, report = self.invoke("os.write(2,b'SYNTHETIC_PRIVATE_CRASH');os._exit(3)")
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "probe_error")

    def test_malformed_oversized_and_nonfinite_results_fail_closed(self) -> None:
        payloads = [b"{bad", b'{"status":"cuda_available","status":"torch_unavailable"}',
                    b'{"status":NaN}', b"x"*(gpu.MAX_RESULT_BYTES+1)]
        for payload in payloads:
            with self.subTest(bytes=len(payload)):
                code, report = self.invoke(
                    f"def _worker(path):\n    open(path,'xb').write({payload!r});return 0")
                self.assertEqual(code, 2)
                self.assertEqual(report["status"], "probe_error")
                self.assertIsNone(report["cuda_available"])

    def test_zero_exit_missing_result_is_not_success(self) -> None:
        code, report = self.invoke("def _worker(path): return 0")
        self.assertEqual(code, 2)
        self.assertEqual(report["probe_error"]["stage"], "probe_result")

    def test_inconsistent_cuda_payload_is_not_trusted(self) -> None:
        code, report = self.invoke(
            "def _worker(path):\n"
            "    report=real.capability();report['device_count']=3\n"
            "    open(path,'xb').write(real.json.dumps(report).encode());return 0")
        self.assertEqual(code, 2)
        self.assertEqual(report["probe_error"]["type"], "InvalidResult")

    def test_metadata_size_limits_fail_without_truncation(self) -> None:
        for body in ("torch.__version__='x'*(real.MAX_TEXT_LENGTH+1)",
                     "cuda.device_count=lambda:real.MAX_DEVICES+1",
                     "cuda.get_device_properties=lambda i:types.SimpleNamespace(name='GPU',total_memory=2**1000,major=8,minor=0)"):
            with self.subTest(body=body):
                code, report = self.invoke(body)
                self.assertEqual(code, 2)
                self.assertEqual(report["status"], "probe_error")

    def test_serialized_result_limit_does_not_publish_truncated_inventory(self) -> None:
        code, report = self.invoke(
            "cuda.device_count=lambda:64\n"
            "cuda.get_device_properties=lambda i:types.SimpleNamespace(name='é'*512,total_memory=1024,major=8,minor=0)")
        self.assertEqual(code, 2)
        self.assertEqual(report["probe_error"], {"stage":"probe_result","type":"OutputLimit"})
        self.assertIsNone(report["cuda_available"])
        self.assertEqual(report["devices"], [])

    def test_filesystem_package_visibility_reaches_worker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vortex synthetic torch search path ") as directory:
            module = Path(directory) / "torch.py"
            module.write_text(
                "import types\n__version__='synthetic-path-visible'\n"
                "version=types.SimpleNamespace(cuda=None)\n"
                "cuda=types.SimpleNamespace(is_available=lambda:False)\n", encoding="utf-8")
            output, errors = io.StringIO(), io.StringIO()
            with patch.object(sys, "path", [directory, *sys.path]), \
                    redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(gpu.main(), 0)
            self.assertEqual(errors.getvalue(), "")
            report = json.loads(output.getvalue())
            self.assertEqual((report["status"], report["torch_version"]),
                             ("cuda_unavailable", "synthetic-path-visible"))

    def test_cleanup_failure_is_sanitized(self) -> None:
        temporary = tempfile.TemporaryDirectory
        class CleanupFailure:
            def __init__(self, **kwargs):
                self.directory = temporary(**kwargs)
            def __enter__(self):
                return self.directory.__enter__()
            def __exit__(self, *args):
                self.directory.__exit__(*args)
                raise OSError("SYNTHETIC_PRIVATE/cleanup")
        def worker(args, **kwargs):
            Path(args[-2]).write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(args, 0)
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(gpu.tempfile, "TemporaryDirectory", CleanupFailure), \
                patch.object(gpu.subprocess, "run", side_effect=worker), \
                redirect_stdout(output), redirect_stderr(errors):
            self.assertEqual(gpu.main(), 2)
        self.assertEqual(errors.getvalue(), "")
        self.assertNotIn("SYNTHETIC_PRIVATE", output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["probe_error"]["type"], "OSError")

    def test_timeout_kills_reaps_worker_and_removes_ipc(self) -> None:
        processes = []
        popen = subprocess.Popen
        def record(*args, **kwargs):
            process = popen(*args, **kwargs);processes.append(process);return process
        with patch.object(gpu.subprocess, "Popen", side_effect=record):
            code, report = self.invoke("time.sleep(20)", timeout=0.4)
        self.assertEqual(code, 2)
        self.assertEqual(report["probe_error"], {"stage":"probe_process","type":"TimeoutExpired"})
        self.assertTrue(processes and all(p.poll() is not None for p in processes))

    def test_launch_and_temp_directory_errors_emit_json_only(self) -> None:
        for target in ("subprocess.run", "tempfile.TemporaryDirectory"):
            output, errors = io.StringIO(), io.StringIO()
            with patch(target, side_effect=OSError("SYNTHETIC_PRIVATE/path")), \
                    redirect_stdout(output), redirect_stderr(errors):
                self.assertEqual(gpu.main(), 2)
            self.assertEqual(errors.getvalue(), "")
            self.assertNotIn("SYNTHETIC_PRIVATE", output.getvalue())
            self.assertEqual(json.loads(output.getvalue())["probe_error"]["type"], "OSError")

    def test_looping_import_path_resolution_is_sanitized(self) -> None:
        output, errors = io.StringIO(), io.StringIO()
        with patch.object(gpu.Path, "resolve", side_effect=RuntimeError("SYNTHETIC_PRIVATE/loop")), \
                redirect_stdout(output), redirect_stderr(errors):
            try:
                code = gpu.main()
            except RuntimeError:
                self.fail("Path-resolution error escaped instead of a sanitized JSON report")
        self.assertEqual(code, 2)
        self.assertEqual(errors.getvalue(), "")
        self.assertNotIn("SYNTHETIC_PRIVATE", output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["status"], "probe_error")


class VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="vortex verifier with spaces ")
        self.addCleanup(temporary.cleanup)
        # Windows temp paths may use 8.3 names; the verifier canonicalizes its root.
        self.root = Path(temporary.name).resolve()
        (self.root / "tests/unit").mkdir(parents=True)
        (self.root / "tests/unit/test_sample.py").write_text('"""Synthetic fixture."""\n')
        (self.root / "scripts").mkdir()
        (self.root / "scripts/publication.py").write_text('print("synthetic publication")\n')
        self.launch = self.enterContext(patch.object(verify.subprocess, "run"))
        self.launch.return_value = subprocess.CompletedProcess([], 0, "", "Ran 1 test in 0.001s\n\nOK\n")
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def invoke(self, root: Path | None = None) -> tuple[int, str]:
        output, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = verify.main(self.root if root is None else root)
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

    def test_empty_suite_exit_five_is_operational_failure(self) -> None:
        (self.root / ".git").mkdir()
        self.launch.return_value = subprocess.CompletedProcess(
            [], 5, "", "Ran 0 tests in 0.000s\n\nNO TESTS RAN\n")
        code, output = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("ERROR", output)
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

    def test_root_with_parent_segments_keeps_scan_in_checkout(self) -> None:
        (self.root / ".git").mkdir()
        code, output = self.invoke(self.root / "scripts" / "..")
        self.assertEqual(code, 0, output)
        self.assertEqual(len(self.launch.call_args_list), 2)
        for call in self.launch.call_args_list:
            self.assertEqual(call.kwargs["cwd"], self.root)
        scan = self.launch.call_args_list[1].args[0]
        self.assertEqual(Path(scan[-2]), self.root / "scripts/publication.py")

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
