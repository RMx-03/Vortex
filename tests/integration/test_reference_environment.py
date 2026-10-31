"""Opt-in installed-profile imports and API inspection; no model execution."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(os.environ.get("VORTEX_REFERENCE_ENVIRONMENT") == "1",
                     "NOT VERIFIED: opt in inside an isolated locked environment")
class InstalledReferenceEnvironmentTests(unittest.TestCase):
    def test_installed_profile_imports_and_api_contract_without_inference(self):
        profile = os.environ.get("VORTEX_REFERENCE_PROFILE", "windows-cpu")
        run = subprocess.run([sys.executable, str(ROOT / "scripts/verify_reference_environment.py"),
                              "--profile", profile], capture_output=True, text=True,
                             timeout=150, check=False)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        report = json.loads(run.stdout)
        self.assertEqual(report["status"], "IMPORTS_VERIFIED")
        self.assertFalse(report["qualified"])
        self.assertEqual(report["forward_smoke"], "NOT_VERIFIED")
        self.assertTrue(report["observation"]["api_contract"])
        self.assertTrue(report["observation"]["pip_check"])
        self.assertEqual(run.stderr, "")


if __name__ == "__main__":
    unittest.main()
