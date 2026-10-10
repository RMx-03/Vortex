#!/usr/bin/env python3
"""Offline CPU checks: exit 0 pass, 1 failed checks, 2 operational failure.

The publication step reads the effective Git index, not untracked proposals.
Source archives without local Git metadata report that step as explicitly skipped.
"""

import os
from pathlib import Path
import re
import subprocess
import sys

TIMEOUT_SECONDS = 300


def run_check(command: list[str], root: Path, label: str) -> subprocess.CompletedProcess[str] | None:
    """Run a bounded noninteractive check without a shell or environment mutation."""
    try:
        result = subprocess.run(command, cwd=root, check=False, capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                stdin=subprocess.DEVNULL, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"ERROR: {label} timed out after {TIMEOUT_SECONDS}s.", file=sys.stderr)
        return None
    except OSError as error:
        print(f"ERROR: cannot launch {label} ({type(error).__name__}); check Python and permissions.",
              file=sys.stderr)
        return None
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result


def main(root: Path | None = None) -> int:
    """Verify tests and, for a checkout, its index; never stage or modify Git."""
    root = Path(__file__).resolve().parents[1] if root is None else root.resolve()
    if not (root / "tests/unit").is_dir():
        print("ERROR: tests/unit is missing; run from a complete project checkout.", file=sys.stderr)
        return 2
    python = [sys.executable, "-B", "-X", "utf8"]
    result = run_check([*python, "-m", "unittest", "discover", "-s", "tests/unit",
                        "-p", "test_*.py", "-v"], root, "CPU unit tests")
    if result is None:
        return 2
    if result.returncode:
        print(f"FAIL: CPU unit tests (child exit {result.returncode}).", file=sys.stderr)
        return 1
    # unittest exits successfully for an empty suite; that is not verification evidence.
    counts = re.findall(r"^Ran ([0-9]+) tests? in ", result.stderr, re.MULTILINE)
    if not counts or int(counts[-1]) == 0:
        print("ERROR: no tests confirmed by unittest discovery; inspect tests/unit and its output.",
              file=sys.stderr)
        return 2
    marker = root / ".git"
    overrides = ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE")
    if marker.exists() or marker.is_symlink() or any(name in os.environ for name in overrides):
        publication = root / "scripts/publication.py"
        if not publication.is_file():
            print("ERROR: Git checkout requires scripts/publication.py; restore the scanner.", file=sys.stderr)
            return 2
        result = run_check([*python, str(publication), "staged"], root, "staged publication check")
        if result is None:
            return 2
        if result.returncode:
            print(f"FAIL: staged publication check (child exit {result.returncode}).", file=sys.stderr)
            return 2 if result.returncode == 2 else 1
        print("PASS: CPU unit tests and staged publication check.")
    else:
        print("SKIP: staged publication check; no local Git checkout metadata.")
        print("PASS: CPU unit tests; publication was not verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
