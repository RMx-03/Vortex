#!/usr/bin/env python3
"""Compare completed schema-1 token evidence offline, without qualifying a golden.

MATCH means exact supplied token equality under the same validated numerical key.
It does not attest asset bytes, actual execution, environment qualification or
pretrained correctness. Independently approved golden evidence is a separate gate.
No record or fixture is written, repaired, accepted as a golden or regenerated.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vortex.executor.reference_manifest import _canonical_bytes, validate_manifest

_MAX_FILE_BYTES = 1_048_576
_VOCABULARY_SIZE = 151936  # Exact pinned Qwen3 configuration; no model import.


@dataclass(frozen=True)
class ComparisonResult:
    """Finite comparison outcome, never raw inputs, prompts, token values or paths.

    first_mismatch is a zero-based generated-token position (EOS included), or
    None for MATCH/INCOMPATIBLE. For unequal lengths, it is the first unequal
    position or the first missing position when one sequence is a strict prefix.
    reason is a stable machine-readable code, not a native exception diagnostic.
    """

    status: str
    first_mismatch: int | None
    reason: str


def _snapshot(record: Mapping[str, object]) -> dict[str, object] | None:
    try:
        # Validate the same owned snapshot we later consume. Reusing the schema's
        # bounded canonicalizer avoids a second traversal of caller-owned data.
        owned = json.loads(_canonical_bytes(record))
        validate_manifest(owned)
        if owned["complete"]:
            for ids in (owned["numerical"]["prompt"]["input_ids"], owned["outputs"]["generated_ids"]):
                if any(not 0 <= token < _VOCABULARY_SIZE for token in ids):
                    return None
        return owned
    except Exception:
        # Mapping implementations have an open exception hierarchy. Reject an
        # ordinary malformed-input failure without exposing data or diagnostics.
        # KeyboardInterrupt/SystemExit remain outside this boundary.
        return None


def compare_reference(expected: Mapping[str, object], actual: Mapping[str, object]) -> ComparisonResult:
    """Return MATCH, MISMATCH or INCOMPATIBLE from bounded schema-1 snapshots.

    Require valid complete records with numerical evidence and equal recomputed
    comparison keys before examining outputs. Invalid/incomplete reference wins
    deterministically without inspecting the candidate. Provenance remains
    required and validated but is excluded from numerical comparison identity.
    Compare generated IDs and counts exactly, including terminal EOS; ignore
    decoded text. Schema validation rejects missing/extra tokens after EOS,
    inconsistent stops, empty successful output and unsupported settings.

    Work is bounded by schema-1's 1 MiB canonical budget/32 nesting levels per
    record, 512 prompt tokens and 64 output tokens. No I/O or inference imports
    occur here; caller data is never mutated. Supplied Mapping methods execute
    in the caller's process: this is not a sandbox for hostile Python code or
    a guarantee against concurrent caller mutation. Interruption propagates.
    """
    reference = _snapshot(expected)
    if reference is None:
        return ComparisonResult("INCOMPATIBLE", None, "invalid_reference")
    if not reference["complete"]:
        return ComparisonResult("INCOMPATIBLE", None, "incomplete_reference")
    candidate = _snapshot(actual)
    if candidate is None:
        return ComparisonResult("INCOMPATIBLE", None, "invalid_candidate")
    if not candidate["complete"]:
        return ComparisonResult("INCOMPATIBLE", None, "incomplete_candidate")
    if reference["comparison_key_sha256"] != candidate["comparison_key_sha256"]:
        return ComparisonResult("INCOMPATIBLE", None, "numerical_identity_mismatch")
    left, right = reference["outputs"]["generated_ids"], candidate["outputs"]["generated_ids"]
    first = next((index for index, (a, b) in enumerate(zip(left, right)) if a != b), None)
    if len(left) != len(right):
        position = min(len(left), len(right)) if first is None else first
        return ComparisonResult("MISMATCH", position, "output_length_mismatch")
    if first is not None:
        return ComparisonResult("MISMATCH", first, "token_mismatch")
    return ComparisonResult("MATCH", None, "identical_tokens")


def _read_record(filename: str) -> object:
    try:
        # Explicit local regular files only; never treat a URL/UNC input as a
        # download, and reject a symlink/FIFO/device before opening it.
        # Windows accepts either slash in a UNC prefix, including mixed pairs.
        if len(filename) > 4096 or filename.replace("\\", "/").startswith("//") or "://" in filename:
            return None
        path = Path(filename)
        if path.is_symlink() or not path.is_file():
            return None
        with path.open("rb") as stream:
            data = stream.read(_MAX_FILE_BYTES + 1)
        if len(data) > _MAX_FILE_BYTES:
            return None
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate JSON field")
                result[key] = value
            return result
        def constant(_):
            raise ValueError("nonfinite JSON number")
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    """Read two JSON files and print one compact deterministic report.

    Exit 0 MATCH, 1 MISMATCH, 2 INCOMPATIBLE, 130 interrupted. Missing/malformed
    files produce finite reason codes without file contents, paths or tracebacks.
    Inputs are read-only; no output-file or golden-update option exists. File
    reads are capped at 1 MiB plus one sentinel byte. Actual filesystem calls
    have no hard wall-time guarantee; use operator-controlled local files.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True, help="Expected schema-1 JSON file")
    parser.add_argument("--actual", required=True, help="Candidate schema-1 JSON file")
    args = parser.parse_args(argv)
    try:
        expected = _read_record(args.expected)
        # Preserve reference-first failure isolation even at the file boundary.
        reference = _snapshot(expected)
        if reference is None:
            result = ComparisonResult("INCOMPATIBLE", None, "invalid_reference")
        elif not reference["complete"]:
            result = ComparisonResult("INCOMPATIBLE", None, "incomplete_reference")
        else:
            result = compare_reference(reference, _read_record(args.actual))
        code = {"MATCH": 0, "MISMATCH": 1, "INCOMPATIBLE": 2}[result.status]
    except KeyboardInterrupt:
        result, code = ComparisonResult("INCOMPATIBLE", None, "interrupted"), 130
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
