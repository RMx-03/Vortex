#!/usr/bin/env python3
"""Read-only publication gate: no network, writes, fetches or pushes.

Exit codes: 0 reviewed content, 1 policy violation, 2 incomplete inspection.
The reviewed policy and this executable are trusted inputs, not a sandbox.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / ".publication" / "policy.json"
ALLOWED_TYPES = {"100644", "100755"}
COMMIT = re.compile(r"^(feat|fix|perf|refactor|test|docs|chore|build|ci)\([a-z][a-z0-9-]*\): [a-z0-9][^\n]{0,70}$")
BRANCH = re.compile(r"^(main|master|develop|(?:feat|fix|perf|refactor|test|docs|chore)/[a-z0-9]+(?:[-/][a-z0-9]+)*)$")
TRAILERS = re.compile(r"(?i)^(?:co-authored-by|generated-by|assisted-by)\s*:")
BANNED_METADATA = re.compile(
    r"(?i)^(?:generated|created|implemented)\s+(?:by|with)\s+(?:an?\s+)?"
    r"(?:ai\b|coding assistant\b|codex\b|claude(?: code)?\b|chatgpt\b|"
    r"(?:github )?copilot\b|gemini\b|(?:google )?antigravity\b)")
PRIVATE = re.compile(r"(?i)^(?:classification\s*:\s*private\b|private\s*[-\u2013\u2014:]\s*do not publish\b)")
SECRETS = [
    re.compile(rb"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9_]{24,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{24,}"),
    re.compile(rb"sk-(?:(?:proj|svcacct)-|ant-(?:api\d+-)?)?[A-Za-z0-9_-]{28,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
]
RESERVED = re.compile(r"(?i)^(?:con|prn|aux|nul|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])(?:\.|$)")
ENV_DIRECTORY = re.compile(r"(?i)^venv(?:[0-9._-].*)?$")
OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


class GuardError(Exception):
    """An inspection cannot establish publication safety."""


def git(*args, input_bytes=None):
    # Inspect original objects even when local replacement objects exist.
    env = os.environ.copy()
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0",
               GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    try:
        run = subprocess.run(["git", "--no-replace-objects", "-C", str(ROOT), *args],
                             input=input_bytes, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, check=False, env=env)
    except OSError as exc:
        raise GuardError("Git executable unavailable; install Git before inspection") from exc
    if run.returncode:
        # Git stderr and supplied ref/path names can contain credentials.
        raise GuardError("Git inspection failed; verify objects, references and local configuration")
    return run.stdout


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def canonical_path(path):
    if not isinstance(path, str) or not path or "\\" in path:
        return False
    if unicodedata.normalize("NFC", path) != path:
        return False
    for part in path.split("/"):
        if not part or part in {".", ".."} or part.endswith((".", " ")):
            return False
        if RESERVED.match(part) or any(c in '<>:"|?*' for c in part):
            return False
        if any(unicodedata.category(c).startswith("C") for c in part):
            return False
    return True


def policy():
    try:
        pol = strict_json(POLICY.read_text(encoding="utf-8"))
        fields = {"schema_version", "maximum_text_bytes", "max_commits_per_push_scan",
                  "allowed_exacts", "allowed_prefix_suffixes", "prohibited_path_segments",
                  "prohibited_basenames"}
        if not isinstance(pol, dict) or set(pol) != fields:
            raise ValueError("policy fields")
        if type(pol["schema_version"]) is not int or pol["schema_version"] != 1:
            raise ValueError("policy version")
        for key in ("maximum_text_bytes", "max_commits_per_push_scan"):
            if type(pol[key]) is not int or pol[key] <= 0:
                raise ValueError("policy limit")
        for key in ("allowed_exacts", "prohibited_path_segments", "prohibited_basenames"):
            values = pol[key]
            if not isinstance(values, list) or not values:
                raise ValueError("policy list")
            if any(not canonical_path(x) or (key != "allowed_exacts" and "/" in x) for x in values):
                raise ValueError("policy path")
            if len({x.casefold() for x in values}) != len(values):
                raise ValueError("policy duplicate")
        prefixes = pol["allowed_prefix_suffixes"]
        if not isinstance(prefixes, dict) or not prefixes:
            raise ValueError("policy prefixes")
        for prefix, suffixes in prefixes.items():
            if not prefix.endswith("/") or not canonical_path(prefix[:-1]):
                raise ValueError("policy prefix")
            if not isinstance(suffixes, list) or not suffixes or any(
                    not isinstance(s, str) or not re.fullmatch(r"\.[a-z0-9]+", s) for s in suffixes):
                raise ValueError("policy suffixes")
        if any(not safe_path(path, pol) for path in pol["allowed_exacts"]):
            raise ValueError("conflicting policy")
        return pol
    except (OSError, UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise GuardError("Publication policy unavailable or invalid; restore the reviewed schema/version and limits") from exc


def secret_present(data):
    return any(pattern.search(data) for pattern in SECRETS)


def label(value):
    if secret_present(value.encode("utf-8", "replace")):
        return "[credential-shaped name redacted]"
    return json.dumps(value, ensure_ascii=True)


def safe_path(path, pol):
    if not canonical_path(path) or secret_present(path.encode("utf-8")):
        return False
    parts = path.split("/")
    denied = {x.casefold() for x in pol["prohibited_path_segments"]}
    if any(part.casefold() in denied for part in parts):
        return False
    if parts[-1].casefold() in {x.casefold() for x in pol["prohibited_basenames"]}:
        return False
    if any(ENV_DIRECTORY.fullmatch(part) for part in parts[:-1]):
        return False
    if path in pol["allowed_exacts"]:
        return True
    # Exact entries permit .github/.publication/.gitignore. Prefix grants
    # never authorize hidden local files, including .venv* variants.
    if any(part.startswith(".") for part in parts):
        return False
    return any(path.startswith(prefix) and path.endswith(tuple(suffixes))
               for prefix, suffixes in pol["allowed_prefix_suffixes"].items())


def content_issue(text):
    if secret_present(text.encode("utf-8")):
        return "credential-shaped material detected (value redacted); remove it and rotate real credentials"
    for line in text.splitlines():
        # Only explicit line labels are prohibited; ordinary prose about tools
        # or private APIs remains publishable.
        normalized = re.sub(r"[*_`]", "", line.lstrip(" \t#>+-")).strip()
        if PRIVATE.match(normalized):
            return "private workspace classification header; remove private material"
        if TRAILERS.match(normalized) or BANNED_METADATA.match(normalized):
            return "unwanted attribution trailer or metadata; remove the prohibited label"
    return None


def decoded_content_issue(value):
    """Inspect every decoded JSON key/value, including escaped metadata."""
    if isinstance(value, str):
        return content_issue(value)
    if isinstance(value, dict):
        values = list(value) + list(value.values())
    elif isinstance(value, list):
        values = value
    else:
        return None
    for item in values:
        if issue := decoded_content_issue(item):
            return issue
    return None


def metadata_valid(metadata, cell_type=None):
    """Validate defined metadata types; custom JSON metadata stays permitted."""
    if not isinstance(metadata, dict):
        return False
    if cell_type is None:
        for key, required in (("kernelspec", ("name", "display_name")), ("language_info", ("name",))):
            if key in metadata:
                info = metadata[key]
                if not isinstance(info, dict) or any(not isinstance(info.get(k), str) for k in required):
                    return False
                if key == "language_info":
                    for field in ("file_extension", "mimetype", "pygments_lexer"):
                        if field in info and not isinstance(info[field], str):
                            return False
                    if "codemirror_mode" in info and not isinstance(info["codemirror_mode"], (str, dict)):
                        return False
        if "title" in metadata and not isinstance(metadata["title"], str):
            return False
        if "orig_nbformat" in metadata and (type(metadata["orig_nbformat"]) is not int or metadata["orig_nbformat"] < 1):
            return False
        if "authors" in metadata and (not isinstance(metadata["authors"], list) or any(
                not isinstance(a, dict) or not isinstance(a.get("name"), str) for a in metadata["authors"])):
            return False
    else:
        if "name" in metadata and (not isinstance(metadata["name"], str) or not metadata["name"]):
            return False
        if "tags" in metadata:
            tags = metadata["tags"]
            if not isinstance(tags, list) or any(not isinstance(t, str) or not t or "," in t for t in tags):
                return False
            if len(set(tags)) != len(tags):
                return False
        if "jupyter" in metadata:
            jupyter = metadata["jupyter"]
            if not isinstance(jupyter, dict) or any(
                    key in jupyter and type(jupyter[key]) is not bool for key in ("source_hidden", "outputs_hidden")):
                return False
        if cell_type == "raw" and "format" in metadata and not isinstance(metadata["format"], str):
            return False
        if cell_type == "code":
            if "collapsed" in metadata and type(metadata["collapsed"]) is not bool:
                return False
            if "scrolled" in metadata and type(metadata["scrolled"]) is not bool and metadata["scrolled"] != "auto":
                return False
            if "execution" in metadata and (not isinstance(metadata["execution"], dict) or any(
                    not isinstance(v, str) for v in metadata["execution"].values())):
                return False
    return True


def notebook_issue(text):
    try:
        obj = strict_json(text)
        if not isinstance(obj, dict) or type(obj.get("nbformat")) is not int or obj["nbformat"] != 4:
            return "unsupported notebook format; use nbformat 4"
        if type(obj.get("nbformat_minor")) is not int or obj["nbformat_minor"] < 0:
            return "invalid notebook version metadata"
        if set(obj) != {"nbformat", "nbformat_minor", "metadata", "cells"}:
            return "invalid notebook top-level fields; remove unrecognized payloads"
        if not metadata_valid(obj.get("metadata")) or not isinstance(obj.get("cells"), list):
            return "invalid notebook metadata/cells"
        if issue := decoded_content_issue(obj):
            return issue
        if obj["metadata"].get("widgets"):
            return "notebook saved widget outputs must be cleared"
        ids = set()
        for cell in obj["cells"]:
            if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown", "raw"}:
                return "invalid notebook cell schema"
            fields = {"cell_type", "metadata", "source", "id"}
            if cell["cell_type"] == "code":
                fields.update({"outputs", "execution_count"})
            else:
                fields.add("attachments")
            if set(cell) - fields:
                return "invalid notebook cell fields; remove unrecognized payloads"
            if obj["nbformat_minor"] >= 5 or "id" in cell:
                cell_id = cell.get("id")
                if not isinstance(cell_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id) or cell_id in ids:
                    return "invalid/missing/duplicate notebook cell id; nbformat 4.5 requires unique cell IDs"
                ids.add(cell_id)
            source = cell.get("source")
            if not isinstance(source, str) and not (isinstance(source, list) and all(isinstance(s, str) for s in source)):
                return "invalid notebook cell source"
            if not metadata_valid(cell.get("metadata"), cell["cell_type"]):
                return "invalid notebook cell metadata"
            if cell["metadata"].get("widgets"):
                return "notebook saved widget outputs must be cleared"
            if "attachments" in cell and (not isinstance(cell["attachments"], dict) or cell["attachments"]):
                return "notebook attachments must be cleared"
            if cell["cell_type"] == "code":
                if not isinstance(cell.get("outputs"), list) or cell["outputs"]:
                    return "notebook execution outputs must be cleared"
                if "execution_count" not in cell or cell["execution_count"] is not None:
                    return "notebook execution counts must be cleared"
            elif "outputs" in cell or "execution_count" in cell:
                return "invalid notebook non-code execution fields"
            issue = content_issue(source if isinstance(source, str) else "".join(source))
            if issue:
                return issue
        return None
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return "invalid notebook JSON; repair duplicate keys, malformed or nonfinite values"


def safe_blob(path, mode, data, pol):
    if mode not in ALLOWED_TYPES:
        return "symlink, submodule, or unsupported Git file mode; publish regular files only"
    if not safe_path(path, pol):
        return "path is not in reviewed publication allowlist; move public material to an approved path"
    if len(data) > pol["maximum_text_bytes"]:
        return "file exceeds reviewed size limit"
    if b"\x00" in data:
        return "binary/non-text content is not approved"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "content is not UTF-8"
    issue = content_issue(text)
    if issue:
        return issue
    if path.endswith(".ipynb"):
        return notebook_issue(text)
    if path.endswith(".json"):
        try:
            return decoded_content_issue(strict_json(text))
        except (ValueError, TypeError, UnicodeError, RecursionError):
            return "invalid JSON; repair malformed, duplicate or nonfinite values"
    return None


def check_text(kind, value):
    issue = content_issue(value)
    if issue:
        return issue
    if kind == "commit":
        line = value.splitlines()[0] if value.splitlines() else ""
        if len(line) > 72 or not COMMIT.fullmatch(line):
            return "commit subject must follow Conventional Commits"
    if kind == "ref":
        if value.startswith("refs/tags/"):
            if not re.fullmatch(r"v\d+\.\d+\.\d+(?:-[a-z0-9.-]+)?", value[len("refs/tags/"):]):
                return "tag does not follow version convention"
        elif not BRANCH.fullmatch(value.removeprefix("refs/heads/")):
            return "branch name does not follow policy"
    return None


def verify_root():
    marker = ROOT / ".git"
    if marker.is_symlink():
        raise GuardError("Symlink Git metadata marker is not approved")
    if marker.is_dir():
        expected_dir = marker.resolve()
    elif marker.is_file():
        # Linked worktrees use a gitdir pointer rather than a .git directory.
        pointer = marker.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: ") or "\n" in pointer:
            raise GuardError("Invalid checkout Git metadata pointer")
        expected_dir = (ROOT / pointer[len("gitdir: "):]).resolve()
    else:
        raise GuardError("Checkout Git metadata marker missing; use the actual Git root")
    git_dir = Path(git("rev-parse", "--absolute-git-dir").decode("utf-8").strip()).resolve()
    if git_dir != expected_dir:
        raise GuardError("Git environment redirects checkout metadata; remove external Git directory overrides")
    common_marker = expected_dir / "commondir"
    expected_common = ((expected_dir / common_marker.read_text(encoding="utf-8").strip()).resolve()
                       if common_marker.is_file() else expected_dir)
    common = Path(git("rev-parse", "--git-common-dir").decode("utf-8").strip())
    if (ROOT / common).resolve() != expected_common:
        raise GuardError("Git environment redirects common metadata; remove external Git common-directory overrides")
    actual = Path(git("rev-parse", "--show-toplevel").decode("utf-8").strip()).resolve()
    if actual != ROOT.resolve():
        raise GuardError("Scanner must reside in the actual Git root; enclosing/nested checkout is not approved")
    if os.environ.get("GIT_INDEX_FILE"):
        index = Path(os.environ["GIT_INDEX_FILE"]).resolve()
        if not index.is_relative_to(git_dir):
            raise GuardError("External Git index override is not approved; inspect the checkout index")
    # Older Git releases do not implement GIT_NO_LAZY_FETCH. Refuse promisor
    # repositories before any object command could contact their remotes.
    for item in git("config", "--null", "--list").split(b"\0"):
        key, _, value = item.partition(b"\n")
        key = key.lower()
        if key == b"extensions.partialclone" or (key.startswith(b"remote.") and key.endswith(b".promisor")
                and value.lower() not in {b"false", b"no", b"off", b"0"}):
            raise GuardError("Partial/promisor checkout cannot be inspected offline; use a complete local clone")


def read_object(kind, oid, pol):
    if int(git("cat-file", "-s", oid)) > pol["maximum_text_bytes"]:
        raise GuardError("Git object exceeds reviewed size limit; split or remove oversized material")
    return git("cat-file", kind, oid)


def entry_issue(path, mode, oid, pol):
    # Preflight mode/path/size before loading object contents.
    if mode not in ALLOWED_TYPES or not safe_path(path, pol):
        return safe_blob(path, mode, b"", pol)
    if int(git("cat-file", "-s", oid)) > pol["maximum_text_bytes"]:
        return "file exceeds reviewed size limit; remove or reduce the file"
    return safe_blob(path, mode, git("cat-file", "blob", oid), pol)


def collision_issue(path, seen):
    # Directory spelling collisions also matter on Windows.
    parts = path.split("/")
    for end in range(1, len(parts) + 1):
        prefix = "/".join(parts[:end])
        key = prefix.casefold()
        if key in seen and seen[key] != prefix:
            return "case-colliding path; use consistent unique spelling"
        seen[key] = prefix
    return None


def inspect_tree(treeish, pol):
    tree = git("rev-parse", "--verify", "--end-of-options", treeish + "^{tree}").decode("ascii").strip()
    issues, seen = [], {}
    for rec in git("ls-tree", "-r", "-z", tree).split(b"\0"):
        if not rec:
            continue
        try:
            metadata, name = rec.split(b"\t", 1)
            mode, _, oid = metadata.decode("ascii").split(" ")
            path = name.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            issues.append("tree: unsupported entry encoding; use UTF-8 paths")
            continue
        for issue in (collision_issue(path, seen), entry_issue(path, mode, oid, pol)):
            if issue:
                issues.append(f"{label(path)}: {issue}")
    return issues


def inspect_staged(pol):
    # Inspect the entire prospective tree, including unchanged tracked blobs.
    issues, seen = [], {}
    for rec in git("ls-files", "--stage", "-z").split(b"\0"):
        if not rec:
            continue
        try:
            meta, name = rec.split(b"\t", 1)
            mode, oid, stage = meta.decode("ascii").split(" ")
            path = name.decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            issues.append("index entry malformed; repair index/UTF-8 paths")
            continue
        if stage != "0":
            issues.append(f"{label(path)}: unresolved merge index stage; resolve the merge")
            continue
        for issue in (collision_issue(path, seen), entry_issue(path, mode, oid, pol)):
            if issue:
                issues.append(f"{label(path)}: {issue}")
    return issues


def complete_history():
    if git("rev-parse", "--is-shallow-repository").strip() != b"false":
        raise GuardError("Shallow history cannot be reviewed; obtain complete local history first")
    graft = Path(git("rev-parse", "--git-path", "info/grafts").decode("utf-8").strip())
    if not graft.is_absolute():
        graft = ROOT / graft
    if graft.exists() or os.environ.get("GIT_GRAFT_FILE"):
        raise GuardError("Grafted history cannot be reviewed; remove history graft configuration first")


def peel_commit(oid, pol, issues=None):
    visited = set()
    while True:
        if oid in visited or len(visited) >= 32:
            raise GuardError("Invalid or excessive annotated tag chain")
        visited.add(oid)
        kind = git("cat-file", "-t", oid).strip()
        if kind == b"commit":
            return oid
        if kind != b"tag":
            raise GuardError("Outgoing reference must resolve to a commit, not a blob/tree")
        data = read_object("tag", oid, pol)
        try:
            text = data.decode("utf-8")
            header, _ = text.split("\n\n", 1)
            lines = header.splitlines()
            if len(lines) != 4 or not all(line.startswith(prefix) for line, prefix in zip(
                    lines, ("object ", "type ", "tag ", "tagger "))):
                raise ValueError("invalid tag headers")
            target, declared, name = lines[0][7:], lines[1][5:], lines[2][4:]
            if not OID.fullmatch(target) or not name or declared not in {"commit", "tag"}:
                raise ValueError("invalid target/type/name")
            if git("cat-file", "-t", target).decode("ascii").strip() != declared:
                raise ValueError("tag target type mismatch")
            if not re.fullmatch(r"tagger [^<>\n]+ <[^<>\n]+> -?\d+ [+-]\d{4}", lines[3]):
                raise ValueError("invalid tagger identity/date")
        except (UnicodeError, ValueError) as exc:
            raise GuardError("Malformed annotated tag; recreate a valid release tag") from exc
        if issues is not None:
            for issue in (content_issue(text), check_text("ref", "refs/tags/" + name)):
                if issue:
                    issues.append(f"tag {oid[:12]}: {issue}")
        oid = target


def inspect_outgoing(stdin, pol):
    complete_history()
    issues, updates, scanned = [], [], set()
    for line in stdin.splitlines():
        parts = line.split()
        if len(parts) != 4:
            raise GuardError("Malformed pre-push reference input; expected four fields per update")
        local_ref, local_sha, remote_ref, remote_sha = parts
        if not OID.fullmatch(local_sha) or not OID.fullmatch(remote_sha) or len(local_sha) != len(remote_sha):
            raise GuardError("Malformed pre-push object IDs; expected full matching Git hashes")
        if not remote_ref.startswith(("refs/heads/", "refs/tags/")):
            issues.append("remote destination: unsupported reference namespace")
        elif (issue := check_text("ref", remote_ref)):
            issues.append(f"remote destination {label(remote_ref)}: {issue}")
        if set(local_sha) == {"0"}:
            continue
        if local_ref not in {"HEAD", local_sha}:
            if not local_ref.startswith(("refs/heads/", "refs/tags/")):
                issues.append("local source: unsupported reference namespace")
            elif (issue := check_text("ref", local_ref)):
                issues.append(f"local source {label(local_ref)}: {issue}")
        local_commit = peel_commit(local_sha, pol, issues)
        remote_commit = None if set(remote_sha) == {"0"} else peel_commit(remote_sha, pol)
        updates.append((local_commit, remote_commit))
    for local_commit, remote_commit in updates:
        args = ["rev-list", "--max-count=" + str(pol["max_commits_per_push_scan"] + 1), local_commit]
        if remote_commit:
            args.append("^" + remote_commit)
        seq = git(*args).decode("ascii").splitlines()
        fresh = [sha for sha in seq if sha not in scanned]
        scanned.update(seq)
        if len(scanned) > pol["max_commits_per_push_scan"]:
            raise GuardError("Push history exceeds reviewed commit limit; split/review the history intentionally")
        for sha in fresh:
            # Author fields can leak too, so inspect raw metadata and message.
            data = read_object("commit", sha, pol)
            try:
                text = data.decode("utf-8")
                _, message = text.split("\n\n", 1)
            except (UnicodeError, ValueError) as exc:
                raise GuardError("Malformed/non-UTF-8 commit metadata; repair before publication") from exc
            for issue in (content_issue(text), check_text("commit", message)):
                if issue:
                    issues.append(f"commit {sha[:12]}: {issue}")
            issues.extend(f"{sha[:12]}/{issue}" for issue in inspect_tree(sha, pol))
    return issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["staged", "tree", "outgoing", "commit-msg", "text", "ref"])
    parser.add_argument("argument", nargs="?")
    parser.add_argument("--kind", choices=["commit", "ref", "prose"], default="prose")
    args = parser.parse_args(argv)
    try:
        pol = policy()
        if args.mode in {"staged", "tree", "outgoing"}:
            verify_root()
        if args.mode == "staged":
            issues = inspect_staged(pol)
        elif args.mode == "tree":
            issues = inspect_tree(args.argument or "HEAD", pol)
        elif args.mode == "outgoing":
            issues = inspect_outgoing(sys.stdin.read(), pol)
        else:
            if not args.argument:
                raise GuardError("Text/reference or commit message file argument is required")
            if args.mode == "commit-msg":
                try:
                    value = Path(args.argument).read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise GuardError("Commit message file unavailable/non-UTF-8; supply a readable message") from exc
                kind = "commit"
            else:
                value, kind = args.argument, "ref" if args.mode == "ref" else args.kind
            issues = [issue] if (issue := check_text(kind, value)) else []
        if issues:
            print("PUBLICATION BLOCKED:")
            for issue in issues[:30]:
                print(" -", issue)
            if len(issues) > 30:
                print(f" - and {len(issues) - 30} more violations")
            return 1
        print(f"PASS: {args.mode} publication check")
        return 0
    except GuardError as exc:
        print("PUBLICATION CHECK ERROR:", str(exc), file=sys.stderr)
        return 2
    except (OSError, UnicodeError, ValueError, RecursionError):
        print("PUBLICATION CHECK ERROR: incomplete inspection; verify local files, encoding and Git objects", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
