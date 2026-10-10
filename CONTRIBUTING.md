# Contribution and Git standards

Use [CODE_QUALITY.md](CODE_QUALITY.md) for implementation and comment standards. Runtime work requires an accepted feature specification and plan; the current scaffold is not an inference server. Keep each change small enough to explain and verify without unrelated cleanup.

## Scope and approval

Inspect the actual Git root, HEAD, branch and staged/unstaged/untracked state before work. Preserve existing changes; do not reset or clean the checkout to make a task easier. Complete the bounded change, verification and review before requesting approval. In the scaffold rollout, stop before staging or committing until the exact file list and message are explicitly approved. Approval of one task does not authorize the next task, a push or a PR.

After approval, stage only named files, inspect the entire staged diff and run the staged publication check. Stop without committing if unexpected files are staged or a check fails. Keep the existing configured Git identity. Any remote push or PR requires separate explicit approval after its contents are reviewable.

## Branches

`feat/<component>-<scope>`, `fix/<component>-<scope>`, `refactor/<component>-<scope>`, `perf/<component>-<scope>`, `test/<component>-<scope>`, `docs/<topic>`, `chore/<scope>`.

Use lowercase letters and digits, with single hyphens between alphanumeric groups; slash-separated components are permitted. Continue an approved existing task branch rather than creating branches for each scaffold document. The current scaffold branch is `chore/scaffold-foundation`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): `<type>(<scope>): <imperative description>`. This repository's [publication scanner](scripts/publication.py) enforces a narrower form: a scope is required, starts with a lowercase letter and contains lowercase letters, digits or hyphens; the description starts with a lowercase letter or digit; the complete subject is at most 72 characters. Allowed types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`, `build`, `ci`. The scanner does not currently accept the `!` subject form. Explain breaking changes and migration in the body and PR; agree on any scanner-policy change before using a different subject form.

For example: `docs(standards): define engineering conventions`. Explain design tradeoffs in the body where relevant. Never add unsolicited co-author trailers, coding-assistant attribution or generation metadata. Maintain truthful Git author metadata, preserve legitimate existing contributor credit and comply with mandatory disclosures; surface any conflict with publication policy before publishing rather than misrepresenting authorship.

## Pull requests

Use the [PR template](.github/PULL_REQUEST_TEMPLATE.md). Lead with the concrete problem and resulting behavior; describe the final change for someone who has not read the development conversation. Keep each PR narrowly scoped to one accepted task or inseparable group. Include design tradeoffs, exact verification commands/results, risk/rollback/compatibility and measured performance impact or an explicit statement that it was not measured. Link only reviewed public evidence. Omit transcripts, abandoned approaches, generic claims and unsolicited coding-assistant attribution.

Reviews must trace the relevant normal, failure, cancellation and cleanup paths, check interface contracts and resource ownership, and examine tests for meaningful assertions and isolation. Match the depth of review to risk: device/cache changes need memory-lifetime evidence; documentation changes need link, example and consistency checks. Give actionable findings with severity, location, rationale and a reproduction or missing verification. Fix blocking findings and verify corrections; resolve disagreements with code, tests or documented tradeoffs rather than silently dismissing feedback.

## Verification

From the repository root, the offline CPU-only starter check is:

```sh
python scripts/verify.py
```

It runs `python -m unittest discover -s tests/unit -p "test_*.py" -v` and, in this Git checkout, `python scripts/publication.py staged`. The staged scan reads the full effective index, not unstaged or untracked working files. Review the exact changed working-file contents during implementation; after approved staging, scan the index again before committing. To inspect committed contents without staging anything, use:

```sh
python scripts/publication.py tree HEAD
```

Neither command proves that untracked proposals are ready for publication, nor does a tree scan inspect outgoing ancestry. The outgoing mode separately reviews commit history during pre-push checks; local hooks can be bypassed and are not an authorization to publish. Do not claim CI, type checking, linting, GPU correctness or runtime performance from these starter checks; report the specific tools and commands actually run. The current checks require no model download, CUDA or external service.

For code changes, add focused tests for changed observable contracts and relevant regressions. Use deterministic, isolated fixtures and record seeds/settings, versions, commands and actual results. For performance changes, benchmark matched hardware/model/tokenizer revision/dtype/workload; warm up, repeat, report percentiles and include [reproducibility details](docs/reproducibility.md). Distinguish CPU fakes from actual device evidence and label skipped or unavailable checks NOT VERIFIED. For documentation-only changes, check links, examples and consistency; do not add tests that merely mirror prose or run unrelated experiments.

## Source/documentation boundaries

Only allowlisted project source, neutral technical documentation and reproducibility utilities belong in public Git under the authoritative [publication policy](.publication/policy.json). Keep local coordination history, internal working notes, tool-specific instructions, credentials, model weights, raw traces and unreviewed notebooks/results out of it. Use synthetic secrets in negative tests and never expose real values in logs or review findings.

Sanitized evidence still needs an allowed path and explicit publication review. The current policy permits only `results/README.md` under results; sanitization alone does not authorize another result file. Approved notebook paths must be valid, output-free and attachment-free. A safe latest tree does not erase a private file from earlier commits: review the outgoing history as well. Do not bypass a scanner failure to publish; resolve the content or seek an explicitly reviewed policy change. Local guards and CI cannot undo an earlier disclosure.
