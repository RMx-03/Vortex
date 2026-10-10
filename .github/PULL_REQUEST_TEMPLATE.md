## Summary
<!-- State the concrete problem and resulting behavior in one or two sentences. -->

## Motivation
<!-- Explain why this change is needed. Link relevant public issues or requirements. -->

## Changes
<!-- Describe the final implementation scope and any intentional exclusions. -->

## Design Decisions
<!-- Explain meaningful tradeoffs, interfaces and ownership choices; omit if not applicable. -->

## Verification
<!-- List exact commands, observed results and relevant environment versions.
Use reviewed public evidence only; omit local coordination notes, credentials and raw sensitive output.
Mark unrun checks NOT VERIFIED, including hosted CI or GPU checks when applicable.
Typical CPU checks: python scripts/verify.py and python scripts/publication.py tree HEAD.
Report CI history-audit results separately; a latest-tree scan does not certify earlier commits.
Final squash/merge messages must follow CONTRIBUTING.md; default GitHub merge subjects may be rejected. -->

## Correctness
<!-- Explain which contracts and normal/failure paths were checked.
For runtime changes, include applicable cancellation, cleanup, determinism and memory-lifetime evidence.
Distinguish CPU mocks from actual device tests; state what remains unverified. -->

## Performance Impact
<!-- Where relevant, give matched baseline/change measurements with hardware, workload,
versions and uncertainty. Otherwise state not measured or not applicable; do not claim a speedup. -->

## Risks and Limitations
<!-- Describe remaining risks, operational impact and the rollback/recovery path. -->

## Compatibility
<!-- Note Python/platform/API/data-format changes and migration needs, or state no known change. -->

## Checklist

- [ ] Scope is small and reviewable
- [ ] Final commit and squash/merge messages follow the repository's Conventional Commit policy
- [ ] Relevant tests pass; evidence included
- [ ] Correctness and compatibility claims match the checks actually performed
- [ ] Comments and docs match implemented behavior
- [ ] No secrets, non-public workspace material or unsolicited attribution metadata
- [ ] No unsupported performance assertions
