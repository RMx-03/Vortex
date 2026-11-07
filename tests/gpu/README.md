# GPU reference qualification

`test_reference_parity.py` implements a bounded, opt-in capture controller and
lightweight CPU regression fixtures. **Actual Linux/T4/pretrained execution is
NOT VERIFIED.** Synthetic tests and wheel metadata do not qualify CUDA execution.
No public golden arrays exist. Captures remain private and unreviewed until an
independent evidence audit and separate human approval.

## Local CPU verification

```sh
python -B -m unittest discover -s tests/gpu -p "test_reference_parity.py" -v
python -B scripts/verify.py
```

With no execution controls, the first command runs the controller fixtures and
skips exactly one GPU capture test with an explicit NOT VERIFIED reason. It
imports no Torch, Transformers, NumPy or Hub code during discovery. Fixtures use
small synthetic records, temporary files and disposable bounded Python children;
they never download weights or start a cloud session. Windows sandbox permissions
may require the normal temporary-file permissions used by the existing CPU suite.
Do not enable the GPU controls on a local laptop.

## Prerequisites for a separately authorized experiment

Use a private, free interactive Colab or Kaggle allocation with one actual
Tesla T4/NVIDIA T4, capability 7.5, selected as `cuda:0`. No paid fallback,
substitution, offload, quantization or relaxed determinism is supported. Require
at least 8 GiB available host RAM, 8 GiB free/total VRAM and 4 GiB free evidence
disk at preflight; estimates do not guarantee loading will fit.

Deliver a reviewed commit containing this harness, with a clean index/worktree
and raw tracked bytes matching that commit. The earlier baseline commit alone
does not contain the harness. Source delivery/publication and cloud execution
need separate authorization. Never upload the private knowledge base.

Use the exact [locked environment](../../docs/reference-environment.md): official
Astral CPython 3.11.18 release 20261026, Linux x86_64/glibc >=2.28, all 43 locked
wheel identities, Torch 2.10.0+cu126, Transformers 4.53.3, CUDA runtime 12.6,
FP16/eager. Driver compatibility floor 560.35.05 is separate from security
clearance. A current execution-specific security review must approve the actual
driver and pinned dependency paths before native imports. No automatic upgrades.

**uv 0.10.7 integrity remains a blocking prerequisite.** Obtain and independently
review the official immutable archive/checksum and resolved executable identity;
record both hashes before using it to create the environment. This harness
verifies approved executable hashes; it does not acquire or independently attest
the archive. A version string, placeholder digest, newest installer or successful
wheel metadata check cannot replace that review. Likewise verify the approved
Python archive before extraction and record the resolved interpreter hash and
full `sys.version`. No installation is performed by this harness.

## Exact opt-in controls

Before starting Python, set these values on the authorized Linux host only:

| Variable | Required value |
|---|---|
| `VORTEX_GPU_EXECUTE` | `1` |
| `VORTEX_GPU_PROFILE` | `t4-fp16` |
| `VORTEX_GPU_SOURCE_SHA` | Full reviewed commit SHA containing the harness |
| `VORTEX_GPU_PRIVATE_ROOT` | Absolute owner-only directory outside source |
| `VORTEX_GPU_CACHE` | Fresh, nonexistent absolute cache directory, disjoint from evidence |
| `VORTEX_GPU_UV` | Approved exact uv executable path outside source |
| `VORTEX_GPU_APPROVAL` | Private approval JSON path outside source |
| `VORTEX_GPU_APPROVAL_SHA256` | SHA256 of the exact reviewed approval JSON bytes |
| `CUDA_VISIBLE_DEVICES` | `0` |
| `PYTHONHASHSEED` | `0` |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| `PYTHONUNBUFFERED` | `1` |

The approval JSON is a trusted operator input, not a golden approval or a public
schema extension. It has exactly these fields:

- `execution_authorized: true`, matching `source_sha`, `profile_id: "t4-fp16"`,
  `provider` (`colab-free` or `kaggle-free`).
- `security_reviewed: true`, exact `driver_version`, `reviewed_utc` in UTC within
  the preceding 24 hours, and integer `remaining_free_seconds` from 1 to 5400.
- `tools`, containing exactly `python_archive_sha256`,
  `python_executable_sha256`, `python_sys_version`, `uv_version: "0.10.7"`,
  `uv_archive_sha256`, `uv_executable_sha256`, `uv_identity_reviewed: true`, and
  `uv_release: "https://github.com/astral-sh/uv/releases/tag/0.10.7"`.

The approved Python archive SHA256 is
`17c4978a36c496dc53c08899ef8dad1ad2734f6325f08d6033a1e66419e0af26`.
All other tool hashes must come from the separate identity review. Symlinked
venv interpreters are checked against their resolved regular executable. The
approval must authorize the actual free session and bounds; setting environment
variables alone is not permission. A missing or incompatible gate fails BLOCKED.

After all prerequisites and controls are approved, use the exact environment's
Python executable with the discovery command above. It runs one real capture
test alongside CPU fixtures. Zero actual GPU tests or a skip is NOT VERIFIED.
A successful capture test still leaves export/readback, independent evidence
review and human golden approval pending.

## Execution and supervision

A fresh offline-negative interpreter proves missing assets fail in a separate
empty cache without network or model construction. The primary interpreter then
qualifies imports, actual hardware and a tiny random FP16/eager forward/cache
transition before acquiring exactly the seven immutable Qwen assets. Acquisition
and streaming verification are separate from pretrained model loading. The
guarded loader rechecks every asset, then loads offline with no remote code or
pickle. Subsequent native operations run with network access blocked.

The primary process loads one model and captures
`A1,A2,A3,B1,B2,B3,C1,C2,C3`. It exits and is reaped before a new offline
interpreter/model captures `A4,B4,C4`. Each request delegates to the existing
generator with a fresh DynamicCache, G=32, P<=512, strict greedy settings and
both EOS IDs 151645/151643. Raw generated IDs, terminal EOS, sequence IDs, actual
length, stopping metadata and complete numerical identities stay in schema-1
private manifests. EOS at G may also set `reached_limit=true`.

Each completed repeat/replay immediately compares `A1-A2,A1-A3,A1-A4` and the
equivalent B/C pairs: twelve captures, nine comparisons. MISMATCH/INCOMPATIBLE
is retained and stops before another request starts. Display text is not an
oracle. There are no retries or changed settings under one attempt identity.

Parent supervision enforces ordered START/DONE messages. A worker waits for an
ACK after the parent's durable start record and after result readback. Each
generation and tiny smoke has a 120-second ceiling; acquisition plus first
load shares one 1200-second absolute deadline. Primary/replay ceilings are
2400/1200 seconds, nested under the lesser of 5400 seconds and the remaining
approved free allowance. Startup/environment and transition deadlines are also
bounded. Timeouts kill the whole Linux process group, including verifier
descendants, then reap the child. Native errors, OOM, malformed events, output
limits and interruptions leave incomplete evidence. A CUDA fault ends the
attempt; clearing cache is not a recovery strategy. Windows child-tree cleanup
exists only for small CPU fixtures; real GPU execution is Linux-only.

Stdout is parsed as bounded JSON events: <=1 MiB per record, <=32 MiB per worker,
with a bounded reader queue. Stderr is discarded after counting, with a 1 MiB
ceiling; no raw native diagnostics, credentials or environment dumps are stored.
Filesystem durability assumes a trusted local filesystem that responds to OS
operations; this controller is not an adversarial-filesystem sandbox.

## Private retention, export and local reopen

Each UUID attempt directory contains `capture`, `review`, `exports` and a separate
immutable primary checkpoint. Prewritten attempt ledgers and external
`qualification.INCOMPLETE` survive interruption; SIGKILL/VM loss cannot be caught.
Records use temporary siblings, fsync, atomic replacement and hash readback.
Per-capture `capture.COMPLETE` markers follow durable manifests. A group marker
requires all execution evidence, twelve manifests and nine matches.

The closed capture tree permits only qualification artifacts, never weights,
caches, notebooks or arbitrary source trees. JSON files are limited to 1 MiB,
the tree to 128 entries and capture payload to 32 MiB. A sorted file/size/SHA256
index excludes itself and its companion hash. All qualification statuses,
export/reopen receipts and human approvals are outside **both** the index and
archive. Those external records bind the immutable capture-index hash and their
supporting hashes. Indexed bytes never change after sealing. A primary checkpoint
has its own index/archive; adding replay does not rewrite that checkpoint.

Export `exports/primary.zip` after the primary group and `exports/capture.zip`
before disconnecting through a separately authorized private channel. The
harness creates closed archives but never uploads, mounts storage or assumes
notebook saves preserve VM files. Retain all failed/incomplete attempts. Transfer
the externally recorded archive/index SHA256 values too; never overwrite an
earlier attempt. Export failure or absent destination readback stays INCOMPLETE.

On Windows, reopen the actual downloaded archive outside the public checkout,
using a new destination directory:

```powershell
python -B tests/gpu/test_reference_parity.py --reopen $DownloadedArchive --destination $NewPrivateDestination --archive-sha256 $ApprovedArchiveSha256 --index-sha256 $ApprovedIndexSha256
```

Reopen validates bounded safe members, archive/index/member hashes, every schema,
all comparisons, complete process grouping and the environment/smoke/assets/tool
qualification records. It re-reads extracted destination bytes before writing an
external receipt and applicable qualification marker. Partial bundles can be
preserved and checked, but return exit 2 and never qualify. Failed writes or
checksums cannot produce `qualification.COMPLETE`. The complete marker concerns
capture/export checks only: independent evidence review remains NOT VERIFIED and
human approval remains NOT_APPROVED. Evidence checks are not host attestation.
Only a later independent audit can reach VERIFIED_PENDING_APPROVAL; the harness
never writes a human golden approval.

## Planned GPU correctness gate

The broader runtime checks below remain future work. This qualification harness
has not been executed on a GPU. An environment probe and CPU tooling do not
verify tensor execution, KV storage or pretrained model parity; missing
hardware/dependencies are **NOT VERIFIED**.

Before timing an implemented variant, record the pinned model/tokenizer revisions, source SHA, GPU/driver/CUDA/PyTorch/Transformers versions, precision, generation settings and seeds from the [run record](../../docs/reproducibility.md#run-record).

Planned checks:

- Fixed greedy output token IDs against trusted sequential HF, plus matched static-batch and Vortex runs. Declare tolerances for intermediate logits where needed; a tolerance is not permission to silently accept different greedy output tokens. Any divergence needs investigation/disclosure before a correctness claim.
- Tensor shapes, masks, token positions, device/dtype and per-request identity as batch rows change; no cross-request cache contamination.
- Physical cache writes/reads and attention parity, populated token counts, measured allocated/reserved GPU memory and declared capacity estimates. CPU block allocation does not prove physical paging.
- Cancellation, EOS, output limits and execution failure: no future scheduling, idempotent cleanup and storage reuse only after confirmed device completion. Stress repeated allocate/retire cycles and declared capacity edges without assuming estimates guarantee no OOM.
- Streamed token order versus final output, per-sequence sampling state and recorded deterministic settings. Stochastic performance scenarios do not replace greedy correctness evidence.

Use the [KV ownership contract](../../docs/kv-cache.md) and [benchmark protocol](../../docs/benchmarks.md). Correctness/profiling runs and timed performance trials are separate, labelled artifacts. Real-device synchronization and safe reuse remain unverified until these tests exist and run. Preserve failures/incomplete tests privately under the [results policy](../../results/README.md); no model weights or unreviewed outputs belong in Git.
