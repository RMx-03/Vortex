# Not yet implemented

Tests in this directory will be introduced with approved runtime feature work.

## Planned GPU correctness gate

No GPU tests have been implemented or executed by this scaffold. An environment probe and the existing CPU tooling suite do not verify tensor execution, KV storage or model parity. These tests will require separately authorized model/dependency setup and suitable hardware; missing hardware/dependencies must be reported as **NOT VERIFIED**, not a passing experiment.

Before timing an implemented variant, record the pinned model/tokenizer revisions, source SHA, GPU/driver/CUDA/PyTorch/Transformers versions, precision, generation settings and seeds from the [run record](../../docs/reproducibility.md#run-record).

Planned checks:

- Fixed greedy output token IDs against trusted sequential HF, plus matched static-batch and Vortex runs. Declare tolerances for intermediate logits where needed; a tolerance is not permission to silently accept different greedy output tokens. Any divergence needs investigation/disclosure before a correctness claim.
- Tensor shapes, masks, token positions, device/dtype and per-request identity as batch rows change; no cross-request cache contamination.
- Physical cache writes/reads and attention parity, populated token counts, measured allocated/reserved GPU memory and declared capacity estimates. CPU block allocation does not prove physical paging.
- Cancellation, EOS, output limits and execution failure: no future scheduling, idempotent cleanup and storage reuse only after confirmed device completion. Stress repeated allocate/retire cycles and declared capacity edges without assuming estimates guarantee no OOM.
- Streamed token order versus final output, per-sequence sampling state and recorded deterministic settings. Stochastic performance scenarios do not replace greedy correctness evidence.

Use the [KV ownership contract](../../docs/kv-cache.md) and [benchmark protocol](../../docs/benchmarks.md). Correctness/profiling runs and timed performance trials are separate, labelled artifacts. Real-device synchronization and safe reuse remain unverified until these tests exist and run. Preserve failures/incomplete tests privately under the [results policy](../../results/README.md); no model weights or unreviewed outputs belong in Git.
