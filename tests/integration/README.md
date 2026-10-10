# Not yet implemented

Tests in this directory will be introduced with approved runtime feature work.

## Planned cross-component gate

This directory contains a plan, not an executable test suite. There is no HTTP runtime or integration harness yet. Future CPU integration checks will combine API/streaming, sequence state, admission, scheduler, batch engine and cache accounting with a deterministic fake executor; they do not establish GPU correctness or performance.

Planned checks against the [runtime architecture](../../docs/architecture.md):

- Validated/tokenized request identity and legal queue/prefill/decode/terminal transitions, including immediate completion on the first output and rejected work with no GPU reservation.
- [Scheduler budgets](../../docs/scheduler.md), deterministic FIFO admission, decode progress, bounded queues/channels, immutable plan validation and no double execution.
- Ordered per-request token/terminal events, disconnect/deadline propagation and backpressure. Do not subtract independent client/server clocks or mistake a sampled token for an emitted response event.
- Cancellation while queued or active, failed materialization/execution, rollback of unused reservations and quarantine/retirement after uncertain device writes. Fake completion events can test ownership logic; only [real GPU tests](../gpu/README.md) verify CUDA lifetime safety.
- Correct metric/outcome accounting for no output, one-token output, rejection, cancellation and failure, plus zero-work/incomplete trial handling and trial drain boundaries.

Benchmark adapters will also need future tests for materialized workload replay, matched generation settings, actual versus requested token counts, timestamp ordering and unavailable metrics. All such assertions require implemented interfaces; no test or benchmark command is advertised here before it exists.

Use synthetic data, pinned fixtures and controlled clocks instead of unreliable timing thresholds. Record the [reproduction metadata](../../docs/reproducibility.md#run-record), exact command and outcome of each future run. Integration correctness, GPU parity and [performance measurements](../../docs/benchmarks.md) are separate evidence; passing CPU integration tests cannot establish throughput or safe device reuse.
