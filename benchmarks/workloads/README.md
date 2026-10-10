# Workloads

Future synthetic workload definitions will be versioned here. **No workload generator, manifests or benchmark runner are implemented yet.** The fields below are a planned contract, not an existing validated JSON schema. Do not store confidential user prompts. Use the [benchmark metric/comparison protocol](../../docs/benchmarks.md) and [run/environment record](../../docs/reproducibility.md#run-record).

## Manifest contract

Record a schema/version identifier, workload ID and content digest, generator version and seed, model/tokenizer revisions and chat-template settings. Materialize the request list before comparing methods; a seed alone is not a replayable workload.

Each request needs a stable synthetic ID, exact prompt text or token IDs with a digest, **actual token length after the pinned chat template**, requested output cap, generation/EOS/stop settings and generation seed if applicable. Record actual output lengths with results rather than pretending every request reached its cap. Any synthetic prompt/fixture intended for public Git still needs publication review.

The scenario record must declare request count, prompt/output-length buckets and sampling distribution, order, load mode, arrival offsets or closed-loop replenishment rule, client concurrency, trial duration/count and drain behavior. Include scheduler/token/KV limits, static batch size/formation wait and a predeclared overload stopping criterion. Check prompt-plus-output/context and configured admission limits; deliberate oversized/rejection cases must be labelled instead of silently dropping them.

Cancellation records identify the selected requests, target ratio and trigger: a fixed time relative to actual arrival or a generated-token count. Record actual signal/observation times and outcomes. Time-based and token-triggered cancellations produce different load and must not be treated as identical.

## Planned scenario matrix

| Scenario | Controls / question |
| --- | --- |
| Length buckets | Short/medium/long prompt and requested output buckets, with exact chosen sizes recorded; compare matched sequential HF, static HF and Vortex work. |
| Concurrency sweep | Closed-loop client counts such as 1, 2, 4, 8, 16 up to the declared capacity/failure stop; these are proposed points, not proven supported capacities. Record actual GPU residency and queueing. |
| Arrival/burst load | Replay a fixed open-loop trace with rates/burst sizes and actual arrival deviation; examine queue growth, rejections and drain, not just successful latency. |
| Mixed lengths | Fixed order of short/long requests; expose FIFO head blocking, decode progress and static-batch padding/drain costs. |
| Repeated prefix | Exact shared token prefixes; separate empty-prefix-cache and deliberately warm-prefix-cache runs. Identify reuse policy and warmup effects. |
| Cancellation/overload | Cancel queued and active work, include timeouts and configured admission violations; check post-cancel work and safe capacity recovery. Mark unsupported baseline cancellation explicitly. |

All compared methods get the same materialized prompts, limits and seeds. Open-loop methods share the intended arrival trace; closed-loop methods share client count/order/replenishment, not identical completion-dependent arrival times. Use greedy comparisons first. If fixed output length is desired, explicitly declare a matched stopping policy; realistic EOS-driven lengths form a separate scenario. Never pad generated-token counts to the requested cap.

Synthetic data avoids user prompt disclosure, but reproducibility still requires exact tokenization/configuration and safe generated output handling. No dataset, measured distribution or performance result is asserted here.
