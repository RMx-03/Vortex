# Reproducing development and GPU experiments

The public package is currently namespace scaffolding with Python 3.11+ support and no inference dependencies. There is no benchmark runner or model-serving command yet. Everything about model/GPU experiments below is a **planned procedure requiring separately authorized feature/experiment work**, not a completed experiment.

## Current CPU checks

From the repository root with a supported Python:

```sh
python scripts/verify.py
python scripts/gpu_capabilities.py
```

The verifier runs existing CPU tooling/publication tests. The probe prints JSON and distinguishes missing PyTorch from unavailable CUDA; a successful CPU/no-GPU probe does not prove model support. These commands do not install packages, download weights or provision a service. The package supports editable installation through `pyproject.toml`; package installation is separate from these offline checks and may need build dependencies.

The existing Colab/Kaggle starter notebooks contain environment probes only; they are **not model or inference experiments**, and this document does not imply they have run on either provider. Do not treat a notebook title or GPU availability probe as benchmark evidence.

## Run record

For each future run, pin intended inputs and record the actual resolved environment before measurement. A package name or model alias without an immutable revision is insufficient. Store these fields together with the [workload manifest](../benchmarks/workloads/README.md) and metric events:

| Group | Required record |
| --- | --- |
| Source | Full Vortex commit SHA, clean/dirty state and any retained patch digest; benchmark adapter/runner version and method/variant configuration. Publish claims from a clean known source or explicitly disclose changes. |
| Model | Repository ID, immutable model and tokenizer revisions, chat-template digest/options, model/config digests, EOS/stop rules, context/output limits and generation settings. |
| Environment | OS/platform, Python, exact package/wheel versions including PyTorch/Transformers, CUDA build/runtime as applicable, NVIDIA driver and dependency lock/installation recipe. Record unavailable fields rather than inventing a CUDA toolkit installation. |
| Hardware | Actual GPU model, device index/count, per-device memory/capability, CPU and host RAM, provider/session/run ID and relevant sharing/resource conditions. Select one GPU initially; two devices' memory is not one addressable pool. |
| Numerics | Weight/compute/KV dtypes, precision/quantization settings, attention backend, compilation flags, inference mode and deterministic backend options. |
| Randomness | Workload and per-request generation seeds, generator implementation/version, tokenized prompt and manifest hashes. Greedy parity first; record remaining nondeterminism. |
| Protocol | Scenario/load mode, intended/actual arrivals, concurrency, batch/scheduler/KV limits, cache warmth, warmup policy, repeated-trial count/order, timing boundary/clock, measurement/drain rules and profiler status. |
| Artifacts | Run/trial IDs, timestamps, complete/incomplete/invalid status and reasons, outcome/sample counts, raw-file checksums, analysis version and metric/quantile definitions. |

Do not dump environment variables or credentials. Capture a whitelist of version/config fields; prompts, generated text, paths and traces remain private until reviewed. Pin the environment used for an experiment rather than relying on provider image defaults. Record actual versions again after installation; if the approved pins cannot be installed, stop that experiment and report the mismatch. No particular lockfile or runtime dependency versions have been approved in this scaffold.

Seeds are necessary records, not a cross-platform reproducibility guarantee. [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html) documents platform/release limitations and deterministic-performance tradeoffs. Record the settings used by every method; correctness and performance runs with different settings are separate evidence.

## Future Colab / Kaggle procedure

GPU environments are temporary execution targets, not permanent infrastructure dependencies. Use an available free tier only; if no suitable GPU or quota remains, defer the experiment. Do not automatically switch to paid resources. Provider quotas, hardware and session duration can change, so record what the session actually supplies rather than promise a device or number of free hours. Confirm current provider rules before an authorized run.

1. Use public source at the pinned commit; do not upload the private development workspace. Install only the approved experiment dependency pins and selected model revision, outside timed inference. Keep account authentication out of notebooks and recorded artifacts.
2. Probe and record the real environment. Run the future correctness checks before timing, choose one available GPU explicitly and validate its model/precision fit. Do not use the free notebook as a permanently hosted service.
3. Execute the [benchmark protocol](benchmarks.md), with matched baselines in the same session where possible. Run short bounded trials, separate profiling, and checkpoint completed artifacts between trials.
4. Write event records, manifests, outputs needed for private parity checks and completion/error markers to session-local storage. Export outside the timed window into operator-controlled persistent **private** storage while the session is alive. Save checksums and verify that the copied files can be reopened; a successful notebook save is not proof that every arbitrary file was retained.
5. Preserve interrupted runs with partial-window/outcome metadata. On a new session, restore source/pins/manifests, re-probe hardware and repeat warmup; never append across sessions as if timing were continuous. Shut down/release the accelerator when finished.

For **Colab**, download the run bundle to local private storage or copy it to an explicitly chosen private persistent location such as Drive, verifying the destination copy. Notebook saving alone does not save arbitrary VM files, and a final export cell cannot protect against abrupt termination. See the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).

For **Kaggle**, write selected artifacts under `/kaggle/working`, use the provider's version/output mechanism, and verify the saved version's output files before downloading a private copy. A `Save & Run All` version executes in a fresh session: re-probe hardware, keep the run bounded and identify it as a new run. Keep notebook/output visibility private; saved outputs do not authorize public evidence. See the [official notebook documentation](https://www.kaggle.com/docs/notebooks). Provider persistence is useful but must be verified; periodic private exports remain necessary.

Loss can occur before the next checkpoint; this procedure reduces loss, not guarantees recovery of unfinished GPU state. A restored artifact bundle proves retained evidence, not resumability of model tensors or an unchanged GPU allocation.

## Publication boundary

Raw logs, prompts/generated text, traces and environment records belong in private storage outside the Git checkout. The ignored `results/` tree is an optional local scratch location, not durable backup or an access-control boundary. Export private artifacts first, then prepare an independently reviewed sanitized copy if publication is requested. Follow the [results policy](../results/README.md): current publication rules do not allow benchmark data files under `results/`, even after sanitization. No model execution, data publication or policy change is authorized by this document.
