# Results policy

**No benchmark results exist yet.** Only separately approved sanitized evidence may be published deliberately. Raw output, retained private evidence and public evidence are different artifacts with different review requirements.

## Current repository restrictions

The current [ignore rules](../.gitignore) exclude `results/**` except this README. The authoritative [publication policy](../.publication/policy.json) allows **only `results/README.md` under `results/`**. Sanitization alone does not make a CSV, JSON, plot or trace publishable here. `git add -f` cannot bypass the publication scanner's allowlist.

Publishing benchmark data later requires a separate approved change that names its formats/paths, updates the necessary ignore exceptions and restrictive publication allowlist, and validates that sanitized evidence. This documentation task changes neither policy nor ignore rules. Existing documentation/workload allowances do not authorize publishing raw runs elsewhere.

## Retain raw evidence privately

Keep per-request events, synthetic prompt/output parity records, run manifests, errors and profiler traces in operator-controlled private storage **outside the public checkout**. Record hashes and completion/incomplete status and verify exported files can be reopened before the notebook session ends. Ignored local `results/` scratch is not a backup or confidentiality guarantee. Avoid full environment-variable dumps and credentials even in retained bundles; capture only required metadata. Follow [reproducibility and retention](../docs/reproducibility.md).

## Prepare a public evidence candidate

When publication is requested, derive a separate sanitized candidate, preserving the raw originals privately. Include the source/model/tokenizer revisions, exact GPU/software/precision and generation configuration, workload digest/seed/length distributions, timing definitions, warmup/repetition/sample counts, outcome/error counts and aggregation method from the [benchmark protocol](../docs/benchmarks.md).

Remove credentials, personal paths/account identifiers, private development context, confidential prompts and unreviewed generated text/notebook output. Inspect traces, embedded metadata and plot labels as well as visible tables. Synthetic prompts are preferable; generated text still needs review. Keep the metadata necessary to interpret measurements and a record of sanitization/analysis transformations. Do not alter measured values, omit failures without disclosure, or fabricate provenance to make a candidate publishable.

A reviewer must check correctness, comparability, missing/incomplete samples, claims and publication safety. Human approval covers the **exact candidate files and proposed publication-policy change**, then the staged/outgoing scanners validate the actual Git content before remote upload. Until those approvals and checks occur, retained or sanitized data remains private and no speedup/capacity claim is established.
