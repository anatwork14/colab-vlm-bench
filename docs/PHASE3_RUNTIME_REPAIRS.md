# Phase 3 runtime repair contract

Scope: RC-01 through RC-06. CPU development only; no inference or Drive mutation
was used to develop these changes. Independent review and actual source/GPU
qualification remain required.

## Execution safety

Every supported pipeline/CLI GPU entrypoint requires the exact boolean
`owner_approved_compute=True`; CLI invocations record explicit operator intent
with `--approve-paid`. This includes `run-job`, `annotate-serving`, the legacy
shell controller, and both remote dispatchers. The flag is not authentication
or proof that a human personally entered it. Agents must never manufacture paid
approval when the owner has not granted the particular execution scope.

HF intake refuses detected GPUs and failed/timed-out probes. A successful empty
probe permits intake only without contradictory NVIDIA driver/device evidence.
Without `nvidia-smi`, intake requires no NVIDIA/Tegra driver or device indicators
and an explicit `torch.cuda.is_available() is False`. Missing torch, probe errors,
unreadable device indicators or non-boolean CUDA results remain blocked. Registry
labels alone do not override unknown accelerator state. Read-only Drive inventory
remains available.

Host invocations carry unique IDs and download artifacts into
`agent_artifacts/<session>/<invocation_id>/`. Completion requires a fresh
ACTION_COMPLETED result matching invocation, action, repository SHA and bundle
SHA256. Missing, invalid, failed and stale artifacts raise on the host.

## Historical sources

The agent CLI defaults to `--source-mode historical`. Supply `--source-map` with
a reviewed JSON manifest on Drive, or place it at
`<independent-drive-root>/source_maps/<model>__<dataset>.json`.
No source map is generated or qualified automatically.

Required manifest fields:

| Field | Meaning |
|---|---|
| model, dataset | Exact selected registry IDs |
| model_revision, dataset_revision | Reviewed immutable 40-character source pins |
| model_root | Defaults to MyDrive/LLM_Weights/models/Qwen/Qwen3.5-{0.8B,4B} |
| dataset_root | Defaults to MyDrive/Research/Topic/Speculative Decoding/STARLIGHT/<dataset> |
| query_path | Reviewed query JSON inside dataset_root; default MemLens dataset_32k.json or AMBER data/query/query_all.json |
| images_root | Inside dataset_root; default MemLens root or AMBER images/ |
| model_files | Relative file -> SHA256 map covering every checkpoint file |
| dataset_files | Relative file -> SHA256 map covering query and all image-source files |

The sole owner-confirmed model metadata exemption is the exact root-relative
`starlight_checkpoint_inventory.json`. Hidden files, other `starlight_*` names,
nested copies, weights, tokenizer/processor assets and code require hashes. If a
standard checkpoint index references even the allowlisted filename as weights,
it loses its exemption and must be covered by the hash inventory.

Paths must stay inside MyDrive. Output namespaces must be separate from original
sources. Model and dataset originals are read and hashed; receipts, normalized
data and derived records are written only to the independent output namespace.
Changed source mappings or bytes block resume. Partial prepared artifacts are
bound to source revisions before reuse. Model staging retains the existing
verified local-SSD copy path.

For the historical `amber/_archives/AMBER.zip`, a reviewed map may additionally
specify `archive_path` and `archive_sha256`. Set dataset_root to exactly
`<independent-drive-root>/source_cache/amber/<archive_sha256>`, with query/image
paths and per-file hashes matching the archive's actual layout. CPU preparation
hash-verifies and safely stages that archive there; it never unpacks into the
original STARLIGHT tree. Traversal, symlinks and excessive expansion are refused.
The map must describe the real upstream layout/schema; unsupported publisher
schema fails rather than inferring labels, IDs or prompts.

The explicit `--source-mode hf` retains the legacy pinned intake workflow for
MemLens/MMLongBench-Doc; it is not an automatic fallback for a missing historical
source. AMBER requires the reviewed map. Revisions and hashes supplied in a map
do not independently establish publisher provenance: DS-07 stays a separate gate.

## Campaign and serving behavior

AMBER + MemLens is an accepted two-dataset campaign; legacy MemLens +
MMLongBench-Doc configs remain valid. Historical campaign generation selects
the prioritized Qwen model, enables all six CCUs and all seven MemLens bins.
The complete model registry remains available for legacy configs; deferred
models are not automatically selected for these generated jobs.

AMBER serving uses `amber_serving`, one native workload without text bins, and
`input_length=null` in its cell identity. It uses real backend-verified effective
token counts for context capacity. It is never relabeled as a MemLens text bin.
MemLens historical preparation calls the tokenizer-based seven-bin builder;
both sources require bounded GPU annotation before serving execution.

Serving requests require exactly 256 generated tokens and an explicit stream
terminal signal (`[DONE]` or a non-null finish_reason). Quality requests use
natural EOS with max_tokens=256 and preserve finish_reason. Generated quality
predictions remain unscored; AMBER proxy results must not be called official.

One wall-clock deadline covers warmup plus measurement. Offered concurrency is
bounded in both phases. Cancellation/error cleanup awaits outstanding tasks and
preserves completed and cancelled per-request evidence on timeout. Warmups do
not contribute to serving metrics.

CPU tests verify these contracts with fake sources, HTTP transports and samplers.
They do not establish live CPU-session operability, source completeness, actual
vLLM model loading, MTP correctness, GPU capacity or official scorer parity.
