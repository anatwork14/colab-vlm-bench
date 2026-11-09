# Standalone-upload Colab notebooks — seven-arm campaign

Each notebook now includes **7 independent methods**, not just Base/MTP: Base, H2O+Base, Declarative Attention+Base, ECRD Grounding+Base, native MTP, H2O+MTP, DA+MTP. Both notebooks have 38 cells; the 0.8B Colab launch badge lives inside the introduction Markdown cell with a method planner, conditional dispatcher, temperature=0/top_p=1 checks and grounded-hallucination scoring from independent judge annotations. Five external-method arms are explicitly BLOCKED until real Qwen3.5+vLLM+A100 implementations are qualified. The notebooks DO NOT fake method results by reusing Base outputs.

Methods manifest: `configs/a100_qwen_7methods_pilot.json` (168 planned cells, 48 native-routable) or `configs/a100_qwen_7methods_full.json` (3,528 planned cells, 1,008 native-routable). Raw native quality uses natural EOS; derived serving uses 256 fixed output tokens. Official quality and hallucination require external evaluation.

Set `RUN_METHOD_MATRIX=True` only after you understand the gates. `METHOD_SELECTION` lists all seven; Base/MTP can run through the existing `run-job`. Do not also enable `RUN_BASE` or `RUN_MTP` if dispatching the same native cells. Five external methods log `BACKEND_UNQUALIFIED` with no measured GPU result. To score hallucination, put audited claim-annotation JSONL files under the configured Drive judge folder and set `RUN_HALLUCINATION_SCORING=True`.


Download one of the two **.ipynb** files and upload to **Google Colab → File → Upload notebook**. They install and clone the public repository at runtime, so you do not need to upload helper Python files.

| Notebook | Model | Experiments |
|---|---|---|
| [Qwen3.5-0.8B.ipynb](Qwen3.5-0.8B.ipynb) | Qwen/Qwen3.5-0.8B | Base + native MTP |
| [Qwen3.5-4B.ipynb](Qwen3.5-4B.ipynb) | Qwen/Qwen3.5-4B | Base + native MTP |

## Run order

1. In Colab select **Runtime → Change runtime type → A100 GPU** (40GB) and open the model-specific notebook.
2. **Edit the paths cell** to your already-downloaded Drive weight directory, MemLens `dataset_32k.json` and `release_images/`, MMLongBench QA JSON, and PDF directory. Do not hardcode credentials.
3. Run setup, Drive mount, pip install, preflight and checkpoint staging. If torch/vLLM changes trigger a kernel import error, restart the runtime and rerun from the setup cell.
4. Prepare native datasets or use existing normalized JSONL. Generate tokenizer-calibrated source-derived serving candidates.
5. Start **Base** vLLM, confirm a real image prompt works, annotate each dataset through actual vLLM usage.prompt_tokens, then shut Base server down.
6. Run serving **Base then MTP** (each mode automatically launches and terminates its own server), initially 6 cells/job. MTP must show its method in the server startup logs. Preserve unsupported cells as failures.
7. Optionally run native full-context quality prediction subsets, proxy F1/EM strictly labeled **not official**, and export runbooks to Drive / download ZIP.

**Important:** Each notebook makes its own reproducible campaign ID, preserving paired Base/MTP comparisons within the same model. The priority pilot combined across two notebooks has **48 planned serving cells**. More than 6 cells/job requires repeat `run-job` invocations until complete.

Quality can fail on A100 when native 32K context plus images exceeds 40GB. Do not truncate and label the result official. The official MemLens LLM judge and MMLongBench scoring are separate upstream workflows. No test run is bundled or claimed; these notebooks have been syntax/contract-validated in CPU CI only.

Source repositories may be fetched and verified outside the Git working tree using `python scripts/checkout_method_sources.py --method all --dest /content/vlmbench/upstream_sources`. This only fetches source code and verifies SHAs/licenses; it is NOT a runtime port.

## RAM-safe MemLens normalization (Qwen3.5-0.8B fix)

After Colab saved commit `1b8542d`, the notebook included a recursive whole-dataset role rewriting step. That step read all 789 examples twice and created `/content/patched_dataset_32k.json`. The replacement implementation streams **one** MemLens question at a time with `ijson`, handles 'USER'/'AI Assistant' role labels inline, and writes normalized JSONL atomically, with no giant temporary JSON or in-memory list. It also reads source-derived prompts using a sorted index of JSONL byte offsets (not `load_jsonl` for the entire native dataset). These are RAM bounds, not measured peak-memory numbers.

For MMLongBench-Doc, the adapter now supports publisher QA `.parquet` via PyArrow small batches as well as JSON, while PDF images remain on disk. Normalizing full documents may consume substantial SSD/Drive space and can exceed the A100 inference context window even if preprocessing succeeds. Dataset-specific publisher prompt/scorer parity remains unverified.

If rerunning an existing Colab session, execute the setup cell to fast-forward the cloned repo to current `main` before retrying normalization. Upload the latest `.ipynb` from GitHub (old Colab tabs do not auto-update). Never commit executed notebook outputs, raw dataset answers, tokens or private runtime diagnostics into GitHub.

For a full code-cell-by-code-cell review and RAM/SSD thresholds, see [COLAB_STEPWISE_AUDIT.md](../docs/COLAB_STEPWISE_AUDIT.md). This checklist distinguishes CPU test PASS from still-unverified A100 capacity.

## Recovered A100 startup — first-image smoke blocked by corrupt Drive JSONL

If vLLM starts normally and reports `GET /v1/models 200 OK` but the first smoke request fails with `JSONDecodeError: Unterminated string` in `load_jsonl(CANDIDATES[first_ds], ...)`, the problem is **invalid candidate JSONL on Drive**, not CUDA or vLLM. Old Step 6 checked only `Path.exists()` and silently reused damaged files.

The new Step 6 uses `prepare_serving_candidates` and `validate_serving_candidates`: it streams **every JSONL line on CPU**, checks unique sample IDs, text bins and image/text content structure. If the candidate is invalid and no annotation/state exists, it rebuilds from the existing normalized dataset **on local SSD first**, validates, copies the original damaged file to a SHA-tagged `.corrupt-<sha>.jsonl` backup on Drive, verifies the new Drive copy's SHA, and prints a receipt. If annotation/state already exists, it **refuses** automatic regeneration to preserve scientific provenance. The paid Step 8 repeats this validation **before** calling `start_server`.

To use this repair in an existing Colab runtime, first run Step 3 so `/content/colab-vlm-bench` checks out the latest fix-branch SHA, then rerun Step 6. Do not rerun costly model-weight staging or normalization if their outputs are intact. Confirm `REBUILT_CORRUPT_CANDIDATES` or `REUSED_VALIDATED_CANDIDATES` for each active dataset, followed by `PRE_GPU_CANDIDATE_PASS` at Step 8. The next failure, if any, will be a real VLM image-request/server issue. Actual A100 image-request success remains unproven until that run.

## Qwen3.5-0.8B bounded A100 Base pilot (branch fix)

The Qwen3.5-0.8B branch notebook is now configured for a **paid first Base pilot when Run All is invoked on an allocated A100**. Step 1 explicitly prints `RUN_GPU_ANNOTATION=True`, `RUN_BASE=True`, `ANNOTATION_CALL_BUDGET=24`, and `MAX_CELLS_PER_JOB=1`. This is at most 24 **new** one-token annotation requests per active dataset, plus one smoke prompt, then at most one ready Base serving cell per dataset. MTP/full quality/external methods stay disabled. Switch both `RUN_GPU_ANNOTATION` and `RUN_BASE` to `False` for preprocessing-only runs. GPU compute units can also be billed during CPU-heavy preparation if the A100 is already allocated.

**Runtime source correctness:** While this notebook repair is in PR #18, the 0.8B setup cell explicitly fetches the fix branch into `/content/colab-vlm-bench` and prints its SHA. Prior notebooks cloned `main` even when opened from a PR branch, silently missing all PR-only runtime patches. Once merged, switch to an immutable release SHA or `main` as appropriate and record the actual SHA in the campaign results.

The shared `run-job` runner now **executes only ready cells**. A cell needing more unique annotated prompts stays pending rather than writing an `INSUFFICIENT_WORKLOAD` failure that could block future measurement. This applies to the agent-operated Colab CLI as well. If Step 9 has no ready bin, the notebook raises an actionable error rather than silently exporting a `NOT_RUN` report. If vLLM fails startup or an image prompt is rejected, Step 8 prints the last 80 vLLM log lines; that is the next runtime evidence required. No successful actual A100 model execution is claimed by this software change.

## Troubleshooting: preflight succeeds but the notebook says `NOT_RUN`

This was the old safe-preparation default, not evidence of successful inference. The 0.8B branch now enables a capped paid Base pilot; the 4B notebook still defaults to preparation-only. `RUN_MTP`, `RUN_METHOD_MATRIX`, and `RUN_NATIVE_QUALITY` remain disabled in the 0.8B pilot. After a successful CPU/SSD gate, weight staging, and normalization, the candidate JSONL still has **no vLLM-verified prompt-token counts**. Step 9 therefore reports `BLOCKED_NO_ANNOTATED_DATASET`; Step 10/11 skip inference when their switches are off.

To obtain a real measurement, verify the new 0.8B Step 1 has `RUN_GPU_ANNOTATION=True` and re-execute that configuration cell. Run Step 8 with the new bound **24 new requests per active dataset** and inspect the recorded `verified_total` and errors. On later executions the file resumes from prior verified rows; repeat only deliberately after checking the receipts. Run Step 9 again: it must show at least one `READY_CELLS_PRESENT` dataset and some eligible cells **before** enabling `RUN_BASE=True` and running Step 10. For a pilot, test Base before MTP and never enable all seven arms expecting unsupported H₂O/DA/ECRD variants to run. Raise annotation budget only with explicit compute approval and clear token-capacity evidence. Missing unique prompts at higher CCU cannot be fixed by GPU retries; leave such cells blocked or expand a source-qualified workload.

The November 9 notebook repair removes a duplicate CPU guard, an extra Colab-badge cell, and saved execution counts/outputs introduced by the `cannot run` commit. Saved outputs fail `tests/test_notebooks.py` and can reveal local Drive paths; keep receipts on Drive rather than committing executed notebooks. **GitHub notebook checks do not demonstrate actual A100/vLLM runtime success.**
