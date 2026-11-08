# Standalone-upload Colab notebooks — seven-arm campaign

Each notebook now includes **7 independent methods**, not just Base/MTP: Base, H2O+Base, Declarative Attention+Base, ECRD Grounding+Base, native MTP, H2O+MTP, DA+MTP. Each .ipynb has 38 cells with a method planner, conditional dispatcher, temperature=0/top_p=1 checks and grounded-hallucination scoring from independent judge annotations. Five external-method arms are explicitly BLOCKED until real Qwen3.5+vLLM+A100 implementations are qualified. The notebooks DO NOT fake method results by reusing Base outputs.

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

## Cost-aware cell audit for 80 GiB RAM / 200 GiB local SSD

A100 VRAM is separate from system RAM. Both notebooks now have **38 cells**, including a CPU-only RAM/SSD/model/dataset source gate before checkpoint staging and PDF rendering. To avoid starting paid A100 time for path issues, open Colab using a CPU runtime, set `CPU_PREFLIGHT_ONLY=True`, run configuration, Drive mount, setup and the resource preflight cell, then STOP. Once `CPU_PREFLIGHT_PASS` is reported, switch to the authorized A100 40GB runtime, set `CPU_PREFLIGHT_ONLY=False` and rerun setup cells. The preflight reads only the Parquet schema / a few QA rows and sampled PDF existence, not all images, and is not a GPU readiness guarantee.

The GPU annotation stage uses `ANNOTATION_CALL_BUDGET=8` **new one-token requests per dataset per invocation** by default, with deterministic state-checked resume on subsequent runs. A valid CCU32 serving cell requires 160 **unique model-token-verified** prompts, so increase the annotation budget only after a small smoke PASS and review the preflight eligibility counts. The runner now returns `BLOCKED_NO_ELIGIBLE_CELLS` **before loading the model or launching vLLM** when no serving cell can run. CPU-only `serving_readiness` counts verified samples in streaming JSONL, not all historical prompts in system RAM.

Pilot PDF QA normalization defaults to at most 200 questions; it is an **engineering subset**, not official full-benchmark quality. Full phase must use the entire original QA population. The RAM/SSD gate verifies actual free resources; it cannot know Google Drive remaining quota or guarantee A100 KV capacity at 16K×CCU32. Native quality uses full evidence and refuses if a question exceeds the configured max images rather than dropping pages. All measured results remain subject to actual A100 qualification.
