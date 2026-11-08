# Standalone-upload Colab notebooks — seven-arm campaign

Each notebook now includes **7 independent methods**, not just Base/MTP: Base, H2O+Base, Declarative Attention+Base, ECRD Grounding+Base, native MTP, H2O+MTP, DA+MTP. Each .ipynb has 36 cells with a method planner, conditional dispatcher, temperature=0/top_p=1 checks and grounded-hallucination scoring from independent judge annotations. Five external-method arms are explicitly BLOCKED until real Qwen3.5+vLLM+A100 implementations are qualified. The notebooks DO NOT fake method results by reusing Base outputs.

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
