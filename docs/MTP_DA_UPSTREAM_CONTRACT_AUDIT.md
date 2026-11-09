# Source audit: Qwen3.5-0.8B MTP and Declarative Attention

**Date:** 2026-11-09  
**Base PR:** #20 at `a5dff1ded497d0760b9c94ae896965b4090439d0`  
**Scope:** Qwen/Qwen3.5-0.8B pinned checkpoint `2fc06364715b967f1860aea9cf38778875588b17`, PyTorch reference engine vs official vLLM/DA.

## Qwen MTP — what is real versus invented

- The original Qwen3.5-0.8B safetensors index carries real `mtp.*` tensors. The checkpoint's **15 MTP keys** are one native MTP prediction-head module (normalizations, concat projection, one decoder layer, final normalization), **not 15 draft steps**. Our full-checkpoint inventory reports 488 keys, 15 MTP and 473 base VL; MTP contains 20,452,864 parameters. This inventory must be identified with its pinned revision.
- Qwen's own serving recipe recommends vLLM `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`. A draft-step count is a **runtime setting**, not a checkpoint key count. Validate this method and version combination on the actual vLLM installation before enabling it.
- The MTP head uses verifier hidden state at position t plus target-selected next-token embedding at t+1 to propose token t+2. At temperature zero, only the target can authorize the next committed token; genuine speculative decoding must be token-equivalent to the same greedy target distribution.
- The **prior 20–25% “acceptance”** was NOT an upstream statistic. The obsolete reference engine used `(step % 2 == 0) or (draft == verifier_after_draft)`, which forced acceptance and compared different positions. These numbers are **RETRACTED**.
- This patch replaces that unsound path with **TARGET_GREEDY_MATCH_ONLY_NO_SPECULATIVE_SERVING**. MTP proposals are compared with the target's next **ordinary** greedy prediction at the correct position; no draft enters KV cache or committed output. Record `proposal_match_rate` separately. `mtp_acceptance_rate=0` reflects **zero actual speculative commits**, NOT zero proposal matches. There is no speculative speedup and no valid production MTP acceptance estimate yet.
- **Fail-closed:** `MTP_CORRECTNESS_PASS=GATED_NATIVE_MTP_VERIFIER_REQUIRED`. Only a separately qualified vLLM native MTP verifier with correct acceptance/residual/rollback and committed-token parity may re-open that gate. All 3 MTP arms remain nonbenchmarkable as speculative *acceleration* until then.

Official references:
- https://huggingface.co/Qwen/Qwen3.5-0.8B
- https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/model.safetensors.index.json
- https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp/
- https://github.com/vllm-project/vllm/tree/main/tests/v1/e2e/spec_decode/mtp/qwen3_5

## Declarative Attention — actual paper contract

- Original paper **Language Models Can Control Their Own Attention**, https://arxiv.org/abs/2609.02737.
- Original public implementation: https://github.com/itsnamgyu/declarative-attention ; see `docs/DESIGN.md`.
- The model **generates** its own `<global>`, `<focus magic_chunks="1">` and `<local>` tags. Focus names pre-segmented addressable document spans; local excludes document context but retains sink, question and generated tail. The FSM applies a recognized transition **on the next forward**, based on real generated text. The original repository uses sink 16, one-based segment IDs, up to three focus IDs, and vLLM block-table metadata compaction.
- Qwen3.5-0.8B has NOT been benchmarked by the paper. Its reported 52.0% (Gemma-4-31B) and 31.1% (Qwen-3.6-27B) reductions are **specific to the paper's models and setup**; neither is an expected fixed saving on our small Qwen VLM.
- The old `0.1895` was **deliberately targeted by code**, not measured or prescribed by the paper. It is RETRACTED. The replacement makes a geometry-derived scripted span mask **for regression diagnostics only**, with source label `SCRIPTED_SEGMENT_DIAGNOSTIC_NOT_MODEL_DECLARED`.
- The current custom DAFSM recognizes `<da:...>` and uses a dense attention matmul followed by masking; it is **not** paper-faithful `<focus magic_chunks>` model-declared DA and does not implement actual KV block-read savings. Report only `attention_mask_restriction_ratio`; never call it HBM saving, KV read saving, or true DA inference speedup. Physical KV residency remains 100%.
- For production DA, port/adapt the upstream FSM/renderer/layouts and block-table scratch-gather path to Qwen3.5 and a validated target GPU profile. Keep native backend gate closed until GPU device-versus-mirror read audit and no-DA parity pass.

## Status and downstream benchmark policy

`SEVEN_ARM_PRETRAINED_PASS` is **pretrained model/ViT forward coverage only**; it does NOT imply MTP speculative verifier or paper-faithful DA. Any earlier readiness receipt with MTP acceptance 20–25%, 18.95% asserted as externally observed, or MTP rollback PASS derived from parity-forced acceptance is historical and **superseded**.

- Base/H2O/ECRD: retain only their validated scope; do not infer vLLM correctness.
- MTP/H2O+MTP/DA+MTP: proposal diagnostic only; keep speculative serving and benchmarking gated.
- DA/DA+MTP: scripted mask diagnostic only; keep model-declared DA and real KV reads gated.
- Never label artificial policy targets as measured findings.
- No Colab A100 authorization is implied by this audit.

## Additional 2026-11-09 numeric integrity repair

- **Baseline MTP overhead (P0):** Former official Qwen forward computed its extra MTP head for every Base, H2O, DA and ECRD step. This contaminated baseline latency with draft-head compute. Now only the three MTP-designated reference arms invoke the MTP module; standard Base does not.
- **MTP positional / cache limitation (P0):** The former MTP proposal rebuilt 3-axis rotary positions from zero on every decode step. Corrected to pass each next-token absolute position into the one-step MTP head. However, **this reference head lacks persistent drafter KV state**. Its `proposal_match_rate` is explicitly nonpublishable and must not be used to claim native MTP acceptance or speedup. Actual vLLM native MTP remains gated.
- **DA measured denominator (P0):** The former `attention_mask_restriction_ratio` used density of the entire causal LxL mask even though only the final query row executes at decode. It now reports the last-row logical restriction ratio and preserves the full-causal-area diagnostic separately. The scripted visual focus is kept active to avoid removing image evidence. Neither metric means HBM reduction.
- **H2O KV bytes (P0):** Legacy `physical_kv_bytes_freed` sums KV tensor size deltas over decode iterations; it does not measure CUDA allocator/free or peak VRAM reductions. The new `kv_memory_accounting` marks this as `SUM_OF_KV_TENSOR_FOOTPRINT_DELTAS_NOT_GPU_VRAM_SAVING`, leaves actual peak VRAM savings null, and reports the final KV tensor payload as a separate quantity. Do not cite 638,976 B or 811,008 B as device memory savings.
- **Correctness:** Per-layer visual masks are separately tracked and compacted; the global original visual mask is not mutated by each full-attention layer. Add multi-layer divergent-index and MTP rollback tests.

## 2026-11-09 multimodal quality score invalidation

Historical Test D four canonical image tasks used substring heuristics instead of objective scoring. In particular, the couch attribute fixture lists `ground_truth_label=Yellow`, but the old rule accepted any of `pink`, `yellow`, `color`, `pattern`, or `couch`. This could PASS a wrong answer. **The 4/4 score is not accuracy, F1, or grounded correctness.** The current Test D only checks that four pretrained ViT/image requests run, records output, and sets `ground_truth_match=null` / `quality_score_qualified=false`. Real image quality awaits AMBER's official scorer with independent labels.

The old artifact `artifacts/qwen35_pretrained_qualification_receipt.json` was based on the invalid forced MTP acceptance, hand-fit DA 0.1895, H2O tensor-footprint deltas miscalled GPU frees, baseline draft-head overhead and permissive four-task substring checks. It has been preserved byte-for-byte under `artifacts/historical/` and replaced at the active path with an explicit supersession notice. **No post-fix pretrained GPU numbers have been rerun or forged.**
