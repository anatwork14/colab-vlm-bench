# P7 Engine audit — 2026-11-09 (base cfccc4b, branch mission/20261109-engine-p7)

Scope: real execution paths live in `src/vlmbench/infrastructure/` (`real_engine.py`, `h2o.py`,
`declarative_attention.py`, `adapters.py`). `src/vlmbench/engine.py` is only the A100 vLLM launch
helper and holds no arm logic. Commands run with `/mnt/ssd/genos2/research_workspace/colab-vlm-bench/.venv/bin/python`,
`PYTHONPATH=src`, `CUDA_VISIBLE_DEVICES=` (CPU only).

## Test counts
- Baseline cfccc4b: 1 failed, 272 passed, 10 skipped.
- After commit 22372dd (tests only): 5 P7 regression tests fail as expected, plus the stale CCU test
  fixed. Engine-file subset: 8 P7 tests + test_c = 9 selected, 5 failed, 4 passed.
- After commit b68cc9f (fixes): full suite 281 passed, 10 skipped, 0 failed.

## Defects (fixed)
| # | Arm/Area | Location | Evidence | Fix |
|---|---|---|---|---|
| 1 | B1 weight load | real_engine.py partial loader | oversized q_proj (128x64 for 64x64) silently sliced and counted mapped (`mapped=4 unmapped=0`) | map projection/visual/MTP tensors only on exact shape |
| 2 | B2 DA geometry | real_engine.py `_da_scripted_tags` | sink_end recomputed from growing KV: 36-token prompt gave 9, 10, 12, 15, 16 across steps | bounds anchored to prefill length |
| 3 | B2 DA parser/mask | declarative_attention.py | window="-4" makes keys unreadable even to self; window="abc" crashes with opaque numpy int() error | non-negative int required, clear ValueError |
| 4 | B3/B4 H2O MTP tx | h2o.py commit | commit(10) after 4 proposed -> seq_len 14 with 4 scored positions | refuse accepted > proposed |
| 5 | B4 synthetic adapter | adapters.py MTP block | commit(1) with no proposal inside transaction (raised after fix 4) | score proposed positions inside transaction |

## Checked, no defect found
- B1: official HF path loads 488/488 tensors with logits parity (existing test passes on CPU).
- B3: zero-eviction below min_budget; protected visual tokens never HH-evicted; protected tokens may exceed budget by design (existing test).
- B4: MTP reference path is token-identical to Base under identical weights (regression test added).
- B5: per-request H2O trackers and ECRD processors; 4 concurrent CPU requests match sequential runs (regression test added; NOT vLLM evidence).

## Qualification boundaries (not fixed; need lead decision or further work)
- DA is applied only to decode steps. Prefill is unmasked, so the first token and prompt KV are unrestricted. Decision needed: mask prefill too?
- DA focus segment is text-position based and is not aligned to actual image-token positions in the official prompt. Remains a scripted diagnostic, not the model-emitted `<da:focus>` protocol.
- DAConfig.enforce_strict_fsm is never read; unmatched closing tags are silently ignored (existing tests expect this tolerance).
- MTP: proposal diagnostic only; zero drafts committed; acceptance 0.0 by design; native verifier gated; the MTP layer attends only to its own position.
- H2O bytes_freed is a sum of per-step tensor deltas, not VRAM saving. Logical fraction is last-step only.
- Partial scaffold maps mtp.fc into mtp_proj (semantically approximate); scaffold only, not official.
- Not verified: whether the hooked forward's attention_mask slice `[..., :k_len]` misaligns after H2O compaction. Reachability on decode is unconfirmed.
- Same request_id reused concurrently would overwrite the H2O tracker (`reset=True`). Not exercised.
- No GPU, no vLLM: nothing here is A100 or serving evidence.
