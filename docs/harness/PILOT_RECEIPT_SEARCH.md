# Search for real Base/MTP pilot receipts (ITER2 D2-2)

Read-only search at HEAD `bd09225` (branch `iter2/p8-pilot-inventory`, base `7d209f5`). Nothing was copied into reports or code.

**Result: no Base/MTP serving receipt exists anywhere** (repository, all refs, or the local research workspace). Nothing in this search is a GPU serving measurement.

## What was searched, and what it found

| Search | Command (read-only) | Result |
|---|---|---|
| Artifact directories | `find artifacts` (the `agent_artifacts/` directory does not exist) | 4 files: 3 JSON, 1 Markdown |
| Added files, all refs | `git log --all --diff-filter=A --name-only` filtered for `cells/`, `requests/`, `run_status`, `summary.json`, `RUNBOOK.md`, `receipt`, `pilot` | none |
| Numeric serving metrics in data files, all refs | `git grep -lE '"(output_tokens_per_s|ttft_p95_ms)"\s*:\s*[0-9]'` on every branch and remote ref, excluding `src/` and `tests/` | none |
| `PASS_MEASURED` in committed files, all refs | `git grep -l PASS_MEASURED` on each ref | 9–20 files per ref, all source or test code that uses the status string. No data file |
| Branch `feat/priority-qwen08b-4b-first-results` (and `origin/` copy) | `git diff --stat main...<branch>`, `git log` | empty diff against `main`; tip `4d629a5` has code and tests only |
| Local research workspace | `grep` for `a100-qwen08b-4b-pilot-v1` and `a100-qwen-priority`; `grep` for `"status": "PASS_MEASURED"` in `*.json` (excluding `.venv`, `.git`, `models/`, `recovery/`) | the campaign ID appears only in configs, docs, runbooks and notebooks. No receipt JSON |
| Legacy notebook outputs | parsed `notebook/Qwen3.5-0.8B.ipynb` (40 cells) for TTFT, tok/s, e2e, PASS_MEASURED | 0 metric-bearing outputs |
| Pytest temporary directories | `find /tmp/pytest-of-genos2 -name run_status.json` | 0. These are test fixtures and are not evidence in any case |

## The receipts that do exist (what they are, and whether they are real)

- `artifacts/model_coverage_and_parity_receipt.json` (timestamp `2026-11-09 08:17:37 UTC`, device `cuda`): a real local forward check on the 0.8B weights. It reports 488/488 tensors mapped in the official Transformers hooked model, and logits parity with max absolute difference `4.77e-06` against tolerance `1e-4`. It contains **no** serving, TTFT, throughput or memory fields. Its own label for the custom scaffold is `PARTIALLY_WEIGHTED_PYTORCH_REFERENCE`.
- `artifacts/qwen35_pretrained_qualification_receipt.json`: a supersession notice. Verdict `SUPERSEDED_NO_POST_REPAIR_PRETRAINED_GPU_RECEIPT`. `NATIVE_VLLM_A100` is BLOCKED and `PHYSICAL_GPU_VRAM_SAVING` is NOT_MEASURED. It lists five earlier results as not publishable: forced MTP acceptance 0.4/0.5, hand-fitted DA 0.1895, H2O byte sums mislabelled as VRAM, Base including MTP-head overhead, and keyword heuristics mislabelled as accuracy.
- `artifacts/historical/qwen35_pretrained_qualification_receipt_20261109_before_scientific_repairs.json` (`08:10:23 UTC`, `cuda_device_name` GTX 1080 Ti): an earlier real local run, superseded by the file above. It is a forward and parity check, not a benchmark.

## Conclusion

- Real measurements exist only for pretrained forward and logits parity (HF, 0.8B). These support no serving, throughput, latency or memory claim.
- No pilot receipt, Base or MTP, exists. There is nothing to report as a pilot result, and no number may be quoted from the old qualification receipts as a benchmark.
- The first real pilot result requires an owner-authorized GPU run. `scripts/harness_inventory.py` (`bd09225`) lists the preconditions.
