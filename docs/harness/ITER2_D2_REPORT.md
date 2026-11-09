# ITER2 D2 report (wF:p8, Benchmark)

Base commit 7d209f5. Last D2 commit 6f3271b. Branch `iter2/p8-pilot-inventory`.

## Commits
| SHA | Item | Summary |
|---|---|---|
| bd09225 | D2-1 | `scripts/harness_inventory.py` + 29 tests: 48 pilot and 3528 full-matrix cells, READY/BLOCKED_*/UNSUPPORTED/NOT_TESTED |
| 1c2bbd8 | D2-2 | Receipt search: no Base/MTP pilot receipts exist (`docs/harness/PILOT_RECEIPT_SEARCH.md`) |
| 7833b47 | D2-4 | 5 guard tests: empty/undefined/partial/all-abstained hallucination data is never zero |
| e3cfe1c | D2-3 | Offline native-scoring audit, MemLens and MMLongBench-Doc (`NATIVE_SCORING_AUDIT.md`) |
| 6f3271b | D2-5 | PyTorch capability probe on GTX 1080 Ti, labelled PyTorch reference (`PYTORCH_REFERENCE_PROBE.md`) |

## Tests
- tests/test_harness_inventory.py: 29 passed.
- tests/test_hallucination_not_measured.py: 5 passed.
- Full suite at 6f3271b: NOT COMPLETED. The run was stopped at about 44% on the lead's instruction, to free CPU under host load (~18 on 8 cores). No full-suite result exists for D2.

## Inventory status (docs/harness/PILOT_INVENTORY.json, schema harness_inventory/1, head 7d209f5)
Status is the first failing check in the ordered list. Each cell's full blocker set is in `blocker_counts`.

| Matrix | Cells | READY | UNSUPPORTED | BLOCKED_REVISION_UNPINNED | BLOCKED_WEIGHTS_MISSING | NOT_TESTED |
|---|---|---|---|---|---|---|
| pilot | 48 | 0 | 0 | 24 | 24 | 0 |
| full | 3528 | 0 | 2520 | 504 | 504 | 0 |

Environment facts behind the blockers: Qwen3.5-0.8B weights present with matching sizes and SHA-256; Qwen3.5-4B absent; both revisions UNPINNED; MemLens and MMLongBench-Doc datasets absent; vLLM not installed; owner authorization file absent; GPU 1080 Ti, compute capability 6.1, below the assumed required 8.0.

## Open items
- No READY cell is possible until owner inputs exist: dataset revision SHAs, 4B pin, vLLM pin, A100 authorization.
