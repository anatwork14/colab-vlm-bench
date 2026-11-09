# Native scoring fidelity audit: MemLens and MMLongBench-Doc (ITER2 D2-3) and hallucination path (D2-4)

Offline audit at HEAD `7833b47` (branch `iter2/p8-pilot-inventory`). No network, no download, no upstream clone (none exists on this host, and the upstream scorers are not vendored). Every "upstream" statement below comes from this repo's docs and is **unverified against the pinned upstream commit**. Missing items are marked BLOCKED. Nothing here is an official score.

## 1. What the repo claims the official metric is

| Source | MemLens | MMLongBench-Doc |
|---|---|---|
| `docs/METRICS.md:7-8` | "publisher LLM judge output, question type breakdown" | "publisher multi-stage scorer, reference answer + format" |
| `docs/audit/NUMERIC_INTEGRITY_LEDGER.md` C1 | "publisher LLM judge, NOT_MEASURED" | "publisher **F1** pipeline, NOT_MEASURED" |
| `README.md:77` | "Official quality uses upstream LLM judge, potentially requiring external resources." | "its official scoring workflow must be run separately, not replaced by proxy F1" |

The sources disagree on the MMLongBench metric (multi-stage scorer versus F1 pipeline). This must be resolved from the pinned upstream code, not from these docs.

## 2. Checklist per component

| # | Component | MemLens | MMLongBench-Doc | Status | Evidence |
|---|---|---|---|---|---|
| 1 | Upstream scorer code pinned and vendored | none vendored; `scripts/checkout_method_sources.py` covers H2O/DA/ECRD only | same | **BLOCKED** (needs owner-approved upstream commit SHA) | `ls` of the repo and `/mnt/ssd` found no MemLens or MMLongBench checkout |
| 2 | Upstream owner identity | `xrenaf/MEMLENS` (GitHub) vs `xiyuRenBill/MEMLENS` (HF) | `mayubo2333` (GitHub) vs `yubo2333` (HF) | **BLOCKED** (owner must confirm authoritative source) | `docs/audit/PINNED_INPUTS.md`, ledger F-13 |
| 3 | Dataset files present locally | absent | absent | **BLOCKED_DATASET_MISSING** (`harness_inventory`, `docs/harness/PILOT_INVENTORY.md`) | config `native_path` points at Drive |
| 4 | Local normalization | `normalize_memlens` (`src/vlmbench/datasets/memlens.py:21`) keeps reference, `question_type`, images; fail-closed on missing image (ledger C1) | `normalize_mmlong` (`src/vlmbench/datasets/mmlong.py:12`) renders every page (`range(page_count)`, line 71, no cap); keeps `answer_format`, `evidence_pages` | PASS (code; tested by `test_datasets.py` and `test_mmlong_render_cache_is_keyed_by_dpi`) | status `NORMALIZED_NEEDS_OFFICIAL_PROMPT_PARITY` (`memlens.py:127`, `mmlong.py:133`) |
| 5 | Prompt parity with the official prompts | adapter builds its own prompt (`memlens.py:93`), provenance "official_prompt_parity_pending" (`memlens.py:109`) | adapter inserts "Document page i/N:" text (`mmlong.py:86`), "official_prompt_parity_pending" (`mmlong.py:117`) | **BLOCKED** (needs upstream prompt text at the pinned commit) | adapter provenance strings |
| 6 | Local proxy scorer | `score_proxy` (`src/vlmbench/quality.py:48`): NFKC token EM and bag-of-words F1, denominator = all references, missing = 0 | same | PASS as **auxiliary** only (`not_official: True`). Not an official score | `test_measurement_regressions.py` (F-01) |
| 7 | Official LLM judge (MemLens) | no judge model, judge prompt, judge revision or judge provenance in code | — | **BLOCKED** (needs judge model + revision + owner approval for any judge calls, and the upstream judge prompt) | `docs/METRICS.md:7` |
| 8 | Official judge or LLM step (MMLongBench) | — | whether the publisher scorer calls an LLM **is not verified offline**. If it does, it needs a judge model + revision + provenance, like item 7 | **BLOCKED_UNVERIFIED_UPSTREAM** | cannot be checked without the pinned upstream code |
| 9 | Format-aware answer matching (MMLongBench) | — | `answer_format` is written to each record (`mmlong.py:113`) and **read by no scorer** (`grep -rn answer_format src tests scripts`) | **BLOCKED** (port or call the upstream matcher) | grep result |
| 10 | Question-type breakdown (MemLens) | `question_type` written (`memlens.py:102`) and **no breakdown computed** (`grep -rn question_type src`) | — | **BLOCKED** (needs an upstream breakdown definition) | grep result |
| 11 | Import of an upstream result | `import_official` (`quality.py:112`) wraps the payload and returns `QUALITY_OFFICIAL_IMPORTED_UNVERIFIED`. It checks only that a judge revision and sample-manifest SHA are non-empty. It does **not** verify scorer parity, denominators or the sample set | same | cannot produce PASS. Needs a separate auditor (`quality.py` docstring line 126) | `quality.py:112-127` |
| 12 | Official score in campaign receipts | — | — | `official_accuracy: "NOT_MEASURED"` constant (`bench.py:387`) | — |
| 13 | Full-context fidelity (no truncation) | config `quality.require_full_native_context` = true; MMLong renders all pages (item 4) | same | PASS in code. The MemLens context-bin fit is not checked here (needs the native file) | `config.validate` |

**Conclusion.** Locally we have normalization, an auxiliary proxy, and an import wrapper that cannot yield PASS. No official MemLens or MMLongBench score can be produced from this repo. The missing parts are the pinned upstream scorer, its prompts, its judge (if any), the owner's judge approval, and the dataset files.

## 3. Owner inputs needed for native scoring

1. Commit SHAs of the official MemLens and MMLongBench-Doc scoring code, and confirmation of the authoritative upstream owner (items 1, 2).
2. The metric definition at that commit, to settle the "judge" versus "F1 pipeline" conflict (section 1).
3. If the scorer needs an LLM judge: which judge model and revision, and owner approval for its cost (items 7, 8). Judge calls are paid, so this needs explicit approval.
4. Dataset files at the pinned revisions (item 3).

## 4. Hallucination evaluation path (D2-4)

- `score_evidence_annotations([])` raises `AnnotationError("No evidence-claim annotations; metrics NOT_MEASURED")` (`src/vlmbench/hallucination.py:31`). The CLI (`src/vlmbench/cli.py:573`) writes no file in that case. **No zero is produced.** LOW: the operator sees a traceback, not an explicit NOT_MEASURED receipt. A receipt would be more auditable. This is a diff request to the CLI owner, not a defect.
- Undefined rates are `None`, never 0.0: `unsupported_claim_rate`, `contradiction_rate`, `hallucinated_claim_rate`, `answer_hallucination_rate`, `false_assertion_rate`, `false_refusal_rate`, `correct_abstention_rate`. Checked with UNVERIFIABLE-only claims.
- `answer_coverage_rate` = 0.0 when every answer abstains. That is a measured count, not a missing value.
- Partial coverage is labelled `PARTIAL_JUDGE_COVERAGE`.
- Guard tests: `tests/test_hallucination_not_measured.py` (5 tests, pass at `7833b47`).
- Still BLOCKED: no independent judge labels exist for any run (the judge is an owner decision, see section 3).

## 5. Verification commands (read-only)

```
grep -rn "answer_format\|question_type" src tests scripts
grep -rn "judge\|official" src/vlmbench/quality.py src/vlmbench/bench.py src/vlmbench/datasets/*.py
ls /mnt/ssd/genos2/research_workspace | head      # no upstream checkout for MemLens or MMLongBench
```
