# Qualification levels for quality numbers (ITER3 D5 / C-11)

Every quality number in this repository carries one of the levels below. A number may never be described with a stronger level
than the one it holds. The level is decided by the code path that produced it, and the status string names that path.

## The levels

| Level | Meaning | Emitted by (status string) | Required before the level applies | May be called |
|---|---|---|---|---|
| **L0 NOT_MEASURED** | no number | `official_accuracy: NOT_MEASURED` (`bench.py`), `NOT_MEASURED` in `hallucination.py:31`, `amber.py` | nothing (this is the default) | "not measured" |
| **L1 PROXY** | an auxiliary metric: NFKC token exact match and bag-of-words F1 over the references. Denominator = every reference; a missing prediction counts as 0. | `QUALITY_PROXY_ONLY` (`quality.py:102`), `not_official: True` (`quality.py:111`) | nothing beyond the predictions and references | "auxiliary proxy", never "accuracy" in a report title, never "official" |
| **L2 PORTED DETERMINISTIC** | a port of the **deterministic** upstream scoring stage (MMLongBench-Doc stage 3: `eval_score`, `eval_acc_and_f1`, `show_results` breakdowns) at a pinned commit, with the deviations listed below. It has no judge. | the port in `src/vlmbench/datasets/mmlong_official_scorer.py`, returning per-stage results; the label `OFFICIAL_SCORED_STAGE3_ONLY` is emitted only with L3 conditions (`mmlong_official_scorer.py:401`) | the upstream file SHA-256s match the pins (verified 2026-11-10, §3); tests: `test_mmlong_fuzz_vs_reference.py`, `test_mmlong_official_scorer.py` | "ported deterministic scoring of stage 3, not official" |
| **L3 OFFICIAL STAGE 3 ONLY** | stage 3 scored by the port, with **stage 2 (the LLM extraction) declared by the caller** and verified against a signed extractor-stage qualification that binds the scorer provenance, the official extractor model (`gpt-4o`), the official extraction prompt hash, the sample manifest and the exact stage-2 outputs (ITER3 C4). | `OFFICIAL_SCORED_STAGE3_ONLY` (`mmlong_official_scorer.py:401`), `OFFICIAL_IMPORTED_QUALIFIED` (`quality.py:152`) | a signed qualification that verifies (`verify_qualification`, `signed_records.verify_signed_record`). The caller-declared stage-2 model is not checked against the run (N-3 limit) | "official stage-3 scoring given a qualified extractor; stage 2 not run here" |
| **L4 OFFICIAL END-TO-END** | the three upstream stages run at the pinned commits: stage 1 model output, stage 2 extraction by the official extractor, stage 3 scoring. | **no label exists yet**: not reachable | all of L3, plus stage 1 and stage 2 receipts for the same samples | "official" |
| **BLOCKED_NO_JUDGE** | an official label was requested and cannot be given. No number is published. | `BLOCKED_NO_JUDGE` (`quality.py:134–150`, `mmlong_official_scorer.py:345`) | — | "blocked: no qualified judge" |
| **J0 HALLUCINATION JUDGE, unscored** | answer and claim labels from an independent judge are absent. Rates are `None`, never 0. | `NOT_MEASURED` (`hallucination.py:31`); `None` rates (`docs/harness/NATIVE_SCORING_AUDIT.md` §4) | — | "no judge labels" |
| **J1 HALLUCINATION JUDGE, partial** | judge labels cover some answers. Rates computed on the covered subset only. | `PARTIAL_JUDGE_COVERAGE` (`hallucination.py:108`) | the covered subset is reported with its count | "partial judge coverage" |
| **J2 HALLUCINATION JUDGE, scored over the frozen population** | labels cover every expected answer of the frozen population. | `EVIDENCE_JUDGE_SCORED` (`hallucination.py:107`) | the judge model, its revision and prompts pinned and owner-approved (`docs/harness/NATIVE_SCORING_AUDIT.md` §3, item 3) | "judge-scored with judge X at revision Y" |
| **J?** `JUDGE_SCORED_POPULATION_UNVERIFIED` | labels exist, but the population is not frozen, so coverage cannot be checked | `hallucination.py:106` | — | "judge labels, population unverified" |

Rules that apply to every level:

1. A level is never raised by a status string alone. The receipt and its inputs must be checked (see the conditions column).
2. A proxy (L1) is never imported as official. `quality.import_official` returns `BLOCKED_NO_JUDGE` for MemLens unconditionally, and for
   MMLongBench-Doc unless the qualification verifies (ITER3 C4).
3. A missing value is `None` or `NOT_MEASURED`, never `0`.
4. Derived workloads (`official_quality_allowed: False`) cannot reach L3 or L4 (`src/vlmbench/derivation.py`).

## Current status per dataset (2026-11-10)

| Dataset / metric | Level reached | Why not higher |
|---|---|---|
| MemLens, auxiliary proxy | **L1** | no ported scorer; LLM judge unpinned (`DATASET_READINESS.md` §2.3) |
| MemLens official | **BLOCKED_NO_JUDGE** | judge not ported, not pinned, not approved |
| MMLongBench-Doc, stage 3 port | **L2** on fixtures | not run against real upstream outputs; no real rows on this host (`DRIVE_INVENTORY.md`) |
| MMLongBench-Doc, official stage 3 | **BLOCKED_NO_JUDGE** | no signed extractor qualification; stage 2 (GPT-4o) not run; the paid service is not authorised |
| MMLongBench-Doc, L4 | **not reachable** | stage 1 and stage 2 receipts do not exist |
| Hallucination metrics | **J0** | no independent judge labels for any run |

## Differential status against the pinned upstream (D5, 2026-11-10)

Upstream commit `d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384`, `eval/eval_score.py` SHA-256 `cafa2683…`, `eval/extract_answer.py`
`fa6a747b…`, `eval/prompt_for_answer_extraction.md` `96a22c30…`. Copies exist in earlier scratch folders. **All three hashes match the pins.**

What was done:
- **Code reading.** The full line-by-line review of `eval_score.py` and `extract_answer.py` against the port was done in the ITER2
  review (`REVIEW_ITER2B.md` §3). This round re-read the `eval` call sites (upstream lines 128–146, 172–182) and checked the port's
  `ast.literal_eval` sites against them. It did not re-read the whole file.
- **Differential fuzz of the port against a transcription** (`tests/mmlong_reference_transcription.py`, which copies the upstream logic
  line for line with the documented deviations). This shows the port matches the transcription, **not** the upstream.
- **Hand-checked cases** in `tests/test_mmlong_fuzz_vs_reference.py`, which check the expected scores by hand.

What was **not** done, and why:
- **The upstream file was not executed.** Upstream `eval_score.py` calls `eval()` on dataset and model strings (lines 138, 142, 178, 179),
  and `show_results` writes to a caller-given path. Executing it on model-produced text runs whatever code those strings contain. The brief
  allows differential execution only in a safe, isolated, no-network scratch, after the owner reviews the upstream code, and only on inputs the
  owner approves. The differential therefore stays at code reading and the transcription.

**Round B update (2026-11-10, C-11):** the three upstream copies in the session scratch folders still match the pins (SHA-256 re-checked:
`eval_score.py` `cafa2683…`, `extract_answer.py` `fa6a747b…`, `prompt_for_answer_extraction.md` `96a22c30…`). **Correction to the line
above:** an isolated no-network namespace **is** available on this host (`unshare -rn` runs, with loopback only). The sandbox is not the
blocker. The execution gates that are still unmet are (1) the owner's review of the upstream code and (2) the owner's approval of the
inputs. Until both are recorded, the upstream file is not executed. The differential is therefore **not completed**; the status stays at
code reading, the transcription fuzz and the hand-checked cases.

Documented deviations of the port from upstream (all in the port's header and the transcription's header):

| # | Upstream | Port | Effect |
|---|---|---|---|
| D-1 | `eval(gt)` and `eval(pred)` on strings starting with `[`; `eval` on `evidence_pages` and `evidence_sources` | `ast.literal_eval` | identical for literal list strings (what MMLong gold contains). **Different for non-literal strings**: `"[2*3]"` gives `[6]` upstream and raises in the port. Not reached by MMLong gold; not silent |
| D-2 | `print(...)` debugging and file writing in `show_results` | removed; results returned as a dict | no effect on scores |
| D-3 | bare `except:` in `is_float_equal`, `eval_score`, `eval_acc_and_f1` | `except Exception:` (ITER2 C-8) | differs only for `KeyboardInterrupt` and `SystemExit` |

## What would move a level

- **L2 to L3:** a signed extractor-stage qualification for real stage-2 outputs, and the owner's approval of the GPT-4o stage (a paid service).
- **L3 to L4:** stage 1 model outputs for the same samples, with receipts, and the same stage-2 qualification.
- **Any level to a claim about the official score:** the level must be L3 or L4 and the receipt must be cited. A proxy number, or a number from
  a derived workload, never supports an official claim.
- **Differential execution:** only in an isolated no-network scratch, after review of the upstream code by the owner, and only on inputs the
  owner approves. Until then, D-1 stays a documented difference.

## Not closed

- No real MMLong or MemLens rows have been scored on this host.
- The stage-2 caller declaration (N-3) is still a limit: the scorer cannot check that the extraction ran.
- The hallucination judge is unpinned, and no independent judge labels exist.
