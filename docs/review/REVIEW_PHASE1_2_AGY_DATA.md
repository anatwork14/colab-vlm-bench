# Review: agy-worker1 data slice, `3fd18a2`

| Field | Value |
|---|---|
| Task | TASK-STARLIGHT-REVIEW-PHASE1-2 |
| Reviewer | argon (independent; not an author of this code) |
| Target | `wt/agy-worker-data`, branch `worker/agy-data-amber-memlens`, commit `3fd18a2` on top of `e83a3c8` |
| Criteria | A1–A8 and B1–B7 in `BASELINE_AUDIT_PHASE0_1.md` |
| Method | Read the full diff; whitespace-insensitive diff of the moved code; ran the test suite; ran targeted reproductions (below). No code changed. |

## Verdict: **BLOCKED**. Do not integrate.

| Part | Verdict |
|---|---|
| Package split (`datasets/` with `memlens.py`, `mmlong.py`, `__init__.py`) | **PASS**. The moved `normalize_memlens` and `normalize_mmlong` differ only in formatting (checked with `diff -w`). |
| AMBER adapter (`normalize_amber`) | **BLOCKED** (DS-02, DS-05, DS-06, DS-07) |
| AMBER scorer (`score_amber*`) | **BLOCKED** (DS-01, DS-02, DS-03) |
| MemLens 7-bin builder (`build_memlens_context_bins`) | **BLOCKED** (DS-04); P2 items DS-08 and DS-09 are not blocking |
| Tests | 22/22 new tests pass. Full suite: 88 passed, 1 failed (the known `nvidia-smi` host test RV-12, fixed separately by sol-lead's slice), 10 skipped. The new tests use only a synthetic schema, so they can't detect DS-01 or DS-02. |

## Reproductions

Run against `3fd18a2` with `PYTHONPATH=src`:

| # | Input | Result |
|---|---|---|
| 1 | Discriminative ground truth stored under `truth` instead of `label`, with perfect answers | accuracy 0.5, F1 0.0 |
| 2 | 3 ground-truth items, 1 answered | count 1, accuracy 1.0 (the 2 missing items disappear) |
| 3 | 1 correct answer + 1 unparseable | accuracy 1.0 (unparseable answers left out of the denominator) |
| 4 | Item with no `type` field | counted under `existence` |
| 5 | Generative ground truth stored under `truth` | CHAIR_i 1.0 for a fully correct caption |
| 6 | Caption hallucinates "traffic light" and "unicorn" | CHAIR_i 0.0 (the multi-word entry never matches; unicorn isn't in the vocabulary) |
| 7 | Only discriminative predictions | `amber_score` 1.0 with status `AMBER_OFFICIAL_SCORED` |
| 8 | `extract_mentioned_words("glasses houses")` | includes `glasse` and `hous` |
| 9 | MemLens row with 300-token history, real binary-search path | `__text1024` … `__text16384` rows each hold **2** text tokens (question only); `__text512` holds 302. All are emitted and counted toward their bins. |

Whether upstream AMBER's annotation files actually use field names such as `truth` must be checked against the pinned upstream repository. Reproductions 1, 4 and 5 show something that doesn't depend on that: **any** schema mismatch produces plausible-looking scores instead of an error.

## Findings

| ID | Sev | Criterion | Finding | Location | Required change |
|---|---|---|---|---|---|
| DS-01 | **P0** | A4 | The scorer is a heuristic re-implementation, not the official AMBER scorer. It uses a hard-coded COCO-style vocabulary plus regex plural stripping, with no upstream NLP pipeline, no hallucination-candidate lists and no Cog metric. Multi-word objects never match, and objects outside the vocabulary are never counted as hallucinations. It still labels its output `AMBER_OFFICIAL_SCORED` and records `official_scoring_eligible: True`. | `amber.py:50-62`, `:388-524`, `:592-593`, `:230-232` | Run the pinned upstream scorer unmodified, or prove parity on an upstream reference output (A4). Until then, label status `AMBER_PROXY_UNVERIFIED`, never "official". |
| DS-02 | **P0** | A1, A2 | Missing schema fields are silently replaced with defaults instead of rejected: `label` → `"None"`, objects → `[]`, `type` → `existence`, task inferred from ID range, image name inferred as `AMBER_{id}.jpg`, several alternative key names accepted. Scores then come out plausible but wrong (reproductions 1, 4, 5). | `amber.py:102-106`, `:124-153`, `:201-209`, `:288-296`, `:419-431`, `:549-578` | Strict validation against the schema of the pinned upstream files: unknown or missing fields raise errors. Add tests built from real upstream-format fixtures. |
| DS-03 | P1 | A7, `METRICS.md` | Denominator and missing-value handling: unanswered ground-truth items are left out; unparseable answers are left out of accuracy; empty denominators return `0.0`; the composite `amber_score` uses defaults `chair=0.0` and `F1=0.0` when a task is absent (reproductions 2, 3, 7). | `amber.py:304-314`, `:357-366`, `:489-510`, `:584-590` | Score over the frozen manifest: missing or failed items count as failures and are reported. Unparseable answers count in accuracy (as wrong) and are reported separately. Return `None`/`NOT_MEASURED` instead of 0. Produce no composite score unless both parts are present. |
| DS-04 | P1 | B1, B4 | The MemLens bin builder never applies `tolerance`. It emits every bin for every row even when the history is too short or the search fails, so question-only rows get labelled `__text16384` (reproduction 9). `counts_per_bin` is therefore false and `max_per_bin` gets used up by invalid rows. | `memlens.py:179-245` (`tolerance` parameter unused) | Emit a row only if the measured text tokens are within ±5%. Count skip reasons per bin. Make the tokenizer mandatory, since the word-count fallback can't satisfy B1. |
| DS-05 | P1 | A2 | An invented generative prompt (`"Please describe this image in detail."`) is used when the query field is missing. | `amber.py:169-174` | Use only upstream query text; a missing query is an error. |
| DS-06 | P1 | A5 | AMBER serving records set `requested_text_length_bin: 256`, mixing up the **output** length with the MemLens **input** bin axis. They also reuse `track: derived_serving`. | `amber.py:235-256` | Give them a separate track (for example, `amber_serving`) with no text-bin field. The 256-token output is enforced by the payload and recorded in the receipt, not in a data field. |
| DS-07 | P1 | A1, A2 | No upstream AMBER pin, file manifest or licence. No per-image SHA256 in records. AMBER isn't registered in `config.DATASETS`, `hf_sources`, the CLI or `agent_pipeline`, so **RV-01 (P0) remains open**. Quality runs still don't record `finish_reason` (**RV-02 open**). | `amber.py` (no pin); `config.py:19` unchanged | A1 and A2. Record `image_sha256` per record. Integrating with the config schema and runner is a separate slice and must be reviewed. |
| DS-08 | P2 | B6 | This adds a **third** derivation implementation instead of consolidating to one. `derivation.py` now uses the same `__text{n}` ID format as `notebook_bridge.py`, so three builders producing different content (first image, last image, word-count based) share identical IDs. Pairing keyed on ID could then silently match different prompts across campaigns. | `memlens.py:130-259`; `derivation.py:54`; `notebook_bridge.py:152` | Keep one canonical builder. Retire the others or make them fail at load. Put the derivation ID into the row ID or the receipt. |
| DS-09 | P2 | B1, cost | The builder loads all native 32K rows into memory (a regression from the byte-offset streaming in `notebook_bridge`; `TROUBLESHOOTING.md` warns about exactly this). It re-encodes the full history 7 times per row. It measures `excerpt + "\n" + question` while annotation joins with `" "` (RV-08 persists). | `memlens.py:152`, `:186-201` | Stream by offset, encode once per row, use the same join rule as annotation. |
| DS-10 | P2 | A4 | The export to official format rewrites discriminative responses to `Yes`/`No` with a local parser before upstream scoring, so the upstream scorer never sees raw outputs. The parser also falls back to searching for *any* yes/no token. | `amber.py:601-627`, `:26-47` | Export raw responses. Apply the upstream extraction rule only inside the pinned scorer. |
| DS-11 | P2 | S1 | Data records carry hard-coded `max_output_tokens: 256`, `stop_sequences` and `temperature`, but the request payload is built from config, not from these fields. They could disagree without anyone noticing. | `amber.py:227-229`, `:250-252` | Remove them, or have the runner check them against the payload. |

## Required for re-review

DS-01 to DS-07 must be fixed, with tests that use fixtures in the real upstream format, including negative tests for missing or renamed fields and for unanswered items. Include the AMBER source pin and manifest (A1). DS-08 to DS-11 may follow in a later slice, but must be tracked.
