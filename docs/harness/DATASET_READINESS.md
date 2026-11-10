# Dataset readiness: MemLens and MMLongBench-Doc (ITER3 C2)

Status: **not ready.** Neither dataset exists on this host, no revision is owner-approved, and the
official judge stage is not qualified. This document says what has to be true before a dataset can feed a
benchmark cell. It makes no download and no installation claim.

Evidence for every fact below comes from one of two sources, named in each line:

- `DF` = `docs/harness/DATASET_FEASIBILITY.json` (read-only metadata resolved 2026-11-10 from the public
  HF API, datasets-server and raw GitHub; not owner-approved; only 82 KB of data and two small code files
  were fetched).
- `LC` = local state read on 2026-11-10 (read-only).

Nothing here is an owner decision. The licence conflict in §3 and the revision approvals in §5 are open.

## 0. Local state (LC, 2026-11-10)

| Item | State |
|---|---|
| MemLens native file (`memlens_native.jsonl`) | not present; the pilot config path is `REPLACE_DRIVE_PATH/memlens_native.jsonl` |
| MMLongBench-Doc native file (`mmlong_native.jsonl`) | not present; path `REPLACE_DRIVE_PATH/mmlong_native.jsonl` |
| Dataset revisions in `configs/a100_qwen_priority_pilot.json` | `null` (no pin) |
| Dataset `native_sha256` in config | `null` (no checksum) |
| MemLens or MMLongBench-Doc under `models/` or `~/.cache/huggingface/hub` | none (the HF hub cache holds only `hf-internal-testing/tiny-random-gpt2`) |
| Free space on `/mnt/ssd` | 15 GB at the time of reading (`df`). Below the harness guard of 20 GiB. |
| `/mnt/hdd` | writable: no (checked with `test -w`, nothing written) |

Consequence: `dataset_present`, `dataset_checksum` and `dataset_revision_pinned` all FAIL for both datasets. The inventory
reports this correctly (`BLOCKED_DATASET_MISSING`, `BLOCKED_DATASET_UNVERIFIED`, `BLOCKED_REVISION_UNPINNED`).

## 1. What must be pinned, per dataset

A pin is an owner-signed value in the record (see `docs/harness/READINESS.md` §2). A value in a config file is not a pin.

For each dataset the signed record must hold:

- `pins.datasets.<id>.revision`: the Hugging Face dataset commit SHA;
- `pins.datasets.<id>.native_sha256`: the SHA-256 of the **native JSONL** this benchmark reads (the normalized file,
  not the upstream file). The normalized file is produced by `vlmbench.datasets` and must be reproducible from the pinned upstream revision.
- The upstream **code** commit and the SHA-256 of each scoring file the benchmark depends on (listed per dataset below).

Revisions and file hashes in `DF` are resolved metadata, not owner pins. They are copied here so that the
owner can check them, and they must be re-resolved at approval time.

## 2. MemLens

### 2.1 Authoritative sources

| Source | Identifier | Resolved (DF) | Licence |
|---|---|---|---|
| Data (Hugging Face) | `xiyuRenBill/MEMLENS` @ `afa101a1907cc37db40b50d649547964387b96b7` | last modified 2026-06-14 | `cc-by-4.0` (dataset card); not gated |
| Code (GitHub) | `xrenaf/MEMLENS` @ `77f3ab9a52fa2d6a17978e2dffe80438a4ecced2` | 2026-07-24 | MIT |

Data facts (DF): total repo 3,885,310,166 bytes, 4,716 files. Four context configs (`32k`, `64k`, `128k`, `256k`), 789 rows each.
Parquet sizes: 52.0 MB (32k), 100.9 MB (64k), 195.3 MB (128k), 387.4 MB (256k). The 32k JSON is 98,085,636 bytes; the upstream
README says "~104 MB" (DF `readme_claim_vs_hf`). Columns: `question_id, question_type, question, answer, question_date, haystack_dates,
haystack_session_ids, haystack_sessions, answer_session_ids`. The DF note measured one 32k row's `haystack_sessions` at ~132 KB of text;
per-row token counts were **not** measured.

### 2.2 Code files to pin (DF `code_sha256`)

| File | SHA-256 |
|---|---|
| `eval.py` | `96ff6fd92975d431477330730864a1f726f83a268aaa7745cfd908f2c4d8cb37` |
| `data.py` | `a05967d62ed21e0a596c7d73021b0af1c5d056604886516f4fa8b0579bc08995` |
| `answer_extraction.py` | `6beecf8ff98538d3d75ef228e22e48932cbcc2293bdf924f0261a48d80a0fe34` |
| `llm_judge.py` | `c2f12050cbf2ae55a8a2c8db73fd932c32c20e9d74f552faa518e02e31b6c992` |
| `eval_api.py` | `6370a33a4d0ec6f22acb2d93a6e2464a8146760e6650fe220b6f425f3bf3e31b` |
| `README.md` | `7095917677f979b48f6238bf9beb9cedc8e616866d4427361e177f5db1c5c277` |

### 2.3 Official scoring and its status

DF `official_scoring`:

1. LLM answer extraction: `answer_extraction.py` (OpenAI-compatible client, cached extraction).
2. LLM judge: `llm_judge.py` with `judge_prompts/*.py` (OpenAI or vLLM judge).
3. Deterministic matching after the LLM steps: `answer_extraction.match_extracted_answer`, token F1 (`_token_f1`).

**Status: BLOCKED_NO_JUDGE for any official number.** The deterministic token F1 is a proxy and is never labelled
official (`src/vlmbench/quality.py`: `score_proxy` returns `QUALITY_PROXY_ONLY`; `import_official` returns
`BLOCKED_NO_JUDGE` for memlens unconditionally, ITER3 C4). The judge model, its version and its prompts are not pinned.

### 2.4 Native versus derived workloads

- **Native:** the 789 rows of a context config, as shipped, with the full `haystack_sessions` for that config. Only native rows may
  feed an official-quality claim, and only after the judge stage is qualified.
- **Derived:** any row that is shortened, re-chunked, sampled from a longer haystack, or truncated to a window. Derived workloads
  carry `official_quality_allowed: False` (`src/vlmbench/derivation.py`) and can only support engineering claims.
- Context fit: `32k_items: 789` is a candidate count from HF metadata. **Whether each item fits in 32K tokens is not verified**
  (DF `eligible_examples_by_protocol`). It needs the pinned tokenizer and an offline count over every row, which this task did not run.

### 2.5 Gold-free request generation

A request sent to the model must contain only the question and the haystack, never `answer`, `answer_session_ids`, or any
evaluator-only field.

- Gold is kept in `reference` (`src/vlmbench/datasets/memlens.py`, normalization), and evidence IDs are kept in
  `evidence_session_ids`.
- `ensure_no_gold_in_request` (`src/vlmbench/workloads.py`) is **structural only**: it rejects a serialized payload
  whose top-level keys include `reference` or `answer`. It does not look at message text. A question or haystack
  that contains the answer string passes the check.
- **Implemented (ITER3 D2).** `native_records.check_record` and `receipt`: every normalized record gets a gold-free check, and
  `normalize_memlens` returns `gold_free_receipt`. A gold key or the gold text as a message part is a structural violation
  (`LEAKED`, the run fails). A gold token inside the haystack context is reported (`GOLD_TOKEN_IN_CONTEXT`) and not hidden. Haystack
  text can contain the answer word legitimately. The question and the constant template wording are not counted. Session headers
  (`[Session s1 — date]`) are template text, so `answer_session_ids` values are not searched in the request. Session ids are
  visible for every session, not only the evidence sessions. Tests: `tests/test_native_records.py`.

## 3. MMLongBench-Doc

### 3.1 Authoritative sources

| Source | Identifier | Resolved (DF) | Licence |
|---|---|---|---|
| Data (Hugging Face) | `yubo2333/MMLongBench-Doc` @ `2ff6aa9237fc777b6627dc57a486e9225ac5fb86` | last modified 2025-11-06 | **not declared on the card** |
| Code (GitHub) | `mayubo2333/MMLongBench-Doc` @ `d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384` | 2025-09-28 | Apache-2.0 |

**Licence conflict (owner decision, open).** The repo's own `README.md` (line 138, as recorded by DF) says
noncommercial research use. The HF card declares no licence. The upstream GitHub declares Apache-2.0. The
owner must decide which licence applies before the data is used. This task does not decide it.

### 3.2 Data facts (DF)

- QA parquet `data/train-00000-of-00001.parquet`: 82,073 bytes, **sha256 `bcdac3c96669634c34184814cede4fe57cf7ac0f98dde0e85936394f6a56a02d`**,
  1,091 rows, 135 distinct `doc_id`.
- Answerable 847; unanswerable 244 (share 0.2237; README says 0.225).
- Answer formats among answerable rows: Int 290, Str 250, Float 160, List 147.
- Evidence pages: 0 for 246 rows, 1 for 485, more than 1 for 360. Cross-page share 0.33 (README 0.33).
  Unanswerable rows with evidence pages: 7.
- Documents: 135 PDFs, largest 45,669,728 bytes, total repo 662,230,109 bytes. **Not downloaded. Page counts not verified.**
- GitHub `data/samples.json`: 1,082 items, sha256 `4ffadad60b18cd7ede864155f3e8a366fe4deec2189b3d0a25691b7df4b839e8`.
- **HF and GitHub question sets differ.** Keyed on `(doc_id, question)`: 26 pairs only in HF, 17 only in GitHub
  (DF `hf_vs_github_question_sets`). The owner must choose which set is native before any run. Until then neither is native.

### 3.3 Code files to pin (DF `code_sha256`)

| File | SHA-256 |
|---|---|
| `eval/eval_score.py` | `cafa268358b08521cfc9a805c67d86732d943677e4408c5751f336123d09631f` |
| `eval/extract_answer.py` | `fa6a747bd6a8e96fb6884eb7201663268a5df700ad71d85db3bc5e04a157722f` |
| `eval/prompt_for_answer_extraction.md` | `96a22c3003c22801f1f19717be45070b1d6b9ce76b42e171e044f63b641de659` |
| `README.md` | `658c0c062c0a4bf10e8ff35e52b6446d981f827cd53d18a7670bffb3b28fa421` |
| `run.sh` | `5e7b2c40a0eeba808c3680f63b0dc5d9bdee1c7a7029f56b21c7841573c5b3e2` |
| `requirements.txt` | `98e959a4d04c8543b3be074237dcce99595a779c36cda092cbc3ab3cf0386d4e` |

### 3.4 Official scoring and its status

DF `official_scoring`, which matches the code:

1. Stage 1: the model's long-form response (not produced by this repo's scorer).
2. Stage 2: `eval/extract_answer.py` calls **GPT-4o** with `eval/prompt_for_answer_extraction.md`. Status: **BLOCKED_NO_JUDGE**.
3. Stage 3: `eval/eval_score.py`, deterministic. Ported in `src/vlmbench/datasets/mmlong_official_scorer.py` and tested
   (fuzz and differential tests). Upstream's `gt[0]` on an empty gold list is preserved and pinned by a test.

Under ITER3 C4, an official stage-3 label needs a **signed extractor-stage qualification** that binds scorer provenance,
the official extractor model and prompt hash, the sample manifest, and the exact stage-2 outputs. Declared metadata is not enough.
Stage 2 needs a GPT-4o run, which is an external paid service and has not been run. No qualification exists.

### 3.5 Native versus derived workloads

- **Native:** full document. Every page of each referenced PDF is rendered (`src/vlmbench/datasets/mmlong.py`: "Render EVERY page of
  each referenced PDF. Never drop evidence for official quality."). A truncated document, a dropped page, or a lower DPI is a
  derived workload.
- Per-document page count must fit the serving context (`max_model_len` 32,768 in the pilot config). **Not verified**: it needs
  the PDFs and the pinned processor.
- Derived workloads cannot carry an official label (`official_quality_allowed: False`).

### 3.6 Gold-free request generation

- The normalized record keeps `reference`, `evidence_pages`, `evidence_sources` and `answer_format`
  (`src/vlmbench/datasets/mmlong.py`, lines 105–113). None of them may enter the request.
- The same structural guard applies, and it has the same limit as §2.5: it checks top-level keys only.
- Unanswerable rows carry the answer string `Not answerable`. That string must not appear in any message.
- **Implemented (ITER3 D2).** The same check as §2.5 applies. `reference`, `answer_format`, `evidence_pages` and
  `evidence_sources` are kept in `meta` and must not appear in `messages` (`LEAKED` if they do). The unanswerable label
  `Not answerable` is a gold text. The adapter's instruction "if not answerable say so" is constant wording for every row. It
  is not counted as a leak, and the test `test_mmlong_template_wording_is_not_a_gold_leak_for_an_unanswerable_row` pins that.

## 4. Validation steps (both datasets, in order)

Each step produces a receipt. A failed step blocks the dataset. Nothing is downloaded in this document; each step runs only after owner approval.

1. **Revision check.** The HF commit equals the signed `revision`. The GitHub commit equals the signed scorer commit.
2. **File hashes.** The upstream files named in §2.2 and §3.3 match their signed hashes.
3. **Row and column check.** MemLens: 789 rows per config, the columns listed in §2.1. MMLongBench-Doc: 1,091 rows,
   the columns listed in DF, 135 distinct `doc_id`.
4. **Native file hash.** The normalized JSONL matches the signed `native_sha256`.
5. **Context fit.** Count tokens per row with the pinned tokenizer; record the maximum, and the number of rows over budget.
   Rows over budget are excluded from the official set and listed, not truncated.
6. **Gold-free test.** §2.5 and §3.6.
7. **Judge qualification.** Stage-2 extraction qualified under ITER3 C4 (MMLongBench-Doc). For MemLens, a judge
   is not ported; official status stays BLOCKED_NO_JUDGE.
8. **Licence decision** recorded by the owner (§3.1).

## 5. Open decisions for the owner

- Approve or reject each revision in §2.1 and §3.1 for the signed record.
- Resolve the MMLongBench-Doc licence conflict (§3.1).
- Choose the native MMLongBench-Doc question set (§3.2): HF or GitHub.
- Authorise the stage-2 GPT-4o extraction and the signed qualification, or leave both datasets without an official label.
- Approve the MemLens judge provenance (pinned `judge_prompts` and judge model), or accept that MemLens is never officially scored.

## 6. Not closed

- No file was downloaded or hashed beyond the DF numbers.
- No token counts, page counts or context-fit results exist.
- The gold-free receipt (§2.5, §3.6) is implemented on fixtures. It has not been run on the real upstream rows, because the rows are
  not on this host (see `DRIVE_INVENTORY.md`).
- The MMLongBench-Doc page count is taken from the renderer. It is not checked against the PDF independently.
- The upstream MemLens judge is unpinned, so no MemLens official number is possible.

## 7. Native record transformation (ITER3 D2)

Implemented in `src/vlmbench/datasets/native_records.py`, used by `normalize_memlens` and `normalize_mmlong`.

- **Ids.** MemLens: the upstream `question_id`, as a string. MMLongBench-Doc: `sha256(doc_id + "\n" + question)[:20]`. The
  MMLong id does not depend on row order. Duplicates fail.
- **Order and repeatability.** Output keeps source order. Two runs on the same input give the same output bytes (test).
- **Missing references.** Every missing image (MemLens) or document (MMLongBench-Doc) is found before anything is written, and
  all are reported together (`MissingReferences`). Nothing is published. Escaping paths still raise at once.
- **Full references.** MemLens meta lists every image with its SHA-256. MMLongBench-Doc records every page file, the page
  count, the PDF SHA-256 and the DPI. Page image bytes are not hashed, because they depend on the renderer.
- **Transform hashes.** The summary has `transform` = version, parameter hash, source SHA-256 and output SHA-256.
- **Native versus derived.** Native records stay on `native_quality`. A derived row is refused by `load_jsonl(track="native_quality")`
  and is not produced here. Derived builders stay in `derivation.py` and `build_memlens_context_bins`.
- **Scorer gate.** `quality.import_official("memlens", ...)` is `BLOCKED_NO_JUDGE` (test). The MMLongBench stage-3 gate is unchanged (ITER3 C4).

Open (not closed by D2): the prompt text is still the adapter's own, so official prompt parity is pending. Upstream prompts are
not on this host and were not fetched. MMLongBench-Doc rendering is tested with an injected renderer, because PyMuPDF
is not installed in the test environment. No real PDF was rendered.

**Round B follow-up (2026-11-10): real local data for the transformation tests — none.** A read-only search of `/mnt/ssd`,
`/mnt/hdd`, `/home`, `/tmp`, `/var/tmp`, `/opt`, `/srv` and `/data` found no MemLens or MMLongBench-Doc data file. No `.pdf` in any
dataset-named path. The only name matches are code, `.pyc` files, and config references to `REPLACE_DRIVE_PATH/*.jsonl`. Two
`tests/fixtures/bench/{memlens,mmlongbench-doc}.json` files exist in other STARLIGHT worktrees (one record each, ~400 bytes). Their
`fixture_label` is `SYNTHETIC FORMAT FIXTURE, NOT BENCHMARK DATA`, so they were **not** used. The transformation tests were not run
on real small files, because there are none. The synthetic-data tests in `tests/test_native_records.py` are the only ones run.
