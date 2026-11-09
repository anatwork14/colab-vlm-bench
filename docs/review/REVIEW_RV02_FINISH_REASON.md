# Review: RV-02 `finish_reason` fix (sol-lead)

| Field | Value |
|---|---|
| Task | TASK-STARLIGHT-REVIEW-RUNTIME-FINISH-REASON |
| Reviewer | argon (independent) |
| Target | `wt/sol-lead-runtime`, uncommitted working tree on `e83a3c8`, branch `lead/sol-runtime-colab-cli` |
| Scope | `src/vlmbench/bench.py`, `src/vlmbench/cli.py` (job manifest), `tests/test_runner_order.py` |
| Evidence | Diff read in full. Suite: 92 passed, 10 skipped. Ruff on the touched files reports no new findings compared with `e83a3c8` (it reports fewer). |

## Verdict: **PASS** for RV-02 (capturing why each answer stopped). Criterion A3 is still **CONDITIONAL** because its summary and gate aren't done yet.

**Confirmed:**
- Successful quality items store `choice["finish_reason"]` exactly as the backend returned it, with no coercion. Failed items store `None`.
- Parametrized tests cover `stop`, `length`, a missing field with HTTP 200, and HTTP 500. They also assert that the quality payload has `max_tokens == 256` and no `ignore_eos` or `min_tokens`.
- Quality and serving receipts, and the job manifest, now record `execution_pass` and `output_contract`.
- `run_performance_cell` refuses any output length other than 256.

## Remaining items (P2, not blocking this fix)

| ID | Finding | Required change |
|---|---|---|
| FR-1 | The quality receipt doesn't summarise `finish_reason`, so the truncation rate and the A3 threshold gate are never computed. | Add `finish_reason_counts` (`stop`, `length`, other, missing) and a truncation rate per receipt; apply the A3 gate in reporting. |
| FR-2 | A successful response with no `finish_reason` is stored as `None`, the same value as a failure, and isn't flagged. | Count it separately as `missing` in FR-1, so a backend that omits the field can't look clean. |
| FR-3 | A quality JSONL started before this change resumes without the field, and nothing reports this. Risk is low because no GPU runs exist yet. | FR-1's `missing` count covers it. Alternatively, refuse to resume when earlier records lack the field. |
| FR-4 | The serving resume key now includes `execution_pass` and `output_contract`, so any earlier `PASS_MEASURED` cell receipts won't resume and would be re-run on GPU. No such receipts exist yet. | Note this in the changelog; no code change needed. |

## Outside this verdict

`scripts/colab_agent.py` also has a new `resolve_session()` and a session-argument rewrite in `command()`. It reads `~/.config/colab-cli/sessions.json`. That change isn't covered here. On a read-through it never allocates a runtime and never prints the registry's contents, but it needs its own review before integration.
