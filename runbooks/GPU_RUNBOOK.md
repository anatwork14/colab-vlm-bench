# GPU runbook — Qwen3.5-0.8B / 4B Base vs native MTP (audit 2026-11-09)

**Status: NOT_RUN. Not approved for paid execution.** Written from the static CPU audit ([NUMERIC_INTEGRITY_LEDGER.md](../docs/audit/NUMERIC_INTEGRITY_LEDGER.md)) at `ebbf60d`. Each command spends owner-approved compute only when it carries `--approve-paid` on that invocation. The detailed protocol is in [docs/AGENT_COLAB_CLI.md](../docs/AGENT_COLAB_CLI.md) and [docs/PRIORITY_EXECUTION.md](../docs/PRIORITY_EXECUTION.md). Use this file for the gate order and stop rules.

## 0. Blocking gates (do not start a paid run until each is closed or the owner accepts it in writing)

| Gate | Why | Owner / status |
|---|---|---|
| **G-A** `run_job` must not skip failed or stale cell receipts (ledger F-05) | Otherwise Base and MTP lose different cells and the pairing is biased | **CODE FIXED** `92c8b68`: only COMPLETED receipts are skipped. Non-completed receipts stop the job before GPU (`BLOCKED_NONCOMPLETE_RECEIPTS`, rc 2). `--requeue-nonpass` quarantines them (nothing deleted) and re-runs those cells. Independent review **OPEN**. |
| **G-B** A run with non-PASS cells must not report success (F-06) | Before the fix, `GPU_JOB_FINISHED` / `ACTION_COMPLETED` appeared even when every cell failed | **CODE FIXED** `00260cc`: distinct statuses and exit codes in `run_status.json` (see §2 step 6). The wrapper returns `ACTION_FAILED` on any non-zero exit. Independent review **OPEN**. |
| **G-C** Do not use `colab_controller.sh --publish` or `remote_entry.py` (F-07) | Pushed from a paid helper; no bundle digest on the legacy path | **CODE FIXED** `4975562`: the controller never commits or pushes, and `--publish` is refused. `remote_entry.py` checks the bundle digest and uses PYTHONPATH, with no runtime `pip install`. The legacy path remains **not recommended**. Use `scripts/colab_agent.py` only. Lead review **OPEN**. |
| **G-D** Revisions and checksums pinned (F-13, F-14, §6 of the manifest) | Otherwise the run is not reproducible | **OPEN, owner.** `docs/audit/PINNED_INPUTS.md` (generated, `e4a30ae`) lists every model revision as `UNPINNED` and every dataset revision as `BLOCKED`. A fabricated or placeholder SHA is refused by the generator. |
| **G-E** Independent review of the E2E and proxy-denominator definition changes (`ebbf60d`) | `AGENTS.md` requires an independent reviewer for metric definitions | Lead. **OPEN.** |
| **G-F** vLLM 0.29.0 and A100 40 GB BF16 verified on the actual session | Not observed in this audit | Verified on the GPU session before benchmark |
| **G-G** One derivation route per serving pool (ledger F-09) | Two MemLens routes produce different prompts for the same dataset | **CODE FIXED** `573be47`: run-job refuses a derived file with mixed or unlabelled rows (`BLOCKED_MIXED_DERIVATION`, rc 2). Receipts and the job manifest record the label. Review **OPEN**. |

Stop rule: if any gate is open, run only the CPU steps in §1 and §2. Gates G-A, G-B, G-C and G-G are code-fixed but still need independent review, so paid execution stays blocked.

## 1. CPU (no GPU charge)

```bash
python scripts/colab_agent.py doctor
python scripts/colab_agent.py provision --session vlmb-cpu --gpu cpu
python scripts/colab_agent.py mount --session vlmb-cpu            # interactive login
python scripts/colab_agent.py bootstrap --session vlmb-cpu --kind cpu
python scripts/colab_agent.py drive-inventory --session vlmb-cpu  # read-only, review before prepare
```

Prepare pilot sources first, with an engineering QA cap. Full publisher scoring uses `--phase full --qa-limit 0` only after the pilot succeeds.

```bash
python scripts/colab_agent.py prepare --session vlmb-cpu --model qwen35_08b --dataset memlens --phase pilot --qa-limit 48
python scripts/colab_agent.py prepare --session vlmb-cpu --model qwen35_08b --dataset mmlongbench_doc --phase pilot --qa-limit 48
```

On the CPU session, also run the checkpoint payload audit and keep its output:

```bash
python scripts/audit_safetensors_payload.py --checkpoint-dir /content/drive/MyDrive/<pinned-model-dir> --hash-shards
```

Check the CPU plan before any A100 allocation:

```bash
PYTHONPATH=src python -m vlmbench.cli plan --config configs/a100_qwen_priority_pilot.json
# Expect: cell_count 48, measured_requests 2816
```

Stop the CPU session when done: `python scripts/colab_agent.py stop --session vlmb-cpu --confirm-stop`.

## 2. A100 (paid; requires G-A to G-F)

1. **Owner approval** for this specific allocation. Then:
   ```bash
   python scripts/colab_agent.py provision --session vlmb-a100 --gpu A100 --high-mem --approve-paid
   python scripts/colab_agent.py mount --session vlmb-a100
   python scripts/colab_agent.py bootstrap --session vlmb-a100 --kind gpu --approve-paid
   ```
2. **Check the machine** before any inference: A100 40 GB, BF16 supported, vLLM `0.29.0`, `environment.json` written with a real `repo_source_sha`. A different GPU is a refusal, not a substitute (`config.py:63`).
3. **Annotate** with the 8-request smoke budget, one model and dataset at a time:
   ```bash
   python scripts/colab_agent.py annotate --session vlmb-a100 --model qwen35_08b --dataset memlens \
     --phase pilot --max-requests 8 --approve-paid
   ```
   Confirm `verified_total` in the receipt. Eight requests are not enough for CCU 8 or 32.
4. **Readiness.** Inspect `preflight/<job>.workload.json`. CCU 1, 8 and 32 need 17, 40 and 160 unique eligible prompts per bin. Zero ready cells means the run stops before vLLM starts.
5. **Serving, one cell at a time** (`--max-cells 1` per call, with `--approve-paid` each time):
   ```bash
   python scripts/colab_agent.py benchmark --session vlmb-a100 --model qwen35_08b --dataset memlens \
     --phase pilot --mode base --max-cells 1 --approve-paid
   python scripts/colab_agent.py benchmark --session vlmb-a100 --model qwen35_08b --dataset memlens \
     --phase pilot --mode mtp  --max-cells 1 --approve-paid
   ```
   Resume and requeue (G-A): a re-run skips only COMPLETED receipts. Any other existing receipt stops the job before GPU start with `BLOCKED_NONCOMPLETE_RECEIPTS`, and nothing is edited. Re-run with `--requeue-nonpass` (or `benchmark ... --requeue-nonpass`) only after you have read those receipts. Requeued receipts are moved to `quarantine/<UTC stamp>/` locally and on Drive, never deleted.
   After each call, read the cell receipt and `server.log`:
   - `status` must be `PASS_MEASURED` for the cell to count. `PARTIAL_FAILURE`, `ERROR`, `TIMEOUT` are recorded results, not retries.
   - For MTP, confirm the speculative config is active in the server log. Do not infer it from the argv alone.
   - Record `gpu_count_max`, `parse_failures`, `query_failures` from the telemetry block. Any non-zero failure count means the memory peak is not trustworthy for that cell.
   - Record the vLLM KV-cache size from `server.log`. It is **not** collected by the code yet (ledger C3-7).
6. **Stop rule (pilot).** The engineering pilot stops after the first runtime failure (`cli.py` circuit breaker, exit 4, `PILOT_STOPPED_ON_FAILURE`). Inspect the receipt before continuing. Do not raise timeouts to make a cell pass. An OOM or capacity failure at CCU 32 is a finding, to be reported as such.

   **Exit codes and statuses** (G-B), written to `run_status.json` as `status` and `exit_code`:

   | Status | Exit | Meaning |
   |---|---|---|
   | `COMPLETE` | 0 | Every planned cell is PASS_MEASURED |
   | `IN_PROGRESS` | 0 | `--max-cells` cap reached, nothing failed; continue with the next call |
   | `PARTIAL_FAILURE` | 6 | Some cells PASS, some non-PASS remain. Read `counts` and the receipts |
   | `ALL_ATTEMPTED_FAILED` | 5 | Every cell attempted in this call failed |
   | `PILOT_STOPPED_ON_FAILURE` | 4 | Engineering circuit breaker fired |
   | `BLOCKED_*`, `BLOCKED_NONCOMPLETE_RECEIPTS`, `BLOCKED_MIXED_DERIVATION` | 2 | Refused before GPU start |

   Only exit 0 is a success. The remote wrapper reports `ACTION_FAILED` for any non-zero exit. Quality runs map to `QUALITY_ALL_FAILED` (5), `QUALITY_PARTIAL_FAILURE` (6) and `QUALITY_STOPPED_AFTER_REPEATED_FAILURES` (4).
7. **Quality (last).** Native full-context, natural EOS, CCU 1, `max_model_len 65536`. Skip the step if the quality preflight reports images over `max_images_per_prompt`; that is `BLOCKED_QUALITY_IMAGE_LIMIT`, not a truncation. Official scoring is a separate upstream run.
8. **Report and export.** Generate the report, pull the small summary, and verify the raw receipts on Drive before stopping the session.
   ```bash
   python scripts/colab_agent.py report --session vlmb-a100 --model qwen35_08b --dataset memlens --phase pilot
   python scripts/colab_agent.py pull --session vlmb-a100 --output agent_artifacts/memlens_qwen08b_runbook.zip
   python scripts/colab_agent.py stop --session vlmb-a100 --confirm-stop
   ```

## 3. Known limits of this runbook

- **Run order is not interleaved.** Base and MTP run in separate invocations, so temporal drift is not balanced (ledger C3-5). Record launch times.
- **MTP changes available KV memory** under the same `gpu_memory_utilization`. High-CCU throughput differences can come from KV capacity, not only speculation (C3-7).
- **Client-side latency only.** TTFT is service-side from the HTTP send. E2E includes the client CCU-slot wait, and excludes vLLM's internal queue, which is not measured (`NOT_MEASURED`).
- **Peak GPU memory** is sampled every 0.5 s and counts total device use, not allocated or KV bytes.
- **Official accuracy and hallucination metrics are not produced here.** They need the publisher judge and independent evidence labels.
- **Derived rows need one route label.** A derived file must carry exactly one `derivation` value (G-G). AMBER rows carry none, and no AMBER config is registered, so AMBER serving is refused until it is labelled.
- **Only the two priority models** are in scope. The other five registered models and the H2O/DA/ECRD arms are not measured by this path.

## 4. What a passing run does and does not show

A `PASS_MEASURED` cell means the 256-token streamed serving contract was met for that cell: the requested number of successful requests, usage-verified output lengths, and the recorded telemetry. It does not show:

- GPU model correctness, native MTP correctness, or that speculation was active unless the server log says so;
- official benchmark quality;
- anything about the other arms;
- any result for a cell that is not `PASS_MEASURED`.
