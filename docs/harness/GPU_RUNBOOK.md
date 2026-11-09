# GPU runbook: 48-cell Qwen3.5 pilot on 1× A100 40GB (BF16, TP=1)

**Status: UNEXECUTED.** Nothing in this file has been run. No GPU, Colab session, paid resource or network
call was used to write it. The commands come from the repo's own files: `docs/PRIORITY_EXECUTION.md`,
`runbooks/QWEN_PRIORITY.md`, `scripts/colab_controller.sh`, `scripts/harness_inventory.py`, and the `vlmbench.cli` subcommands.
Values in `<ANGLE_BRACKETS>` are for the owner to supply. Do not run any paid step below without the owner's explicit approval
for GPU compute (`--approve-paid`).

## 0. Gates that must hold first

Status column is as of 2026-11-10, from the repo's own files.

| # | Gate | Status | Where it is defined |
|---|---|---|---|
| G1 | Owner approval for paid A100 time | **not given** | `scripts/colab_controller.sh` refuses without `--approve-paid`; `AGENTS.md` step 4 |
| G2 | Signed owner record with pins for both models, both datasets, vLLM, GPU, receipt index | **absent** | `docs/harness/READINESS.md` §2 |
| G3 | Dataset revisions and native JSONL hashes | **absent** (`REPLACE_DRIVE_PATH`) | `docs/harness/DATASET_READINESS.md` §0 |
| G4 | Qwen3.5-4B revision, weights manifest, tokenizer check | **absent** | `docs/harness/QWEN35_4B_STAGING_PLAN.md` §1–2 |
| G5 | MMLongBench-Doc licence decision | **open (owner)** | `docs/harness/DATASET_READINESS.md` §3.1 |
| G6 | Inventory: every pilot cell reaches at least `LIVE_ENVIRONMENT_VERIFIED` | **not met** (0 READY, ladder NONE in the committed inventory) | `docs/harness/PILOT_INVENTORY.md` |
| G7 | Clean git state at the commit to be run | to check at run time | `colab_controller.sh` refuses a dirty tree |
| G8 | Colab session `vlmbench-a100` exists, is an A100 40GB, Drive is mounted | **not checked** (no Colab access in this task) | `docs/PRIORITY_EXECUTION.md` "Execution on an existing approved Colab A100 session" |
| G9 | Judge qualification for official quality (C4) | **not met**: native subset is a proxy until then | `src/vlmbench/datasets/mmlong_official_scorer.py` |

If any gate is not met, do not run §5 onward. §1–§4 are read-only or local.

## 1. Owner-local config (never committed)

Copy the frozen config to a path **outside** the repository. Commit nothing from it.

```bash
cp configs/a100_qwen_priority_pilot.json <LOCAL_DIR>/a100_qwen_priority_pilot.local.json
```

Edit only these fields in the local copy, with values from the signed record:

- `dry_run`: `true` → `false` (the controller refuses `dry_run=true`).
- `models[*].revision` for `qwen35_08b` and `qwen35_4b`.
- `datasets[*].revision`, `datasets[*].native_path` (the owner's Drive path), `datasets[*].native_sha256`.

Do **not** change the campaign ID (`a100-qwen08b-4b-pilot-v1`), the hardware block (A100 40GB, BF16, TP=1), the serving block
(`backend_version` 0.29.0, `max_model_len` 32768, `max_num_seqs` 32), seeds, or the cell grid. A change to those
is a new campaign with a new ID (`docs/PRIORITY_EXECUTION.md`).

## 2. Static checks (local, no GPU)

```bash
cd <REPO_ROOT>
PYTHONPATH=src python -m vlmbench.cli validate --config <LOCAL_DIR>/a100_qwen_priority_pilot.local.json
PYTHONPATH=src python -m vlmbench.cli plan --config <LOCAL_DIR>/a100_qwen_priority_pilot.local.json
```

Expected from `plan`: **48 serving cells**. Per (model, mode, dataset) combination that is 6 cells:
2 input lengths (2048, 16384) × 3 concurrency levels (1, 8, 32) × 1 repetition. Native quality subset: 48 IDs per dataset,
per model and mode, seed 42 (up to 384 predictions in total). If the plan prints any other count, stop.

## 3. Readiness inventory (read-only; run where the A100 is)

```bash
PYTHONPATH=src python scripts/harness_inventory.py \
  --pilot-config <LOCAL_DIR>/a100_qwen_priority_pilot.local.json \
  --owner-record <SIGNED_OWNER_RECORD.json> \
  --allowed-signers <OWNER_SIGNERS_FILE> \
  --receipt-index <SIGNED_RECEIPT_INDEX.json> \
  --receipts-dir <RECEIPTS_DIR> \
  --models-root <MODELS_ROOT> \
  --min-free-gib 20 \
  --out-json <SCRATCH_DIR>/PILOT_INVENTORY.json \
  --out-md <SCRATCH_DIR>/PILOT_INVENTORY.md
```

Write to a scratch path, not `docs/harness/`, so the committed inventory is not overwritten by a run.

Expected before the paid run: the pilot cells reach `LIVE_ENVIRONMENT_VERIFIED`. `GPU_SMOKE_VERIFIED` (READY) is not
expected yet, because the smoke receipt comes from §6, which runs first. A `READY` count of 0 at this point is the correct result.

**Open item.** The repo does not say whether this preflight runs on the Colab session or on the local host. The live probe needs
`nvidia-smi` on the machine that runs the GPU work. The owner must decide before §3 is run.

## 4. Clean git state

```bash
git status --porcelain        # must print nothing
git rev-parse HEAD            # record this SHA with the run
```

The controller packages `git archive HEAD`, so the run uses exactly that commit. Do not commit the local config.

## 5. Colab session check (before any paid job)

```bash
colab status -s vlmbench-a100
```

Confirm: A100 40GB, BF16 available, Drive mounted (`docs/PRIORITY_EXECUTION.md` requires exact A100 40GB, vLLM MTP and the
256-output-token policy). The repo does not define an automated check for this. The owner confirms it by reading the output.

## 6. GPU smoke (PRIORITY step 1): 0.8B Base, then 0.8B MTP

One cell each. The smoke must show a real image and long prompt, the output token usage, and the MTP backend trace.

```bash
bash scripts/colab_controller.sh --session vlmbench-a100 \
  --config <LOCAL_DIR>/a100_qwen_priority_pilot.local.json \
  --model qwen35_08b --mode base --dataset memlens --max-cells 1 --approve-paid

bash scripts/colab_controller.sh --session vlmbench-a100 \
  --config <LOCAL_DIR>/a100_qwen_priority_pilot.local.json \
  --model qwen35_08b --mode mtp --dataset memlens --max-cells 1 --approve-paid
```

**Assumption.** The repo does not say which cell `--max-cells 1` selects. The owner must confirm it is acceptable.

Stop rule: if the MTP trace does not show MTP active, or MTP output equals Base output by accident, record the arm as `BLOCKED_MTP`.
Do not report an MTP speedup. Do not run §7 until the owner has read the smoke results.

## 7. The 48-cell pilot (PRIORITY step 3): 8 jobs of 6 cells

Order: 0.8B before 4B, Base before MTP, MemLens before MMLongBench-Doc. The controller caps a job at 6 cells
(`src/vlmbench/agent_pipeline.py`: `1 <= max_cells <= 6`), so the 48 cells run as eight jobs.

```bash
C=<LOCAL_DIR>/a100_qwen_priority_pilot.local.json
S=vlmbench-a100

bash scripts/colab_controller.sh --session $S --config $C --model qwen35_08b --mode base --dataset memlens          --max-cells 6 --approve-paid
bash scripts/colab_controller.sh --session $S --config $C --model qwen35_08b --mode base --dataset mmlongbench_doc   --max-cells 6 --approve-paid
bash scripts/colab_controller.sh --session $S --config $C --model qwen35_08b --mode mtp  --dataset memlens          --max-cells 6 --approve-paid
bash scripts/colab_controller.sh --session $S --config $C --model qwen35_08b --mode mtp  --dataset mmlongbench_doc   --max-cells 6 --approve-paid
bash scripts/colab_controller.sh --session $S --config $C --model qwen35_4b  --mode base --dataset memlens          --max-cells 6 --approve-paid
bash scripts/colab_controller.sh --session $S --config $C --model qwen35_4b  --mode base --dataset mmlongbench_doc   --max-cells 6 --approve-paid
bash scripts/colab_controller.sh --session $S --config $C --model qwen35_4b  --mode mtp  --dataset memlens          --max-cells 6 --approve-paid
bash scripts/colab_controller.sh --session $S --config $C --model qwen35_4b  --mode mtp  --dataset mmlongbench_doc   --max-cells 6 --approve-paid
```

The 4B jobs (last four) run only after the 4B gates G4 and §3 are met for the 4B cells.

Each job's outcome is recorded whatever it is: PASS, FAIL, UNSUPPORTED, OOM, timeout, INSUFFICIENT_WORKLOAD. An OOM or
capacity boundary is a result, not a reason to re-run silently. Requeueing is available through `run-job --requeue-nonpass` and
is an explicit owner choice.

## 8. Native quality subset (PRIORITY step 4): 8 jobs with `--quality`

The sample cap (48 per dataset, model and mode, seed 42) is in the config. `--quality` runs the deterministic native-context subset.
It is **not** a full official score.

```bash
C=<LOCAL_DIR>/a100_qwen_priority_pilot.local.json
S=vlmbench-a100
for MODEL in qwen35_08b qwen35_4b; do
  for MODE in base mtp; do
    for DATASET in memlens mmlongbench_doc; do
      bash scripts/colab_controller.sh --session $S --config $C --model $MODEL --mode $MODE --dataset $DATASET --quality --approve-paid
    done
  done
done
```

These predictions are unscored until a judge stage is qualified (G9). Until then they are labelled as predictions or as proxy metrics,
never official accuracy.

## 9. Report and review

```bash
PYTHONPATH=src python -m vlmbench.cli report --config <LOCAL_DIR>/a100_qwen_priority_pilot.local.json --root <RUN_ROOT>
```

`--publish` was removed from the controller (audit F-07). Copy `runbooks/a100-qwen08b-4b-pilot-v1/` after the owner has reviewed it,
then commit the copy by hand. Raw receipts stay on the owner's Google Drive.

## 10. Stop rules (from `docs/PRIORITY_EXECUTION.md`)

- Never label MTP equal to Base unless MTP activation and correctness were observed. If not supported, mark `BLOCKED_MTP`.
- Never silently switch to text-only mode. Never drop images.
- Never score a truncated or image-dropped sample as official native accuracy.
- Never call a proxy metric official.
- Owner decides after each job whether to continue. The repo defines no automatic stop threshold.

## 11. Planned, not measured

From `runbooks/QWEN_PRIORITY.md`, planned values only:

- Pilot: 48 serving cells, 2,816 measured requests planned, up to 384 native predictions.
- Confirmatory (a new campaign ID, separate config): 1,008 cells, 45,696 requests, up to 7,520 predictions.

No GPU result exists. Nothing in this runbook shows a latency, a throughput or an accuracy.

## 12. Known gaps

- G1–G9 in §0 are unmet, so §5 onward cannot run.
- Where the inventory preflight runs (§3) is undefined.
- The cell chosen by `--max-cells 1` (§6) is undefined.
- The EOS / stop-token question for both models is open (`docs/harness/QWEN35_4B_STAGING_PLAN.md` §4). Check `stop_reason` per request.
- vLLM 0.29.0 loading of the 4B architecture and MTP activation are unverified.
- The gold-free test for requests (`docs/harness/DATASET_READINESS.md` §2.5, §3.6) does not exist.
- The `colab` CLI session was not checked in this task.
