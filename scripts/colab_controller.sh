#!/usr/bin/env bash
# Authenticated local Linux/macOS controller. Never allocates/stops billable sessions.
set -euo pipefail
SESSION=""; CONFIG=""; MODEL=""; MODE=""; DATASET=""; MAX_CELLS=0; QUALITY=0; PUBLISH=0
APPROVE_PAID=0
while (($#)); do
  case "$1" in
    --session) SESSION="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --max-cells) MAX_CELLS="$2"; shift 2;;
    --quality) QUALITY=1; shift;;
    --publish) PUBLISH=1; shift;;
    --approve-paid) APPROVE_PAID=1; shift;;
    *) echo "Unrecognized option: $1" >&2; exit 2;;
  esac
done
(( APPROVE_PAID )) || { echo "GPU execution requires --approve-paid" >&2; exit 2; }
[[ -n "$SESSION" && -f "$CONFIG" && -n "$MODEL" && -n "$MODE" && -n "$DATASET" ]] || {
  echo "Usage: bash scripts/colab_controller.sh --session NAME --config LOCAL.json --model MODEL_ID --mode base|mtp --dataset memlens|mmlongbench_doc [--max-cells N] [--quality] [--publish]" >&2; exit 2;
}
command -v colab >/dev/null || { echo "Colab CLI missing" >&2; exit 2; }
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
[[ -z "$(git status --porcelain)" ]] || { echo "Dirty git state; commit/stash before GPU job" >&2; exit 2; }
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$CONFIG" "$TMP/campaign.json"
python - "$TMP/campaign.json" "$MODEL" "$MODE" "$DATASET" "$MAX_CELLS" "$QUALITY" > "$TMP/job.json" <<'PY'
import json, subprocess, sys
cfg=json.load(open(sys.argv[1], encoding="utf8"))
if cfg.get("dry_run", True):
    raise RuntimeError("dry_run=true; GPU work refused")
assert sys.argv[2] in [x["id"] for x in cfg["models"]]
assert sys.argv[3] in cfg["modes"]
assert sys.argv[4] in [x["id"] for x in cfg["datasets"]]
assert int(sys.argv[5]) >= 0
print(json.dumps({"campaign_id":cfg["campaign_id"],"model":sys.argv[2],
    "mode":sys.argv[3],"dataset":sys.argv[4],"max_cells":int(sys.argv[5]),
    "quality":sys.argv[6]=="1", "owner_approved_compute": True,
    "source_sha":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}))
PY
colab status -s "$SESSION"
git archive --format=zip -o "$TMP/source.zip" HEAD
colab upload -s "$SESSION" "$TMP/source.zip" /content/vlmbench_checkout.zip
colab upload -s "$SESSION" "$TMP/campaign.json" /content/vlmbench_campaign.json
colab upload -s "$SESSION" "$TMP/job.json" /content/vlmbench_job.json
colab exec -s "$SESSION" -f scripts/remote_entry.py
CID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["campaign_id"])' "$TMP/campaign.json")"
colab download -s "$SESSION" "/content/vlmbench_export_$CID.zip" "$TMP/results.zip"
mkdir -p "$TMP/results"
python - "$TMP/results.zip" "$TMP/results" <<'PY'
from pathlib import Path
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    for f in z.infolist():
        p=Path(f.filename)
        if p.is_absolute() or ".." in p.parts: raise ValueError("Unsafe archive")
    z.extractall(sys.argv[2])
PY
OUT="$ROOT/runbooks/$CID"
mkdir -p "$OUT"
for item in RUNBOOK.md summary.json environment.json; do
    [[ ! -f "$TMP/results/$item" ]] || cp "$TMP/results/$item" "$OUT/$item"
done
[[ -s "$OUT/summary.json" ]] || { echo "Missing summary bundle" >&2; exit 1; }
echo "Report: $OUT/RUNBOOK.md"
if (( PUBLISH )); then
    git add -- "runbooks/$CID/RUNBOOK.md" "runbooks/$CID/summary.json"
    [[ ! -f "$OUT/environment.json" ]] || git add -- "runbooks/$CID/environment.json"
    if ! git diff --cached --quiet; then
      git commit -m "results: A100 campaign $CID (observed receipts only)"
      git push origin HEAD
    fi
fi
echo "Verify raw receipts on Drive before manually stopping Colab."
