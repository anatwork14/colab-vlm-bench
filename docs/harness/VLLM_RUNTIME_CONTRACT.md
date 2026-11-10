# vLLM runtime contract (ITER3 D4)

**Status: UNEXECUTED, except the local read-only checks named here.** No vLLM package is installed on this host, no GPU run was
made, and no model was loaded. Every other command below is a specification to run later, under owner approval, on the exact
A100 session. Expected outputs marked UNKNOWN must be recorded from a real run, not assumed.

Local checks that were run on 2026-11-10 (read-only, against the local Qwen3.5-0.8B directory and `configs/a100_qwen_priority_pilot.json`):
the §3.3 config snippet, the §3.4 preprocessor snippet, the §3.6 `server_argv` snippet, and the §3.1 `verify_vllm` call. Their output
matches the expected values stated in those sections. The §3.1 call returns `verified: false`, because no signed pin exists and
no vLLM is installed. The §3.2 GPU query, the install, the server start, and all smoke requests have not been run.

Scope: the pilot serving path `vlmbench.engine.server_argv` (vLLM 0.29.0, BF16, tensor parallel 1, Base and MTP modes),
for `qwen35_08b` (local, verified by MD5 against Drive copies where noted) and `qwen35_4b` (Drive candidate, not staged).

## 1. Pins and what is and is not pinned

| Item | Pin in the repo | Status |
|---|---|---|
| vLLM package | `vllm==0.29.0` (`pyproject.toml` extra `gpu`; `requirements-agent-gpu.txt`; `requirements-colab.txt` as a candidate profile) | pinned by version. Wheel hashes **not recorded** |
| torch and CUDA build for vLLM 0.29.0 | none | **UNKNOWN.** Must be chosen and recorded during qualification on the A100 image |
| Python | `requires-python >= 3.10` | pinned as a range |
| vLLM RECORD SHA-256 | owner-signed `pins.vllm.distribution_record_sha256` (ITER3 C1) | **not signed**. The capture step below produces the value for the owner to sign. It is not a pin |
| GPU name and uuid | owner-signed `pins.gpu` | not signed |
| Model revisions | owner-signed `pins.models.<id>` | not signed (`UNPINNED` in the config) |

`requirements-colab.txt` says itself that it is a candidate profile and that the wheel and CUDA must be qualified on the exact session.
This contract is that qualification, written down. Nothing here qualifies a wheel.

## 2. Environment install (UNEXECUTED)

Run on the A100 host. The environment lives outside the repository, and nothing is installed into the repo's `.venv`.

```bash
# UNEXECUTED. Path and interpreter are placeholders.
PY=/usr/bin/python3.11                      # any Python >= 3.10; record the exact version
ENV=/content/vlmbench-vllm-env              # outside the repository
$PY -m venv "$ENV"
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install -r requirements-agent-gpu.txt   # pulls vllm==0.29.0 and its torch
"$ENV/bin/python" -m pip freeze > "$ARTIFACTS/vllm_env/pip_freeze.txt"
"$ENV/bin/python" -m pip show vllm > "$ARTIFACTS/vllm_env/pip_show_vllm.txt"
```

Recording (required; the install itself is not evidence): `pip freeze`, the torch and CUDA versions (`python -I -c "import torch;
print(torch.__version__, torch.version.cuda)"`), and the wheel files' SHA-256 from the pip cache or `pip download` into a scratch folder.
Hash-locked installs need those hashes, and they are UNKNOWN until this step has run.

## 3. Validation commands (UNEXECUTED)

Each check writes one JSON line into `$ARTIFACTS/RECEIPT.jsonl`. A check passes only when its own output meets the expected
value stated here. Starting a process, or a server reaching "ready", is not a pass.

### 3.1 vLLM version and distribution (isolated interpreter)

The harness's check runs with `python -I` (ITER3 C-2). It is the same check that `scripts/harness_inventory.py` runs:

```bash
# UNEXECUTED
PYTHONPATH=src "$ENV/bin/python" -c "
import sys, json; sys.path.insert(0, 'scripts')
import harness_inventory as h
print(json.dumps(h.verify_vllm('0.29.0', pinned_record_sha256=sys.argv[1] if len(sys.argv) > 1 else None)))
" "$RECORD_SHA256_IF_SIGNED"
```

Expected: `module_version` = `0.29.0`, `dist_version` = `0.29.0`, `record_sha256` present. Without a signed RECORD hash the
result is `verified: false` with the reason, which is the correct output at this stage. The RECORD hash the run captures
is recorded as an artifact for the owner to sign (it is not a pin).

Stub check: the same command, run with a `PYTHONPATH` directory that contains a fake `vllm/` package and `vllm-0.29.0.dist-info`,
must not return `verified: true` (regression: `tests/test_harness_r2_regressions.py`, `test_C2_*`).

### 3.2 GPU identity and capability

```bash
# UNEXECUTED. Absolute path, as the harness requires (G-PATH); do not resolve through PATH.
/usr/bin/nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total --format=csv,noheader,nounits
```

Expected for the pilot: name contains `A100`, `compute_cap` >= 8.0 (`REQUIRED_CAPABILITY`), `memory.total` >= 39000 MiB
(`REQUIRED_VRAM_MIB`). The uuid must equal the owner-signed GPU pin, which does not exist yet. The local GTX 1080 Ti (capability 6.1)
fails BF16 and fails this check.

Note: `src/vlmbench/engine.py` `a100_inventory()` still calls bare `nvidia-smi` through PATH (the G-PATH issue fixed in the harness
in round 3, not fixed there). Fix before the paid run.

### 3.3 Model configuration (per model directory)

```bash
# UNEXECUTED. MODEL_DIR is the staged directory (for example /content/models/Qwen3.5-0.8B).
python -I -c "
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
c = json.loads((d / 'config.json').read_text(encoding='utf8'))
t = c.get('text_config', {})
out = {
  'architectures': c.get('architectures'),
  'model_type': c.get('model_type'),
  'text_vocab_size': t.get('vocab_size'),
  'text_max_position_embeddings': t.get('max_position_embeddings'),
  'text_eos_token_id': t.get('eos_token_id'),
  'mtp_num_hidden_layers': t.get('mtp_num_hidden_layers'),
  'generation_config_present': (d / 'generation_config.json').is_file(),
}
print(json.dumps(out, sort_keys=True))
" "$MODEL_DIR"
```

Expected for `qwen35_08b` (local reference, read on 2026-11-10): `architectures` = `["Qwen3_5ForConditionalGeneration"]`,
`model_type` = `qwen3_5`, `text_vocab_size` = 248320, `text_max_position_embeddings` = 262144, `text_eos_token_id` = 248044,
`mtp_num_hidden_layers` = 1, `generation_config_present` = **false**. For the 4B the values are UNKNOWN until its config is read
(the Drive copy's `config.json` could not be read, see `DRIVE_INVENTORY.md` §5).

**EOS discrepancy (open).** The tokenizer's `eos_token` is `<|im_end|>` (id 248046), and `config.json` has 248044 under `text_config`.
With no `generation_config.json`, the stop token that vLLM uses is not fixed by the files. The smoke run must record `stop_reason` and
the token that ended each reply. A mismatch is a finding, not a fix.

### 3.4 Vision processor

```bash
# UNEXECUTED
python -I -c "
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
p = json.loads((d / 'preprocessor_config.json').read_text(encoding='utf8'))
print(json.dumps({k: p.get(k) for k in ('processor_class', 'image_processor_type', 'patch_size', 'temporal_patch_size', 'merge_size', 'image_mean', 'image_std')}, sort_keys=True))
" "$MODEL_DIR"
```

Expected for the 0.8B (local): `processor_class` = `Qwen3VLProcessor`, `image_processor_type` = `Qwen2VLImageProcessorFast`,
`patch_size` = 16, `temporal_patch_size` = 2, `merge_size` = 2, mean and std 0.5. The 4B's `preprocessor_config.json` on Drive has the
**same MD5** as the local 0.8B file (`7bcaedf7…`, `DRIVE_INVENTORY.md` §4a), so these values are expected for the 4B as well. Identity by
MD5 is a Drive-metadata fact, not a verification of the staged file: the staged file's SHA-256 must be checked after the transfer.

Image token accounting is checked by the smoke image request (§3.7): the reported `prompt_tokens` must include the image tokens, and the
count must be recorded with the image size. It must not be inferred.

### 3.5 MTP checks

Weights: the index must list MTP tensors. For the 0.8B the local index lists 15 `mtp` tensors (`QWEN35_4B_STAGING_PLAN.md` §3). For the 4B the
count is UNKNOWN until its index is read.

Serving: the MTP arm starts vLLM with
`--speculative-config {"method": "mtp", "num_speculative_tokens": 1}` (from `server_argv(..., mode="mtp")`). Base starts without it.

Evidence that MTP is active, which must come from the run and is UNKNOWN until it has happened:
- the server log lines that vLLM writes when speculative decoding is enabled (record the exact text);
- a difference in the backend's speculative metrics between Base and MTP for the same request set, if the backend exposes them.

Rule (`docs/PRIORITY_EXECUTION.md`): an MTP arm is never labelled equal to Base unless MTP activation and correctness are observed. If
MTP is not active or fails, the arm is `BLOCKED_MTP`. The run does not silently change mode.

### 3.6 Serving start

```bash
# UNEXECUTED. Built by vlmbench.engine.server_argv; do not hand-edit flags.
PYTHONPATH=src "$ENV/bin/python" -c "
import json, sys
from vlmbench.engine import server_argv
cfg = json.load(open(sys.argv[1]))
model = next(m for m in cfg['models'] if m['id'] == sys.argv[2])
print(json.dumps(server_argv(cfg, model, sys.argv[3], sys.argv[4])))
" configs/a100_qwen_priority_pilot.json qwen35_08b mtp "$MODEL_DIR" > "$ARTIFACTS/serve/argv.json"
```

Then start the server with that argv, log to `$ARTIFACTS/serve/server.log`, and wait for `GET /v1/models` to return 200 with the
served model name. Record the start time, the exit status if it stops, and the last 200 log lines.

Known config discrepancy (open): `quality.max_model_len` is 65536 and `quality.max_num_seqs` is 1, while `serving.max_model_len` is 32768
and `serving.max_num_seqs` is 32. A quality run must use a server with the quality values. The two are not reconciled here.

### 3.7 Smoke commands (UNEXECUTED)

```bash
# Text, greedy, thinking disabled (matches vlmbench.bench.request_payload)
curl -sS http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen35_08b",
  "messages": [{"role": "user", "content": "Reply with the single word OK."}],
  "temperature": 0, "top_p": 1.0, "max_tokens": 8,
  "chat_template_kwargs": {"enable_thinking": false}
}' | tee "$ARTIFACTS/serve/smoke_text.json"
```

Expected: HTTP 200; `choices[0].finish_reason` and the stop token recorded; `usage.prompt_tokens` and `usage.completion_tokens` present.
The reply content is recorded, not judged.

Image smoke: one PNG or JPEG from a scratch folder, sent as a `data:image/...;base64,` `image_url` part (the adapter path is
`vlmbench.workloads.resolve_messages`). Expected: HTTP 200 and `usage.prompt_tokens` larger than the text-only request by the image
token count. That count is recorded as a measurement, not assumed.

Repeat both smoke requests on the MTP server. The outputs at temperature 0 are compared with the Base outputs. A difference is recorded
as a finding. It is not a pass or a fail by itself.

### 3.8 Failure reporting

Each check writes one line to `$ARTIFACTS/RECEIPT.jsonl`:

```json
{"check": "3.1 vllm_version", "status": "PASS|FAIL|BLOCKED|NOT_RUN", "reason": "...", "artifact": "<path>", "artifact_sha256": "<sha>", "started_utc": "...", "exit_status": 0}
```

- `BLOCKED` names the missing input (for example "no owner-signed RECORD hash", "no A100 device").
- `FAIL` keeps the artifact. Nothing is retried silently, and a retry is a new line.
- A server that is up but reports the wrong model name, wrong `max_model_len`, or no MTP activity is `FAIL`, not `PASS`.
- A run that was not started is `NOT_RUN`. The absence of a line is not a pass.

## 4. Expected artifacts (listed; none exist yet)

| Path under `$ARTIFACTS` | Content | Created by |
|---|---|---|
| `vllm_env/pip_freeze.txt`, `vllm_env/pip_show_vllm.txt` | installed versions | §2 |
| `vllm_env/verify_vllm.json` | module and dist version, RECORD SHA-256 | §3.1 |
| `gpu/nvidia_smi.csv` | name, uuid, capability, memory | §3.2 |
| `model/config_summary.json`, `model/preprocessor_summary.json`, `model/index_summary.json` | §3.3, §3.4, and the tensor-group counts | §3.3–3.5 |
| `serve/argv.json`, `serve/server.log` | the exact command, and the log | §3.6 |
| `serve/smoke_text.json`, `serve/smoke_image.json` | HTTP responses and usage | §3.7 |
| `RECEIPT.jsonl` | one line per check, with the status rules of §3.8 | all |

Artifacts are kept out of the repository. Only the receipt's summary and hashes may be cited in the ledger.

## 5. Not closed

- No environment has been installed. The torch and CUDA pair is UNKNOWN.
- vLLM 0.29.0 has not been shown to load `Qwen3_5ForConditionalGeneration`, for either model (`QWEN35_4B_STAGING_PLAN.md` §7).
- MTP activation is not observed for any model.
- The EOS and `generation_config.json` question is open (§3.3).
- The quality and serving `max_model_len` and `max_num_seqs` values disagree (§3.6).
- `src/vlmbench/engine.py` still calls bare `nvidia-smi` (§3.2).
- The RECORD SHA-256, the GPU pin and the model revisions are not signed. Until they are, `READY` cannot be reached (`READINESS.md`).
