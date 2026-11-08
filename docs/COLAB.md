# Colab CLI operating manual — A100 40GB

Official CLI supports Linux/macOS; Windows users should use WSL/Linux/macOS controller. Confirm exact `colab <command> --help` before use.

## 0. Read-only discovery
```bash
colab version
colab sessions
colab usage
colab new --help
colab drivemount --help
colab exec --help
```

## 1. Paid resource approval
User decides approved A100 compute-unit budget, lease and start. **No script in this repo creates or stops a paid runtime automatically.**

```bash
# Only after user approves and CLI is authenticated:
colab new -s vlmbench-a100 --gpu A100
colab status -s vlmbench-a100
colab drivemount -s vlmbench-a100
```

First Drive mount may require interactive login. Check GPU name, VRAM, disk, host RAM, installed Python, CUDA, torch, vLLM, AIPerf; when wrong GPU returned, fail closed (do not accept A100 80GB as 40GB silently).

## 2. Runtime boot and model weights
Stage existing checkpoint from Drive to local VM SSD then verify SHA256; do not benchmark from FUSE-mounted Drive (significant I/O variance). Do not automatically download pretrained models or unpin revisions. vLLM candidate 0.29.0, but agent must verify exact wheel/driver/CUDA compatibility on session before full benchmark.

```bash
colab install -s vlmbench-a100 -r requirements-colab.txt
# inspect installed version and kernel compatibility first
```

Use [scripts/colab_controller.sh](../scripts/colab_controller.sh) after validating config. The controller uploads a source ZIP and runs a Python entrypoint via `colab exec`, pulls a small summary, and leaves GPU session ownership to user. Large evidence is periodically synchronized to mounted Drive.

## 3. Jobs and resume
- Assign immutable `campaign_id` and per-cell receipt path; do not reuse ID for different config hashes.
- Keep job duration shorter than available session allowance; save after each cell.
- Resume skips only verified identical PASS receipts; failed cells remain visible and can be retried with attempt suffix.
- Export large AIPerf raw logs/results, engine launch trace, environment JSON and model manifests to Drive. No API credentials in repo.

## 4. Integrity controls
Refuse non-A100 40GB, incompatible version, BF16 unavailable, wrong model hash, dataset missing/mismatched, context bin without token counts, MTP mode not visibly active, image count unsupported, or any part of full native question context missing. Mark unsupported cells and continue where safe; never manufacture numbers.

## 5. Shutdown
After receipts are durable and controller downloaded report:
```bash
colab stop -s vlmbench-a100
```
User/operator confirms session actually stopped to avoid charges.
