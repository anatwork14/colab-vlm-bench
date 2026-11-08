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

## 2b. Workload annotation and quality

Native full-data JSONL is created with prepare-memlens or prepare-mmlong. Derived serving candidates are generated using derive-serving and annotated per target model with annotate-serving **after** staging the matching processor/checkpoint. These extra 1-token requests consume A100 time. The annotation produces measured backend effective prompt tokens and tokenizer-derived text tokens; image overhead is separately recorded. Do not confuse derived serving inputs with official native quality questions.

For official native quality, use the upstream MemLens/MMLongBench scoring tools and import-official-score; proxy token F1 is not official.

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

## Prioritize low-cost failure discovery (Colab RAM ~80 GiB, SSD ~200 GiB)

1. **CPU runtime first**: use the Qwen notebook `CPU_PREFLIGHT_ONLY=True`; mount Drive; validate model/config/tokenizer and small QA/PDF samples, actual **available** RAM and free SSD. No CUDA model, no vLLM server. Minimum available RAM 10 GiB and free local SSD max(18 GiB, 10 GiB + 1.35×checkpoint GiB); choose a larger margin for full PDF rendering. Drive space is separate.
2. **GPU smoke**: after CPU PASS manually select A100 40GB, set `CPU_PREFLIGHT_ONLY=False`; validate driver/vLLM and one actual multimodal image request before bulk annotation.
3. **Annotation budget**: begin with 8 new one-token requests per active dataset per invocation. Resume partial verified annotations. Check readiness per 2K/16K text bin and offered CCU 1/8/32; 160 unique requests required for CCU32, including warmups.
4. **Serving**: Base before MTP, max six cells per invocation; no experiment with zero eligible cells starts vLLM. Do not conclude full GPU readiness from CPU tests.
5. **Native quality**: run only with explicit switch, verify full-context image count fits configured max; report unanswerable, OOM and not-measured rather than truncating PDF/history.
6. **Durability**: per-cell results and each expensive native quality prediction are copied to Drive. Do not remove failed receipts or re-use a campaign ID with edited protocol.

A100 GPU time and system RAM are independent resources. A 200 GiB disk *total* does not imply 200 GiB is free. Always read the measured preflight report.
