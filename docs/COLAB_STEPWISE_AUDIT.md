# Step-by-step Colab notebook audit — A100 40GB / ~80GB system RAM / ~200GB SSD

Scope: [Qwen3.5-0.8B.ipynb](../notebook/Qwen3.5-0.8B.ipynb) and [Qwen3.5-4B.ipynb](../notebook/Qwen3.5-4B.ipynb), evaluated 2026-11-08. Both have 38 notebook cells including **18 executable code cells**. GitHub CPU tests confirm code syntax and cost-guard contracts, not A100 execution.

**Important:** Colab compute units may be consumed from the moment A100 is allocated, even during CPU-only parsing. For zero-GPU source/path checking, start in a *separate CPU runtime* with `CPU_PREFLIGHT_ONLY=True`, run through the added CPU resource gate, then STOP. Reopen or switch to A100 only after PASS.

| Code cell (0-based notebook index) | Operation | RAM/SSD/GPU risk | Audit / policy |
|---:|---|---|---|
| 2 | Configure paths/switches | CPU trivial | Default all GPU switches OFF; choose active datasets; phase and budget are explicit |
| 4 | Drive mount and hardware check | CPU / GPU allocation | CPU_ONLY skips GPU check; GPU mode requires single A100 40GB, never silently substitutes L4/T4 |
| 6 | Clone/update repo, install dependencies | SSD/network; vLLM wheel may be large | CPU_ONLY installs only source/schema dependencies; GPU mode pins vLLM 0.29.0 and records versions/SHA |
| **8** | **CPU resource + dataset preflight** | Reads only source file metadata and small QA sample | Measure actual available RAM and SSD; min 10 GiB RAM, min max(18 GiB, 10GiB+1.35×checkpointGiB) SSD; verify weights, tokenizer, MemLens images and sampled Parquet doc_id PDFs |
| 10 | Stage model weights locally | Local disk + Drive I/O | Disabled in CPU_ONLY; stage SHA-verified weights, no new Hugging Face download |
| 12 | Native dataset normalization | Long-context system RAM + PDF rendering disk | MemLens streams one QA at a time via ijson; MMLong accepts JSON/Parquet and small QA batches, renders PDF pages on disk; pilot limited to first 200 source QA for engineering only |
| 14 | Source-derived serving prompts | Tokenizer CPU memory/time | Reads JSONL via bounded offsets, keeps one real image and capped two-document PDF text cache; cannot be officially quality-scored |
| 16 | Campaign freeze | CPU trivial | Validate model/mode/dataset/grid, temperature=0, top_p=1, revision and paths; different campaign IDs |
| 18 | Seven method plan | CPU trivial | Stock vLLM Base/MTP only. H2O/DA/ECRD plus combinations BLOCKED until engine-qualified |
| **20** | **Base server smoke + token annotation** | **First expensive A100 launch**, 16K prefill | Start ONLY when `RUN_GPU_ANNOTATION=True`, real-image smoke first, then at most 8 new calls/dataset by default; source-SHA checked resumable annotations; always stop server |
| **22** | **CPU verified-workload readiness** | Lightweight JSONL scan | Report eligibility per text length and CCU; 17/40/160 unique prompts required for CCU 1/8/32 including warmups. Zero eligible prevents server startup |
| 24 | Base serving | GPU decode/memory/CCU | Disabled by default. Maximum 6 cells/invocation. Runner refuses zero-ready or already-recorded jobs before CUDA. Pilot circuit breaker after runtime failure |
| 26 | Native MTP serving | GPU extra KV/draft/verifier | Disabled by default. Same input, seed, temperature and output tokens; MTP logs/activation still need real A100 review |
| 28 | Seven-method dispatcher | GPU only for Base/MTP | Disabled by default. Other five arms log BACKEND_UNQUALIFIED, no fabricated latency/quality |
| 30 | Native full-context quality | Very high 32K+images / 65K server | Disabled by default; images > configured max cause CPU pre-launch block. No context truncation, errors recorded; no official score absent upstream scorer |
| 32 | Evidence-hallucination scoring | CPU unless external judge separately invoked | Disabled by default, needs independent claim-level evidence annotations and exact sample IDs; missing labels = NOT_MEASURED |
| 34 | Report, paired comparisons | RAM/Drive receipts | CPU, status-preserving, sample-ID paired speedups only when measured; host RAM/SSD telemetry separate from VRAM |
| 36 | Export runbook ZIP | Small local disk output | Download summary only on opt-in; raw receipts remain on Drive |

## Honest resource/capacity limitations

- *Available* vs total: 80GB system RAM is not 80GB free, and 200GB SSD total is not 200GB available. The preflight uses measured live availability. Google Drive quota is a different filesystem not guaranteed by this gate.
- GPU: A100 40GB BF16 TP1; 16,384 text tokens **plus** visual tokens at offered CCU 32 can still OOM despite 80GB system RAM. The pilot's errors are scientifically meaningful capacity data; do not enlarge timeouts indefinitely.
- QA: MMLongBench pilot cap 200 is not the 1,091-question official full evaluation. A full run must use the full source and upstream scorer.
- Annotation: Eight one-token requests are a first-smoke budget, NOT enough for a CCU32 measurement. Raise explicitly in later, approved batches only after the first small run succeeds; receipts survive runtime restarts if the source hash remains frozen.
- GPU model-load/MTP, vLLM streaming schema, device memory, official quality and independent hallucination judges cannot be certified by GitHub CPU tests alone.
- Native MemLens + MMLong 65K quality and high-image-count docs may exceed A100 capacity; never silently drop visual evidence or substitute text-only performance.

## Recommended cost-preserving order

1. **CPU runtime:** config → Drive → CPU-only setup → CPU gate. STOP and resolve missing paths/schema/free resources if blocked.
2. **A100 session, explicit approval:** setup → stage model → normalized sources → derived prompts → campaign freeze / seven-arm plan.
3. **Bounded GPU smoke:** only 8 token-annotation calls/dataset; inspect server logs, first real image response, token counts.
4. **Scale annotation intentionally:** collect enough verified unique 2K and 16K prompts; look at readiness at CCU 1, 8 and 32. Do not start full serving with only 8 verified examples.
5. **Serving pilot:** Base before MTP, 6 cells/invocation, inspect after first failure and per-cell Drive receipts. Only then consider native quality.
6. **Report:** results and failures to Drive, independent official scorer and hallucination judge separately, and only expand to 4B/other methods with budget-aware evidence.

No A100 computation was performed during this software audit.
