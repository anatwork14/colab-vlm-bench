# Seven-arm benchmark protocol — Qwen3.5 0.8B and 4B (prospective)

**Research questions**: Can we improve long-context VLM grounded accuracy, reduce hallucinated claims, TTFT/TPOT/E2E, increase committed output tokens/s and supported CCU, and decrease peak GPU memory while retaining comparable output quality?

## Method arms
| ID | Arm | Required real implementation | Known qualification gap |
| --- | --- | --- | --- |
| B0 | Base / FullKV | stock vLLM, BF16 | Verify 0.8B/4B real VLM on A100 |
| B1 | H2O + Base | heavy-hitter + recent-token KV eviction, attention score accumulation | Old HF/FlexGen code, Qwen3.5 hybrid DeltaNet requires **attention-layer-specific** port; no reclaim of recurrent states claimed |
| B2 | DA + Base | declarative FSM/selective attention **read mask** over retained KV | upstream vLLM 0.29 profiles SM100, not A100; check prompt layout and mask correctness |
| B3 | ECRD + Base | visual evidence pool + candidate logits reweighting + optional GRIT decider | original implementation HF Qwen2.5-VL, not plug-and-play vLLM Qwen3.5. Auxiliary calls, VRAM/latency included |
| B4 | MTP SD | vLLM native MTP and correct verifier | validate MTP activation, greedy correctness vs B0 |
| B5 | H2O + MTP | B1 eviction + B4 draft/verify, safe KV rollback, cache residency across proposals | no verified combination |
| B6 | DA + MTP | B2 read masks + B4 draft/verify; shared declaration state through draft, target verify and rollback | no verified combination |

Sources (inspect / pin, never indiscriminately vendor):
- H2O https://github.com/FMInference/H2O @ \`ac75c2a8a9e76832b2a4139b9363373b56336bfb\`
- DA https://github.com/itsnamgyu/declarative-attention @ \`24110b7b441ad9423f55952a2c02d23eac80ec52\`
- ECRD https://github.com/uuuuZYC/ECRD @ \`0bc209f599bfef0d28d19e9a34024464406a9295\`

**Do not label ECRD as KV evidence protection.** ECRD is a decoding-time quality control (not a DA mask or H2O internal signal). This matters for attributing quality/VRAM claims. DA's upstream explicitly preserves original KV; do NOT claim DA lowers allocated KV capacity just because it reduces attended/read blocks. H2O can sacrifice evidence if eviction is wrong; track that failure mode directly.

## Frozen controls
- GPU: 1× Colab A100 40GB, BF16, TP1; vLLM version/build/driver/processor revision pinned.
- \`temperature=0\`, \`top_p=1.0\`, no stochastic sampling; \`enable_thinking=False\` where supported. Greedy decoding does not guarantee bitwise equality on GPU with batched scheduling; record divergence, do not fabricate equivalence.
- Same prompt IDs, image bytes, source context, output cap, seeds, CCU, CPU/client placement and server runtime for comparable arms. Stage weights from Drive to local SSD with SHA256.
- Performance uses *derived real-source* text bins 128/512/1K/2K/4K/8K/16K, plus separately measured vision-token overhead; EOS disabled for exactly 256 output tokens. **No official quality from these truncated derived prompts.**
- Native quality uses complete MemLens 32K (if fit) and full MMLongBench-Doc PDF pages with natural EOS, max output 256. Official quality needs upstream scorer. Evidence-grounded judge is independent and evidence-localized.
- Pilot: 2 models × 7 methods × 2 datasets × 2 context bins × 3 CCUs × 1 rep = **168 planned serving cells**, of which at most 48 are currently routable via native Base/MTP runner (others method-blocked).
- Full: 2×7×2×7×6×3 = **3,528 planned serving cells**. A100 40GB may fail 16K×CCU32; preserve failures/capacity boundaries. These are planned counts, not GPU results.
- In quality experiments choose native samples and judge thresholds *before* comparing methods; judge versions, annotation rubric, exact image evidence and sample manifests frozen.

## Endpoints / metrics
Quality: official accuracy and F1; refusal/abstention accuracy and false refusal rate on answerable questions (avoid over-abstention); answer coverage; answer hallucination rate and unsupported claim rate; contradiction rate; false assertion rate on unanswerable; visual evidence citation precision/recall when labels available. Always report judge coverage and \`UNVERIFIABLE\` separately; never equate any incorrect answer with hallucination.

Serving: TTFT p50/p95/p99, TPOT p50/p95, E2E p50/p95/p99, requests/s, committed output tok/s, goodput under declared SLO, offered CCU and actual active GPU sequences if instrumented, GPU peak MiB, physical KV/cache bytes if instrumented, decode/eviction/protected KV stats for qualifying arms, MTP acceptance/rollback if instrumented. No invented measurements: \`NOT_MEASURED\` for missing.

## Gate policy
- B0 and B4: code route exists, GPU method activation/correctness not proven.
- B1/B2/B3/B5/B6: BLOCKED_EXTERNAL_ENGINE_PORT_REQUIRED until source-pinned backend tests pass. A plain \`vllm serve\` process with renamed method **does not count**.
- Introduce an adapter only with real patch commit, tested startup and inference flag, isolated unit tests, image+text parity, KV/attention/decoding trace, resource attribution, exact dependency versions, smoke and negative tests. A100 must be explicitly qualified.
- ECRD can change greedy output and may increase compute/VRAM and worsen CCU. MTP should be checked against its **same target** decoding algorithm; ECRD changes logits and breaks naive lossless-SD assumptions (not one of the frozen 7 arms).
- H2O/DA+SD must verify draft/target cache semantics, speculative rejection rollback and retained evidence after each step. Unqualified combinations are not silently simulated.

## Statistical comparisons
Primary: each arm vs B0 on same model and dataset; B5 vs B1 and B4; B6 vs B2 and B4. Report quality deltas with confidence intervals on matched native question IDs, serving speedup with repeated same-cell receipts. Report percentage of attempted cells that fit, not just successful throughput averages. No claim of universal Pareto improvement without data.
