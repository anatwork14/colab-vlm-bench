# A100 campaign resource/time budget — prospective planning only

## Expanded seven-method research budget (not a timing measurement)

Two Qwen3.5 models × seven method arms × two datasets × seven input lengths × six CCUs × three repeats = **3,528 proposed serving cells**. Pilot (2K/16K × CCU1/8/32 × one repetition) = **168 planned cells**. The original Base/MTP 1,008 confirmatory cells are the **2/7 subset**, not a separate comparable 7-arm study.

A proportional extrapolation from the earlier two-arm 25–65 GPU-hour planning budget suggests approximately **88–228 A100 GPU-hours**, **before** uncertain H2O/DA/ECRD porting, extra evidence extraction/decider inference, judges and restart overhead. This is only a rough capacity upper/lower planning exercise and is **not validated**; external arm runtime could be far slower or blocked completely. Do not reserve a full paid 7-arm campaign until individual method GPU pilots produce real timing receipts. Stop/go via docs/SEVEN_ARM_PROTOCOL.md.



## First-results priority: Qwen3.5-0.8B + Qwen3.5-4B

- **Pilot:** 2 models × 2 modes × 2 datasets × 2 text lengths (2K/16K) × 3 offered CCUs (1/8/32) × 1 repetition = **48 cells**, **2,816 measured requests**, **656 warmups**, **720,896 requested output tokens**.
- **Confirmatory:** 2 models × 2 modes × 2 datasets × 7 lengths × 6 CCUs × 3 repetitions = **1,008 cells**, **45,696 measured requests**, **10,584 warmups** and **11,698,176 requested output tokens**.
- **Native quality:** approximately 1,880 dataset questions × 4 model/mode configs = **up to 7,520 complete-context predictions**, actual publisher release frozen at execution. Pilot uses a deterministic 48-question subset per dataset/config (384 predictions total), explicitly **not an official full score**.
- **Provisional GPU budget (unverified):** pilot around **3–10 A100 hours** including GPU qualification and annotation; two-model confirmatory serving **~14–30 A100 hours**, quality **~8–25 hours**, plus **~3–8 hours** for setup/restarts and Drive export. **Total first-priority full campaign approximately 25–65 GPU hours** before external judge and additional replay. Re-estimate from actual pilot receipts; no Colab runtime guarantee.
- **Stop/go:** do not allocate GPU for Qwen 2B/9B or Gemma E4B until the priority pair has paired throughput/latency, capacity boundaries, failure receipts and quality evaluation status documented.


Hardware: Google Colab 1× A100 40GB BF16 TP1. Colab GPU and per-session runtime are best-effort, not guaranteed; verify actual CU balance and availability on the owner's authenticated account before running.

## Planned volume

- 5 checkpoints × Base/MTP = 10 configs
- 2 datasets × 7 context bins × 6 CCUs × 3 reps = 2520 serving cells
- Exactly 114,240 measured requests and 26,460 warmup requests according to the frozen count function, before quality evaluation
- Nominal 29,245,440 requested output tokens (=114,240×256), provided no cell is blocked and all output lengths are honored
- Native quality: MemLens 789 + MMLongBench-Doc **publisher current 1091** = 1880 responses per configuration, or up to 18,800 responses across 10 configs. Publisher releases can change; freeze actual IDs rather than trusting README totals.

## GPU-hour forecast (not measured on this repository)

| Component | Low | High |
|---|---:|---:|
| Serving grid (two datasets, three reps) | 44.7h | 99h |
| Native quality 18,800 outputs | 20h | 60h |
| GPU qualification, restarts, storage verification | 5h | 14h |
| Optional long-running replay | 4h | 12h |
| Combined | ~74h | ~185h |

**The above is a budgeting assumption, not a guaranteed Colab duration.** Missing native full-context fits, different streaming output sizes, media decode and model loading can alter runtime substantially. External MemLens LLM judge costs/GPU are not included. A single session should run small bounded pilots and persist receipts before it expires.

## Calibration pilot — no scientific claims
Start with Qwen3.5-0.8B Base/MTP, then Gemma E4B Base/MTP on a handful of 2K and 16K bins, CCU 1/8/32. Record actual elapsed, model load and stage seconds, success/OOM/timeout, and how much Drive export time each cell uses. Recompute projected GPU-hours from real receipts before enabling full 42-cell runs.

Campaign config starts with max_cells_per_session=6 for a capped pilot invocation. Set to 0 for full matrix only after explicitly approving an adequate session budget and artifact recovery plan. The controller does not allocate or release Colab GPU; the operator owns lifecycle and compute-unit spending.
