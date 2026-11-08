# Qwen3.5 Seven-arm research campaign

**Status: PLANNED_NOT_RUN** — no 7-arm A100 run and no hallucination judge score have been produced.

Models: Qwen3.5-0.8B / Qwen3.5-4B. Two datasets: MemLens / MMLongBench-Doc.
Methods: Base, H2O+Base, DA+Base, ECRD+Base, native MTP SD, H2O+MTP, DA+MTP.

| Campaign | Planned serving cells | Base/MTP cells supported by existing runner | External methods (blocked until port) |
|---|---:|---:|---:|
| [Pilot](../configs/a100_qwen_7methods_pilot.json) | 168 | 48 (not GPU-qualified) | 120 |
| [Confirmatory](../configs/a100_qwen_7methods_full.json) | 3,528 | 1,008 (not GPU-qualified) | 2,520 |

Method status **does not imply measured performance**. H2O eviction and Declarative Attention masking must be implemented and independently qualified for Qwen3.5 + A100. ECRD is inference-time evidence-constrained reweighting, not KV marking. Native quality requires full evidence plus official accuracy and independently judged claim-support hallucination rates.

Temperature: **0**, top_p: **1.0**, both frozen for all seven methods. Quality uses natural EOS; fixed-length serving uses 256 actual tokens.

[Full method protocol](../docs/SEVEN_ARM_PROTOCOL.md).
