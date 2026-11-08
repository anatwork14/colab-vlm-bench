# Agent task queue — seven independent baseline methods

**Owner update (2026-11-08):** Benchmark only Qwen3.5 0.8B and 4B. Extend seven arms over MemLens + MMLongBench-Doc, separate native quality and derived serving performance; A100 40GB only; temperature=0, top_p=1.0.

Methods (canonical): base, h2o_base, da_base, ecrd_base, mtp, h2o_mtp, da_mtp.

1. Run CPU plan: \`python -m vlmbench.cli methods-plan --config configs/a100_qwen_7methods_pilot.json\`. This does not use GPU.
2. For model 0.8B then 4B, first qualify Base + MTP native arms with a real image, inspect activation and correct numerical cache/resume behavior.
3. For H2O, use upstream [FMInference/H2O](https://github.com/FMInference/H2O) *source pinned*, port to Qwen3.5 **hybrid attention layers**, implement real KV evictions and trace physical cache occupancy; do not change the recurrent state of DeltaNet without separate study.
4. For DA, use upstream [itsnamgyu/declarative-attention](https://github.com/itsnamgyu/declarative-attention), add A100 and Qwen3.5 compatible backend/masks, verify original KV is not removed, compare attended/read bytes (do not relabel as VRAM saved).
5. For ECRD, use upstream [uuuuZYC/ECRD](https://github.com/uuuuZYC/ECRD), verify Qwen3.5 token logits processor compatibility; include visual decider and all auxiliary model costs. Do not substitute evidence-augmented prompting for ECRD.
6. H2O+MTP and DA+MTP require real hybrid engine and cache/rollback correctness qualification before latency measurements. No flag-only simulation.
7. Report native official accuracy/F1 and separate evidence-grounded factual hallucination metrics via independent judge, with evidence/session/page IDs. Do not claim official hallucination metrics from token F1 alone.
8. Only after independent GPU qualification, permit each method's full 7 context×6 CCU×3 repeat matrix; if capacity/compatibility fails, report blockers. Never fill blanks with baseline values.
9. Publish both favorable and negative results with method/model/source SHA, data revisions, image assets, actual A100 memory, accuracy denominator, judge agreement, TTFT, TPOT, throughput, VRAM and offered/active CCU when measured.

Detailed protocol: [SEVEN_ARM_PROTOCOL.md](SEVEN_ARM_PROTOCOL.md).
Hallucination metrics: [HALLUCINATION_EVALUATION.md](HALLUCINATION_EVALUATION.md).
Third-party adapter audit: [THIRD_PARTY_METHODS.md](THIRD_PARTY_METHODS.md).

Note: H2O and DA target cache/attention runtime; ECRD changes token selection and can increase overhead. Accuracy improvements are research hypotheses, not guaranteed outcomes. A100 40GB may OOM at 16K×CCU32.
