# Metrics and evaluation contracts

Metrics are divided into quality, evidence-grounded hallucination, end-to-end serving and internal inference diagnostics. **All arms use `temperature=0`, `top_p=1.0`** for greedy decoding; speculative decoding is not presumed bitwise identical under all GPU schedules. Hallucination metrics require independently judged factual claims, not token F1. See [HALLUCINATION_EVALUATION.md](HALLUCINATION_EVALUATION.md). Every metric has an operational definition and a measurement scope. Do not fill a missing metric with zero.

| Metric | Definition / data | Requires | Scope |
|---|---|---|---|
| MemLens official accuracy | publisher LLM judge output, question type breakdown | independent judge + scorer revision | native |
| MMLongBench official F1 | publisher multi-stage scorer, reference answer + format | upstream scorer revision + whole document | native |
| Unsupported factual claim rate | UNSUPPORTED / evidence-evaluable claims; evidence required | independent evidence judge + provenance | native |
| Answer hallucination rate | responses with >=1 unsupported/contradicted claim / evidence-evaluable answered responses | independent claim/evidence labels | native |
| Contradiction rate | CONTRADICTED / evidence-evaluable claims | cited evidence IDs | native |
| False assertion rate / correct abstention | assert or abstain on independently labeled unanswerable items | publisher answerability + independent abstention labels | native |
| False refusal rate / answer coverage | abstention on answerable items; fraction of requests actually answered | independent answerability + abstention labels | native |
| Judge coverage, unverifiable count | annotated / expected; unknowns kept separate | frozen manifest and judge revision | native |
| Proxy EM / token F1 | normalized exact match and overlap, failures count zero | references held evaluator-only | auxiliary only |
| TTFT | HTTP send to first nonempty streamed content chunk (service-side; excludes client CCU-slot wait) | streaming server + client clock | serving |
| TPOT | (response finish - first content)/ (actual generated tokens-1) | completion usage metadata | serving |
| ITL | inter-token emission intervals, not SSE message intervals | engine token-level instrumentation | NOT_MEASURED currently |
| E2E p50/p95/p99 | submission (before waiting for a CCU slot) to completed streamed response; **includes client-side slot wait** (`client_queue_wait_ms`) | client clock | serving |
| Client queue wait p50/p95 | submission to CCU-slot acquisition, before the HTTP send; reported separately so E2E can be decomposed | client clock | serving |

**E2E comparability (review F-02, R3-F).** E2E now includes the client CCU-slot wait, so it is **not comparable** with receipts written before that change. Those receipts carry no `client_queue_wait_p50_ms` / `client_queue_wait_p95_ms` fields. Detection: `vlmbench.analysis.e2e_definition(receipt)` returns `INCLUDES_CLIENT_QUEUE_WAIT` or `LEGACY_NO_QUEUE_FIELD`. Rules:
- `report` withholds the E2E value of a legacy receipt (`e2e_p95_status = NOT_COMPARABLE_LEGACY_DEFINITION`) and sets `e2e_comparability = MIXED_NOT_COMPARABLE` when a campaign mixes definitions.
- `paired_speedups` does not pool a Base/MTP pair whose two receipts have different definitions. Those pairs are counted in `excluded_mixed_e2e_definition`.
- Throughput and TTFT are not affected by this change. Only E2E is.
| Throughput | committed *successful user-facing* output tokens / wall clock | verified completion_tokens | serving |
| Requests/s | successful requests / complete measured-cell wall clock | per request PASS | serving |
| Offered CCU | max simultaneous in-flight client requests | client concurrency semaphore | serving |
| Active GPU sequences | actual running sequences, queued requests | vLLM scheduler metrics | NOT_MEASURED currently |
| Peak VRAM | max sampled nvidia-smi total used MiB, summed over visible devices (0.5 s sampling; transient peaks between samples can be missed; includes CUDA context/cache, not allocated bytes). `parse_failures` / `query_failures` are reported; a non-zero count means the peak is not trustworthy | nvidia-smi interval | serving |
| Physical KV read/resident bytes | kernel/cache allocator measured bytes | internal backend instrumentation | NOT_MEASURED currently |
| MTP acceptance rate | accepted drafts / proposed drafts | engine trace + correct accounting | NOT_MEASURED currently |

**Fairness:** each model's Base/MTP arms must use same dataset digest, row IDs, image payload, warmup count, prompt lengths, random seed and target outputs; only inference mode changes. Across different model families, input content is maintained where possible but tokenizers and image embedding budgets differ. Full effective input token count must be reported, not inferred from text bins. No cross-hardware speedup aggregates from different GPU SKUs.

**Capacity:** FAIL, OOM, TIMEOUT, UNSUPPORTED_CONTEXT and INSUFFICIENT_WORKLOAD are not zero latency values; they are reported separately. A model that only reaches CCU8 cannot be credited as CCU32 success. Median per-repetition Base/MTP throughput ratios are computed only on matched PASS rows, with descriptive bootstrap CI for n>=3.

**Output:** Performance fixed target 256 actually generated tokens; if usage tokens are missing/mismatch, cell is PARTIAL_FAILURE even if HTTP 200. Native quality uses natural stopping.

| Host RAM minimum available / peak used | Periodic system RAM available/used during measured cells | `psutil` on Colab Linux; independent of GPU memory | serving |
| Local SSD minimum free | Periodic free space for `/content/vlmbench` filesystem | Linux disk usage; **not** Google Drive quota | serving |

**Cost protection (pilot):** any ERROR, TIMEOUT or PARTIAL_FAILURE stops remaining cells in that invocation so owners can inspect the first failure before exhausting A100 compute units. The fixed confirmatory protocol retains all attempted capacity/failure cells, but never substitutes successful values for failed cells. 
