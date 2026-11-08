# Troubleshooting and negative result handling

| Symptom | Interpretation | Action |
|---|---|---|
| A100 not available, or wrong VRAM | No matched hardware | Stop; do not change to T4/L4/H100 and combine results |
| vLLM wheel fails CUDA import | backend incompatible | Pin correct CUDA/Python/torch wheel; record startup failure; no benchmark PASS |
| Gemma assistant MTP treated as draft_model | incorrect speculative path | Require method=mtp, matching assistant model and validated GPU backend trace |
| MTP server starts but slower than Base | possible scheduler/verification overhead | Keep negative throughput result; measure at CCU strata |
| MemLens native 32K plus images exceeds max model length | native quality infeasible at current cap | Qualify separate 65,536-token server; if still fails mark UNSUPPORTED_CONTEXT |
| Incomplete PDF rendering / missing pages | invalid MMLong official quality | Block; never score subset as publisher official |
| INSUFFICIENT_WORKLOAD | not enough model-token verified unique inputs at this bin | Generate/review more real-derived samples; do not replace with random IDs |
| GPU OOM at CCU32 | capacity result | Record exact request mix/model and queueing; no silent CCU change |
| Missing completion_tokens usage | cannot verify 256 output tokens | Mark PARTIAL_FAILURE, inspect vLLM streaming usage |
| Colab disconnect/restart | lost local SSD | Stage same model, restore receipt folder from Drive, skip verified PASS cells |
| Proxy token F1 available but no official scorer | cannot claim benchmark accuracy | Run upstream scorer and import its evidence separately |
| GitHub report shows NOT_RUN | no observed receipt | Check Drive export / config+dataset hash, not just server HTTP logs |

Always preserve raw logs and annotate failures under immutable run IDs. Do not merge synthetic/debug receipts into final scientific claims.
