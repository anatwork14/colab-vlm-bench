# GPU cost gates — audit addendum

## Cost-preserving quality and warmup circuit breakers

- Serving warmups execute **at offered CCU**, rather than 32 slow sequential 16K prefill requests. This tests the concurrency-specific scheduler without excessive A100 wall time.
- Engineering-pilot serving stops the current invocation after the first runtime ERROR/TIMEOUT/PARTIAL_FAILURE and preserves that failure receipt; unattempted cells stay NOT_RUN.
- Native 32K+/many-image quality pilot stops after **two consecutive HTTP/inference failures**, writes a \`QUALITY_STOPPED_AFTER_REPEATED_FAILURES\` receipt and synchronizes each expensive prediction to Drive. Fix model/context/visual limits before rerunning with a new vetted campaign. A pilot aborted early is NOT an official quality score.
- Successful all-cells and already-completed quality runs skip vLLM startup entirely on repeat invocation. Colab resource checks precede model staging.
