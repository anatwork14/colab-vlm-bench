# Grounded hallucination and accuracy evaluation (v1)

## What counts as hallucination?
A factual assertion unsupported by the evidence provided in **the evaluated multimodal record**, or contradicted by evidence in that record. Wrong reasoning with a fully supported premise, formatting errors and ordinary lexical mismatches **must not** automatically be labeled hallucination.

## Independent annotation protocol
1. Run full original native MemLens or MMLongBench prompt, images/PDF pages retained, temperature 0, greedy; never use truncated serving workloads.
2. Record response, sample ID, model, method, evidence session/page IDs, image hashes and full context digest. Do not give gold labels to target model.
3. Independent evidence judge or humans break each non-abstaining response into *atomic factual claims*; mark each as SUPPORTED, UNSUPPORTED, CONTRADICTED or UNVERIFIABLE. Include evidence IDs for supported and contradicted claims. Unverifiable is not silently counted as hallucinated.
4. Independently decide response abstained (boolean), question answerability (ANSWERABLE/UNANSWERABLE) from publisher metadata and approved annotator. Don't treat the mere presence of "don't know" as sufficient without a rubric.
5. Freeze judge checkpoint/revision, prompt rubric, modality accessibility, evidence digest, coverage and reviewer agreement. Blind review should include stratified random sample. Import exact JSONL via `vlmbench hallucination-score`.
6. Report all denominators, coverage and failure rate; incomplete judge output does not qualify as official hallucination evidence.

## Primary metrics
- **Unsupported claim rate**: UNSUPPORTED / (SUPPORTED+UNSUPPORTED+CONTRADICTED). Evidence contradiction reported separately.
- **Contradiction rate**: CONTRADICTED / evaluable claims.
- **Hallucinated claim rate**: (UNSUPPORTED+CONTRADICTED) / evaluable claims.
- **Answer hallucination rate**: non-abstained responses containing ≥1 unsupported/contradicted claim / non-abstained responses with ≥1 evaluable factual claim.
- **False assertion rate (FAR)**: non-abstentions on verified UNANSWERABLE items / all verified UNANSWERABLE items.
- **Correct abstention rate**: abstentions on UNANSWERABLE / all verified UNANSWERABLE.
- **Coverage / UNVERIFIABLE**: judge-annotated responses / frozen expected responses; unverifiable claims separate, not zero.
- **Quality**: publisher official MemLens accuracy, MMLongBench-Doc overall Accuracy/F1, grouped by refusal/unanswerable + evidence type/page. Proxy EM/F1 remains auxiliary.

The judge scores a statement relative to available evidence; it cannot claim ground truth from arbitrary external world facts not present in the evidence packet. An independent judge that is also one of the evaluated models requires bias disclosure, and an external judge is preferable.

## Sample annotation
See `configs/hallucination_annotation.example.jsonl`. All fields except a schema placeholder must be replaced. The example is a schema illustration, NOT an evaluated model output.

## Scope limitations
MemLens's answer-refusal and MMLongBench's unanswerable subsets support **answerability-specific** metrics. The datasets' official metrics are not hallucination claims by themselves. Raw model predictions alone cannot produce an audited grounded hallucination rate without evidence access and judge labels.

Inference at temperature 0 is greedy, not necessarily bit-for-bit deterministic across different batch sizes and fused kernels. Check source hashes and stability.

When no independently judged labels are present, report `hallucination_rate = NOT_MEASURED`, never 0.
