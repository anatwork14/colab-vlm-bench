# Independent dataset repair audit — DS-01 through DS-07

**Current verdict: CONDITIONAL.** Re-review of repair commit `555f870` passes
the DS-02/DS-03/DS-04 repair scope. The original blocking reproductions are
resolved. DS-07 upstream provenance remains conditional as recorded below.

The original audit follows as historical evidence; its BLOCKED verdict is
superseded by the re-review at the end of this document.

Reviewer: sol-lead, temporarily assigned independent dataset review by owner via
agy-orchestrator. The reviewer did not author the dataset implementation or
repairs. This verdict excludes the reviewer's runtime changes.

Reviewed dataset repair commit: `5d917ffd710ef4dff3e5dfbfe04293c822cf96d8`.
Validated integrated source: `baca7a06d5a5372a64a9000aa51de079fdb76507` in
`/mnt/ssd/genos2/research_workspace/colab-vlm-bench`.
The dataset diff between these commits changes one MemLens exception from
ValueError to TypeError; the findings below apply to both versions.

## Validation

Requested command, run against integrated source:

```sh
/mnt/ssd/genos2/.local/bin/uv run --extra dev pytest tests/test_amber.py tests/test_datasets.py
```

Result: **23 passed in 0.23s**, Python 3.14.7, pytest 9.2.0.
The initial sandbox attempt could not create uv's cache lock; the same command
then succeeded through the tool's approved escalation path.

Additional independent CPU probes are recorded at
`/tmp/sol-lead-data-audit-probes.json`. No dataset/runtime code was modified,
no Drive was accessed, and no model inference or allocation was performed.

## Findings requiring repair

### DS-02 — P0: dictionary ground truth bypasses schema validation

Location: `src/vlmbench/datasets/amber.py:364-365`, with label interpretation
at `:405-435`. List inputs validate labels; dictionary inputs are accepted as-is.

Reproduction:

```python
score_amber_discriminative(
    [{"id": "1", "prediction": "No"}],
    {"1": {"label": "INVALID", "dimension": "existence"}},
)
```

Observed: accuracy **1.0**, hallucination_rate **0.0**. The invalid label is
silently treated as a negative answer rather than rejected.
Also, `normalize_amber` accepts an unsupported root object `{}` and returns a
successful empty normalization (`count=0`) because of the wrapper fallback at
`:164-165`. This can hide an upstream schema mismatch.

Required: normalize and validate every supported input representation through
the same strict schema; reject invalid labels, missing dimensions, unexpected
wrapper objects and duplicate annotation IDs. Add negative tests for dictionary
inputs and unsupported roots. Validate real pinned upstream-format fixtures
before claiming publisher-schema compatibility.

### DS-03 — P1: missing responses still disappear from recall/F1

Location: `amber.py:409-425`, `:456-468`, `:668-675`.

Reproduction: two positive existence ground-truth items, with only one answered
`Yes`. Observed count=2, missing_count=1, accuracy=0.5, but precision=recall=F1=1.0.
Replacing the missing answer with an unparseable answer follows the same
early-continue path and excludes it from the positive-class denominator.

With an additional correctly described generative `dog` item, the unified
scorer reports **amber_score=1.0**, despite the unanswered discriminative item.
The repair correctly fixes accuracy and blocks composites when an entire task
is absent, but does not prevent a perfect composite from incomplete responses.

Required: define and enforce failure treatment across all reported metrics,
using the frozen ground-truth population for recall and coverage. Missing and
unparseable positive items must not disappear from recall. Suppress the
composite unless its component metrics have valid documented failure treatment.
Add missing-positive and unparseable-positive regression cases.

Scientific note: `score_amber_generative` increments its hallucinated-description
counter when a response is missing (`:561-564`). Missing predictions establish
execution failure, not evidence of hallucination; expose a separate failure or
coverage metric rather than labeling that event as hallucination.

### DS-04 — P1: caller can relax or disable the mandatory 5% limit

Location: `src/vlmbench/datasets/memlens.py:146-158`, `:229-232`.

Tokenizer presence and default 5% rejection work. However, `tolerance` has no
finite/range validation. The sole comparison uses the caller's value.

Independent deterministic test: history `"H" * 2000`, question `"Q?"`, target
128; a character tokenizer whose decode returns the first half of its input
models a non-round-trip token conversion. Best candidate has **92 tokens**.

| tolerance | emitted records | skip reason |
|---|---:|---|
| 0.05 | 0 | tolerance_mismatch_for_128 |
| 0.5 | 1 | none |
| NaN | 1 | none |

92 is outside [121.6, 134.4], yet both relaxed and NaN values admit the row.
This probe is synthetic; it tests enforcement at the public API boundary, not
the behavior of a qualified Qwen tokenizer.

Required: reject non-finite tolerances and values outside the allowed interval
(at most 0.05), or freeze tolerance to 0.05. Validate target lengths as positive
supported bins. Add tests for NaN, infinity, negative and >0.05 values.

## Per-finding disposition

| Finding | Disposition | Evidence / limitation |
|---|---|---|
| DS-01 | PASS for proxy labeling | Unified result AMBER_PROXY_UNVERIFIED; component statuses include PROXY_UNVERIFIED; normalized quality metadata sets official_scoring_eligible=False. No official parity claim is justified. |
| DS-02 | BLOCKED | List validation improved; dictionary bypass and unsupported wrapper remain. |
| DS-03 | BLOCKED | Accuracy fixed; recall/F1 and composite still exclude missing positives. |
| DS-04 | BLOCKED | Mandatory tokenizer and default skip accounting work; caller can defeat the 5% limit. |
| DS-05 | PASS | Missing/blank query raises ValueError; no replacement description prompt. |
| DS-06 | PASS | AMBER fixed-serving records use amber_serving and omit requested_text_length_bin at record and metadata levels. This does not establish campaign integration. |
| DS-07 | CONDITIONAL | Manifest contains a 40-character pin and per-image SHA256 is computed from file bytes and cached by resolved path. Pin existence, expected-file schema and license provenance were not independently established. Attempts to fetch the exact pinned upstream README/query/commit through the web tool returned cache misses. |

## Scope and next gate

The existing tests cover useful normal and error paths, but are based on
synthetic fixtures and do not cover the reproductions above. Readability and
package boundaries are adequate for this review; resolved-path image containment
and hashing are present. The proxy's vocabulary/parser limitations remain
explicit and cannot establish official scorer parity.

Return the three blocking items to the dataset owner, add regression evidence,
and perform independent re-review. This audit does not authorize Phase 4 paid
compute or certify data completeness, upstream scorer parity, or GPU readiness.

## Independent re-review — repair commit 555f870

Target: `555f870a239b8f08d7b2b6ed68531472d0e4df99`, inspected directly in
`wt/agy-worker-data`. HEAD matched that commit. Reviewed the implementation and
test diff from `5d917ff`; no implementation files were changed by the reviewer.

**Repair verdict: PASS for DS-02, DS-03 and DS-04. Overall audit:
CONDITIONAL solely on the outstanding DS-07 provenance qualification.**

### Re-run validation

```sh
PYTHONPATH=src /mnt/ssd/genos2/research_workspace/colab-vlm-bench/.venv/bin/python -m pytest tests/test_amber.py tests/test_datasets.py -q
```

Run directory: `/mnt/ssd/genos2/research_workspace/wt/agy-worker-data`.
Result: **29 passed in 0.20s**. This uses the existing dev environment with
imports from the repair worktree; no installation or resource allocation.

Independent assertions and outputs:
`/tmp/sol-lead-data-rereview-555f870.json`.

| Probe | Observed result | Disposition |
|---|---|---|
| Dictionary label INVALID | ValueError | DS-02 repaired |
| Missing dictionary dimension | ValueError | DS-02 repaired |
| Non-dictionary entry | TypeError | DS-02 repaired |
| Duplicate list IDs 1 and "1" | ValueError | DS-02 repaired |
| Unsupported normalization root `{}` | ValueError | DS-02 repaired |
| One correct positive, one missing | accuracy=0.5, recall=0.5, F1=2/3, missing_count=1 | DS-03 repaired |
| One correct positive, one unparseable | accuracy=0.5, recall=0.5, F1=2/3, unparseable_count=1 | DS-03 repaired |
| Correct generative item plus incomplete discriminative pair | composite=5/6 rather than 1.0 | DS-03 repaired |
| One correct generative caption, one missing | count=2, evaluated_count=1, missing_count=1, coverage_rate=0.5, CHAIR_s=0 | Missing/failure separation repaired |
| Tolerances 0.5, NaN, infinity, zero, negative, bool | ValueError for every case | DS-04 repaired |
| Targets empty, zero, negative, noncanonical, float, bool | ValueError for every case | DS-04 repaired |
| Prior lossy-tokenizer probe at default 0.05 | zero emitted rows; tolerance_mismatch_for_128=1 | Strict bin rejection preserved |

The changed code validates both dictionary and list scorer inputs before
scoring, adds positive missing/unparseable items to false negatives, and keeps
missing generative predictions outside hallucination counts with explicit
coverage. The tolerance guard is applied before reading native sources and
rejects non-finite values and bounds above 0.05. Canonical positive integer
target validation is present.

DS-01, DS-05 and DS-06 remain PASS within their original scope. DS-07 remains
CONDITIONAL: manifest/hash mechanics exist, but the pinned upstream source and
real publisher schema/license were not independently qualified in this review.
Proxy scoring labels must remain in place; report generative coverage alongside
conditional-on-response hallucination metrics. Re-review does not establish
full campaign wiring, official scorer parity, real dataset completeness, or GPU
readiness, and does not certify the reviewer's own runtime code.
