# STARLIGHT Stage 1 — Independent Invariants and Adversarial Acceptance Oracle

- Author: wF:p9 (Scientific Auditor)
- Base: 7d209f5 (iteration 2, ITER2 TASK D3)
- Status: **SPECIFICATION, not evidence.** Nothing in this document is a test result or a PASS.
- Independence: written from the owner contract in `/mnt/ssd/genos2/research_workspace/wt/MISSION_BRIEF_ITER2.md` (lines 11–15) and the Qwen3.5 architecture facts recorded below. It was **not** derived from, and the author did not read, the wF:p7 Stage 1 implementation. Where the owner contract is silent, the item is marked **AMBIGUITY** and needs an owner decision; the auditor does not silently choose.
- Use: wF:p7 tests and the controller are judged against this oracle. Where the implementation and this oracle disagree, the disagreement is reported to the lead; neither side is assumed right.

---

## 0. Scope

In scope: the Stage 1 controller only (non-speculative base decoding; no MTP, no draft/verify). Arms: `starlight_s1`, `starlight_s1_no_da`, `starlight_s1_no_grounding`. Comparators (IDs unchanged): `base`, `h2o_base`, `da_base`, `ecrd_base`, `mtp`, `h2o_mtp`, `da_mtp`.

Out of scope: Stage 2 (MTP), vLLM/A100 serving, physical VRAM measurement, paper-faithful model-declared DA (see `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md`; the custom DAFSM is not that protocol).

---

## 1. Definitions (the oracle's vocabulary)

| Term | Definition used by this oracle |
|---|---|
| **entry** | One token position `p` in one layer `l` (full-attention only) in one request `r`, including its KV for every KV head of that layer. Entries are identified by `(request_id, layer, position)`, never by array index. |
| **FA set** | Full-attention layers of Qwen3.5-0.8B: `{3, 7, 11, 15, 19, 23}` = `{i : (i+1) % 4 == 0}`, `i ∈ [0, 23]`. |
| **GDN layers** | The other 18 layers (Gated-DeltaNet, linear attention). Their recurrent/conv state is **not** token-addressable KV. |
| **DA-protected** | Entry named by a *valid, unexpired, current* declaration under the DA protocol of this repo (`<da:...>`-style declaration as implemented by the custom DAFSM). |
| **grounding-protected** | Entry that is an active, unexpired, current grounding-evidence span for the request (e.g., the image-token span or text evidence span cited by the answer). |
| **internally_protected** | `DA-protected ∪ grounding-protected`. (Owner contract, verbatim intent.) |
| **h2o_candidate** | Entry proposed for eviction by the H2O scorer (heavy-hitter score + recent window rule) for this step. External signal. |
| **eligible_to_evict** | `h2o_candidate ∧ ¬internally_protected`. (Owner contract.) |
| **budget B** | Hard maximum number of retained entries per (request, FA layer, step). See AMBIGUITY A-1 on the granularity. |
| **retained set R** | Entries kept after the step. |
| **status** | One of `OK`, `NO_ELIGIBLE_CANDIDATES_BUDGET_UNMET` (AMBIGUITY A-2), `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET` (owner contract, exact string). |

### 1.1 Qwen3.5-0.8B hazards that the oracle depends on

These are facts from the checkpoint and the repo's own audits, not from the controller:

- **H-1 Hybrid stack.** 24 layers: 6 FA layers and 18 GDN layers. A "KV cache" of this model is not a homogeneous per-layer KV tensor. Any controller that indexes a cache list by layer number without skipping GDN layers is wrong.
- **H-2 GQA.** 16 query heads : 2 KV heads. H2O scores are per KV head. Any eviction decision that is per token must say whether it applies to both KV heads; otherwise one head can keep a token the other head evicted (see AMBIGUITY A-3).
- **H-3 GDN state is not token KV.** Evicting a token from a GDN layer is impossible in the recurrent state (the state summarises all past tokens). The oracle therefore requires GDN layers to be **exempt** and the exemption to be documented in the decision record, not silently skipped.
- **H-4 Position ids.** Qwen3.5 uses position ids (including multimodal rotary positions for image tokens). Removing tokens must **not** renumber surviving tokens; RoPE positions of retained entries must be unchanged. Re-indexing the cache compactly after eviction shifts positions and silently changes the model's computation.
- **H-5 Vision spans.** Image pad token id 248056; `vision_start` 248053; `vision_end` 248054. Partial eviction inside an image span changes the visual token grid the model was trained on (see AMBIGUITY A-4).
- **H-6 Cache object.** The HF hybrid cache carries KV for FA layers and conv/recurrent state for GDN layers. Evicting KV tensors without a consistent GDN state (or with a cache object whose length counter is not updated) produces silent divergence, not an exception.
- **H-7 Prefill vs decode.** Eviction applied at prefill changes the attention every later decode step sees; eviction applied only at decode does not. The oracle requires the step at which eviction happens to be stated in the decision record.

---

## 2. Core invariants (must hold on every step, every arm that uses the protection)

Each invariant states what is checked, how, and what a violation looks like.

**I-1 Eligibility formula.** For every entry `e` at every step:
`e ∈ eligible_to_evict ⇔ (e.h2o_candidate = true) ∧ ¬(e.da_protected ∨ e.grounding_protected)`.
Check: recompute the formula from the per-entry record; any mismatch is a violation.

**I-2 Protected entries are never evicted.** For every step, `evicted ∩ internally_protected = ∅`. There is **no** exception, including budget pressure. The only permitted outcome for protection > budget is the status `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET` (I-4).
Violation: any protected entry absent from R after the step with no status change.

**I-3 No silent discard.** Every entry in the pre-step set is either (a) in R, or (b) in `evicted`, and each evicted entry carries a reason code from a closed enum (e.g. `H2O_CANDIDATE_UNPROTECTED`). An entry that disappears without a record is a violation. Count check: `|pre| = |R| + |evicted|` per (request, layer, step).

**I-4 Infeasibility is explicit.** If the number of internally protected entries (plus any entries that may not be evicted under I-5) exceeds the budget, then the step's status is `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET`, `R` must contain **all** protected entries, and the step must not evict any protected entry to meet the budget. A controller that truncates protected entries to meet B, or that reports `OK` while `|R| > B`, is a violation.

**I-5 Budget when feasible.** If status is `OK`, then `|R_fa_layer| ≤ B` for every FA layer and step. Entries not proposed by H2O are retained by default (no silent extra eviction; the controller must not evict unproposed entries to hit the budget — that would make the controller an unlabelled extra eviction policy). If the budget cannot be met that way, the status is the separate state in AMBIGUITY A-2, never `OK`.

**I-6 Layer scope.** Protection and eviction apply only to FA layers. For GDN layers: zero evicted entries, zero token-level state modification, and a decision record line `GDN_EXEMPT_NOT_TOKEN_ADDRESSABLE`. The GDN recurrent and conv state tensors are **byte-identical** before and after a controller step (checked by hash).

**I-7 Position preservation.** For every retained entry, its RoPE / multimodal position id after the step equals its position id before the step. Check: the position-id tensor restricted to R equals the pre-step tensor restricted to R (same order or with an explicit index map).

**I-8 Structural tokens.** `vision_start` and `vision_end` tokens (H-5) are never evicted. This is **not** in the owner contract; it is an auditor requirement derived from H-5 and must be ratified (AMBIGUITY A-4). Until ratified, a violation is reported as "structural eviction: policy-dependent", not as PASS.

**I-9 Determinism.** Same inputs (scores, declarations, budget, prompt, seed) give identical R and identical decision records across two runs in the same process and across process restarts.

**I-10 Decision records are complete.** Each step emits, per (request, layer): step index, prefill/decode flag, `|pre|`, `|R|`, `|evicted|`, the counts `n_da`, `n_grounding`, `n_h2o_candidates`, `n_eligible`, `n_protected_outside_budget`, status, and a per-entry list (entry id, flags, decision, reason). Missing fields are a violation, not a default of zero.

---

## 3. Activation assertions (the arm must actually use what it claims)

An arm whose label says "DA" or "grounding" must demonstrably consume that signal. Each assertion below must be checked from the decision records, not from the arm's configuration.

**A-DA** (`starlight_s1`, `starlight_s1_no_grounding`, and the `da_*` comparators): on at least one hard-case fixture with a DA/H2O conflict, `n_da > 0` **and** at least one H2O candidate with `da_protected = true` is retained. If `n_da = 0` on all fixtures, the arm is labelled `DA_NOT_ACTIVATED`, not PASS.

**A-GROUND** (`starlight_s1`, `starlight_s1_no_da`): on the grounding/H2O conflict fixture, `n_grounding > 0` and at least one H2O candidate with `grounding_protected = true` is retained.

**A-NODA** (`starlight_s1_no_da`): `n_da = 0` on every step of every fixture. Any non-zero value means the ablation leaked the DA signal.

**A-NOGROUND** (`starlight_s1_no_grounding`): `n_grounding = 0` on every step of every fixture.

**A-H2O** (all STARLIGHT arms): `n_h2o_candidates > 0` on the long-context fixture. An arm that never sees a candidate is not exercising eviction and cannot be compared.

**A-TOGGLE** (causal check): on the DA/H2O conflict fixture, `starlight_s1` and `starlight_s1_no_da` produce **different** retained sets (the DA signal changes R). If identical, the DA signal is inert. Same for grounding on the grounding/H2O fixture. Identical outputs must be reported as "signal inert", not as "no difference observed".

**A-MASK** (DA mechanism, separate from KV): DA preserves the original KV (upstream DA behaviour; see `docs/THIRD_PARTY_METHODS.md`). The oracle requires that a DA-protected entry is **not** physically freed. Any reported "KV bytes freed" for DA must be 0 unless measured physically; a logical attention-mask restriction is reported as `attention_mask_restriction_ratio`, never as KV saving.

---

## 4. Hard-case fixtures (required tests)

Each row is a test specification. "Assert" lists what must be true for PASS. Each case also states the failure it is designed to catch, so that a green run means something.

| ID | Hard case | Setup | Assert | Catches |
|---|---|---|---|---|
| HC-1 | DA/H2O conflict | An entry is both H2O top-score and DA-declared; budget forces some eviction. | Entry retained (I-2); `da_protected` recorded; A-TOGGLE holds. | Controller letting H2O score override declarations. |
| HC-2 | Grounding/H2O conflict | Same as HC-1 with a grounding span. | Entry retained; `grounding_protected` recorded. | Grounding flag ignored, or only DA consulted. |
| HC-3 | Protection > budget | Protected count = B+k, k ≥ 1. | Status `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET`; all protected retained; zero protected evicted (I-4). | Silent truncation to fit budget. |
| HC-3b | Protection = budget exactly | Protected count = B, plus candidates. | Status `OK`; all candidates evicted; `|R| = B` (I-5). | Off-by-one in infeasibility test. |
| HC-4 | Expired evidence | Declaration with expiry step < current step. | Entry **not** protected; reason `EXPIRED_EVIDENCE`; eligible if H2O candidate. | Stale evidence keeping memory forever. |
| HC-4b | Stale evidence (wrong request/turn) | Declaration whose request ID or turn ID does not match the entry's. | Not protected; reason `STALE_DECLARATION`. | Cross-turn leakage of protection. |
| HC-5 | Irrelevant evidence wrongly protected | Protected set includes entries a labelled relevance judge marks irrelevant to the answer. | Entries are protected (declaration is honoured) **and** the metric `protected_irrelevant_fraction` is reported. The test fails if the metric is absent. | Hiding protection cost. The point is measurement, not PASS/FAIL on quality. |
| HC-6 | Long-context pressure | Context ≥ 4× budget, many candidates. | I-3 count check holds at every step; A-H2O holds; no protected evicted; decision records complete (I-10). | Count drift over many steps. |
| HC-7 | Missing declaration | DA arm with no declaration for a request. | Behaviour is **fail-closed**: either an explicit error or conservative protection with reason `MISSING_DECLARATION`. Never silent "no protection, all eligible". | Missing declaration silently disabling DA. |
| HC-8 | Malformed declaration | Declaration with bad span (start > end, out-of-range, non-integer, NaN score). | Entry treated as protected (fail-closed) with reason `MALFORMED_DECLARATION`, or a raised error. Never eviction of the entry on the grounds that the declaration was unparseable. | Parser errors quietly becoming "unprotected". |
| HC-9 | Multi-request isolation | Requests A and B interleaved in one batch/loop; B's decisions compared with B run alone. | B's R and records identical to solo run (I-9 extended). A's declarations never appear in B's records. | Global protection state shared across requests. |
| HC-10 | Multi-turn continuity | Turn 1 protects entries E; turn 2 appends tokens and calls the controller again. | Entries in E still protected in turn 2 unless expired (HC-4); evicted entries from turn 1 do not return; positions unchanged (I-7). | Protection reset per turn; evicted tokens resurrected. |
| HC-11 | No eligible candidates | H2O proposes nothing, or every candidate is protected, budget exceeded by non-protected non-candidates. | Status per AMBIGUITY A-2 (not `OK` when budget unmet); no eviction; record shows `n_eligible = 0`. | Controller inventing evictions when nothing is eligible. |
| HC-12 | GDN exemption | Any multi-step run with eviction on. | I-6 hash identity on GDN state; decision record contains `GDN_EXEMPT_NOT_TOKEN_ADDRESSABLE`. | Pretending GDN state is KV. |
| HC-13 | Vision span integrity | Image span with candidates inside; `vision_start/end` included. | I-8 (ratified form); no change to image-grid-related position ids (I-7). | Partial eviction breaking the visual grid. |
| HC-14 | Cache consistency | Eviction at step t; continue decoding. | Logits after eviction equal logits of an independent recomputation of the same retained set **or** the difference is reported and attributed (H-6). | Length counter / GDN state drift. |
| HC-15 | Determinism | Run twice with the same seed and inputs. | I-9. | Hidden ordering or set iteration. |

Test-design rules (auditor requirements):

- Every hard case must have a **negative control**: the same fixture run through the H2O-only arm must show the eviction the protection is supposed to prevent (otherwise the fixture does not exercise the rule).
- Assertions use the decision records (I-10), not the final text output. Output text equality is not evidence of protection.
- A test that passes because no candidate existed is not a pass for that case: HC-x fixtures must have `n_h2o_candidates ≥ 1` checked as a precondition.

---

## 5. Ablation fairness

1. All arms in a comparison use the same prompt set, image bytes, tokenizer, budget `B`, H2O scorer, seed and decoding (greedy, `temperature=0`, `top_p=1.0`).
2. `starlight_s1` vs `starlight_s1_no_da`: the **only** difference is the DA signal. `starlight_s1` vs `starlight_s1_no_grounding`: the **only** difference is the grounding signal. `starlight_s1_no_da` and `starlight_s1_no_grounding` are the two ablations; `h2o_base` is the H2O-only comparator with no protection signal.
3. An arm that is `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET` on a fixture is **not** silently dropped from the comparison. Its quality on that fixture is reported as "not comparable", with the count.
4. Quality deltas use matched fixture IDs. Deltas computed only on the rows where the arm did not return `INFEASIBLE` are labelled as such.
5. No arm may receive information the others do not (e.g., the grounding labels must not leak into `starlight_s1_no_grounding` via the prompt or a side channel). A-NOGROUND is the check.
6. Comparison against the seven existing arms uses the **same** budget definition. The existing `h2o_*` arms currently place protected tokens outside the budget (`docs/audit/MISSION_QUALIFICATION_20261109.md:31`, a labelled variant). That is a different budget semantics from I-4/I-5. Any table that puts `h2o_base` and `starlight_s1` side by side must state the difference or use a common definition.

---

## 6. Misreadable retention metrics (what must not be reported as the headline)

- **"Retention rate" / "KV kept %"** computed over all entries inflates the number when protected entries are many. It must be reported split into: protected-retained, H2O-candidate-retained, non-candidate-retained, and budget used.
- **"Protected fraction"** without the relevance label hides HC-5. Report `protected_irrelevant_fraction` next to it.
- **"KV bytes saved"** summed from tensor-size deltas is not peak VRAM (see the historical receipt's `638976B/811008B`, retracted in `artifacts/qwen35_pretrained_qualification_receipt.json`). Any byte figure must be physical allocator measurement or labelled `LOGICAL_ENTRIES_REMOVED`.
- **"attention_mask_restriction_ratio"** (e.g., the retracted `0.1895`) is not KV saving and not an accuracy result.
- **Token-level accuracy or keyword match** is not an answer-quality measure (`artifacts/qwen35_pretrained_qualification_receipt.json`: "4 canonical image task keyword heuristics did not measure accuracy"). Quality needs a scorer with provenance.
- **Output-equality rate vs base** is not "quality retained" unless the base itself is shown to be reproducible under the same settings.
- **Count of INFEASIBLE** is a required field, not an optional note.

---

## 7. Citations: verified vs unverified

| Source | Used for | Status in this iteration |
|---|---|---|
| H2O, arXiv 2306.14048 (upstream FMInference/H2O, pinned in the matrix) | heavy-hitter + recent-window eviction semantics | **Not re-fetched in this iteration.** Treat as unverified here. |
| Declarative Attention, arXiv 2609.02737, v1 | reported 52.0% / 31.1% | Verified in an earlier audit round (per auditor notes); not re-fetched here. |
| Declarative Attention, arXiv 2609.02737, v2 | reported 45.8% / 19.6% | Verified in an earlier audit round (per auditor notes); not re-fetched here. |
| Grounding-evidence protection | the concept of grounding-protected entries | **No citation exists in the repo.** Any claim that grounding protection follows a published method is unverified. |
| Qwen3.5-0.8B architecture facts (H-1..H-6) | FA set, GQA 16:2, token ids | Checkpoint files and the repo coverage receipt (488 tensors). Architecture hazards are the auditor's reading and must be checked against the pinned `config.json` before a test relies on them. |

Rule: no README, report or test name may state a paper result as reproduced unless the reproduction receipt exists. Both DA numbers above are paper-reported figures, not local results.

---

## 8. Open ambiguities (owner or lead decision required)

- **A-1 Budget granularity.** Is `B` per (request, FA layer), per request total across FA layers, or per KV head? The brief does not say. Oracle default: per (request, FA layer), applied identically to both KV heads; any other choice must be written down before tests run.
- **A-2 Budget unmet with no eligible candidates and no protection violation.** Example: `|entries| > B`, H2O proposes nothing, nothing is protected. The brief defines only `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET`. Oracle proposal: a separate status `BUDGET_UNMET_NO_ELIGIBLE_CANDIDATES`, reported and never returned as `OK`. Needs ratification.
- **A-3 Per-head protection.** A DA or grounding span protects a token position. Oracle default: the protection applies to that position in **all** KV heads of the layer (conservative). Per-head protection must be stated and tested, otherwise a head can evict a protected token.
- **A-4 Structural and image-span tokens.** I-8 and HC-13 (vision_start/end never evicted; partial image-span eviction policy) are auditor requirements, not in the owner contract. Must be ratified or dropped explicitly.
- **A-5 Missing declarations (HC-7).** Fail-closed with conservative protection vs raised error. Oracle accepts either, not silent eviction. Owner to choose.
- **A-6 Eviction timing.** Prefill-time vs decode-time eviction (H-7). The contract is silent. Must be stated per arm; the oracle requires the choice to appear in the decision record.

---

## 9. Verdict rule for the Stage 1 implementation

The Stage 1 controller is **not** accepted unless all of the following hold on the fixtures in section 4, with the negative controls in section 4:

1. I-1..I-10 hold on every step recorded.
2. A-DA, A-GROUND, A-NODA, A-NOGROUND, A-H2O, A-TOGGLE hold for the arms that claim them.
3. HC-1..HC-15 pass or are reported with the specific failing assertion. A skipped case is reported as skipped, with the reason, and cannot be counted as a pass.
4. AMBIGUITIES A-1..A-6 are resolved in writing by the owner or the lead. Until then, the implementation may be labelled "prototype, pending ambiguity resolution" and no quality claim may use it.
5. No receipt labels the run as qualified. The label is computed from the rules above, not written as a literal.

Failure of any single item is a failure of the Stage 1 acceptance, regardless of the headline quality number.
