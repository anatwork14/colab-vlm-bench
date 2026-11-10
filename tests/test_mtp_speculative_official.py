"""R3-2: CPU greedy k=1 MTP speculative reference on the OFFICIAL Qwen3.5-0.8B checkpoint.

Label: cpu_greedy_k1_reference_not_vllm_mtp. These tests check exact token equivalence with
non-speculative greedy decoding, rollback of KV and Gated-DeltaNet states, and isolation across
calls. They make NO speed claims and are NOT vLLM MTP evidence.

Forced accept/reject paths use ORACLE hooks: the proposal is the target's own greedy token (accept)
or a corrupted one (reject). This exercises the real verify/rollback code without depending on how
often the MTP head happens to be right.
"""
from pathlib import Path

import pytest
import torch

from vlmbench.infrastructure.real_engine import (
    RealSevenArmEngine,
    _restore_dynamic_cache,
    _snapshot_dynamic_cache,
)

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
pytestmark = pytest.mark.skipif(
    not (CKPT / "config.json").is_file(), reason=f"official checkpoint not present at {CKPT}")

LABEL = "cpu_greedy_k1_reference_not_vllm_mtp"
TEXT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
TEXT_2 = " Rome. What is the capital of Japan?"


@pytest.fixture(scope="module")
def engine():
    eng = RealSevenArmEngine(method_arm="mtp", device="cpu", checkpoint_dir=CKPT)
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def prompt(engine):
    if engine.tokenizer is None:
        pytest.skip("official tokenizer not available")
    return list(engine.tokenizer.encode(TEXT))


def _pos(start, n):
    return torch.arange(start, start + n, dtype=torch.long).view(1, 1, n).expand(3, 1, n).clone()


def _reference_greedy(engine, prompt_ids, n):
    """Non-speculative greedy decode through the same hooked official forward."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    cache = modeling_qwen3_5.DynamicCache(config=engine.model.hf_model.config)
    L = len(prompt_ids)
    out = engine.model(torch.tensor([list(prompt_ids)]), past_key_values=cache,
                       position_ids=_pos(0, L), logits_to_keep=1)
    tok = int(torch.argmax(out["logits"][0]).item())
    toks, fed = [tok], L
    while len(toks) < n:
        out = engine.model(torch.tensor([[tok]]), past_key_values=cache,
                           position_ids=_pos(fed, 1), logits_to_keep=1)
        tok = int(torch.argmax(out["logits"][0]).item())
        toks.append(tok)
        fed += 1
    return toks


def test_r3_spec_mtp_equals_greedy_with_natural_mtp_drafts(engine, prompt):
    ref = _reference_greedy(engine, prompt, 16)
    spec = engine._speculative_mtp_greedy_k1(prompt, max_new_tokens=16)
    assert spec["label"] == LABEL
    assert spec["tokens"] == ref
    assert spec["proposed"] > 0, "speculative path proposed nothing: test would be vacuous"
    assert spec["accepted"] + spec["rejected"] == spec["proposed"]


def test_r3_spec_mtp_equals_greedy_when_every_draft_is_rejected(engine, prompt):
    ref = _reference_greedy(engine, prompt, 16)
    vocab = engine.config.vocab_size
    spec = engine._speculative_mtp_greedy_k1(
        prompt, max_new_tokens=16,
        draft_override=lambda draft, step, n: (draft + 1) % vocab)
    assert spec["proposed"] > 0
    assert spec["accepted"] == 0
    assert spec["rejected"] == spec["proposed"]
    assert spec["tokens"] == ref


def test_r3_spec_mtp_equals_greedy_with_oracle_accepts_and_rejects(engine, prompt):
    """Even proposals are the target's true next token (accept); odd proposals are corrupted (reject)."""
    n = 16
    ref = _reference_greedy(engine, prompt, n)
    vocab = engine.config.vocab_size

    def oracle(draft, step, n_emitted):
        true_next = ref[n_emitted]  # target greedy token that follows the committed prefix
        return true_next if step % 2 == 0 else (true_next + 1) % vocab

    spec = engine._speculative_mtp_greedy_k1(prompt, max_new_tokens=n, draft_override=oracle)
    assert spec["proposed"] > 0 and spec["accepted"] > 0 and spec["rejected"] > 0
    assert spec["tokens"] == ref


def test_r3_rejected_verify_is_rolled_back_to_exact_snapshot(engine, prompt):
    """A rejected verify must leave KV and Gated-DeltaNet conv/recurrent tensors bit-identical
    to the pre-verify snapshot. The verify must visibly change the state, so the test is not vacuous."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    cache = modeling_qwen3_5.DynamicCache(config=engine.model.hf_model.config)
    L = len(prompt)
    out = engine.model(torch.tensor([prompt]), past_key_values=cache,
                       position_ids=_pos(0, L), logits_to_keep=1)
    pending = int(torch.argmax(out["logits"][0]).item())
    before = _snapshot_dynamic_cache(cache)
    wrong = (pending + 7) % engine.config.vocab_size
    engine.model(torch.tensor([[pending, wrong]]), past_key_values=cache, logits_to_keep=2,
                 position_ids=_pos(L, 2))
    mid = _snapshot_dynamic_cache(cache)
    _restore_dynamic_cache(cache, before)
    after = _snapshot_dynamic_cache(cache)

    def _equal(x, y):
        if isinstance(x, torch.Tensor):
            return torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(_equal(x[k], y[k]) for k in x)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(_equal(a, b) for a, b in zip(x, y))
        return x == y

    kinds = {b[0] for b in before}
    assert {"dynamic", "linear"} <= kinds, "test must cover both attention and DeltaNet layers"
    assert not _equal(before, mid), "verify did not change cache state: test would be vacuous"
    assert _equal(before, after), "rollback did not restore the exact pre-verify state"


def test_r3_multi_call_isolation_matches_fresh_greedy(engine, prompt):
    """Shared model state must not leak between calls: a second turn that concatenates the first
    turn's output with a new question must match a fresh greedy reference on that prompt."""
    turn1 = engine._speculative_mtp_greedy_k1(prompt, max_new_tokens=8)
    prompt2 = list(prompt) + turn1["tokens"] + list(engine.tokenizer.encode(TEXT_2))
    ref2 = _reference_greedy(engine, prompt2, 12)
    spec2 = engine._speculative_mtp_greedy_k1(prompt2, max_new_tokens=12)
    assert spec2["tokens"] == ref2
    assert spec2["proposed"] > 0


def test_r3_execute_request_mtp_arm_reports_verifier_derived_acceptance(engine, prompt):
    """ITER2 Stage 2(a) (disclosed change): the mtp arm on text-only official requests now runs the
    verifier, so acceptance is derived from verifier counts (accepted / proposed), never a literal.
    Gate is the CPU-reference label, not a serving PASS."""
    res = engine.execute_request("r3_mtp_arm", prompt_token_ids=prompt,
                                 max_new_tokens=4, image_present=False)
    spec = res.telemetry["speculation"]
    expected = spec["accepted"] / spec["proposed"] if spec["proposed"] else None
    assert res.mtp_acceptance_rate == expected
    assert res.gates["MTP_CORRECTNESS_PASS"] == "CPU_REFERENCE_SPECULATION_EXECUTED_NOT_VLLM"
    # ITER2 D10 N-7: the speculative route carries the memory-field scope glossary too.
    scopes = res.telemetry["memory_field_scopes"]
    assert "not per decode step" in scopes["final_retained_kv_payload_bytes"]
    assert "None" in scopes["physical_kv_bytes_freed"]
