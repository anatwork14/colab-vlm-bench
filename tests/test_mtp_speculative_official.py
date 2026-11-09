"""R3-2: CPU greedy k=1 MTP speculative reference on the OFFICIAL Qwen3.5-0.8B checkpoint.

Label: cpu_greedy_k1_reference_not_vllm_mtp. These tests check exact token equivalence with
non-speculative greedy decoding, rollback of KV and Gated-DeltaNet states, and isolation across
calls. They make NO speed claims and are NOT vLLM MTP evidence.
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
PROMPT_A = list(range(100, 140))  # 40 text tokens, no image


@pytest.fixture(scope="module")
def engine():
    eng = RealSevenArmEngine(method_arm="mtp", device="cpu", checkpoint_dir=CKPT)
    yield eng
    eng.close()


def _pos(engine, start, n):
    return torch.arange(start, start + n, dtype=torch.long).view(1, 1, n).expand(3, 1, n).clone()


def _reference_greedy(engine, prompt_ids, n):
    """Non-speculative greedy decode through the same hooked official forward."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    cache = modeling_qwen3_5.DynamicCache(config=engine.model.hf_model.config)
    L = len(prompt_ids)
    out = engine.model(torch.tensor([list(prompt_ids)]), past_key_values=cache,
                       position_ids=_pos(engine, 0, L), logits_to_keep=1)
    tok = int(torch.argmax(out["logits"][0]).item())
    toks, fed = [tok], L
    while len(toks) < n:
        out = engine.model(torch.tensor([[tok]]), past_key_values=cache,
                           position_ids=_pos(engine, fed, 1), logits_to_keep=1)
        tok = int(torch.argmax(out["logits"][0]).item())
        toks.append(tok)
        fed += 1
    return toks


def test_r3_spec_mtp_equals_greedy_with_natural_drafts(engine):
    ref = _reference_greedy(engine, PROMPT_A, 16)
    spec = engine._speculative_mtp_greedy_k1(PROMPT_A, max_new_tokens=16)
    assert spec["label"] == LABEL
    assert spec["tokens"] == ref
    assert spec["proposed"] > 0, "speculative path proposed nothing: test would be vacuous"
    assert spec["accepted"] + spec["rejected"] == spec["proposed"]


def test_r3_spec_mtp_equals_greedy_when_every_draft_is_rejected(engine):
    ref = _reference_greedy(engine, PROMPT_A, 16)
    vocab = engine.config.vocab_size
    spec = engine._speculative_mtp_greedy_k1(
        PROMPT_A, max_new_tokens=16,
        draft_override=lambda draft, step: (draft + 1) % vocab)
    assert spec["proposed"] > 0
    assert spec["accepted"] == 0
    assert spec["rejected"] == spec["proposed"]
    assert spec["tokens"] == ref


def test_r3_spec_mtp_equals_greedy_with_forced_accepts_and_rejects(engine):
    """Accept the natural draft on even proposals, corrupt it on odd proposals."""
    ref = _reference_greedy(engine, PROMPT_A, 16)
    vocab = engine.config.vocab_size
    spec = engine._speculative_mtp_greedy_k1(
        PROMPT_A, max_new_tokens=16,
        draft_override=lambda draft, step: draft if step % 2 == 0 else (draft + 1) % vocab)
    assert spec["proposed"] > 0 and spec["accepted"] > 0 and spec["rejected"] > 0
    assert spec["tokens"] == ref


def test_r3_rejected_verify_is_rolled_back_to_exact_snapshot(engine):
    """A rejected verify must leave KV and Gated-DeltaNet conv/recurrent tensors bit-identical
    to the pre-verify snapshot. The verify must visibly change the state, so the test is not vacuous."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    cache = modeling_qwen3_5.DynamicCache(config=engine.model.hf_model.config)
    L = len(PROMPT_A)
    out = engine.model(torch.tensor([PROMPT_A]), past_key_values=cache,
                       position_ids=_pos(engine, 0, L), logits_to_keep=1)
    pending = int(torch.argmax(out["logits"][0]).item())
    before = _snapshot_dynamic_cache(cache)
    wrong = (pending + 7) % engine.config.vocab_size
    engine.model(torch.tensor([[pending, wrong]]), past_key_values=cache, logits_to_keep=2,
                 position_ids=_pos(engine, L, 2))
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


def test_r3_multi_call_isolation_matches_fresh_greedy(engine):
    """Shared model state must not leak between calls: a second turn that concatenates the first
    turn's output with a new question must match a fresh greedy reference on that prompt."""
    turn1 = engine._speculative_mtp_greedy_k1(PROMPT_A, max_new_tokens=8)
    question2 = [300, 301, 302, 303, 304]
    prompt2 = PROMPT_A + turn1["tokens"] + question2
    ref2 = _reference_greedy(engine, prompt2, 12)
    spec2 = engine._speculative_mtp_greedy_k1(prompt2, max_new_tokens=12)
    assert spec2["tokens"] == ref2
    assert spec2["proposed"] > 0


def test_r3_execute_request_mtp_arm_reports_no_verified_acceptance(engine):
    """The serving-style mtp arm does not run a verifier, so acceptance must be None, not 0.0."""
    res = engine.execute_request("r3_mtp_arm", prompt_token_ids=PROMPT_A,
                                 max_new_tokens=4, image_present=False)
    assert res.mtp_acceptance_rate is None
    assert res.gates["MTP_CORRECTNESS_PASS"] == "GATED_NATIVE_MTP_VERIFIER_REQUIRED"
