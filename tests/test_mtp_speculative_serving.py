"""ITER2 Stage 2(a): the serving-style `mtp` arm runs CPU speculative decoding on the OFFICIAL
Qwen3.5-0.8B checkpoint (text-only requests), through the verified helper
`_speculative_mtp_greedy_k1`. Label: cpu_greedy_k1_reference_not_vllm_mtp.

Checks:
  * committed tokens equal non-speculative greedy of the same arm (>= 32 tokens), with natural
    MTP drafts and with FORCED rejects (every draft corrupted) and mixed accept/reject;
  * per-request verifier counts are measured, not literals: verifier_calls equals the number of
    model calls counted by an instrumented wrapper, and obeys 1 + proposed + rejected;
  * EOS is never emitted and stops generation, for natural drafts and for an accepted EOS draft;
  * multi-turn and multi-request calls do not leak state into each other.
Speed is NOT claimed. This is NOT vLLM MTP evidence.
"""
from pathlib import Path

import pytest
import torch

from vlmbench.infrastructure.real_engine import RealSevenArmEngine

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
pytestmark = pytest.mark.skipif(
    not (CKPT / "config.json").is_file(), reason=f"official checkpoint not present at {CKPT}")

LABEL = "cpu_greedy_k1_reference_not_vllm_mtp"
GATE_VALUE = "CPU_REFERENCE_SPECULATION_EXECUTED_NOT_VLLM"
TEXT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
TEXT_B = "Water boils at one hundred degrees Celsius at sea level. Ice melts at"
TEXT_2 = " Rome. What is the capital of Japan?"
N = 40


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


def _greedy(engine, prompt_ids, n, stops):
    """Non-speculative greedy through the same hooked official forward (single-token decode).

    `stops` = token ids that end generation and are NOT emitted (empty set = no EOS stop)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    cache = modeling_qwen3_5.DynamicCache(config=engine.model.hf_model.config)
    L = len(prompt_ids)
    with torch.no_grad():
        out = engine.model(torch.tensor([list(prompt_ids)]), past_key_values=cache,
                           position_ids=_pos(0, L), logits_to_keep=1)
        tok = int(torch.argmax(out["logits"][0]).item())
        toks, fed = [], L
        while True:
            if tok in stops:
                return toks, "eos"
            toks.append(tok)
            if len(toks) >= n:
                return toks, "max_new_tokens"
            out = engine.model(torch.tensor([[tok]]), past_key_values=cache,
                               position_ids=_pos(fed, 1), logits_to_keep=1)
            tok = int(torch.argmax(out["logits"][0]).item())
            fed += 1


def _ref_no_stop(engine, prompt_ids, n):
    return _greedy(engine, prompt_ids, n, set())[0]


def _ref_with_stop(engine, prompt_ids, n):
    stops = set(engine.eos_token_ids) if engine.stop_at_eos else set()
    return _greedy(engine, prompt_ids, n, stops)


class _CountingModel:
    """Instrumented wrapper: records the width of every model call (independent of the helper)."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "widths", [])

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __call__(self, *args, **kwargs):
        self.widths.append(int(args[0].shape[1]))
        return self._inner(*args, **kwargs)


def _assert_counts_consistent(spec, n_tokens):
    assert spec["committed"] == n_tokens
    assert spec["proposed"] == spec["accepted"] + spec["rejected"]
    assert spec["verifier_calls"] == 1 + spec["proposed"] + spec["rejected"]


def test_text_only_mtp_arm_commits_greedy_stream_with_natural_drafts(engine, prompt):
    res = engine.execute_request("s2a_natural", prompt_token_ids=prompt,
                                 max_new_tokens=N, image_present=False)
    ref, reason = _ref_with_stop(engine, prompt, N)
    assert len(ref) >= 32, "reference too short to satisfy the >=32-token equality requirement"
    assert res.tokens_generated == ref
    spec = res.telemetry["speculation"]
    assert spec["label"] == LABEL
    assert res.telemetry["speculation_executed"] is True
    assert spec["stop_reason"] == reason
    _assert_counts_consistent(spec, len(res.tokens_generated))
    expected_rate = spec["accepted"] / spec["proposed"] if spec["proposed"] else None
    assert res.mtp_acceptance_rate == expected_rate
    assert res.gates["MTP_CORRECTNESS_PASS"] == GATE_VALUE


def test_forced_rejects_commit_greedy_stream(engine, prompt):
    ref = _ref_no_stop(engine, prompt, N)
    vocab = engine.config.vocab_size

    def corrupt_every_draft(draft, i, n_emitted):
        # ref[n_emitted] is the target's next token after the committed prefix; +1 mod vocab
        # is guaranteed to differ from it, so every proposal is a forced reject.
        return (ref[n_emitted] + 1) % vocab

    spec = engine._speculative_mtp_greedy_k1(prompt, max_new_tokens=N,
                                             draft_override=corrupt_every_draft)
    assert spec["tokens"] == ref
    assert spec["proposed"] >= 1
    assert spec["accepted"] == 0
    assert spec["rejected"] == spec["proposed"]
    _assert_counts_consistent(spec, len(spec["tokens"]))


def test_mixed_forced_accept_and_reject_commit_greedy_stream(engine, prompt):
    ref = _ref_no_stop(engine, prompt, N)
    vocab = engine.config.vocab_size
    calls = {"i": 0}

    def alternate(draft, i, n_emitted):
        calls["i"] += 1
        if calls["i"] % 2 == 1:
            return ref[n_emitted]  # forced accept
        return (ref[n_emitted] + 1) % vocab  # forced reject

    spec = engine._speculative_mtp_greedy_k1(prompt, max_new_tokens=N, draft_override=alternate)
    assert spec["tokens"] == ref
    assert spec["accepted"] >= 1 and spec["rejected"] >= 1
    _assert_counts_consistent(spec, len(spec["tokens"]))


def test_verifier_counts_match_instrumented_model_calls(engine, prompt, monkeypatch):
    ref = _ref_no_stop(engine, prompt, N)
    vocab = engine.config.vocab_size
    proxy = _CountingModel(engine.model)
    monkeypatch.setattr(engine, "model", proxy)
    # Mixed block: some rejects so the refeed term of the invariant is exercised.
    flip = {"i": 0}

    def half_reject(draft, i, n_emitted):
        flip["i"] += 1
        return ref[n_emitted] if flip["i"] % 2 else (ref[n_emitted] + 1) % vocab

    spec = engine._speculative_mtp_greedy_k1(prompt, max_new_tokens=N, draft_override=half_reject)
    assert spec["tokens"] == ref
    # Every model call the helper made is a verifier call: prefill + block(s) + refeed(s).
    assert spec["verifier_calls"] == len(proxy.widths)
    # Blocks are the 2-token calls (prompt is longer than 2 tokens, so prefill is never width 2).
    assert len(prompt) != 2
    assert sum(1 for w in proxy.widths if w == 2) == spec["proposed"]
    _assert_counts_consistent(spec, len(spec["tokens"]))


def test_eos_is_never_emitted_and_stops_generation(engine, prompt, monkeypatch):
    ref = _ref_no_stop(engine, prompt, N)
    # Choose an EOS id that first appears at index k >= 2, so a prefix is emitted before it.
    k = next(i for i in range(2, N) if ref[i] not in ref[:i])
    eos = ref[k]
    monkeypatch.setattr(engine, "eos_token_ids", (eos,))
    monkeypatch.setattr(engine, "stop_at_eos", True)
    expected = ref[:k]

    natural = engine._speculative_mtp_greedy_k1(prompt, max_new_tokens=N)
    assert natural["tokens"] == expected
    assert natural["stop_reason"] == "eos"
    assert eos not in natural["tokens"]

    # Oracle proposes the true next token, so the EOS draft is ACCEPTED by the verifier; it must
    # still not be emitted and must stop generation.
    oracle = engine._speculative_mtp_greedy_k1(
        prompt, max_new_tokens=N, draft_override=lambda d, i, n: ref[n])
    assert oracle["tokens"] == expected
    assert oracle["stop_reason"] == "eos"
    assert oracle["accepted"] >= 1
    assert eos not in oracle["tokens"]

    res = engine.execute_request("s2a_eos", prompt_token_ids=prompt,
                                 max_new_tokens=N, image_present=False)
    assert res.tokens_generated == expected
    assert res.telemetry["speculation"]["stop_reason"] == "eos"


def test_multi_turn_and_multi_request_isolation(engine, prompt):
    r1 = engine.execute_request("s2a_turn1", prompt_token_ids=prompt,
                                max_new_tokens=24, image_present=False)
    prompt_b = list(engine.tokenizer.encode(TEXT_B))
    rb = engine.execute_request("s2a_req_b", prompt_token_ids=prompt_b,
                                max_new_tokens=24, image_present=False)
    turn2 = prompt + list(r1.tokens_generated) + list(engine.tokenizer.encode(TEXT_2))
    r2 = engine.execute_request("s2a_turn2", prompt_token_ids=turn2,
                                max_new_tokens=24, image_present=False)
    assert r2.tokens_generated == _ref_with_stop(engine, turn2, 24)[0]
    assert rb.tokens_generated == _ref_with_stop(engine, prompt_b, 24)[0]
    # Re-running turn 1 after other requests must reproduce it bit-for-bit, including counts.
    r1_again = engine.execute_request("s2a_turn1_again", prompt_token_ids=prompt,
                                      max_new_tokens=24, image_present=False)
    assert r1_again.tokens_generated == r1.tokens_generated
    for key in ("proposed", "accepted", "rejected", "verifier_calls", "committed"):
        assert r1_again.telemetry["speculation"][key] == r1.telemetry["speculation"][key]
