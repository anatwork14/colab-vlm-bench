"""ITER3 B1: RoPE consistency and logit recomputation after physical KV compaction, on the OFFICIAL
Qwen3.5-0.8B checkpoint (hooked HF forward, real RoPE). The scaffold model used by tiny-config tests
applies NO rotary embedding, so these checks cannot be done there.

Two test kinds, labelled in each test name:
  * pinv_  POLICY-INVARIANT: state identity. Retained KV entries after compaction are bit-identical to the
           same entries computed without eviction at their ORIGINAL positions.
  * infc_  INFERENCE-CONSISTENCY: numerical agreement of logits. Physical compaction (index_select) must
           give the same next-token logits as a non-compacting reference that masks evicted positions
           with the original position ids (within float tolerance).

Method: teacher-forced decoding with fixed tokens. Path A = the real policy path (H2O plan, physical
compaction). Path B = no compaction, full history kept, `da_masks` restricted to the SAME retained
original positions. Path B's physical index equals the original position, so RoPE is by construction the
original one.

cpu_pytorch_reference_not_vllm. No speed, quality or GPU claim.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from vlmbench.infrastructure.real_engine import RealSevenArmEngine

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
pytestmark = pytest.mark.skipif(
    not (CKPT / "config.json").is_file(), reason=f"official checkpoint not present at {CKPT}")

PROMPT = list(range(1000, 1024))  # 24 text tokens
FORCED = [1234, 567, 89, 4321, 765, 2468, 97]  # teacher-forced decode tokens (fixed ids)
ATOL = 1e-4


@pytest.fixture(scope="module")
def eng():
    e = RealSevenArmEngine(method_arm="h2o_base", device="cpu", checkpoint_dir=CKPT,
                           stop_at_eos=False, h2o_min_budget=8, h2o_budget_ratio=0.25)
    yield e
    e.close()


def _run_both(eng, monkeypatch, b_shift=0):
    """Drive path A (compacting) and path B (masked, non-compacting) through the same tokens."""
    from vlmbench.infrastructure.starlight import compact_orig_positions

    captured: list[dict[int, np.ndarray]] = []
    real_apply = eng._pol_apply

    def spy(st, plans, next_orig):
        captured.append({l: np.asarray(plans[l]).copy() for l in plans})
        return real_apply(st, plans, next_orig)

    monkeypatch.setattr(eng, "_pol_apply", spy)
    stA = eng._pol_new_state("h2o", len(PROMPT))
    logitsA, _ = eng._pol_prefill(stA, PROMPT)
    stB = eng._pol_new_state("h2o", len(PROMPT))
    logitsB, _ = eng._pol_prefill(stB, PROMPT)
    full = tuple(stA.full_layers)
    orig = {l: np.arange(len(PROMPT), dtype=np.int64) for l in full}
    rows = []
    for token in FORCED:
        P = stA.P
        assert stB.P == P
        logA, _mtp = eng._pol_single(stA, token)
        plans = captured[-1]
        # Original positions of the new token's cache row, then the retained set after compaction.
        appended = {l: np.append(orig[l], P) for l in full}
        orig = {l: np.asarray(compact_orig_positions(appended[l], plans[l].tolist(), None), dtype=np.int64)
                for l in full}
        allowed = {}
        for l in full:
            m = torch.zeros(1, P + 1, dtype=torch.bool)
            m[0, torch.as_tensor(orig[l])] = True
            allowed[l] = m
        outB = eng.model(torch.tensor([[int(token)]], device=eng.device, dtype=torch.long),
                         past_key_values=stB.cache, da_masks=allowed,
                         position_ids=torch.full((3, 1, 1), P + b_shift, device=eng.device, dtype=torch.long),
                         logits_to_keep=1)
        stB.P += 1
        rows.append({"logA": logA.detach().clone(), "logB": outB["logits"][0].detach().clone(),
                     "orig": {l: orig[l].copy() for l in full},
                     "cacheA": {l: stA.cache.layers[l].keys.detach().clone() for l in full},
                     "cacheB": {l: stB.cache.layers[l].keys.detach().clone() for l in full},
                     "evicted": any(len(plans[l]) < P + 1 for l in full)})
    return rows


def test_pinv_retained_keys_equal_original_position_keys_at_first_full_attention_layer(eng, monkeypatch):
    # [POLICY-INVARIANT] Layer 3 is the first full-attention layer; its keys depend only on tokens
    # (the layers below are Gated-DeltaNet, which do not read attention), so compaction must leave
    # each retained key bit-identical to the key of the same token at the same original position.
    rows = _run_both(eng, monkeypatch)
    assert any(r["evicted"] for r in rows), "test would be vacuous: no eviction happened"
    layer = eng.config.full_attention_layers[0]
    for step, r in enumerate(rows):
        keysA = r["cacheA"][layer]
        keysB = r["cacheB"][layer]
        idx = torch.as_tensor(r["orig"][layer])
        assert keysA.shape[2] == len(idx), step
        assert torch.equal(keysA, keysB[:, :, idx]), f"layer {layer} key mismatch at decode step {step}"


def test_infc_compacted_logits_match_masked_reference_with_original_positions(eng, monkeypatch):
    # [INFERENCE-CONSISTENCY] Next-token logits after physical compaction equal the non-compacting
    # reference that masks evicted positions, step by step, under teacher forcing.
    rows = _run_both(eng, monkeypatch)
    assert any(r["evicted"] for r in rows), "test would be vacuous: no eviction happened"
    worst = 0.0
    for step, r in enumerate(rows):
        a, b = r["logA"].float(), r["logB"].float()
        diff = float((a - b).abs().max())
        worst = max(worst, diff)
        assert torch.allclose(a, b, atol=ATOL, rtol=ATOL), f"step {step}: max|logit diff|={diff:.3e}"
        assert int(torch.argmax(a)) == int(torch.argmax(b)), step
    # Keep the measured worst case visible in the test output (not a quality claim).
    print(f"infc max|logit diff| over {len(rows)} steps = {worst:.3e}")


def test_infc_control_wrong_query_positions_are_detected(eng, monkeypatch):
    # [INFERENCE-CONSISTENCY, sensitivity control] The same comparison with the reference's query
    # positions shifted by +1 must FAIL the tolerance. Shows the oracle can see a RoPE/position error.
    rows = _run_both(eng, monkeypatch, b_shift=1)
    diffs = [float((r["logA"].float() - r["logB"].float()).abs().max()) for r in rows]
    assert max(diffs) > 10 * ATOL, f"control not detected: max diff {max(diffs):.3e}"
