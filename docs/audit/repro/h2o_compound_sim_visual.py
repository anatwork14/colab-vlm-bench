import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("h2o_mod", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["h2o_mod"] = m; spec.loader.exec_module(m)
tr = m.H2OCacheTracker(m.H2OConfig(), num_layers=1, num_kv_heads=2)
rng = np.random.default_rng(0)
L = 300  # prompt length already in cache (no visual tokens protected)
# seed scores as if prefill happened
w = rng.exponential(size=(2, L, L)).astype(np.float32)
w = np.tril(w); w /= w.sum(-1, keepdims=True).clip(1e-9)
vis=np.zeros(L,dtype=bool); vis[20:220]=True; tr.set_visual_tokens(vis)
tr.update_attention_scores(0, w)
lens = [L]
for step in range(1, 201):
    cur = L_cache = lens[-1] + 1          # cache length including the new token
    retained, metrics = tr.compute_eviction_plan(0, cur)
    tr.compact_layer_state(0, np.where(retained)[0])
    kept = int(retained.sum())
    # new token's attention row over the kept keys (+ itself)
    row = rng.exponential(size=(2, 1, kept + 1)).astype(np.float32)
    row /= row.sum(-1, keepdims=True)
    tr.update_attention_scores(0, row)
    lens.append(kept)
    if step in (1, 2, 3, 5, 10, 20, 50, 100, 200):
        print(f"step {step:3d}: cache_len_before_eviction={cur:4d} retained_after={kept:4d} ratio_vs_prev={metrics['retained_tokens']/max(1,cur):.3f}")
print("final cached length:", lens[-1], " start:", L)
