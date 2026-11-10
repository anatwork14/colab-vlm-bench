import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("da_mod", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["da_mod"] = m; spec.loader.exec_module(m)
gen = m.DAMaskGenerator()
# Mirror real_engine scripted tags for a text-only decode step (image_present=False)
for L in (100, 400):
    sink_end = min(16, max(1, L // 4))
    tags = [m.DATag(m.DATagType.GLOBAL, 0, sink_end)]
    tags.append(m.DATag(m.DATagType.LOCAL, sink_end, L, window=8))
    mask = gen.generate_attention_mask(seq_len=L, tags=tags, active_focus_id=None)
    last = mask[-1]
    readable = np.nonzero(last)[0]
    print(f"text-only L={L}: last-row readable keys={len(readable)} "
          f"-> indices {readable.min()}..{readable.max()} ; question-region keys (16..L-9) readable: "
          f"{int(last[16:L-9].sum())}")
