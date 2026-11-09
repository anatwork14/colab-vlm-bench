import importlib.util, sys
import numpy as np
spec = importlib.util.spec_from_file_location("ge", sys.argv[1])
m = importlib.util.module_from_spec(spec); sys.modules["ge"] = m; spec.loader.exec_module(m)
class Tok:
    vocab = {0: "reduce", 1: "green", 2: "the", 3: "red"}
    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.vocab[i] for i in ids)
    def encode(self, s):
        return [k for k, v in self.vocab.items() if v == s.strip()]
for alpha in (1.5, 20.0):
    pool = m.EvidencePool("r")
    pool.add_evidence(m.EvidenceItem("e1", "text", "red car parked", "a"*64), keywords=["red"])
    proc = m.ECRDLogitsProcessor(pool, Tok(), m.ECRDConfig(reweight_alpha=alpha))
    logits = np.array([0.9, 1.0, -5.0, -5.0])  # reduce=0.9 (unsupported), green=1.0 (raw argmax)
    out = proc([], logits)
    print(f"alpha={alpha}: argmax={Tok.vocab[int(np.argmax(out))]!r} (raw argmax green); citations={proc.citations}")
