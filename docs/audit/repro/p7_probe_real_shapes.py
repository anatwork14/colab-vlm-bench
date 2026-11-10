import sys, json
sys.path.insert(0, sys.argv[1] + "/src")
import torch
from safetensors.torch import save_file
from pathlib import Path
from vlmbench.infrastructure.model_config import Qwen35ModelConfig
from vlmbench.infrastructure.real_engine import RealSevenArmEngine
ck = Path(sys.argv[2]); ck.mkdir(parents=True, exist_ok=True)
shapes = {
 "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": (6144, 1024),
 "model.language_model.layers.0.linear_attn.out_proj.weight": (1024, 2048),
 "model.language_model.layers.3.self_attn.q_proj.weight": (4096, 1024),
 "model.language_model.layers.3.self_attn.k_proj.weight": (512, 1024),
 "model.language_model.layers.3.self_attn.v_proj.weight": (512, 1024),
 "model.language_model.layers.3.self_attn.o_proj.weight": (1024, 2048),
 "mtp.fc.weight": (1024, 2048),
 "model.visual.merger.linear_fc2.weight": (1024, 3072),
 "model.visual.merger.linear_fc2.bias": (1024,),
}
save_file({k: torch.randn(*v) * 0.01 for k, v in shapes.items()}, str(ck / "model.safetensors"))
cfg = Qwen35ModelConfig.qwen35_08b()
eng = RealSevenArmEngine(method_arm="base", config=cfg, device="cpu", checkpoint_dir=ck)
s = eng.pretrained_weight_summary
print(json.dumps({k: s.get(k) for k in ["mapped_tensors_count","unmapped_tensors_count","parameter_coverage_ratio"]}))
print("mapped:", sorted(s.get("mapped_tensor_keys", []))[:20])
print("unmapped:", sorted(s.get("unmapped_tensor_keys", []))[:20])
print("is_pretrained", eng.is_pretrained, "architectural_mode", getattr(eng, "architectural_mode", None))
