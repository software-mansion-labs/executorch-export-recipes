import importlib.util
import json
from pathlib import Path

import torch

root = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("vk_export", root / "vulkan/export.py")
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
torch.set_num_threads(4)
model = m._load_model()
from depth_anything_3.model.dinov2.layers.rope import RotaryPositionEmbedding2D

for layer in model.modules():
    if hasattr(layer, "fused_attn"):
        layer.fused_attn = False
    if isinstance(layer, RotaryPositionEmbedding2D):
        layer._export_max_position = 38
        layer.forward = m._static_rope.__get__(layer)
x = torch.zeros(1, 3, 518, 518)
with torch.no_grad():
    model(x)
    graph = (
        torch.export.export(model, (x,), strict=False).run_decompositions({}).module()
    )
    m._fold_constants(graph)
    ep = torch.export.export(graph, (x,), strict=False)
    assert not ep.graph_signature.buffers_to_mutate
    exported = ep.module()
    cases = []
    for path in sorted(Path("/private/tmp/depth_anything_v3/comparison").glob("*.pt")):
        data = torch.load(path, weights_only=True)
        actual = exported(data["input"]).clone()
        expected = data["v3_reference"]
        assert torch.isfinite(actual).all()
        error = actual - expected
        check = {
            "case": path.stem,
            "max_abs_error": error.abs().max().item(),
            "range_normalized_rmse": error.square().mean().sqrt().item()
            / (expected.max() - expected.min()).item(),
        }
        cases.append(check)
        print(check, flush=True)
        assert torch.allclose(actual, expected, atol=5e-5, rtol=5e-5)
        again = exported(data["input"]).clone()
        assert torch.equal(actual, again), "Repeated execution changed constant buffers"
report = {
    "passed": True,
    "cases": cases,
    "delegate_arity": m.verify_delegate_arity(
        Path(
            "/private/tmp/depth_anything_v3/vulkan/depth_anything_v3_small_vulkan_fp16.pte"
        ).read_bytes()
    ),
}
Path("/private/tmp/depth_anything_v3/vulkan_fix_verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
