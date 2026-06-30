import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent

SEQ_LEN = 256
GROUP_SIZE = 64


def _load_pf_model():
    name = "pf_model_openai_mlx"
    spec = importlib.util.spec_from_file_location(name, _MODEL_DIR / "_pf_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class PrivacyFilterMoE(torch.nn.Module):
    """MLX static-MoE: per-expert weights packed into SwitchLinear (int4
    gather_qmm). GLU gate + expert biases applied explicitly (SwitchMLP drops
    biases). Biases stay fp32 stacked buffers, gathered via F.embedding."""

    def __init__(self, src, limit, alpha):
        super().__init__()
        import torch.nn.functional as F  # noqa: F401  (used in forward)
        from executorch.backends.mlx.llm.switch import SwitchLinear

        E = src.num_experts
        H = src.gate_up_proj.shape[1]
        twoF = src.gate_up_proj.shape[2]
        F_ = twoF // 2
        self.num_experts, self.hidden, self.inter = E, H, F_
        self.limit, self.alpha = limit, alpha

        self.gate_up = SwitchLinear(H, twoF, E, bias=False)
        self.down = SwitchLinear(F_, H, E, bias=False)
        with torch.no_grad():
            for e in range(E):
                self.gate_up.experts[e].weight.copy_(src.gate_up_proj[e].t())
                self.down.experts[e].weight.copy_(src.down_proj[e].t())
        self.register_buffer("gate_up_bias", src.gate_up_proj_bias.data.clone())
        self.register_buffer("down_bias", src.down_proj_bias.data.clone())

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        import torch.nn.functional as F

        x = hidden_states.unsqueeze(-2).unsqueeze(-2)             # [T,1,1,H]
        gu = self.gate_up(x, router_indices)                     # [T,k,1,2F]
        gu = gu + F.embedding(router_indices, self.gate_up_bias).unsqueeze(-2)
        gate, up = gu.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated = (up + 1) * glu
        out = self.down(gated, router_indices)                   # [T,k,1,H]
        out = out + F.embedding(router_indices, self.down_bias).unsqueeze(-2)
        out = out.squeeze(-2)
        out = out * routing_weights.unsqueeze(-1).float()
        return out.sum(dim=-2).to(hidden_states.dtype)


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx.partitioner import MLXPartitioner
    from executorch.extension.llm.export.quantize import quantize_model_
    from executorch.backends.mlx.llm.switch import pack_all_switch_linears
    from executorch.exir import to_edge_transform_and_lower

    m = _load_pf_model()
    # fp32 compute dtype is REQUIRED for the iOS MLX delegate — bf16 corrupts
    # the MoE gather_qmm path (every token labeled PII). fp32 is essentially
    # free here since the expert weights are int4-quantized regardless.
    model, tok = m.load_model(torch.float32)

    for layer in model.model.layers:
        src = layer.mlp.experts
        layer.mlp.experts = PrivacyFilterMoE(src, limit=src.limit, alpha=src.alpha)

    quantize_model_(
        model,
        qlinear_config="4w",
        qlinear_group_size=GROUP_SIZE,
        qembedding_config="4w",
        qembedding_group_size=GROUP_SIZE,
    )
    pack_all_switch_linears(model)

    ids, mask = m.example_inputs(tok, SEQ_LEN)
    wrapper = m.ClassifierWrapper(model).eval()

    with torch.no_grad():
        ep = torch.export.export(wrapper, (ids, mask), strict=False)
        et = to_edge_transform_and_lower(
            {"forward": ep}, partitioner=[MLXPartitioner()]
        ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et.buffer)
    return str(out_path)
