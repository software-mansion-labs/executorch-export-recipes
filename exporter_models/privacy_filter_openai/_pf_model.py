"""Shared privacy-filter (OpenAI MoE token classifier) model code.

The stock OpenAIPrivacyFilterExperts.forward loops over data-dependently-hit
experts (torch.where + index_add_), which torch.export can't trace. Both
backends replace it with a STATIC top-k gather form; they differ only in the
expert linear primitive:

  - MLX: SwitchLinear (int4-packed mlx::gather_qmm). See mlx/export.py.
  - XNNPACK: PrivacyFilterMoEDense below — plain nn.Linear experts, gathered
    densely per top-k slot so 8da4w quantize_() + XnnpackPartitioner lower them.

Forward: (input_ids[1,S] long, attention_mask[1,S] long) -> logits[1,S,num_labels]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "openai/privacy-filter"


class ClassifierWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


class PrivacyFilterMoEDense(nn.Module):
    """Exportable static-MoE for XNNPACK: per-expert nn.Linear, top-k gather.

    Mirrors the MLX PrivacyFilterMoE math (GLU gate + expert biases) but keeps
    plain nn.Linear experts (no gather_qmm) so 8da4w quantize_() hits them and
    XNNPACK lowers them. Experts are gathered per top-k slot via index_select.
    """

    def __init__(self, src, limit, alpha):
        super().__init__()
        E = src.num_experts
        H = src.gate_up_proj.shape[1]
        twoF = src.gate_up_proj.shape[2]
        F_ = twoF // 2
        self.num_experts, self.hidden, self.inter = E, H, F_
        self.limit, self.alpha = limit, alpha

        self.gate_up = nn.ModuleList(
            [nn.Linear(H, twoF, bias=True) for _ in range(E)]
        )
        self.down = nn.ModuleList([nn.Linear(F_, H, bias=True) for _ in range(E)])
        with torch.no_grad():
            for e in range(E):
                self.gate_up[e].weight.copy_(src.gate_up_proj[e].t())
                self.gate_up[e].bias.copy_(src.gate_up_proj_bias[e])
                self.down[e].weight.copy_(src.down_proj[e].t())
                self.down[e].bias.copy_(src.down_proj_bias[e])

    def _expert(self, e, x):
        gu = self.gate_up[e](x)
        gate, up = gu.chunk(2, dim=-1)
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return self.down[e]((up + 1) * glu)

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        # hidden_states [T, H]; router_indices/routing_weights [T, k].
        # Static: run every expert on every token, mask by whether the expert
        # is in that token's top-k, then weight + sum. O(E) per token but fully
        # static and XNNPACK-lowerable; E=128 is acceptable for the single
        # prefill the runner does (no autoregression).
        T = hidden_states.shape[0]
        out = hidden_states.new_zeros(T, self.hidden)
        for e in range(self.num_experts):
            # weight for expert e per token = sum over the k slots where it was
            # picked (0 if not picked). No data-dependent branch — the loop runs
            # all E experts unconditionally so the graph stays static/traceable.
            sel = (router_indices == e).float()                # [T, k]
            w = (sel * routing_weights.float()).sum(dim=-1)     # [T]
            out = out + self._expert(e, hidden_states) * w.unsqueeze(-1)
        return out.to(hidden_states.dtype)


def load_model(compute_dtype=torch.float32):
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, dtype=compute_dtype
    ).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    return model, tok


def example_inputs(tok, seq_len):
    enc = tok(
        "My email is john@example.com and SSN 123-45-6789.",
        return_tensors="pt",
        padding="max_length",
        max_length=seq_len,
        truncation=True,
    )
    return enc["input_ids"], enc["attention_mask"]
