"""Build the exportable LFM3 decoder from a local HF snapshot."""

import json
from pathlib import Path

import torch

from .model import Lfm3TextConfig, Lfm3TextModel


def read_safetensors(path: Path) -> dict:
    from safetensors.torch import load_file

    files = sorted(path.glob("model*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no model*.safetensors under {path}")
    state = {}
    for f in files:
        state.update(load_file(str(f)))
    return state


def convert(hf_state: dict, cfg: Lfm3TextConfig, prefix: str = "model.") -> dict:
    """HF `Lfm3Model` names -> `Lfm3TextModel` names. `prefix` is where the
    text trunk lives in the checkpoint (`model.` for the causal LM,
    `model.language_model.` for the multimodal wrappers)."""
    d = cfg.hidden_size
    out = {}
    for name, t in hf_state.items():
        if not name.startswith(prefix):
            continue
        key = name[len(prefix):]
        if key.endswith("conv.in_proj.weight"):
            base = key[: -len("in_proj.weight")]
            b, c, x = t.split(d, dim=0)
            out[base + "B_proj.weight"] = b
            out[base + "C_proj.weight"] = c
            out[base + "x_proj.weight"] = x
            continue
        if key.endswith("conv.in_proj.bias"):
            base = key[: -len("in_proj.bias")]
            b, c, x = t.split(d, dim=0)
            out[base + "B_proj.bias"] = b
            out[base + "C_proj.bias"] = c
            out[base + "x_proj.bias"] = x
            continue
        out[key] = t
    if "lm_head.weight" not in out:
        out["lm_head.weight"] = out["embed_tokens.weight"]
    return out


def build(
    source: Path, *, prefix: str = "model.", hf_state: dict = None,
    dtype: torch.dtype = torch.float32, **runtime,
) -> Lfm3TextModel:
    hf_cfg = json.loads((Path(source) / "config.json").read_text())
    cfg = Lfm3TextConfig.from_hf(hf_cfg, **runtime)
    model = Lfm3TextModel(cfg)
    if hf_state is None:
        hf_state = read_safetensors(Path(source))
    head = hf_state.get("lm_head.weight")
    state = convert(hf_state, cfg, prefix)
    if head is not None and not cfg.tie_word_embeddings:
        state["lm_head.weight"] = head
    # dtype=None keeps the checkpoint's own dtype (bf16) for the weights and
    # leaves buffers (RoPE tables, caches) fp32: callers upcast layer by layer
    # (see export_lib.quantize) instead of holding a full fp32 copy.
    if dtype is not None:
        state = {k: v.to(dtype) for k, v in state.items()}
    del hf_state
    if dtype not in (None, torch.float32):
        model.to(dtype)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # Caches, conv states and RoPE tables are buffers, not weights (MLX's
    # KVCache registers its caches as persistent ones).
    missing = [k for k in missing if not k.endswith(("k_cache", "v_cache"))]
    if missing or unexpected:
        raise RuntimeError(f"weight mismatch: missing={missing} unexpected={unexpected}")
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    return model.eval()


@torch.no_grad()
def reset_state(model: torch.nn.Module) -> None:
    for name, buf in model.named_buffers():
        if name.endswith(("k_cache", "v_cache", "conv_state")):
            buf.zero_()
