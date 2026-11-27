"""Echo -> ExecuTorch QNN (Qualcomm HTP): the int4 backbone react-native-executorch ships for Reverb
(Reverb-TTS model/executorch/qnn/int4/reverb_qnn_int4_v<hexagon>.pte, one per SoC: export_reverb_qnn.sh). prefill and
step, in static shapes with int32 ids; the caller runs v3's polish (_echo/v3.py) as two step calls (polish_with_step)
with no arithmetic of its own.

  prefill : (text i32[1024], reference i32[8, 1024], lengths i32[2] = (L, R)) -> k_0..k_23, v_0..v_23   48 x f16[6, 1024, 128]
  step    : (tokens i32[8, 512], k_0..k_23, v_0..v_23, lengths i32[4] = (L, R, frames, reference_frames),
             prior f16[512, 4, 1024]) -> confidence f32[2, 512], prediction i32[2, 512], held f32[2, 512],
            fine i32[6, 512], logits f16[512, 4, 1024]
  step_t128, step_t256 : the same at T = 128 / 256

As in v3, the text (L <= 512 prompt tokens) and the reference (R <= 512 frames) come separately, each laid out over the
1024 cache slots: text ids in slots [0, L), reference codes in [512, 512 + R), anything elsewhere (prefill blends the
two embeddings with a constant slot mask, as v1's encode_prefix does; embedding them apart and concatenating computed
the reference slots wrong on the HTP inside the full program). The cache is 48 tensors, the 24 layers' keys
[6, 1024, 128] then their values: prefill's outputs, passed to step as they are, so a caller keeps 48 buffers and
never copies or slices them (QNN's HTP compiler rejects a 48-way concat in the graph and segfaults on the rank-5
stack, so prefill can't return one tensor). The static buckets pad: text and reference to 512 each (cache slots
[0, 512) text, [512, 1024) reference), the target to T. `lengths` gives the real sizes; the graph derives positions
(text 0..L-1, reference L..L+R-1, target from L+R) and the -1e4 masks of padded keys from them, so the outputs match
v3 on the real slots (padded target slots are garbage).

`step` is one pass. Main passes (reference_frames 0) read its codebook 0-1 outputs: best code, its log-probability, the
log-probability of the held code; they ignore `fine` and `logits`, and `prior` can hold anything. Polish passes
(reference_frames = how many of the last reference frames to bias towards, +1 on their scores) read `fine`, the codes
of codebooks 2-7: codebooks 2-3 are the plain argmax, 4-7 the argmax of 1.5 x logits - 0.5 x prior, where `logits`
(also returned) are this pass's codebook 4-7 logits. polish_with_step runs two of them: codebooks 2-3 from the first,
4-7 from the second with the first's `logits` as its `prior`, which is v3's polish. Polish is not one graph because
the HTP expands each layer's int4 weights to fp16 at run time, and a graph running both passes keeps all 24 layers'
expansions alive from the first pass to the second (552 MB of spill-fill on SM8650; 23 MB for step).

The graph is _echo/model.py's HTP one (fused attention with the scale folded into q_norm, aten.rms_norm, one
audio-embedding lookup, exp clamped at -30 for SM8650's fp16 exp). The weights are the released Reverb-TTS int4
checkpoint (hub.checkpoint("int4"): the w4a16 QAT codes, blocks of 32), lowered as the HTP's fp16-activation block
quantization (QNN_QUANTIZATION_ENCODING_BW_FLOAT_BLOCK), which needs patches/bw_float_block.patch in ExecuTorch.
Inputs are not memory-planned and shared_buffer is on, so a caller may bind them to rpcmem (the 48 cache buffers
serving prefill's outputs and every step's inputs); outputs are memory-planned, so a caller that runs a method and
copies its outputs out needs no output binding (the cost: the program's own copy of each output, ~9 ms per prefill on
a OnePlus 12).

usage: python qnn/export_qnn_v3.py <out.pte> --soc SM8650     (from exporter_models/echo)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _echo import hub  # noqa: E402
from _echo.model import MAX_POSITIONS, Attention, Echo, RMSNorm, fold_attention_scale  # noqa: E402

TEXT, REFERENCE = 512, 512       # v3's MAX_TEXT / MAX_REFERENCE: the prefix bucket
PREFIX = TEXT + REFERENCE
STEP_BUCKETS = {"step_t128": 128, "step_t256": 256, "step": 512}
NEG = -1e4                       # additive mask on padded keys
POLISH_BIAS, POLISH_GUIDANCE = 1.0, 0.5


def int4_block_spec(group: int):
    """int4 (in int8 containers) with a scale per block of `group` input weights, symmetric absmax / 7: gives back the
    checkpoint's codes and scales exactly (every block's max |code| is 7)."""
    from executorch.backends.qualcomm.quantizer.observers.per_block_param_observer import PerBlockParamObserver
    from torchao.quantization.pt2e.quantizer import QuantizationSpec

    return QuantizationSpec(
        dtype=torch.int8, quant_min=-7, quant_max=7, qscheme=torch.per_channel_symmetric, ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=PerBlockParamObserver.with_args(
            dtype=torch.int8, block_size=(1, group), quant_min=-7, quant_max=7),
    )


def weight_only(mod: torch.nn.Module, example: tuple, spec) -> tuple:
    """llm.layers.* projection weights quantized with `spec`, activations left float (fp16 on the HTP).
    Returns the converted module and the number of quantized linears."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torchao.quantization.pt2e.quantizer import QuantizationAnnotation, Quantizer

    class LayerProjections(Quantizer):
        count = 0

        def annotate(self, gm):
            for node in gm.graph.nodes:
                if node.op != "call_function" or node.target != torch.ops.aten.linear.default:
                    continue
                w = node.args[1]
                if w.op == "get_attr" and "llm.layers." in w.target and w.target.endswith("_proj.weight"):
                    node.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map={w: spec}, _annotated=True)
                    self.count += 1
            return gm

        def validate(self, gm):
            pass

    quantizer = LayerProjections()
    prepared = prepare_pt2e(torch.export.export(mod, example, strict=False).module(), quantizer)
    prepared(*example)
    return convert_pt2e(prepared), quantizer.count


class V3(torch.nn.Module):
    """Shared layout and layers over _echo/model.Echo."""

    def __init__(self, m: Echo) -> None:
        super().__init__()
        self.m = m
        slot = torch.arange(PREFIX, dtype=torch.int32)
        self.register_buffer("slot", slot, persistent=False)
        self.register_buffer("is_ref_col", (slot >= TEXT).float()[:, None], persistent=False)
        w = m.audio_heads.weight.detach().view(m.n_cb, m.vocab, -1)  # centered per codebook by from_pretrained
        mid = m.mask_id
        self.register_buffer("main_real", w[:2, :mid].flatten(0, 1).contiguous(), persistent=False)
        self.register_buffer("main_mask", w[:2, mid].contiguous(), persistent=False)
        self.register_buffer("polish_first", w[2:, :mid].flatten(0, 1).contiguous(), persistent=False)
        self.register_buffer("codes", torch.arange(mid, dtype=torch.int32), persistent=False)

    def prefix_layout(self, text_len: torch.Tensor, ref_len: torch.Tensor):
        """Cache slots: [0, 512) text, [512, 1024) reference. -> positions [P], valid [P], is_ref [P] (0/1 floats),
        j (reference index). Arithmetic, not where: the HTP can't select between bool tensors."""
        is_ref = (self.slot >= TEXT).float()
        j = self.slot - TEXT
        valid = is_ref * (j < ref_len).float() + (1.0 - is_ref) * (self.slot < text_len).float()
        # in float (exact: positions stay <= 2048): QNN's validator rejects the int32 multiply by a negative scalar
        pos = (self.slot.float() + is_ref * (text_len.float() - TEXT)).to(torch.int32).clamp(0, MAX_POSITIONS - 1)
        return pos, valid, is_ref, j

    def layers(self, x, cos, sin, cache, key_bias, self_bias):
        for i, layer in enumerate(self.m.llm.layers):
            h = layer.input_layernorm(x)
            k, v = layer.self_attn.kv(h, cos, sin)
            past_k, past_v = cache[i], cache[len(cache) // 2 + i]  # the layers' keys, then their values
            x = x + layer.self_attn(h, cos, sin, k, v, (past_k.float(), past_v.float(), key_bias), self_bias)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.m.llm.norm(x)

    def target(self, tokens, cache, lengths, key_bias):
        frames = tokens.shape[1]
        t = torch.arange(frames, dtype=torch.int32)
        pos = (lengths[0] + lengths[1] + t).clamp(0, MAX_POSITIONS - 1)
        self_bias = (t < lengths[2]).float().sub(1.0).mul(-NEG)
        cos, sin = self.m.rope(pos)
        return self.layers(self.m.embed_audio(tokens), cos, sin, cache, key_bias, self_bias)


class Prefill(V3):
    def forward(self, text, reference, lengths):
        pos, valid, _, _ = self.prefix_layout(lengths[0], lengths[1])
        key_bias = valid.sub(1.0).mul(-NEG)
        x = self.m.llm.embed_tokens(text) * (1.0 - self.is_ref_col) + self.m.embed_audio(reference) * self.is_ref_col
        cos, sin = self.m.rope(pos)
        keys, values = [], []
        for layer in self.m.llm.layers:
            h = layer.input_layernorm(x)
            k, v = layer.self_attn.kv(h, cos, sin)
            x = x + layer.self_attn(h, cos, sin, k, v, self_bias=key_bias)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
            keys.append(k)
            values.append(v)
        return tuple(t.half() for t in keys + values)


class Step(V3):
    def forward(self, tokens, cache, lengths, prior):
        _, valid, is_ref, j = self.prefix_layout(lengths[0], lengths[1])
        towards = is_ref * valid * (j >= lengths[1] - lengths[3]).float()   # none when reference_frames is 0
        x = self.target(tokens, cache, lengths, valid.sub(1.0).mul(-NEG) + towards * POLISH_BIAS)
        real = F.linear(x, self.main_real).view(-1, 2, self.m.mask_id)  # [T, 2, 1024]
        mask = F.linear(x, self.main_mask)                                 # [T, 2]
        top = real.amax(-1)
        rel = real - top[..., None]
        # log-sum-exp relative to the top logit, exp only of arguments >= -30 (SM8650's fp16 exp), then the mask code.
        lse_real = torch.log(torch.exp(rel.clamp(min=-30)).sum(-1))       # lse(real) - top
        z = mask - top - lse_real
        confidence = -lse_real - (F.relu(z) + torch.log(1 + torch.exp(-z.abs().clamp(max=30))))  # top - lse(all)
        # The held code's logit relative to the top, without gather (as v3): masked slots hold no real code.
        held_rel = rel.masked_fill(self.codes != tokens[:2].t().unsqueeze(-1), NEG).amax(-1)
        held = held_rel + confidence
        logits = F.linear(x, self.polish_first).view(-1, 6, self.m.mask_id)  # codebooks 2-7, [T, 6, 1024]
        late = logits[:, 2:]                                                 # codebooks 4-7: guided by the prior
        guided = (1 + POLISH_GUIDANCE) * late - POLISH_GUIDANCE * prior.float()
        fine = torch.cat([logits[:, :2].argmax(-1), guided.argmax(-1)], dim=1).to(torch.int32)  # [T, 6]
        return (confidence.t().float(), real.argmax(-1).to(torch.int32).t(), held.t().float(), fine.t(),
                late.half())


def polish_with_step(step: torch.nn.Module, tokens: torch.Tensor, reference_frames: torch.Tensor,
                     cache: list[torch.Tensor], lengths: torch.Tensor) -> torch.Tensor:
    """v3's polish (tokens i32[8, T], reference_frames i32[1], cache, lengths i32[3]) -> tokens i32[6, T] from two
    `step` calls: what a caller of the QNN program does. The first call's prior is ignored (zeros here)."""
    lengths4 = torch.cat([lengths[:3], reference_frames.view(1)]).to(torch.int32)
    prior = torch.zeros(tokens.shape[1], 4, step.m.mask_id, dtype=torch.float16)
    fine, logits = step(tokens, cache, lengths4, prior)[3:]
    middle = fine[:2]                                                      # codebooks 2-3, [2, T]
    tokens = torch.cat([tokens[:2], middle, tokens[4:]])
    fine = step(tokens, cache, lengths4, logits)[3]
    return torch.cat([middle, fine[2:]])


def build() -> tuple:
    """The model with the HTP graph choices, and {method: (module, example inputs)}."""
    RMSNorm.mode = "native"
    Attention.fused = True
    Echo.embed_mode = "one"
    model = Echo.from_pretrained(hub.checkpoint("int4"))
    fold_attention_scale(model)
    cfg = model.cfg
    layers, kv_heads, head_dim = len(model.llm.layers), cfg["num_key_value_heads"], cfg["head_dim"]
    i32 = torch.int32
    cache = [torch.zeros(kv_heads, PREFIX, head_dim, dtype=torch.float16) for _ in range(2 * layers)]
    lengths4 = torch.tensor([100, 200, 64, 0], dtype=i32)
    methods = {"prefill": (Prefill(model).eval(), (torch.zeros(PREFIX, dtype=i32), torch.zeros(8, PREFIX, dtype=i32),
                                                    torch.tensor([100, 200], dtype=i32)))}
    for name, frames in STEP_BUCKETS.items():
        methods[name] = (Step(model).eval(), (torch.full((8, frames), model.mask_id, dtype=i32), cache, lengths4,
                                              torch.zeros(frames, 4, model.mask_id, dtype=torch.float16)))
    return model, methods


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--soc", default="SM8650")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    _, methods = build()

    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
        get_qnn_context_binary_alignment,
        to_edge_transform_and_lower_to_qnn,
    )
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    specs = generate_qnn_executorch_compiler_spec(
        soc_model=getattr(QcomChipset, args.soc),
        backend_options=generate_htp_compiler_spec(use_fp16=True, use_weight_sharing=True),
        shared_buffer=True,
    )
    group = json.loads((Path(hub.checkpoint("int4")) / "config.json").read_text())["quantization"]["group_size"]
    modules, inputs = {}, {}
    for name, (mod, ex) in methods.items():
        mod, n = weight_only(mod, ex, int4_block_spec(group))
        print(f"{name}: {n} linears quantized (int4, blocks of {group})", flush=True)
        modules[name], inputs[name] = mod, ex
    t0 = time.time()
    edge = to_edge_transform_and_lower_to_qnn(modules, inputs, specs)
    print(f"lowering ok ({time.time() - t0:.0f}s)", flush=True)
    prog = edge.to_executorch(config=ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False, alloc_graph_output=True),
        segment_alignment=get_qnn_context_binary_alignment(),
    ))
    Path(args.out).write_bytes(prog.buffer)
    print(f"wrote {args.out} ({len(prog.buffer) / 2**20:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
