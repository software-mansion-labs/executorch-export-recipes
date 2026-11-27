"""Echo's v3 backbone contract for MLX and Vulkan: prefill / step / polish, from a checkpoint's own weights.

  prefill : (text i64[L], reference i64[8, R]) -> cache f16[2, 24, 6, P, 128]   (keys and values, P = L + R)
  step    : (tokens i64[8, T], cache) -> confidence f32[2, T], prediction i64[2, T], held f32[2, T]
  polish  : (tokens i64[8, T], reference_frames i64[1], cache) -> tokens i64[6, T]

`step` is one main sampler pass reduced in-graph to codebooks 0-1: each slot's best code, its log-probability, and
the log-probability of the code the slot holds now (`held`, meaningless for masked slots): remasking withdraws the
held codes that became least likely. `polish` runs both polish passes with +1 added to the target's scores towards
the R reference frames (the prefix's last `reference_frames` positions) and returns codebooks 2-7: 2-3 from the first
pass, 4-7 from the second, guided by 0.5 x (second - first). The caller keeps the cache opaque, passing it back whole;
the Gumbel top-k and remasking stay in the caller.

fp16 throughout but the audio heads, in fp32: the logits reach hundreds, where fp16 rounds to a quarter. A quantized
checkpoint (w8a16, w4a16) keeps its own int codes, which MLX runs in its quantized matmul. The Vulkan classes at the
end export the same contract from the fp32 checkpoint.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

DTYPE = torch.float16

# Upper bounds of the dynamic lengths, in positions: the prefix (prompt tokens + reference frames) and the target.
MAX_TEXT = 512       # prompt tokens: template, reference transcript and text
MAX_REFERENCE = 512  # reference frames (20 s is 500)
MAX_PREFIX = MAX_TEXT + MAX_REFERENCE
MAX_FRAMES = 1024

# The polish passes: attention bias towards the reference frames, and the second pass's guidance against the first
# (MiniMax multilingual: 0.5 beats 1.0 by +0.001 SIM-o at the same WER, knowledge/benchmarks/minimax_multilingual.md).
POLISH_BIAS = 1.0
POLISH_GUIDANCE = 0.5


def frozen(t: torch.Tensor) -> torch.nn.Parameter:
    """A constant held as a parameter: ExecuTorch drops the parameters a method doesn't use, but keeps every buffer."""
    return torch.nn.Parameter(t, requires_grad=False)


class Linear(torch.nn.Module):
    """A projection with fp16 weights, stored transposed: aten.linear lowers to a transpose and a matmul, x @ W to one
    node."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = frozen(weight.t().contiguous().to(DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class QuantizedLinear(torch.nn.Module):
    """A projection with the checkpoint's symmetric int weights, as is: the MLX backend fuses the dequantization into
    its quantized matmul. One scale per output channel (int8) or per group of inputs (int4)."""

    def __init__(self, values: torch.Tensor, scale: torch.Tensor, bits: int) -> None:
        super().__init__()
        self.values = frozen(values)
        # fp16: a quantized matmul runs in the dtype of its scales.
        self.scale = frozen(scale.to(DTYPE))
        self.zero = frozen(torch.zeros_like(scale, dtype=torch.int8))
        self.block = [1, values.shape[1] // scale.shape[1]]
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        limit = 2 ** (self.bits - 1)
        weight = torch.ops.torchao.dequantize_affine(
            self.values, self.block, self.scale, self.zero, torch.int8, -limit, limit - 1, output_dtype=DTYPE
        )
        return F.linear(x, weight)


def load_projections(weights: dict, quantization: Optional[dict]) -> dict:
    """The decoder projections, popped from `weights`: float weights, or int values and their scales."""
    projections = {}
    for key in [key for key in weights if key.startswith("llm.layers.") and key.endswith("_proj.weight")]:
        weight = weights.pop(key)
        if quantization is None:
            projections[key] = (weight.float(),)
            continue
        scale = weights.pop(key + "_scale").float()
        if quantization["bits"] == 4:
            # Two values per byte, the first one in the low nibble, offset by 8.
            weight = torch.stack([weight & 0xF, weight >> 4], dim=-1).flatten(-2).to(torch.int8) - 8
        projections[key] = (weight, scale)
    return projections


def make_linear(parts: List[tuple], bits: Optional[int]) -> torch.nn.Module:
    """One projection from the stacked output channels of `parts`, e.g. q, k and v as a single matmul."""
    if bits is None:
        return Linear(torch.cat([part[0] for part in parts]))
    return QuantizedLinear(torch.cat([part[0] for part in parts]), torch.cat([part[1] for part in parts]), bits)


class Layer(torch.nn.Module):
    def __init__(self, prefix: str, projections: dict, weights: dict, bits: Optional[int], config: dict) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config["heads"], config["kv_heads"], config["head_dim"]
        self.eps, self.theta = config["eps"], config["theta"]

        def linear(*names: str) -> torch.nn.Module:
            return make_linear([projections[f"{prefix}{name}.weight"] for name in names], bits)

        # q, k and v in one matmul: ~10% faster per pass in fp16. Gate and up fused gain nothing.
        self.qkv = linear("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
        self.o = linear("self_attn.o_proj")
        self.gate = linear("mlp.gate_proj")
        self.up = linear("mlp.up_proj")
        self.down = linear("mlp.down_proj")
        self.input_norm = frozen(weights[prefix + "input_layernorm.weight"].to(DTYPE))
        self.post_norm = frozen(weights[prefix + "post_attention_layernorm.weight"].to(DTYPE))
        self.q_norm = frozen(weights[prefix + "self_attn.q_norm.weight"].to(DTYPE))
        self.k_norm = frozen(weights[prefix + "self_attn.k_norm.weight"].to(DTYPE))

    def forward(
        self, x: torch.Tensor, position: int, past: Optional[Tuple[torch.Tensor, torch.Tensor]], mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[n, hidden] -> [n, hidden], and the n positions' keys and values, [1, kv heads, n, head dim] each."""
        n, d = x.shape[0], self.head_dim
        # aten.rms_norm, fused by MLX: the residual stream reaches |x| ~ 1.1e4, so a decomposed x * x overflows fp16.
        h = F.rms_norm(x, (x.shape[1],), self.input_norm, self.eps)
        q, k, v = self.qkv(h).split([self.heads * d, self.kv_heads * d, self.kv_heads * d], dim=-1)
        q = F.rms_norm(q.view(1, n, self.heads, d), (d,), self.q_norm, self.eps).transpose(1, 2)
        k = F.rms_norm(k.view(1, n, self.kv_heads, d), (d,), self.k_norm, self.eps).transpose(1, 2)
        v = v.view(1, n, self.kv_heads, d).transpose(1, 2)
        q = torch.ops.mlx.rope(q, d, position, False, self.theta, 1.0, None)
        k = torch.ops.mlx.rope(k, d, position, False, self.theta, 1.0, None)
        keys, values = (k, v) if past is None else (torch.cat([past[0], k], dim=2), torch.cat([past[1], v], dim=2))
        a = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, enable_gqa=True)
        x = x + self.o(a.transpose(1, 2).reshape(n, self.heads * d))
        h = F.rms_norm(x, (x.shape[1],), self.post_norm, self.eps)
        return x + self.down(F.silu(self.gate(h)) * self.up(h)), k, v


class Echo(torch.nn.Module):
    """The checkpoint's weights: the decoder in fp16 (or its int codes), the audio heads in fp32."""

    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        config = json.loads(Path(checkpoint, "config.json").read_text())
        quantization = config.get("quantization")
        if quantization and (quantization.get("activations") or quantization.get("rotation")):
            raise NotImplementedError("MLX has no int8 activations nor a Hadamard op: export the weight-only checkpoints")
        llm = config["llm_config"]
        geometry = {
            "heads": llm["num_attention_heads"],
            "kv_heads": llm["num_key_value_heads"],
            "head_dim": llm["head_dim"],
            "eps": llm["rms_norm_eps"],
            "theta": float(llm["rope_parameters"]["rope_theta"]),
        }
        self.books, self.codes, self.mask_id = config["num_audio_codebook"], config["audio_vocab_size"], config["audio_mask_id"]
        # The mask is the last code, so dropping the last column drops it from every prediction.
        assert self.mask_id == self.codes - 1

        weights = load_file(Path(checkpoint, "model.safetensors"))
        projections = load_projections(weights, quantization)
        # Embedding tables stored in int8, one scale per row: dequantized, as the program keeps them in fp16.
        for key in [key for key in weights if key.endswith("_scale")]:
            weights[key[: -len("_scale")]] = weights[key[: -len("_scale")]].float() * weights.pop(key).float()
        weights = {key: value.float() if value.is_floating_point() else value for key, value in weights.items()}
        bits = quantization["bits"] if quantization else None
        self.layers = torch.nn.ModuleList(
            Layer(f"llm.layers.{i}.", projections, weights, bits, geometry) for i in range(llm["num_hidden_layers"])
        )
        self.eps = geometry["eps"]
        self.norm = frozen(weights["llm.norm.weight"].to(DTYPE))
        # The text embedding stays factorized, as the checkpoint stores it: a [vocab, rank] table and a projection.
        self.table = frozen(weights["llm.embed_tokens.table.weight"].to(DTYPE))
        self.project = frozen(weights["llm.embed_tokens.project.weight"].t().contiguous().to(DTYPE))
        self.audio = frozen(weights["audio_embeddings.weight"].to(DTYPE))
        self.offsets = frozen(self.codes * torch.arange(self.books)[:, None])

        # Centering each codebook's rows leaves its softmax as it is, and keeps the logits small.
        heads = weights["audio_heads.weight"].view(self.books, self.codes, -1)
        heads = (heads - heads.mean(1, keepdim=True)).flatten(0, 1).t()
        self.main_heads = frozen(heads[:, : 2 * self.codes].contiguous())
        self.polish_heads = frozen(heads[:, 2 * self.codes :].contiguous())

    def embed_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """[8, n] codes -> [n, hidden]: each frame is the sum of its 8 codebook embeddings."""
        return F.embedding(codes + self.offsets, self.audio).sum(0)

    def run(
        self, x: torch.Tensor, position: int, past: Optional[Tuple[torch.Tensor, torch.Tensor]], mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        keys, values = [], []
        for i, layer in enumerate(self.layers):
            x, k, v = layer(x, position, None if past is None else (past[0, i : i + 1], past[1, i : i + 1]), mask)
            keys.append(k)
            values.append(v)
        return F.rms_norm(x, (x.shape[1],), self.norm, self.eps), keys, values

    def logits(self, h: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        """[n, hidden] -> [n, codebooks, codes], in fp32."""
        return (h.float() @ heads).view(h.shape[0], -1, self.codes)


# RoPE only sees relative positions, so the whole sequence is shifted by the prefix length: mlx::rope takes only a
# symbolic offset, and the prefix length is one that prefill and the passes both know. The prefix sits at [P, 2P),
# the target from 2P on.


class Prefill(torch.nn.Module):
    def __init__(self, echo: Echo) -> None:
        super().__init__()
        self.echo = echo

    def forward(self, text: torch.Tensor, reference: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([F.embedding(text, self.echo.table) @ self.echo.project, self.echo.embed_audio(reference)])
        _, keys, values = self.echo.run(x, x.shape[0], None, None)
        return torch.stack([torch.cat(keys), torch.cat(values)])


class Step(Prefill):
    """One main pass. The softmax counts the mask, but the mask is never a prediction."""

    def forward(self, tokens: torch.Tensor, cache: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        h, _, _ = self.echo.run(self.echo.embed_audio(tokens), 2 * cache.shape[3], cache, None)
        logits = self.echo.logits(h, self.echo.main_heads)
        normalizer = torch.logsumexp(logits, dim=-1)
        logits = logits[..., : self.echo.mask_id]
        # The held code's logit without gather (the MLX backend has none): the largest logit at its id.
        codes = torch.arange(self.echo.mask_id, device=tokens.device)
        held = logits.masked_fill(codes != tokens[:2].t().unsqueeze(-1), -torch.inf).amax(-1)
        return (logits.amax(-1) - normalizer).t(), logits.argmax(-1).t(), (held - normalizer).t()


class Polish(Prefill):
    """Both polish passes. The log-softmax normalizers are constant per slot, so the guided argmax runs on logits."""

    def forward(self, tokens: torch.Tensor, reference_frames: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        prefix, mid = cache.shape[3], self.echo.mask_id
        # +POLISH_BIAS on the scores towards the reference frames, the prefix's last `reference_frames` positions.
        keys = torch.arange(prefix, device=tokens.device)
        bias = (keys >= prefix - reference_frames).to(DTYPE) * POLISH_BIAS
        mask = torch.cat([bias, bias.new_zeros(tokens.shape[1])]).view(1, 1, 1, -1)
        h, _, _ = self.echo.run(self.echo.embed_audio(tokens), 2 * prefix, cache, mask)
        first = self.echo.logits(h, self.echo.polish_heads)[..., :mid]
        middle = first[:, :2].argmax(-1)
        tokens = torch.cat([tokens[:2], middle.t(), tokens[4:]])
        h, _, _ = self.echo.run(self.echo.embed_audio(tokens), 2 * prefix, cache, mask)
        second = self.echo.logits(h, self.echo.polish_heads[:, 2 * self.echo.codes :])[..., :mid]
        fine = ((1 + POLISH_GUIDANCE) * second - POLISH_GUIDANCE * first[:, 2:]).argmax(-1)
        return torch.cat([middle, fine], dim=1).t()


def export(echo: Echo) -> dict:
    """{"prefill", "step", "polish"} ExportedPrograms with dynamic lengths."""
    import executorch.backends.mlx.custom_ops  # noqa: F401  registers mlx::rope
    import torchao  # noqa: F401  registers torchao::dequantize_affine

    echo = echo.eval()
    Dim = torch.export.Dim
    layers, kv_heads, head_dim = len(echo.layers), echo.layers[0].kv_heads, echo.layers[0].head_dim
    prefix, frames = Dim("prefix", min=4, max=MAX_PREFIX), Dim("frames", min=4, max=MAX_FRAMES)
    text, reference = Dim("text", min=2, max=MAX_TEXT), Dim("reference", min=2, max=MAX_REFERENCE)
    cache = torch.zeros(2, layers, kv_heads, 300, head_dim, dtype=DTYPE)
    tokens = torch.zeros(echo.books, 120, dtype=torch.long)
    ex = torch.export.export
    with torch.no_grad():
        return {
            "prefill": ex(
                Prefill(echo),
                (torch.zeros(60, dtype=torch.long), torch.zeros(echo.books, 240, dtype=torch.long)),
                dynamic_shapes=({0: text}, {1: reference}),
                strict=False,
            ),
            "step": ex(Step(echo), (tokens, cache), dynamic_shapes=({1: frames}, {3: prefix}), strict=False),
            "polish": ex(
                Polish(echo),
                (tokens, torch.tensor([240]), cache),
                dynamic_shapes=({1: frames}, None, {3: prefix}),
                strict=False,
            ),
        }


def schema(echo: Echo) -> str:
    """The `get_model_schema` JSON (react-native-executorch's ModelSpec): the dynamic lengths' bounds, which the
    program's metadata gives as fixed sizes, and the target length each method keeps. P = L + R can't be declared."""
    layer = echo.layers[0]

    def tensor(dtype: str, *shape) -> dict:
        dims = [d if isinstance(d, dict) else {"kind": "constant", "value": d} for d in shape]
        return {"kind": "Tensor", "dtype": dtype, "shape": dims}

    def length(low: int, high: int) -> dict:
        return {"kind": "range", "range": {"min": low, "max": high, "step": 1}}

    def same_frames(*refs: Tuple[str, int]) -> dict:
        return {"kind": "equality", "dims": [{"paramSide": side, "tensorIdx": i, "dimIdx": 1} for side, i in refs]}

    frames = length(4, MAX_FRAMES)
    tokens = tensor("int64", echo.books, frames)
    cache = tensor("float16", 2, len(echo.layers), layer.kv_heads, length(4, MAX_PREFIX), layer.head_dim)
    return json.dumps({
        "prefill": {
            "inputs": [tensor("int64", length(2, MAX_TEXT)), tensor("int64", echo.books, length(2, MAX_REFERENCE))],
            "outputs": [cache],
            "runtimeConstraints": [],
        },
        "step": {
            "inputs": [tokens, cache],
            "outputs": [tensor("float32", 2, frames), tensor("int64", 2, frames), tensor("float32", 2, frames)],
            "runtimeConstraints": [same_frames(("input", 0), ("output", 0), ("output", 1), ("output", 2))],
        },
        "polish": {
            "inputs": [tokens, tensor("int64", 1), cache],
            "outputs": [tensor("int64", echo.books - 2, frames)],
            "runtimeConstraints": [same_frames(("input", 0), ("output", 0))],
        },
    })


def lower(methods: dict, model_schema: str) -> bytes:
    """The .pte bytes, every op inside the MLX delegate, with `model_schema` as the `get_model_schema` method."""
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.exir.passes import MemoryPlanningPass

    edge = to_edge_transform_and_lower(
        methods,
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        constant_methods={"get_model_schema": model_schema},
    )
    for name in methods:
        stray = {
            str(node.target)
            for node in edge.exported_program(name).graph.nodes
            if node.op == "call_function"
            and "executorch_call_delegate" not in str(node.target)
            and "getitem" not in str(node.target)
        }
        if stray:
            raise RuntimeError(f"{name}: ops left outside the MLX delegate: {sorted(stray)}")

    # Inputs are not copied into planned memory: the K/V cache goes back in on every pass.
    config = ExecutorchBackendConfig(
        extract_delegate_segments=True, memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
    )
    return edge.to_executorch(config).buffer


# Vulkan (Android GPUs). The cache keeps the MLX shape f16[2, 24, 6, P, 128] but holds [2, 24, P, 6, 128] data, the
# layout llama.custom_sdpa reads: the caller passes it back whole and never looks inside.


class VulkanLayer(torch.nn.Module):
    """A layer on Vulkan's own ops: weights as [out, in] (its linear prepacks them), et_vk RoPE, and llama.custom_sdpa,
    which takes non-causal attention only as an additive [n, keys] mask."""

    def __init__(self, prefix: str, projections: dict, weights: dict, config: dict) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config["heads"], config["kv_heads"], config["head_dim"]
        self.eps = config["eps"]

        def weight(*names: str) -> torch.nn.Parameter:
            return frozen(torch.cat([projections[f"{prefix}{name}.weight"][0] for name in names]).to(DTYPE))

        self.qkv = weight("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")
        self.o = weight("self_attn.o_proj")
        self.gate = weight("mlp.gate_proj")
        self.up = weight("mlp.up_proj")
        self.down = weight("mlp.down_proj")
        self.input_norm = frozen(weights[prefix + "input_layernorm.weight"].to(DTYPE))
        self.post_norm = frozen(weights[prefix + "post_attention_layernorm.weight"].to(DTYPE))
        self.q_norm = frozen(weights[prefix + "self_attn.q_norm.weight"].to(DTYPE))
        self.k_norm = frozen(weights[prefix + "self_attn.k_norm.weight"].to(DTYPE))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position,
        past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """[n, hidden] -> [n, hidden], and the n positions' keys and values, [1, n, kv heads, head dim] each."""
        n, d = x.shape[0], self.head_dim
        h = F.rms_norm(x, (x.shape[1],), self.input_norm, self.eps)
        q, k, v = F.linear(h, self.qkv).split([self.heads * d, self.kv_heads * d, self.kv_heads * d], dim=-1)
        q = F.rms_norm(q.reshape(1, n, self.heads, d), (d,), self.q_norm, self.eps)
        k = F.rms_norm(k.reshape(1, n, self.kv_heads, d), (d,), self.k_norm, self.eps)
        v = v.reshape(1, n, self.kv_heads, d)
        q, k = torch.ops.et_vk.apply_rotary_emb_hf(q, k, cos, sin, position)
        keys, values = (k, v) if past is None else (torch.cat([past[0], k], dim=1), torch.cat([past[1], v], dim=1))
        a = torch.ops.llama.custom_sdpa(q, keys, values, 0, mask, 0.0, False, None)
        x = x + F.linear(a.reshape(n, self.heads * d), self.o)
        h = F.rms_norm(x, (x.shape[1],), self.post_norm, self.eps)
        # SiLU spelled out: the Vulkan delegate has sigmoid but no silu.
        g = F.linear(h, self.gate)
        return x + F.linear(g * torch.sigmoid(g) * F.linear(h, self.up), self.down), k, v


class VulkanEcho(torch.nn.Module):
    """The fp32 checkpoint's weights for Vulkan: the decoder in fp16, the audio heads in fp32."""

    def __init__(self, checkpoint: str) -> None:
        super().__init__()
        config = json.loads(Path(checkpoint, "config.json").read_text())
        if config.get("quantization"):
            raise NotImplementedError("the Vulkan export takes the fp32 checkpoint")
        llm = config["llm_config"]
        geometry = {
            "heads": llm["num_attention_heads"],
            "kv_heads": llm["num_key_value_heads"],
            "head_dim": llm["head_dim"],
            "eps": llm["rms_norm_eps"],
        }
        self.books, self.codes, self.mask_id = config["num_audio_codebook"], config["audio_vocab_size"], config["audio_mask_id"]
        assert self.mask_id == self.codes - 1

        weights = load_file(Path(checkpoint, "model.safetensors"))
        projections = load_projections(weights, None)
        weights = {key: value.float() if value.is_floating_point() else value for key, value in weights.items()}
        self.layers = torch.nn.ModuleList(
            VulkanLayer(f"llm.layers.{i}.", projections, weights, geometry) for i in range(llm["num_hidden_layers"])
        )
        self.eps = geometry["eps"]
        self.norm = frozen(weights["llm.norm.weight"].to(DTYPE))
        self.table = frozen(weights["llm.embed_tokens.table.weight"].to(DTYPE))
        self.project = frozen(weights["llm.embed_tokens.project.weight"].to(DTYPE))
        self.audio = frozen(weights["audio_embeddings.weight"].to(DTYPE))
        self.offsets = frozen(self.codes * torch.arange(self.books)[:, None])

        # RoPE tables for the prefix at [0, P) and the target from P on.
        d = geometry["head_dim"]
        inverse = 1.0 / (float(llm["rope_parameters"]["rope_theta"]) ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
        angles = torch.arange(MAX_PREFIX + MAX_FRAMES, dtype=torch.float64)[:, None] * inverse
        angles = torch.cat([angles, angles], dim=-1)
        self.cos = frozen(angles.cos().to(DTYPE))
        self.sin = frozen(angles.sin().to(DTYPE))

        heads = weights["audio_heads.weight"].view(self.books, self.codes, -1)
        heads = heads - heads.mean(1, keepdim=True)
        # Each main codebook padded to a multiple of 4 rows for the vectorized linear; the -1e4 bias keeps the padding
        # out of the softmax, which still counts the mask.
        self.padded = (self.codes + 3) // 4 * 4
        main = torch.zeros(2, self.padded, heads.shape[-1])
        main[:, : self.codes] = heads[:2]
        self.main_heads = frozen(main.flatten(0, 1))
        bias = torch.full((2, self.padded), -1e4)
        bias[:, : self.codes] = 0
        self.main_bias = frozen(bias.flatten())
        # The polish passes only take argmaxes of logits, so their heads drop the mask row.
        self.middle_heads = frozen(heads[2:4, : self.mask_id].flatten(0, 1).contiguous())
        self.fine_heads = frozen(heads[4:, : self.mask_id].flatten(0, 1).contiguous())

    def embed_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """[8, n] codes -> [n, hidden]. keepdim: Vulkan reduces over the first dim only with it."""
        return F.embedding(codes + self.offsets, self.audio).sum(0, keepdim=True)[0]

    def past(self, cache: torch.Tensor) -> torch.Tensor:
        """The cache as [2 * layers, P, kv heads, head dim]. Through a flat view: torch.export guards a direct one."""
        layer = self.layers[0]
        return cache.flatten().view(2 * len(self.layers), cache.shape[3], layer.kv_heads, layer.head_dim)

    def run(
        self, x: torch.Tensor, position, past: Optional[torch.Tensor], mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        keys, values = [], []
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            layer_past = None if past is None else (past[i : i + 1], past[n + i : n + i + 1])
            x, k, v = layer(x, self.cos, self.sin, position, layer_past, mask)
            keys.append(k)
            values.append(v)
        return F.rms_norm(x, (x.shape[1],), self.norm, self.eps), keys, values


class VulkanPrefill(torch.nn.Module):
    def __init__(self, echo: VulkanEcho) -> None:
        super().__init__()
        self.echo = echo

    def forward(self, text: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F.linear(F.embedding(text, self.echo.table), self.echo.project), self.echo.embed_audio(reference)])
        n = x.shape[0]
        _, keys, values = self.echo.run(x, 0, None, torch.zeros(n, n, dtype=DTYPE))
        layer = self.echo.layers[0]
        return torch.cat(keys + values).flatten().view(2, len(self.echo.layers), layer.kv_heads, n, layer.head_dim)


class VulkanStep(VulkanPrefill):
    """One main pass. The softmax counts the mask, but the mask is never a prediction."""

    def forward(self, tokens: torch.Tensor, cache: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        prefix, frames = cache.shape[3], tokens.shape[1]
        mask = torch.zeros(frames, prefix + frames, dtype=DTYPE)
        h, _, _ = self.echo.run(self.echo.embed_audio(tokens), prefix, self.echo.past(cache), mask)
        logits = F.linear(h.float(), self.echo.main_heads, self.echo.main_bias)
        logp = torch.log_softmax(logits.view(frames, 2, self.echo.padded).transpose(0, 1), dim=-1)
        real = logp[..., : self.echo.mask_id]
        # Dim 2, not -1: the Vulkan squeeze's resize ignores negative dims.
        held = logp.gather(2, tokens[:2].unsqueeze(2)).squeeze(2)
        return real.amax(-1), real.argmax(-1), held


class VulkanPolish(VulkanPrefill):
    """Both polish passes. The log-softmax normalizers are constant per slot, so the guided argmax runs on logits."""

    def forward(self, tokens: torch.Tensor, reference_frames: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        prefix, frames, mid = cache.shape[3], tokens.shape[1], self.echo.mask_id
        # +POLISH_BIAS on the scores towards the prefix's last `reference_frames` positions. The distance from the end
        # is a flipped arange: the Vulkan arange's resize miscounts negative steps.
        bias = (torch.arange(prefix).flip(0) < reference_frames).to(DTYPE) * POLISH_BIAS
        mask = torch.cat([bias, torch.zeros(frames, dtype=DTYPE)]).unsqueeze(0).expand(frames, -1)
        past = self.echo.past(cache)
        h, _, _ = self.echo.run(self.echo.embed_audio(tokens), prefix, past, mask)
        middle = F.linear(h.float(), self.echo.middle_heads).view(frames, 2, mid).argmax(-1)
        first = F.linear(h.float(), self.echo.fine_heads).view(frames, 4, mid)
        tokens = torch.cat([tokens[:2], middle.t(), tokens[4:]])
        h, _, _ = self.echo.run(self.echo.embed_audio(tokens), prefix, past, mask)
        second = F.linear(h.float(), self.echo.fine_heads).view(frames, 4, mid)
        fine = ((1 + POLISH_GUIDANCE) * second - POLISH_GUIDANCE * first).argmax(-1)
        return torch.cat([middle, fine], dim=1).t()


def export_vulkan(echo: VulkanEcho) -> dict:
    """{"prefill", "step", "polish"} ExportedPrograms with dynamic lengths."""
    import executorch.backends.vulkan.custom_ops_lib  # noqa: F401  registers et_vk ops
    import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401  registers llama.custom_sdpa

    echo = echo.eval()
    Dim = torch.export.Dim
    layer = echo.layers[0]
    prefix, frames = Dim("prefix", min=4, max=MAX_PREFIX), Dim("frames", min=4, max=MAX_FRAMES)
    text, reference = Dim("text", min=2, max=MAX_TEXT), Dim("reference", min=2, max=MAX_REFERENCE)
    cache = torch.zeros(2, len(echo.layers), layer.kv_heads, 300, layer.head_dim, dtype=DTYPE)
    tokens = torch.zeros(echo.books, 120, dtype=torch.long)
    ex = torch.export.export
    with torch.no_grad():
        return {
            "prefill": ex(
                VulkanPrefill(echo),
                (torch.zeros(60, dtype=torch.long), torch.zeros(echo.books, 240, dtype=torch.long)),
                dynamic_shapes=({0: text}, {1: reference}),
                strict=False,
            ),
            "step": ex(VulkanStep(echo), (tokens, cache), dynamic_shapes=({1: frames}, {3: prefix}), strict=False),
            "polish": ex(
                VulkanPolish(echo),
                (tokens, torch.tensor([240]), cache),
                dynamic_shapes=({1: frames}, None, {3: prefix}),
                strict=False,
            ),
        }


def lower_vulkan(methods: dict, model_schema: str) -> bytes:
    """The .pte bytes, each method one Vulkan delegate (the cache's sym_size stays outside), with `model_schema` as the
    `get_model_schema` method."""
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
    from executorch.backends.vulkan.patterns.pattern_registry import fusable_patterns
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.exir.passes import MemoryPlanningPass

    # The fork's causal_sdpa pattern takes every custom_sdpa for an LLM's update_cache + custom_sdpa pair and asserts on
    # the missing update_cache; the plain ones here are left to the partitioner's own check.
    entry = fusable_patterns["causal_sdpa"]
    detect = entry.detector_fn
    entry.detector_fn = lambda node: m if (m := detect(node)) is not None and m.update_key_cache_node is not None else None
    try:
        edge = to_edge_transform_and_lower(
            methods,
            # Options of the patched runtime, which older runtimes ignore. shared_workspace: the methods never run at
            # once, so they get one set of working memory. fp16_round_to_nearest: fp16 results round to nearest even
            # where the GPU can choose (Samsung's Xclipse driver rounds toward zero by default).
            partitioner=[
                VulkanPartitioner(
                    {"texture_limits": (2048, 2048, 2048), "shared_workspace": True, "fp16_round_to_nearest": True}
                )
            ],
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            constant_methods={"get_model_schema": model_schema},
        )
    finally:
        entry.detector_fn = detect
    for name in methods:
        stray = {
            str(node.target)
            for node in edge.exported_program(name).graph.nodes
            if node.op == "call_function"
            and "executorch_call_delegate" not in str(node.target)
            and "getitem" not in str(node.target)
            and "sym_size" not in str(node.target)
        }
        if stray:
            raise RuntimeError(f"{name}: ops left outside the Vulkan delegate: {sorted(stray)}")
    config = ExecutorchBackendConfig(
        extract_delegate_segments=True, memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
    )
    return edge.to_executorch(config).buffer
