"""Exports the W8A8 Echo checkpoint to ExecuTorch for the XNNPACK backend (CPU): the checkpoint's int8 weights, and the
inputs of every decoder linear quantized to int8 on the fly, one asymmetric scale per token. The embedding tables stay
int8 (quantized_decomposed::embedding_byte); everything else is fp32.

The program has the interface of the MLX and Vulkan programs, react-native-executorch's dynamic Reverb backbone, and
their get_model_schema:
- prefill(text [L], reference [8, R]) -> cache fp16 [2, layers, kv heads, P, head dim]: keys, then values.
- step(tokens [8, T], cache) -> confidence, prediction, held [2, T]: one main pass, codebooks 0-1 only.
- polish(tokens [8, T], reference_frames [1], cache) -> [6, T]: both polish passes, codebooks 2-7, with the attention
  bias towards the R reference frames and the guidance (0.5) in the graph.
L and R stop at 512 each: ExecuTorch plans prefill's outputs for the largest P = L + R.

The attention's matmuls, softmax and the cache are fp16, on values that stay below 200; everything else is as above.
fp16 matmuls need a CPU with fp16 arithmetic (ARMv8.2): XNNPACK creates none without it.

The graph is laid out for XNNPACK's cost model, where every copy, layout change and broadcast is an operator:
- q, k and v come from one int8 matmul, its rows every head's first halves (q's heads, then k's), their second
  halves, then v: RoPE combines the halves elementwise, on flat tensors.
- Tensors stay 2-D or 3-D: a 4-D mean would be run channels-last, with a transpose on each side. Means are matmuls.
- The attention reads keys as [kv heads, P, head dim] through XNNPACK's transposed batch matmul, with no transpose.
- The cache is one tensor only at the methods' edges: each layer cuts its keys and values out as it runs.

Three patches of ExecuTorch's XNNPACK lowering make that a single delegate per method: slices of tensors with dynamic
shapes, the merging of partitions around a quantized tensor that several linears read, and the transposed batch matmul.

Run the program with one thread per performance core (6 on an M4 Pro, an iPhone 17 and a Snapdragon 8 Gen 3).
"""

import collections
import json
import operator
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.export import Dim, export

import torchao  # noqa: F401 - registers torchao's quantization ops

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes import _quant_patterns_and_replacements  # noqa: F401 - registers embedding_byte

# Upper bounds of the dynamic lengths, in positions: the prompt tokens, the reference frames, and the target.
MAX_TEXT = 512
MAX_REFERENCE = 512
MAX_FRAMES = 1024

# The polish passes: attention bias towards the reference frames, and the second pass's guidance against the first.
POLISH_BIAS = 1.0
POLISH_GUIDANCE = 0.5


def frozen(t: torch.Tensor) -> torch.nn.Parameter:
    """
    A constant held as a parameter: ExecuTorch drops the parameters a method doesn't use, but keeps every buffer.
    """

    return torch.nn.Parameter(t, requires_grad=False)


def quantize_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    x through int8, one asymmetric scale per row. One per tensor, shared by every linear that reads it: XNNPACK runs
    it and those linears as int8 matmuls.
    """

    block = [1] * (x.dim() - 1) + [x.shape[-1]]
    scale, zero = torch.ops.torchao.choose_qparams_affine(x, "ASYMMETRIC", block, torch.int8, -128, 127, None, None, torch.int8)
    x = torch.ops.torchao.quantize_affine(x, block, scale, zero, torch.int8, -128, 127)
    return torch.ops.torchao.dequantize_affine(x, block, scale, zero, torch.int8, -128, 127)


class QuantizedLinear(torch.nn.Module):
    """
    A projection with the checkpoint's int8 weights, one scale per output channel, of an input through quantize_tokens.
    """

    def __init__(self, values: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.values = frozen(values.contiguous())
        self.scale = frozen(scale.reshape(-1, 1).float())
        self.zero = frozen(torch.zeros_like(self.scale, dtype=torch.int8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block = [1, self.values.shape[1]]
        weight = torch.ops.torchao.dequantize_affine(self.values, block, self.scale, self.zero, torch.int8, -128, 127)
        return F.linear(x, weight)


class Mean(torch.nn.Module):
    """
    The mean of [n, d] rows as a matmul with 1 / d, plus eps: [n, 1]. XNNPACK's own mean wants a 4-D tensor in
    channels-last layout, a transpose away.
    """

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = frozen(torch.full((1, width), 1.0 / width))
        self.bias = frozen(torch.tensor([eps]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def halves(t: torch.Tensor, heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The rows of a per-head projection, [heads * head dim, ...], split into every head's first and second halves.
    """

    t = t.view(heads, 2, -1, *t.shape[1:])
    return t[:, 0].flatten(0, 1), t[:, 1].flatten(0, 1)


class Layer(torch.nn.Module):
    def __init__(self, prefix: str, weights: dict, config: dict) -> None:
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = config["heads"], config["kv_heads"], config["head_dim"]
        self.fp16_attention = config["fp16_attention"]
        half, hidden = self.head_dim // 2, weights[prefix + "input_layernorm.weight"].shape[0]

        def linear(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
            return weights[f"{prefix}{name}.weight"], weights[f"{prefix}{name}.weight_scale"]

        (q, q_scale), (k, k_scale), (v, v_scale) = linear("self_attn.q_proj"), linear("self_attn.k_proj"), linear("self_attn.v_proj")
        (q1, q2), (q1_scale, q2_scale) = halves(q, self.heads), halves(q_scale, self.heads)
        (k1, k2), (k1_scale, k2_scale) = halves(k, self.kv_heads), halves(k_scale, self.kv_heads)
        # One matmul: XNNPACK quantizes and packs the input again for every linear that reads it. Its rows: every
        # head's first halves (q's heads, then k's), their second halves, which RoPE pairs elementwise, then v.
        self.qkv = QuantizedLinear(torch.cat([q1, k1, q2, k2, v]), torch.cat([q1_scale, k1_scale, q2_scale, k2_scale, v_scale]))
        self.o = QuantizedLinear(*linear("self_attn.o_proj"))
        self.gate = QuantizedLinear(*linear("mlp.gate_proj"))
        self.up = QuantizedLinear(*linear("mlp.up_proj"))
        self.down = QuantizedLinear(*linear("mlp.down_proj"))
        self.input_norm = frozen(weights[prefix + "input_layernorm.weight"])
        self.post_norm = frozen(weights[prefix + "post_attention_layernorm.weight"])
        self.input_mean, self.post_mean = Mean(hidden, config["eps"]), Mean(hidden, config["eps"])
        # Each head is normed over its halves: the mean of a^2 + b^2 over a half is twice the full mean, which the
        # weights' factor sqrt(2) makes up for. The attention's 1 / sqrt(head dim) goes into the q norm.
        q_norm = weights[prefix + "self_attn.q_norm.weight"] * 2**0.5 / self.head_dim**0.5
        k_norm = weights[prefix + "self_attn.k_norm.weight"] * 2**0.5
        self.norm1, self.norm2 = (
            frozen(torch.cat([qw.repeat(self.heads), kw.repeat(self.kv_heads)]).contiguous())
            for qw, kw in zip(q_norm.chunk(2), k_norm.chunk(2))
        )
        self.head_mean = Mean(half, 2 * config["eps"])

    def forward(
        self,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor, torch.Tensor],
        past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [n, hidden] -> [n, hidden], and the n positions' keys and values, [kv heads, n, head dim] each.
        """

        d, groups, heads = self.head_dim, self.kv_heads, self.heads + self.kv_heads
        cut, permute = torch.ops.aten.slice_copy.Tensor, torch.ops.aten.permute_copy
        h = quantize_tokens(x * torch.rsqrt(self.input_mean(x * x)) * self.input_norm)
        qkv, width = self.qkv(h), heads * d // 2
        a, b, v = cut(qkv, 1, 0, width), cut(qkv, 1, width, 2 * width), cut(qkv, 1, 2 * width, 2 * width + groups * d)
        # Per-head RMSNorm of q's and k's heads together: its scale from the two halves' squares, [heads, n, 1].
        r = torch.rsqrt(self.head_mean((a * a + b * b).view(-1, d // 2)))
        r = permute(r.view(-1, heads), [1, 0]).reshape(heads, -1, 1)
        # RoPE on the halves, heads first: [heads, n, head dim / 2].
        cos, sin = rope
        a = permute((a * self.norm1).view(-1, heads, d // 2), [1, 0, 2])
        b = permute((b * self.norm2).view(-1, heads, d // 2), [1, 0, 2])
        qk = torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1) * r
        if self.fp16_attention:  # before the tensors are cut and moved: half the bytes
            qk, v = qk.half(), v.half()
        # Each kv head serves 2 q heads: their rows are stacked, so one batched matmul per kv head needs no repeat.
        q = cut(qk, 0, 0, self.heads).view(groups, -1, d)
        k = cut(qk, 0, self.heads, heads)
        v = permute(v.view(-1, groups, d), [1, 0, 2])
        keys, values = (k, v) if past is None else (torch.cat([past[0], k], dim=1), torch.cat([past[1], v], dim=1))
        # The lowering turns this transpose into XNNPACK's transposed batch matmul: no keys are moved.
        scores = torch.bmm(q, permute(keys, [0, 2, 1]))
        if mask is not None:
            scores = scores + mask
        a = torch.bmm(torch.softmax(scores, dim=-1), values)
        if self.fp16_attention:
            a = a.float()
        a = permute(a.view(groups, self.heads // groups, -1, d), [2, 0, 1, 3]).reshape(-1, self.heads * d)
        x = x + self.o(quantize_tokens(a))
        h = quantize_tokens(x * torch.rsqrt(self.post_mean(x * x)) * self.post_norm)
        gate = self.gate(h)
        return x + self.down(quantize_tokens(gate * torch.sigmoid(gate) * self.up(h))), k, v


class Echo(torch.nn.Module):
    """
    The checkpoint's weights: the decoder linears and the embedding tables in int8, everything else in fp32.
    """

    def __init__(self, checkpoint: str, fp16_attention: bool = True) -> None:
        super().__init__()
        config = json.loads(Path(checkpoint, "config.json").read_text())
        quantization = config.get("quantization") or {}
        if quantization.get("bits") != 8 or quantization.get("activations") != "per_token" or quantization.get("rotation"):
            raise NotImplementedError("this export is for the W8A8 checkpoint: int8 weights, int8 inputs, no rotation")
        if quantization.get("embeddings") != "int8_per_row":
            raise NotImplementedError("this export takes the checkpoint's embedding tables in int8, one scale per row")
        llm = config["llm_config"]
        geometry = {
            "heads": llm["num_attention_heads"],
            "kv_heads": llm["num_key_value_heads"],
            "head_dim": llm["head_dim"],
            "eps": llm["rms_norm_eps"],
            "fp16_attention": fp16_attention,
        }
        self.books, self.codes, self.mask_id = config["num_audio_codebook"], config["audio_vocab_size"], config["audio_mask_id"]
        # The mask is the last code, so dropping the last column drops it from every prediction.
        assert self.mask_id == self.codes - 1

        weights = load_file(Path(checkpoint, "model.safetensors"))
        tables = {}
        for name in ("llm.embed_tokens.table", "audio_embeddings"):
            tables[name] = (frozen(weights.pop(f"{name}.weight")), frozen(weights.pop(f"{name}.weight_scale").float().view(-1)))
        weights = {key: value.float() if value.dtype in (torch.bfloat16, torch.float16) else value for key, value in weights.items()}
        self.layers = torch.nn.ModuleList(
            Layer(f"llm.layers.{i}.", weights, geometry) for i in range(llm["num_hidden_layers"])
        )
        self.eps, self.heads, self.kv_heads = geometry["eps"], geometry["heads"], geometry["kv_heads"]
        self.norm = frozen(weights["llm.norm.weight"])
        # The text embedding stays factorized, as the checkpoint stores it: a [vocab, rank] table and a projection.
        self.table, self.table_scale = tables["llm.embed_tokens.table"]
        self.project = frozen(weights["llm.embed_tokens.project.weight"])
        self.audio, self.audio_scale = tables["audio_embeddings"]
        self.offsets = frozen(self.codes * torch.arange(self.books)[:, None])
        half = geometry["head_dim"] // 2
        self.inv_freq = frozen(1.0 / float(llm["rope_parameters"]["rope_theta"]) ** (torch.arange(half) / half))
        self.mean = Mean(self.norm.shape[0], self.eps)

        # Centering each codebook's rows leaves its softmax as it is, and keeps the logits small. The heads are split by
        # use, so that every pass multiplies by whole parameters. The polish passes drop the mask's rows: they only take
        # the argmax of the logits, and the mask is never a prediction.
        heads = weights["audio_heads.weight"].view(self.books, self.codes, -1)
        heads = heads - heads.mean(1, keepdim=True)
        self.main_heads = frozen(heads[:2].flatten(0, 1).contiguous())
        self.middle_heads = frozen(heads[2:4, : self.mask_id].flatten(0, 1).contiguous())
        self.fine_heads = frozen(heads[4:, : self.mask_id].flatten(0, 1).contiguous())

    def embed_text(self, ids: torch.Tensor) -> torch.Tensor:
        """
        [n] prompt tokens -> [n, hidden].
        """

        lookup = torch.ops.quantized_decomposed.embedding_byte.default
        return F.linear(lookup(self.table, self.table_scale, None, -128, 127, ids), self.project)

    def embed_audio(self, codes: torch.Tensor) -> torch.Tensor:
        """
        [8, n] codes -> [n, hidden]: each frame is the sum of its 8 codebook embeddings.
        """

        rows = codes + self.offsets
        lookup = torch.ops.quantized_decomposed.embedding_byte.default
        x = lookup(self.audio, self.audio_scale, None, -128, 127, rows[0])
        for book in range(1, self.books):
            x = x + lookup(self.audio, self.audio_scale, None, -128, 127, rows[book])
        return x

    def rope(self, start: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        cos and sin of positions [start, start + n), [1, n, head dim / 2]: they broadcast over the heads in front, as
        whole runs of n positions, and stay in the cache between layers.
        """

        angles = (torch.arange(n, dtype=torch.float32) + start)[:, None] * self.inv_freq[None, :]
        return angles.cos().view(1, -1, angles.shape[1]), angles.sin().view(1, -1, angles.shape[1])

    def run(
        self, x: torch.Tensor, past: Optional[torch.Tensor], mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        [n, hidden] through the layers, after the positions of 'past', the cache [2, layers, kv heads, P, head dim].
        """

        rope = self.rope(0 if past is None else past.shape[3], x.shape[0])
        cut, cache = torch.ops.aten.slice_copy.Tensor, []
        for i, layer in enumerate(self.layers):
            pair = None
            if past is not None:
                # A layer cuts its own keys and values out when it needs them: cut up front, the whole cache would
                # stay alive through the pass, a second time, at the size of the longest prefix.
                both = cut(past, 1, i, i + 1)
                pair = tuple(cut(both, 0, j, j + 1).view(past.shape[2], -1, past.shape[4]) for j in range(2))
            x, k, v = layer(x, rope, pair, mask)
            cache += [k, v]

        return x * torch.rsqrt(self.mean(x * x)) * self.norm, cache

    def logits(self, h: torch.Tensor, heads: torch.Tensor, codes: int) -> torch.Tensor:
        """
        [n, hidden] -> [n, codebooks, codes].
        """

        return F.linear(h, heads).view(h.shape[0], -1, codes)


class Prefill(torch.nn.Module):
    def __init__(self, echo: Echo) -> None:
        super().__init__()
        self.echo = echo

    def forward(self, text: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        echo = self.echo
        _, cache = echo.run(torch.cat([echo.embed_text(text), echo.embed_audio(reference)]), None, None)
        # Every layer's keys, then every layer's values. XNNPACK concatenates at most 5 tensors.
        pieces = cache[0::2] + cache[1::2]
        while len(pieces) > 1:
            pieces = [torch.cat(pieces[i : i + 5]) if len(pieces[i : i + 5]) > 1 else pieces[i] for i in range(0, len(pieces), 5)]
        return pieces[0].view(2, len(echo.layers), echo.kv_heads, -1, pieces[0].shape[-1])


class Step(Prefill):
    """
    One main pass: codebooks 0-1's best code in every slot, its log-probability, and the log-probability of the code the
    slot holds now (meaningless for masked slots). The softmax counts the mask, but the mask is never a prediction.
    """

    def forward(self, tokens: torch.Tensor, cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        echo = self.echo
        h, _ = echo.run(echo.embed_audio(tokens), cache, None)
        log_probs = torch.log_softmax(echo.logits(h, echo.main_heads, echo.codes), dim=-1)
        held = log_probs.gather(-1, tokens[:2].t().unsqueeze(-1)).squeeze(-1)
        log_probs = log_probs[..., : echo.mask_id]
        return log_probs.amax(-1).t(), log_probs.argmax(-1).t(), held.t()


class Polish(Prefill):
    """
    Both polish passes, with codebooks 0-1 final and POLISH_BIAS added to the attention scores towards the reference
    frames: codebooks 2-3 from the first, then 4-7 from the second, guided by POLISH_GUIDANCE x (second - first). The
    log-softmax normalizers are constant per slot, so the guided argmax runs on the logits.
    """

    def forward(self, tokens: torch.Tensor, reference_frames: torch.Tensor, cache: torch.Tensor) -> torch.Tensor:
        echo = self.echo
        start, codes = cache.shape[3], echo.mask_id
        # 1 on the prefix's last `reference_frames` positions, 0 before them.
        positions = torch.arange(start, dtype=torch.float32)
        bias = (positions + reference_frames.float() - start + 1).clamp(0, 1) * POLISH_BIAS
        mask = torch.cat([bias, bias.new_zeros(tokens.shape[1])]).view(1, 1, -1)
        if echo.layers[0].fp16_attention:
            mask = mask.half()
        h, _ = echo.run(echo.embed_audio(tokens), cache, mask)
        middle = echo.logits(h, echo.middle_heads, codes).argmax(-1)
        first = echo.logits(h, echo.fine_heads, codes)
        tokens = torch.cat([tokens[:2], middle.t(), tokens[4:]])
        h, _ = echo.run(echo.embed_audio(tokens), cache, mask)
        second = echo.logits(h, echo.fine_heads, codes)
        fine = ((1 + POLISH_GUIDANCE) * second - POLISH_GUIDANCE * first).argmax(-1)
        return torch.cat([middle, fine], dim=1).t()


def export_methods(echo: Echo) -> dict:
    layers, kv_heads, head_dim = len(echo.layers), echo.kv_heads, echo.layers[0].head_dim
    prefix, frames = Dim("prefix", min=4, max=MAX_TEXT + MAX_REFERENCE), Dim("frames", min=4, max=MAX_FRAMES)
    text, reference = Dim("text", min=2, max=MAX_TEXT), Dim("reference", min=2, max=MAX_REFERENCE)
    dtype = torch.float16 if echo.layers[0].fp16_attention else torch.float32
    cache = torch.zeros(2, layers, kv_heads, 300, head_dim, dtype=dtype)
    tokens = torch.zeros(echo.books, 120, dtype=torch.long)

    with torch.no_grad():
        return {
            "prefill": export(
                Prefill(echo),
                (torch.zeros(60, dtype=torch.long), torch.zeros(echo.books, 240, dtype=torch.long)),
                dynamic_shapes=({0: text}, {1: reference}),
                strict=False,
            ),
            "step": export(Step(echo), (tokens, cache), dynamic_shapes=({1: frames}, {3: prefix}), strict=False),
            "polish": export(
                Polish(echo),
                (tokens, torch.tensor([240]), cache),
                dynamic_shapes=({1: frames}, None, {3: prefix}),
                strict=False,
            ),
        }


def schema(echo: Echo) -> str:
    """
    The get_model_schema JSON (react-native-executorch's ModelSpec), as the MLX and Vulkan programs carry it: every
    tensor's dtype and shape with the dynamic lengths' bounds, and the frame counts that must agree.
    """

    def length(low: int, high: int) -> dict:
        return {"kind": "range", "range": {"min": low, "max": high, "step": 1}}

    def tensor(dtype: str, *shape) -> dict:
        dims = [dim if isinstance(dim, dict) else {"kind": "constant", "value": dim} for dim in shape]
        return {"kind": "Tensor", "dtype": dtype, "shape": dims}

    def same_frames(outputs: int) -> list:
        dims = [{"paramSide": "input", "tensorIdx": 0, "dimIdx": 1}]
        dims += [{"paramSide": "output", "tensorIdx": i, "dimIdx": 1} for i in range(outputs)]
        return [{"kind": "equality", "dims": dims}]

    text, reference, frames = length(2, MAX_TEXT), length(2, MAX_REFERENCE), length(4, MAX_FRAMES)
    cache = tensor("float16", 2, len(echo.layers), echo.kv_heads, length(4, MAX_TEXT + MAX_REFERENCE), echo.layers[0].head_dim)
    tokens = tensor("int64", echo.books, frames)
    return json.dumps({
        "prefill": {
            "inputs": [tensor("int64", text), tensor("int64", echo.books, reference)],
            "outputs": [cache],
            "runtimeConstraints": [],
        },
        "step": {
            "inputs": [tokens, cache],
            "outputs": [tensor("float32", 2, frames), tensor("int64", 2, frames), tensor("float32", 2, frames)],
            "runtimeConstraints": same_frames(3),
        },
        "polish": {
            "inputs": [tokens, tensor("int64", 1), cache],
            "outputs": [tensor("int64", echo.books - 2, frames)],
            "runtimeConstraints": same_frames(1),
        },
    })


def merge_until_stable() -> None:
    """
    ExecuTorch's group partitioner merges partitions in one greedy pass: a pair that would form a cycle is never retried
    once the nodes between them have joined. A quantized tensor shared by several linears is one group spanning
    branches, and leaves a method in many delegates. Repeating the pass until nothing merges gives one per method.
    """

    from executorch.exir.backend.canonical_partitioners.group_partitioner import GroupBasedPartitioner

    merge = GroupBasedPartitioner._merge_partitions

    def repeated(self, partitions_by_id, *args) -> None:
        count = None
        while count != len(partitions_by_id):
            count = len(partitions_by_id)
            merge(self, partitions_by_id, *args)

    GroupBasedPartitioner._merge_partitions = repeated


def allow_dynamic_slices() -> None:
    """
    XNNPACK slices tensors of dynamic shape (xnn_define_static_slice_v3, also in react-native-executorch's runtime): an
    end at or below 0 counts from the end of its dim, so a dynamic dim kept whole goes out as offset 0, size 0.
    ExecuTorch's partitioner rejects every slice with a dynamic dim, and its serializer writes the traced sizes: both are
    patched for slices along a static dim that keep every dynamic dim whole.
    """

    import executorch.backends.xnnpack.operators  # noqa: F401 - registers the serializers
    from executorch.backends.xnnpack.operators.node_visitor import _node_visitor_dict
    from executorch.backends.xnnpack.partition.config.generic_node_configs import SliceCopyConfig
    from executorch.backends.xnnpack.utils.utils import get_input_node

    visitor = _node_visitor_dict["aten.slice_copy.Tensor"]
    check, define = SliceCopyConfig.check_constraints, visitor.define_node

    def check_constraints(self, node: torch.fx.Node, ep) -> bool:
        source, out = get_input_node(node, 0).meta["val"].shape, node.meta["val"].shape
        if all(isinstance(size, int) for size in out):
            return check(self, node, ep)
        dim = node.args[1] % len(source)
        if (node.args[4] if len(node.args) > 4 else 1) != 1 or not isinstance(out[dim], int) or not isinstance(source[dim], int):
            return False
        whole = all(i == dim or str(a) == str(b) for i, (a, b) in enumerate(zip(source, out)))
        return whole and self.check_common_constraints(node, ep)

    def define_node(self, node: torch.fx.Node, xnn_graph, *args) -> None:
        define(self, node, xnn_graph, *args)
        sizes = xnn_graph.xnodes[-1].xnode_union.sizes
        for i, size in enumerate(node.meta["val"].shape):
            if not isinstance(size, int):
                sizes[i] = 0

    SliceCopyConfig.check_constraints = check_constraints
    visitor.define_node = define_node


def transpose_in_matmul() -> None:
    """
    bmm(a, b transposed) goes out as XNNPACK's batch matmul with XNN_FLAG_TRANSPOSE_B, which reads b as [..., N, K]: the
    transpose is not serialized. ExecuTorch's runtime passes a node's flags on (also react-native-executorch's).
    """

    import executorch.backends.xnnpack.operators  # noqa: F401 - registers the serializers
    from executorch.backends.xnnpack.operators.node_visitor import _node_visitor_dict
    from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import XNNBatchMatrixMultiply, XNode
    from executorch.backends.xnnpack.utils.utils import get_input_node

    transposed_b = 1  # XNN_FLAG_TRANSPOSE_B
    matmul, transpose = _node_visitor_dict["aten.bmm.default"], _node_visitor_dict["aten.permute_copy.default"]
    define_transpose = transpose.define_node

    def is_matmul(node: torch.fx.Node) -> bool:
        return node.op == "call_function" and getattr(node.target, "__name__", "") == "aten.bmm.default"

    def folds(node: torch.fx.Node) -> bool:
        """
        Whether a permute swaps the last two of three dims and only feeds batch matmuls, as their second operand.
        """

        swaps = getattr(node.target, "__name__", "") == "aten.permute_copy.default" and list(node.args[1]) == [0, 2, 1]
        return swaps and all(is_matmul(user) and user.args[1] is node and user.args[0] is not node for user in node.users)

    def define_matmul(self, node: torch.fx.Node, xnn_graph, vals_to_ids, debug_handle: int) -> None:
        a, b = get_input_node(node, 0), get_input_node(node, 1)
        flags = 0
        if folds(b):
            b, flags = get_input_node(b, 0), transposed_b
        for tensor in (node, a, b):
            self.define_tensor(tensor, xnn_graph, vals_to_ids)
        union = XNNBatchMatrixMultiply(input1_id=vals_to_ids[a], input2_id=vals_to_ids[b], output_id=vals_to_ids[node], flags=flags)
        xnn_graph.xnodes.append(XNode(xnode_union=union, debug_handle=debug_handle))

    def define_permute(self, node: torch.fx.Node, xnn_graph, vals_to_ids, debug_handle: int) -> None:
        if not folds(node):
            define_transpose(self, node, xnn_graph, vals_to_ids, debug_handle)

    matmul.define_node = define_matmul
    transpose.define_node = define_permute


def lower(methods: dict, model_schema: str) -> bytes:
    merge_until_stable()
    allow_dynamic_slices()
    transpose_in_matmul()
    edge = to_edge_transform_and_lower(
        methods,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        constant_methods={"get_model_schema": model_schema},
    )
    for name in methods:
        kinds = collections.Counter(
            "delegate" if node.target is executorch_call_delegate else getattr(node.target, "__name__", str(node.target))
            for node in edge.exported_program(name).graph.nodes
            if node.op == "call_function" and node.target is not operator.getitem
        )
        print(f"{name}: " + ", ".join(f"{count} {kind}" for kind, count in kinds.most_common()), flush=True)
    # Inputs are not copied into planned memory: the cache goes back in on every pass.
    config = ExecutorchBackendConfig(
        extract_delegate_segments=True, memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False)
    )
    return edge.to_executorch(config).buffer


def build(checkpoint: str) -> bytes:
    """The program from a W8A8 checkpoint directory (config.json and model.safetensors)."""
    echo = Echo(checkpoint).eval()
    return lower(export_methods(echo), schema(echo))
