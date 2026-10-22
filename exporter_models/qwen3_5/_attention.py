"""Gated DeltaNet state restoration behind the standard token/position interface."""

import logging
from contextlib import contextmanager

import torch
from executorch.examples.models.llama.attention import (
    ATTENTION_REGISTRY,
    AttentionGatedDeltaNet,
)
from torch.nn import functional as F

_LOG = logging.getLogger(__name__)


class RewindableGatedDeltaNet(AttentionGatedDeltaNet):
    def __init__(self, args, layer_id, rope, **kwargs):
        super().__init__(args, layer_id, rope, **kwargs)
        # Cache layer inputs rather than a full recurrent matrix at every token.
        self.register_buffer(
            "hidden_history",
            torch.zeros(
                args.max_context_len, args.max_batch_size, args.dim, device="cpu"
            ),
        )
        self.register_buffer(
            "state_pos", torch.zeros(1, dtype=torch.int64, device="cpu")
        )

    def _advance_state(self, x, conv_state, recurrent_state):
        if hasattr(self, "state_step"):
            return self.state_step(x, conv_state, recurrent_state)[:2]
        return _DeltaNetStep.forward(self, x, conv_state, recurrent_state)[:2]

    def _output(self, x, conv_state, recurrent_state):
        conv_state, recurrent_state, core = self.state_step(
            x, conv_state, recurrent_state
        )
        self.conv_state.copy_(conv_state)
        self.recurrent_state.copy_(recurrent_state)
        z = self.in_proj_z(x).reshape(-1, self.head_v_dim)
        core = self.norm(core.reshape(-1, self.head_v_dim), z)
        return self.out_proj(core.reshape(x.shape[0], 1, -1)), None

    def forward(self, x, freqs_cos, freqs_sin, **kwargs):
        position = kwargs["input_pos"]
        assert x.shape[1] == 1
        restore = (position != self.state_pos) & (position > 0)
        clear = restore | (position == 0)
        conv_state = torch.where(
            clear, torch.zeros_like(self.conv_state), self.conv_state
        )
        recurrent_state = torch.where(
            clear, torch.zeros_like(self.recurrent_state), self.recurrent_state
        )

        from torch._higher_order_ops.scan import scan

        def restore_state(conv_state, recurrent_state):
            length = position.clamp_min(1)[0].item()
            torch._check_is_size(length)
            torch._check(length > 0)
            torch._check(length <= self.hidden_history.shape[0])
            indices = torch.arange(length, device=position.device)

            def replay(carry, index):
                hidden = self.hidden_history.index_select(
                    0, index.reshape(1)
                ).transpose(0, 1)
                return self._advance_state(hidden, *carry), ()

            states, _ = scan(replay, (conv_state, recurrent_state), indices)
            return states

        def unchanged(conv_state, recurrent_state):
            return conv_state.clone(), recurrent_state.clone()

        conv_state, recurrent_state = torch.cond(
            restore.all(),
            restore_state,
            unchanged,
            (conv_state, recurrent_state),
        )
        self.conv_state.copy_(conv_state)
        self.recurrent_state.copy_(recurrent_state)
        if hasattr(self, "state_step"):
            output = self._output(x, conv_state, recurrent_state)
        else:
            output = super().forward(x, freqs_cos, freqs_sin, **kwargs)
        self.hidden_history.index_put_((position,), x.transpose(0, 1))
        self.state_pos.copy_(position + 1)
        return output


class _DeltaNetStep(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        for name in (
            "in_proj_qkv",
            "in_proj_a",
            "in_proj_b",
            "conv1d",
            "A_log",
            "dt_bias",
            "conv_kernel_size",
            "conv_dim",
            "key_dim",
            "value_dim",
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
            "head_repeat",
        ):
            setattr(self, name, getattr(layer, name))

    def forward(self, x, conv_state, recurrent_state):
        mixed = self.in_proj_qkv(x).transpose(1, 2)
        conv_input = torch.cat((conv_state, mixed), dim=-1)
        conv_state = conv_input[:, :, -self.conv_kernel_size :].contiguous()
        mixed = F.conv2d(
            conv_input.unsqueeze(-2),
            self.conv1d.weight.unsqueeze(-2),
            groups=self.conv_dim,
        ).squeeze(-2)[:, :, -1:]
        mixed = (mixed * mixed.sigmoid()).transpose(1, 2)
        query, key, value = mixed.split(
            (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        batch = x.shape[0]
        query = query.reshape(batch, self.num_k_heads, self.head_k_dim)
        query = query * torch.rsqrt(query.square().sum(dim=-1, keepdim=True) + 1e-6)
        query = query * (self.head_k_dim**-0.5)
        key = key.reshape(batch, self.num_k_heads, self.head_k_dim)
        key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
        value = value.reshape(batch, self.num_v_heads, self.head_v_dim)
        if self.head_repeat > 1:
            query = query.repeat_interleave(self.head_repeat, dim=1)
            key = key.repeat_interleave(self.head_repeat, dim=1)
        beta = self.in_proj_b(x).sigmoid().reshape(batch, self.num_v_heads, 1)
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias
        )
        recurrent_state = recurrent_state * g.exp().reshape(
            batch, self.num_v_heads, 1, 1
        )
        delta = (value - (recurrent_state * key.unsqueeze(-1)).sum(dim=-2)) * beta
        recurrent_state = recurrent_state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        core = (recurrent_state * query.unsqueeze(-1)).sum(dim=-2)
        return conv_state, recurrent_state, core


def _lower_projections(model, backend):
    model.requires_grad_(False)
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower

    if backend == "mlx":
        from executorch.backends.mlx import MLXPartitioner
        from executorch.backends.mlx.passes import get_default_passes

        partitioners = [MLXPartitioner()]
        passes = get_default_passes()
    else:
        from executorch.extension.llm.export.partitioner_lib import (
            get_xnnpack_partitioner,
        )

        partitioners = [get_xnnpack_partitioner(True), get_xnnpack_partitioner(False)]
        passes = []

    # Reuse compiled recurrence during replay: different kernel rounding can
    # change dynamic activation quantization downstream of a restored state.
    for layer in list(model.modules()):
        if not isinstance(layer, RewindableGatedDeltaNet):
            continue
        if backend == "xnnpack":
            step = _DeltaNetStep(layer)
            inputs = (
                torch.zeros(1, 1, layer.hidden_size),
                torch.zeros_like(layer.conv_state),
                torch.zeros_like(layer.recurrent_state),
            )
            with torch.no_grad():
                program = torch.export.export(step, inputs, strict=True)
                program = program.run_decompositions({})
            edge = to_edge_transform_and_lower(
                program,
                partitioner=partitioners,
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,
                ),
            )
            layer.state_step = edge.exported_program().module()
            _LOG.info("Lowered recurrent step for Qwen3.5 layer %s", layer.layer_id)
            continue
        for name in ("in_proj_qkv", "in_proj_a", "in_proj_b"):
            projection = torch.nn.Sequential()
            projection.add_module(
                f"layer_{layer.layer_id}_{name}", getattr(layer, name)
            )
            inputs = (torch.zeros(1, 1, layer.hidden_size),)
            with torch.no_grad():
                program = torch.export.export(projection, inputs, strict=True)
                program = program.run_decompositions({})
            edge = to_edge_transform_and_lower(
                program,
                partitioner=partitioners,
                transform_passes=passes,
                compile_config=EdgeCompileConfig(
                    _check_ir_validity=False,
                    _skip_dim_order=True,
                ),
            )
            setattr(layer, name, edge.exported_program().module())
        _LOG.info("Lowered recurrent projections for Qwen3.5 layer %s", layer.layer_id)
    return model


@contextmanager
def install(backend):
    from executorch.examples.models.llama import export_llama_lib
    from executorch.exir.emit._emitter import _Emitter
    from executorch.exir.schema import Bool, EValue, KernelCall, MoveCall, Tensor
    from executorch.extension.llm.export import builder
    from torchao.utils import unwrap_tensor_subclass

    previous = ATTENTION_REGISTRY["gated_deltanet"]
    source_transforms = export_llama_lib._get_source_transforms
    node_source = torch.fx.traceback.NodeSource.__init__
    backend_config = builder.ExecutorchBackendConfig
    emit_output = _Emitter.output

    def copy_branch_outputs(self, *args, **kwargs):
        start = len(self.chain.instructions)
        emit_output(self, *args, **kwargs)
        for instruction in self.chain.instructions[start:]:
            move = instruction.instr_args
            if not isinstance(move, MoveCall):
                continue
            if not isinstance(self.emitter_state.values[move.move_from].val, Tensor):
                continue
            # MoveCall aliases branch scratch storage, which later layers reuse.
            # Copy tensor results into their planned buffers across invocations.
            operator, _ = self._get_operator(name="aten::copy_")
            non_blocking = self._emit_evalue(EValue(Bool(False)))
            instruction.instr_args = KernelCall(
                operator,
                [move.move_to, move.move_from, non_blocking.id, move.move_to],
            )

    def runtime_config(*args, **kwargs):
        # The delegated linears already own their quantized weights. The optional
        # quant-fusion rewriter corrupts nested scan submodules in this ET version.
        kwargs["do_quant_fusion_and_const_prop"] = False
        return backend_config(*args, **kwargs)

    def shallow_source(self, node, *args, **kwargs):
        # PyTorch 2.14 recursively copies provenance across nested control flow,
        # exhausting the Python/C stack on the full decoder. Keep one generation.
        if node is not None:
            node.meta.pop("from_node", None)
        node_source(self, node, *args, **kwargs)

    def transforms(*args, **kwargs):
        return [
            *source_transforms(*args, **kwargs),
            unwrap_tensor_subclass,
            lambda model: _lower_projections(model, backend),
        ]

    ATTENTION_REGISTRY["gated_deltanet"] = RewindableGatedDeltaNet
    export_llama_lib._get_source_transforms = transforms
    torch.fx.traceback.NodeSource.__init__ = shallow_source
    builder.ExecutorchBackendConfig = runtime_config
    if backend == "xnnpack":
        _Emitter.output = copy_branch_outputs
    try:
        # Fake-tensor dispatch caching repeatedly deep-copies nested replay graphs.
        with torch._dynamo.config.patch(fake_tensor_cache_enabled=False):
            yield
    finally:
        ATTENTION_REGISTRY["gated_deltanet"] = previous
        export_llama_lib._get_source_transforms = source_transforms
        torch.fx.traceback.NodeSource.__init__ = node_source
        builder.ExecutorchBackendConfig = backend_config
        _Emitter.output = emit_output
