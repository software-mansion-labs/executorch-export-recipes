"""Edge passes shared by the exporters (adapted from lfm2_5_vl's Vulkan exporter)."""

import logging

import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RemoveInversePermutePairs(ExportPass):
    """Collapse `permute_copy(permute_copy(x, p), p_inv)` back to `x`.

    `CustomKVCache.update` returns the cache transposed and `SDPACustom`
    transposes it straight back, so every decode step copies the whole cache
    twice per layer. On XNNPACK those transposes are ~40% of a decode step. On
    Vulkan the pair also hides `llama::update_cache` from the `causal_sdpa`
    matcher (it looks among the users of `custom_sdpa`'s key argument), and the
    export dies in `FusePatternsPass`; with the pair gone each attention layer
    fuses into one `llama::sdpa_with_kv_cache` and the cache lives on the GPU.
    """

    _PERMUTE = exir_ops.edge.aten.permute_copy.default

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        removed = 0
        for node in list(graph_module.graph.nodes):
            if node.op != "call_function" or node.target is not self._PERMUTE:
                continue
            inner = node.args[0]
            if (
                not isinstance(inner, torch.fx.Node)
                or inner.op != "call_function"
                or inner.target is not self._PERMUTE
            ):
                continue
            outer_dims, inner_dims = list(node.args[1]), list(inner.args[1])
            if len(outer_dims) != len(inner_dims):
                continue
            if any(inner_dims[d] != i for i, d in enumerate(outer_dims)):
                continue
            node.replace_all_uses_with(inner.args[0])
            graph_module.graph.erase_node(node)
            removed += 1

        if removed:
            logging.info("RemoveInversePermutePairs: removed %d permute pairs", removed)
            graph_module.graph.eliminate_dead_code()
            graph_module.recompile()
        return PassResult(graph_module, removed > 0)
