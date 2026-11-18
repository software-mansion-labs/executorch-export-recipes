"""Check Gemma ring-cache rewind/reset in eager and exported XNNPACK execution.

Uses deterministic small random weights, so no checkpoint is required.
Run in a Gemma export environment; --export also lowers and executes two PTEs.
"""
import argparse
from pathlib import Path
import sys

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--export', action='store_true')
args = parser.parse_args()
sys.path.insert(0, str(args.source_root / 'exporter_models/gemma4'))

from executorch.runtime import Runtime  # noqa: E402
import executorch.kernels.quantized  # noqa: F401,E402
import executorch.extension.llm.custom_ops.custom_ops  # noqa: F401,E402
from _gemma4_edge.model.text import Gemma4Attention, Gemma4EdgeConfig, Gemma4EdgeForCausalLM, RMSNorm  # noqa: E402
from _gemma4_edge.model.transforms.kv_quant import replace_kv_cache_with_quantized_kv_cache  # noqa: E402
from _gemma4_edge.model.transforms.kv_ring import replace_sliding_kv_cache_with_ring_kv_cache  # noqa: E402


def make_model(quantized, shared):
    torch.manual_seed(123)
    Gemma4Attention.use_bshd_kv = True
    Gemma4Attention.use_bool_mask = True
    Gemma4Attention.use_jit_mask = True
    Gemma4Attention.use_dynamic_kv_slice = True
    Gemma4Attention.use_shared_ring_state = shared
    Gemma4Attention.skip_redundant_rope_slice = True
    RMSNorm.use_rsqrt = True
    model = Gemma4EdgeForCausalLM(Gemma4EdgeConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=10, num_attention_heads=4, num_key_value_heads=1,
        head_dim=16, global_head_dim=32, ple_dim=8, ple_vocab_size=128,
        num_kv_shared_layers=5, sliding_window=16, max_position_embeddings=128,
    )).eval()
    model.model.split_ple_for_export()
    model.setup_caches(1, 128)
    for layer in model.model.layers:
        layer.self_attn.use_custom_sdpa = True
        layer.self_attn.use_broadcast_gqa = True
    if quantized:
        replace_kv_cache_with_quantized_kv_cache(model)
    replace_sliding_kv_cache_with_ring_kv_cache(model)
    return model


def eager_runner(model, multimodal):
    def execute(ids, pos):
        if multimodal:
            embeds, ple = model.embed_for_export(ids)
            return model.decode_from_embeds(embeds, ple, pos).clone()
        return model(ids, pos).clone()
    return execute


def check(make_runner, label):
    generator = torch.Generator().manual_seed(456)
    ids = torch.randint(1, 128, (1, 64), generator=generator)
    branch = torch.randint(1, 128, (1, 8), generator=generator)
    expected, actual = make_runner(), make_runner()
    # Cross a complete 32-slot ring, then rewind within retained history.
    for start, end in ((0, 12), (12, 24), (24, 36)):
        positions = torch.arange(start, end)
        torch.testing.assert_close(actual(ids[:, start:end], positions), expected(ids[:, start:end], positions))
    actual(ids[:, 36:41], torch.arange(36, 41))
    wanted = expected(branch, torch.arange(36, 44))
    got = actual(branch, torch.arange(36, 44))
    torch.testing.assert_close(got, wanted, atol=1e-5, rtol=1e-5, msg=label + ': rewind')
    # App reset changes only caller input_pos; it does not clear model buffers.
    fresh = make_runner()
    wanted = fresh(branch, torch.arange(8))
    got = actual(branch, torch.arange(8))
    torch.testing.assert_close(got, wanted, atol=1e-5, rtol=1e-5, msg=label + ': reset')
    # Append after reset to catch stale future slots becoming visible later.
    torch.testing.assert_close(
        actual(ids[:, 8:16], torch.arange(8, 16)),
        fresh(ids[:, 8:16], torch.arange(8, 16)),
        atol=1e-5, rtol=1e-5, msg=label + ': append after reset',
    )
    print('PASS:', label, flush=True)


def exported_factory(quantized):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.exir.passes.init_mutable_pass import InitializedMutableBufferPass
    from _gemma4_edge.export.patches import apply_patches
    from _gemma4_edge.export.pipeline import _INIT_MUTABLE_BUFFERS

    apply_patches(dynamic_kv_slice=True)
    model = make_model(quantized, True)
    ids = torch.ones((1, 8), dtype=torch.int64)
    seq = torch.export.Dim('seq', min=1, max=16)
    ep = torch.export.export(model, (ids, torch.arange(8)),
        dynamic_shapes={'input_ids': {1: seq}, 'input_pos': {0: seq}}, strict=False)
    model.reset_caches()
    edge = to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False))
    et = edge.to_executorch(ExecutorchBackendConfig(
        passes=[InitializedMutableBufferPass(_INIT_MUTABLE_BUFFERS)],
        do_quant_fusion_and_const_prop=True,
    ))
    plan = next(p for p in et.executorch_program.execution_plan if p.name == 'forward')
    assert plan.delegates, 'Expected XNNPACK delegation'
    blob = et.buffer
    def factory():
        program = Runtime.get().load_program(blob)
        method = program.load_method('forward')
        def execute(ids, pos):
            _ = program  # Keep the program and its backing storage alive.
            return method.execute([ids.contiguous(), pos])[0].clone()
        return execute
    return factory


def main():
    torch.set_num_threads(4)
    with torch.no_grad():
        for quantized in (False, True):
            for shared in (False, True):
                for multimodal in (False, True):
                    check(lambda: eager_runner(make_model(quantized, shared), multimodal),
                        f'eager int8={quantized} shared={shared} multimodal={multimodal}')
        if args.export:
            for quantized in (False, True):
                check(exported_factory(quantized), f'XNNPACK PTE int8={quantized} shared=True')


if __name__ == '__main__':
    main()
