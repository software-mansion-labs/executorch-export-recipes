"""Check packed CPU embeddings through the Gemma omni Vulkan export recipe.

Uses a small, deterministic Gemma model without a checkpoint. Inspects the
serialized operators and compares runtime outputs with the quantized eager
reference at dynamic sequence boundaries. This does not validate full-model
quality or device memory. Run in a Gemma Vulkan export environment.
"""
import argparse
import importlib
from pathlib import Path
import sys
import tempfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import order registers the portable quantized runtime kernels.
from executorch.runtime import Runtime  # noqa: E402
import executorch.kernels.quantized  # noqa: F401,E402
from executorch.exir._serialize._program import deserialize_pte_binary  # noqa: E402
from exporter_models.gemma4_omni.vulkan import export as omni  # noqa: E402


def check(out_path):
    omni._load_edge()
    package = omni._EDGE_PKG
    pipeline = importlib.import_module(f"{package}.export.pipeline")
    wrappers = importlib.import_module(f"{package}.export.wrappers")
    config = importlib.import_module(f"{package}.export.config")
    quant = importlib.import_module(f"{package}.export.quant")
    text = importlib.import_module(f"{package}.model.text")
    omni._adapt_multimodal(pipeline, wrappers)

    torch.manual_seed(1613)
    model = text.Gemma4EdgeForCausalLM(text.Gemma4EdgeConfig(
        vocab_size=128, ple_vocab_size=128, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1,
        head_dim=16, global_head_dim=16, intermediate_size=128,
        ple_dim=32, num_kv_shared_layers=0, tie_word_embeddings=False,
    )).to(omni.BASE_DTYPE).eval()
    model.model.split_ple_for_export()
    model = quant.quantize_text_model(
        model,
        linear_group_size=omni.LINEAR_GROUP_SIZE,
        quantize_main_embeddings=omni.QUANTIZE_MAIN_EMBEDDINGS,
        embedding_group_size=omni.EMBEDDING_GROUP_SIZE,
        quantize_ple_embeddings=omni.QUANTIZE_PLE_EMBEDDINGS,
        ple_embedding_bits=omni.PLE_EMBEDDING_BITS,
        ple_embedding_group_size=omni.PLE_EMBEDDING_GROUP_SIZE,
    )
    quant.retarget_quantized_to_fp16_with_fp32_scales(model)
    wrapper = wrappers.TokenEmbeddingExport(model).eval()
    seq = torch.export.Dim("seq_len", min=1, max=omni.PREFILL_LEN)
    example = torch.tensor([[0, 1, 3, 127, 4, 1, 64, 7]])
    program = torch.export.export(
        wrapper, (example,), dynamic_shapes={"input_ids": {1: seq}}, strict=True,
    )
    recipe = config.Recipe(
        name="cpu-embedding-smoke", backend="vulkan", modality="multimodal",
    )
    partitioners = pipeline._build_partitioners(recipe)
    pipeline._lower_and_write(
        {"token_embedding": program},
        {"token_embedding": partitioners["token_embedding"]},
        recipe, out_path,
    )

    # Inspect the actual PTE, after quant fusion and constant propagation.
    serialized = deserialize_pte_binary(out_path.read_bytes()).program
    plan = next(p for p in serialized.execution_plan if p.name == "token_embedding")
    operators = [(op.name, op.overload) for op in plan.operators]
    assert not plan.delegates, "Embedding tables must not enter a delegate"
    assert ("quantized_decomposed::embedding_4bit", "dtype_out") in operators
    assert not any("dequantize_affine" in name for name, _ in operators)
    print("Serialized CPU operators:", operators)

    method = Runtime.get().load_program(str(out_path)).load_method("token_embedding")
    assert method is not None
    lengths = (1, 8, omni.PREFILL_LEN, 3, 1)
    for length in lengths:
        ids = torch.arange(length).remainder(128).reshape(1, length)
        with torch.no_grad():
            reference = wrapper(ids)
        actual = method.execute([ids])
        assert len(actual) == 2
        for expected, result in zip(reference, actual):
            assert result.dtype == torch.float32 and result.shape == expected.shape
            assert torch.isfinite(result).all()
            torch.testing.assert_close(result, expected, atol=0, rtol=0)
    print(f"PASS: packed CPU embeddings; exact eager parity at lengths {lengths}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Keep the small generated PTE here")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="gemma-cpu-embedding-") as temporary:
        check(args.output or Path(temporary) / "token_embedding.pte")
