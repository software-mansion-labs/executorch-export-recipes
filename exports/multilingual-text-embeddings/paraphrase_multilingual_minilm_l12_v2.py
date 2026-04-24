"""
This script exports the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
model to Executorch format.

The model is wrapped in a custom PyTorch module that handles the postprocessing:
 - mean pooling (attention-mask weighted average)
 - L2 normalization

Although this model is a BERT architecture, it was initialized from XLM-R and uses
the XLM-R tokenizer convention: the sequence is wrapped with `<s>` (id=0) at the
start and `</s>` (id=2) at the end.

Max sequence length is set from `sentence_bert_config.json` (128), minus 2 to
account for the start/end tokens -> dynamic `tokens` dim max = 126.

Supported (backend, precision) combinations:

  xnnpack + fp32    baseline (~470 MB)
  xnnpack + 8da4w   Int8 dynamic activation + Int4 weight via torchao source
                    transform, group_size=32 (~397 MB; embeddings stay fp32)
  coreml  + fp32    baseline (~470 MB)
  coreml  + fp16    compute_precision=FLOAT16 at CoreML compile time (~235 MB)

Combinations that are NOT supported:

  xnnpack + fp16    `model.to(torch.float16)` produces NaN runtime output for
                    this class of transformer (softmax/LayerNorm overflow).
  coreml  + 8da4w   coremltools has no mapping for torch.int8 tensors emitted
                    by torchao's source transform. Use CoreML fp16 for size
                    savings on Apple platforms; CoreML-native quantization
                    (ct.optimize.coreml) would be a separate follow-up.

Embedding quantization was attempted (torchao `IntxWeightOnlyConfig` on
`nn.Embedding`, both 4w and 8w) but fails at `.to_executorch()`:

  RuntimeError: Missing out variants: {'torchao::dequantize_affine'}

ExecuTorch 1.2 doesn't register an out-variant for that op. The Llama path in
`examples/models/llama/source_transformation/quantize.py` works around this with
a bespoke `EmbeddingQuantHandler` that swaps `nn.Embedding` for a hand-rolled
quantized module emitting only out-variant-backed ops. Porting that pattern is
a follow-up — for now the scripts stop at `nn.Linear` quantization.

Usage:
    python paraphrase_multilingual_minilm_l12_v2.py \\
        --backend xnnpack --precision fp32
"""

SUPPORTED_COMBOS = {
    ("xnnpack", "fp32"),
    ("xnnpack", "8da4w"),
    ("coreml", "fp32"),
    ("coreml", "fp16"),
}

import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch.export import Dim, export
from transformers import AutoModel, AutoTokenizer

from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_BASENAME = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_TOKENS = 126  # 128 model max - 2 for <s>/</s>


class ParaphraseMultilingualMiniLmL12V2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(MODEL_ID)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        # XLM-R style start/end tokens
        start_tokens = torch.tensor([[0]], dtype=input_ids.dtype)
        end_tokens = torch.tensor([[2]], dtype=input_ids.dtype)
        input_ids = torch.cat([start_tokens, input_ids, end_tokens], dim=1)

        start_attn = torch.tensor([[1]], dtype=attention_mask.dtype)
        end_attn = torch.tensor([[1]], dtype=attention_mask.dtype)
        attention_mask = torch.cat([start_attn, attention_mask, end_attn], dim=1)

        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


def apply_quantization(model, precision):
    if precision != "8da4w":
        return
    from executorch.extension.llm.export.quantize import quantize_model_

    quantize_model_(model, qlinear_config="8da4w", qlinear_group_size=32)


def lower_xnnpack(exported_program):
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    return to_edge_transform_and_lower(
        exported_program, partitioner=[XnnpackPartitioner()]
    ).to_executorch()


def lower_coreml(exported_program, precision):
    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner

    compute_precision = (
        ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32
    )
    compile_specs = CoreMLBackend.generate_compile_specs(
        compute_precision=compute_precision,
        compute_unit=ct.ComputeUnit.ALL,
        model_type=CoreMLBackend.MODEL_TYPE.MODEL,
    )
    return to_edge_transform_and_lower(
        exported_program,
        partitioner=[CoreMLPartitioner(compile_specs=compile_specs)],
    ).to_executorch()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["xnnpack", "coreml"], default="xnnpack"
    )
    parser.add_argument(
        "--precision", choices=["fp32", "fp16", "8da4w"], default="fp32"
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if (args.backend, args.precision) not in SUPPORTED_COMBOS:
        raise SystemExit(
            f"Combination backend={args.backend}, precision={args.precision} is "
            f"not supported. See the module docstring for the working matrix."
        )

    torch.set_printoptions(precision=8)

    # Input Preparation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    sentences = ["Hola, mundo! Ciao, mondo! Hello, World!"]
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    print("Input IDs:", encoded["input_ids"].shape, encoded["input_ids"].dtype)
    print("Attn mask:", encoded["attention_mask"].shape, encoded["attention_mask"].dtype)

    # Dump tokenizer.json so the JS runtime can consume it via ResourceSource.
    # Named with the model slug so multiple models can coexist in one dir.
    tokenizer_path = os.path.join(
        args.output_dir.rstrip("/"), f"{MODEL_BASENAME}_tokenizer.json"
    )
    if not os.path.exists(tokenizer_path):
        tokenizer.backend_tokenizer.save(tokenizer_path)
        print(f"Wrote {tokenizer_path}")

    # Model Preparation
    model = ParaphraseMultilingualMiniLmL12V2()
    model.eval()

    apply_quantization(model, args.precision)

    inputs = (encoded["input_ids"], encoded["attention_mask"])
    with torch.no_grad():
        eager_out = model(*inputs)
    print("Eager embeddings:", eager_out.shape, eager_out.dtype)

    # Export
    dynamic_shapes = {
        "input_ids": {1: Dim("tokens", min=1, max=MAX_TOKENS)},
        "attention_mask": {1: Dim("tokens", min=1, max=MAX_TOKENS)},
    }
    exported_program = export(model, inputs, dynamic_shapes=dynamic_shapes)

    # Lower
    if args.backend == "xnnpack":
        executorch_program = lower_xnnpack(exported_program)
    else:
        executorch_program = lower_coreml(exported_program, args.precision)

    out_name = f"{MODEL_BASENAME}_{args.backend}_{args.precision}.pte"
    out_path = f"{args.output_dir.rstrip('/')}/{out_name}"
    with open(out_path, "wb") as f:
        f.write(executorch_program.buffer)
    print(f"Wrote {out_path} ({len(executorch_program.buffer) / 1e6:.2f} MB)")

    if args.skip_verify:
        return

    # Runtime check
    runtime = Runtime.get()
    program = runtime.load_program(out_path)
    method = program.load_method("forward")
    outputs = method.execute([encoded["input_ids"], encoded["attention_mask"]])
    print("Runtime out:", outputs[0].shape, outputs[0].dtype)

    # Precision thresholds differ by config:
    if args.precision == "fp32":
        atol, rtol = 1e-4, 1e-4
    elif args.precision == "fp16":  # CoreML fp16 only
        atol, rtol = 5e-3, 5e-3
    else:  # 8da4w
        atol, rtol = 5e-2, 5e-2
    ref = eager_out.to(outputs[0].dtype)
    close = torch.allclose(outputs[0], ref, atol=atol, rtol=rtol)
    mse = ((outputs[0].float() - ref.float()) ** 2).mean().sqrt().item()
    print(f"Runtime vs eager: close={close}, RMSE={mse:.6f} (atol={atol})")

    # Benchmark
    t0 = time.time()
    for _ in range(50):
        method.execute([encoded["input_ids"], encoded["attention_mask"]])
    print(f"50 iters: {time.time() - t0:.3f}s")


if __name__ == "__main__":
    main()
