"""
This script exports the `sentence-transformers/distiluse-base-multilingual-cased-v2`
model to Executorch format.

Unlike the plain DistilBERT encoder, this sentence-transformers model appends a
Dense projection (768 -> 512 with Tanh) on top of mean-pooled embeddings. The
Dense weights ship in the repo at `2_Dense/model.safetensors` and are loaded here
so the exported .pte reproduces the full sentence-transformers pipeline.

Postprocessing order:
 - mean pooling (attention-mask weighted average)
 - Dense(768 -> 512) + Tanh
 - L2 normalization

Tokenization uses the BERT convention: `[CLS]` (id=101) at the start and `[SEP]`
(id=102) at the end. Max sequence length is 128 (from `sentence_bert_config.json`),
minus 2 for CLS/SEP -> dynamic `tokens` dim max = 126.

Supported (backend, precision) combinations:

  xnnpack + fp32    baseline (~540 MB)
  xnnpack + 8da4w   Int8 dynamic activation + Int4 weight via torchao source
                    transform, group_size=32 (~393 MB; embeddings stay fp32)
  coreml  + fp32    baseline (~540 MB)
  coreml  + fp16    compute_precision=FLOAT16 at CoreML compile time (~270 MB)

Combinations that are NOT supported (see paraphrase script docstring for
the same analysis):

  xnnpack + fp16    numerical overflow in softmax/LayerNorm -> NaN at runtime.
  coreml  + 8da4w   coremltools cannot consume torch.int8 tensors emitted by
                    torchao's source transform.

Embedding quantization was attempted (see paraphrase script docstring for the
full analysis) but fails at `.to_executorch()` with
`Missing out variants: {'torchao::dequantize_affine'}`. A follow-up could port
the Llama-style `EmbeddingQuantHandler` from
`examples/models/llama/source_transformation/quantize.py` to avoid that op.

Usage:
    python distiluse_base_multilingual_cased_v2.py \\
        --backend coreml --precision fp16
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
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.export import Dim, export
from transformers import AutoModel, AutoTokenizer

from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


# PyTorch nightly bug: `torch.utils._pytree.LeafSpec` declares `type/_context/
# _children` fields with `init=False` defaults but the frozen+slots dataclass
# never assigns them — `leaf.type` / `leaf._context` / `leaf._children` raise
# AttributeError inside torch.export's decomposition and deepcopy paths. Patch
# by wrapping `__new__` to preset the slots on every fresh instance, and patch
# `_dataclass_getstate`'s attribute access path so deserialized instances don't
# explode on missing slots either.
def _patch_leafspec_missing_fields():
    from torch.utils._pytree import LeafSpec

    original_new = LeafSpec.__new__

    def patched_new(cls, *args, **kwargs):
        inst = original_new(cls)
        for name, default in (
            ("type", None),
            ("_context", None),
            ("_children", []),
        ):
            try:
                object.__setattr__(inst, name, default)
            except AttributeError:
                pass
        return inst

    LeafSpec.__new__ = staticmethod(patched_new)

    # Backstop: any lingering instances (e.g. produced by deepcopy reconstruct
    # via __reduce__) still see AttributeError on `.type`. Add __getattr__.
    def __getattr__(self, name):
        if name in ("type", "_context"):
            return None
        if name == "_children":
            return []
        raise AttributeError(name)

    LeafSpec.__getattr__ = __getattr__


_patch_leafspec_missing_fields()


MODEL_ID = "sentence-transformers/distiluse-base-multilingual-cased-v2"
MODEL_BASENAME = "distiluse-base-multilingual-cased-v2"
MAX_TOKENS = 126  # 128 model max - 2 for [CLS]/[SEP]
DENSE_IN = 768
DENSE_OUT = 512


def load_dense_layer():
    path = hf_hub_download(MODEL_ID, "2_Dense/model.safetensors")
    state = load_file(path)
    dense = torch.nn.Linear(DENSE_IN, DENSE_OUT, bias=True)
    dense.weight.data.copy_(state["linear.weight"])
    dense.bias.data.copy_(state["linear.bias"])
    return dense


class DistiluseBaseMultilingualCasedV2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(MODEL_ID)
        self.dense = load_dense_layer()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(self, input_ids, attention_mask):
        # BERT-style start/end tokens
        start_tokens = torch.tensor([[101]], dtype=input_ids.dtype)
        end_tokens = torch.tensor([[102]], dtype=input_ids.dtype)
        input_ids = torch.cat([start_tokens, input_ids, end_tokens], dim=1)

        start_attn = torch.tensor([[1]], dtype=attention_mask.dtype)
        end_attn = torch.tensor([[1]], dtype=attention_mask.dtype)
        attention_mask = torch.cat([start_attn, attention_mask, end_attn], dim=1)

        # Pass attention_mask=None to skip HF's mask-to-4D conversion
        # (eq/any/logical_not/where/expand_copy), which XNNPACK can't delegate.
        # Safe at inference: RNE never pads, so the mask is always all-ones.
        model_output = self.model(input_ids=input_ids, attention_mask=None)
        pooled = self.mean_pooling(model_output, attention_mask)
        projected = torch.tanh(self.dense(pooled))
        # Return a 1-tuple so `out_spec.type` resolves to `tuple` for the
        # torch.export decomposition pass (a plain LeafSpec output hits a C++
        # binding that doesn't expose `.type` in PyTorch nightly).
        return (F.normalize(projected, p=2, dim=1),)


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
    model = DistiluseBaseMultilingualCasedV2()
    model.eval()

    apply_quantization(model, args.precision)

    inputs = (encoded["input_ids"], encoded["attention_mask"])
    with torch.no_grad():
        eager_out = model(*inputs)[0]
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
