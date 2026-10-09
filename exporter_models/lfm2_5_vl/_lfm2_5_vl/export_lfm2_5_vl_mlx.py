# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Export LFM2.5-VL as a single multi-method PTE delegated to the MLX backend
(Apple GPU). MLX variant of export_lfm2_5_vl.py.

Differences from the XNNPACK export:
  - text decoder uses attention_type="mlx" (construct_transformer builds the
    MLX-optimized attention via et_attention, replacing the ET custom
    SDPA / KV-cache source transforms).
  - all three methods lower with MLXPartitioner instead of XnnpackPartitioner.

Methods (D = text hidden dim: 2048 for 1.6B, 1024 for 450M):
  vision_encoder  : [1, 3, 512, 512] f32 NCHW pixels [0,255] -> [1, 256, D] f32
  token_embedding : [1, seq_len] i64                          -> [1, seq_len, D] f32
  text_decoder    : ([1, seq_len, D] f32, [seq_len] i64)      -> [1, 65536] f32

Usage:
    python examples/models/lfm2_5_vl/export_lfm2_5_vl_mlx.py \
        --model_dir LiquidAI/LFM2.5-VL-450M \
        [--dtype bf16|fp16|fp32] [--output lfm2_5_vl_mlx.pte]
"""

import json
import logging
import os
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Optional

import torch

# Registers the "mlx" attention type used by construct_transformer.
import executorch.backends.mlx.llm.et_attention  # noqa: F401
from executorch.backends.mlx.partitioner import MLXPartitioner
from executorch.backends.mlx.passes import get_default_passes
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)

from .model import (
    Lfm2p5VlModel,
    MAX_SEQ_LEN,
    IMAGE_SIZE,
)

# Upper bound on a single text_decoder prefill chunk. The multimodal runner
# chunks decoder prefill at get_max_seq_len, so capping the decoder's dynamic
# seq dim (and the reported max_seq_len) here bounds per-step activation memory
# and avoids OOM on long prompts. The KV cache / context stays at MAX_SEQ_LEN,
# and token_embedding still runs once over the full fused sequence.
PREFILL_SEQ_CAP = 64
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from executorch.exir.passes.sym_shape_eval_pass import ConstraintBasedSymShapeEvalPass
from executorch.extension.llm.export.builder import DType, LLMEdgeManager
from torch.export import Dim
from torch.nn.attention import SDPBackend

FORMAT = "[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def _resolve_params_path(model_dir: str, params: Optional[str]) -> Optional[str]:
    if params is not None:
        return params
    name = model_dir.lower()
    if "450m" in name:
        return os.path.join(_CONFIG_DIR, "lfm2_5_vl_450m_config.json")
    if "1.6b" in name or "1_6b" in name:
        return os.path.join(_CONFIG_DIR, "lfm2_5_vl_1_6b_config.json")
    return None


class Lfm2p5VlEdgeManager(LLMEdgeManager):
    """strict=False + SDPBackend.MATH, like the XNNPACK export."""

    def export(self) -> "Lfm2p5VlEdgeManager":
        dynamic_shape = self._get_dynamic_shape()
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            self.export_program = torch.export.export(
                self.model,
                self.example_inputs,
                dynamic_shapes=dynamic_shape,
                strict=False,
            )
            self.pre_autograd_graph_module = self.export_program.module()
        return self


def export_image_encoder(
    lfm2, dtype: DType = DType.fp32
) -> torch.export.ExportedProgram:
    class ImageEncoder(torch.nn.Module):
        def __init__(self, lfm2):
            super().__init__()
            self.lfm2 = lfm2

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.lfm2.image_embedding(images)

    encoder = ImageEncoder(lfm2)
    example_pixels = torch.randint(
        0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32
    ).to(dtype.to_torch_dtype())
    logging.info(f"Exporting vision encoder ({dtype.value})...")
    with torch.no_grad():
        ep = torch.export.export(encoder, (example_pixels,), strict=False)
    return ep


def export_text_decoder(lfm2, dtype: DType = DType.fp32) -> torch.export.ExportedProgram:
    class TextDecoder(torch.nn.Module):
        def __init__(self, text_model):
            super().__init__()
            self.text_model = text_model

        def forward(
            self, embeddings: torch.Tensor, input_pos: torch.Tensor
        ) -> torch.Tensor:
            out = self.text_model(None, {"input_pos": input_pos}, embeddings)
            return out.to(torch.float32) if isinstance(out, torch.Tensor) else out

    decoder = TextDecoder(lfm2.text_model)
    dim = lfm2.text_model_args.dim
    dummy_seq = 8
    dummy_embeddings = torch.randn(1, dummy_seq, dim, dtype=dtype.to_torch_dtype())
    dummy_input_pos = torch.arange(dummy_seq, dtype=torch.int64)
    # Decoder prefill is chunked by the runner; cap the seq dim at the chunk
    # size so per-step activation memory stays bounded (KV cache stays at
    # MAX_SEQ_LEN via the model's attention).
    token_dim = Dim("token_dim", min=1, max=PREFILL_SEQ_CAP)
    dynamic_shapes = ({1: token_dim}, {0: token_dim})

    manager = Lfm2p5VlEdgeManager(
        model=decoder,
        modelname="lfm2_5_vl_text_decoder",
        max_seq_len=PREFILL_SEQ_CAP,
        dtype=dtype,
        use_kv_cache=True,
        example_inputs=(dummy_embeddings, dummy_input_pos),
        dynamic_shapes=dynamic_shapes,
    )

    # No ET custom-op source transforms: attention_type="mlx" already built the
    # MLX attention (with its own KV cache) inside construct_transformer.
    logging.info(f"Exporting text decoder ({dtype.value}, MLX attention)...")
    manager = manager.set_output_dir("./").to_dtype(dtype).export()
    return manager.export_program


def export_token_embedding(lfm2, max_seq_len: int = PREFILL_SEQ_CAP) -> torch.export.ExportedProgram:
    language_model = lfm2.model_.model.language_model
    token_dim = Dim("token_dim_1", min=1, max=max_seq_len)
    dynamic_shapes = [{1: token_dim}]
    logging.info("Exporting token embedding (fp32)...")
    embed_module = language_model.get_input_embeddings()
    example_ids = torch.zeros(1, max_seq_len, dtype=torch.int64)
    with torch.no_grad():
        ep = torch.export.export(
            embed_module, (example_ids,), dynamic_shapes=dynamic_shapes, strict=False
        )
    return ep


def export_all(
    model_dir: str,
    output: Optional[str],
    dtype: DType = DType.fp32,
    vision_dtype: DType = DType.fp32,
    vision_backend: str = "mlx",
    quantize: bool = False,
    max_seq_len: int = MAX_SEQ_LEN,
    max_context_len: int = MAX_SEQ_LEN,
    params_path: Optional[str] = None,
    _return_program: bool = False,
):
    logging.info(f"Loading {model_dir}...")
    lfm2_model = Lfm2p5VlModel(
        # MLX attention manages its own KV cache; skip the ET custom-op
        # replacement that use_sdpa_with_kv_cache_op would trigger.
        use_sdpa_with_kv_cache_op=False,
        model_dir=model_dir,
        max_seq_len=max_seq_len,
        max_context_len=max_context_len,
        params_path=params_path,
    )
    # Build the full_attention layers with the MLX-optimized attention.
    lfm2_model.text_model_args.attention_type = "mlx"

    lfm2 = lfm2_model.get_eager_model()
    if dtype != DType.fp32:
        lfm2 = lfm2.to(dtype.to_torch_dtype())
    # The vision tower mis-lowers in bf16 on the iOS MLX delegate (cosine ~0.40
    # vs fp32), so keep the vision tower + multimodal projector in fp32 even
    # when the decoder runs bf16. encode_images already casts feats to the
    # projector weight dtype, so the fp32 tower feeds cleanly into the decoder.
    if vision_dtype == DType.fp32 and dtype != DType.fp32:
        lfm2.model_.model.vision_tower.float()
        lfm2.model_.model.multi_modal_projector.float()

    if quantize:
        from executorch.backends.mlx.llm.quantization import quantize_model_

        # int4 weight-only (4w) for the decoder linears + token embedding, via
        # TorchAO — same path that fully delegates for the Llama MLX export.
        # Vision encoder weights stay unquantized (runs once per image; quant
        # gives little win) but still follow the bf16 compute dtype below.
        logging.info("Quantizing text decoder linears (4w)...")
        quantize_model_(lfm2.text_model, qlinear_config="4w", qlinear_group_size=32)
        logging.info("Quantizing token embedding (4w)...")
        quantize_model_(
            lfm2.model_.model.language_model,
            qembedding_config="4w",
            qembedding_group_size=32,
        )

    logging.info("[1/3] Exporting vision encoder...")
    vision_ep = export_image_encoder(lfm2, vision_dtype)
    logging.info("[2/3] Exporting text decoder...")
    decoder_ep = export_text_decoder(lfm2, dtype)
    logging.info("[3/3] Exporting token embedding...")
    token_ep = export_token_embedding(lfm2, max_seq_len=max_seq_len)

    logging.info("Lowering to Edge IR + MLX...")
    lowered = to_edge_transform_and_lower(
        {
            "vision_encoder": vision_ep,
            "token_embedding": token_ep,
            "text_decoder": decoder_ep,
        },
        transform_passes={
            "text_decoder": get_default_passes(),
        },
        partitioner={
            # vision_encoder can be delegated to XNNPACK (the device-validated
            # vision path) while the decoder stays on MLX for GPU decode speed.
            "vision_encoder": [
                XnnpackPartitioner()
                if vision_backend == "xnnpack"
                else MLXPartitioner()
            ],
            "token_embedding": [MLXPartitioner()],
            "text_decoder": [MLXPartitioner()],
        },
        constant_methods={
            # get_max_seq_len is the prefill chunk size the runner steps by
            # (capped to bound per-chunk activation memory); the KV cache /
            # history is get_max_context_len.
            "get_max_seq_len": PREFILL_SEQ_CAP,
            "get_max_context_len": lfm2.text_model_args.max_context_len,
            "get_n_layers": lfm2.text_model_args.n_layers,
            "get_vocab_size": lfm2.text_model_args.vocab_size,
            "use_kv_cache": lfm2.text_model_args.use_kv_cache,
            "use_sdpa_with_kv_cache": False,
            # Report True so the multimodal runner budgets the full prefill
            # against get_max_context_len (2048) and chunks by get_max_seq_len
            # (64), instead of capping total prefill at 64 and rejecting image
            # prompts (256+ tokens). The graph itself is still chunk-static
            # (token_dim max=PREFILL_SEQ_CAP); the model's actual
            # text_model_args.enable_dynamic_shape stays False to avoid the
            # .item() in rope.get_freqs that breaks export. Matches gemma4 MM.
            "enable_dynamic_shape": True,
            "get_eos_ids": [7],  # <|im_end|>
            "get_model_schema": json.dumps({
                "vision_encoder": {
                    "inputs": [
                        {
                            "kind": "Tensor",
                            "dtype": "float32",
                            "shape": [
                                {"kind": "constant", "value": 1},
                                {"kind": "constant", "value": 3},
                                {"kind": "constant", "value": IMAGE_SIZE},
                                {"kind": "constant", "value": IMAGE_SIZE},
                            ],
                        }
                    ],
                    "outputs": [
                        {
                            "kind": "Tensor",
                            "dtype": "float32",
                            "shape": [
                                {"kind": "constant", "value": 1},
                                {"kind": "constant", "value": 256},
                                {"kind": "constant", "value": lfm2.text_model_args.dim},
                            ],
                        }
                    ],
                    "runtimeConstraints": [],
                },
                "token_embedding": {
                    "inputs": [
                        {
                            "kind": "Tensor",
                            "dtype": "int64",
                            "shape": [
                                {"kind": "constant", "value": 1},
                                {
                                    "kind": "range",
                                    "range": {
                                        "min": 1,
                                        "max": PREFILL_SEQ_CAP - 1,
                                        "step": 1,
                                    },
                                },
                            ],
                        }
                    ],
                    "outputs": [
                        {
                            "kind": "Tensor",
                            "dtype": "bfloat16" if dtype == DType.bf16 else "float32",
                            "shape": [
                                {"kind": "constant", "value": 1},
                                {
                                    "kind": "range",
                                    "range": {
                                        "min": 1,
                                        "max": PREFILL_SEQ_CAP - 1,
                                        "step": 1,
                                    },
                                },
                                {"kind": "constant", "value": lfm2.text_model_args.dim},
                            ],
                        }
                    ],
                    "runtimeConstraints": [],
                },
                "text_decoder": {
                    "inputs": [
                        {
                            "kind": "Tensor",
                            "dtype": "bfloat16" if dtype == DType.bf16 else "float32",
                            "shape": [
                                {"kind": "constant", "value": 1},
                                {
                                    "kind": "range",
                                    "range": {
                                        "min": 1,
                                        "max": PREFILL_SEQ_CAP - 1,
                                        "step": 1,
                                    },
                                },
                                {"kind": "constant", "value": lfm2.text_model_args.dim},
                            ],
                        },
                        {
                            "kind": "Tensor",
                            "dtype": "int64",
                            "shape": [
                                {
                                    "kind": "range",
                                    "range": {
                                        "min": 1,
                                        "max": PREFILL_SEQ_CAP - 1,
                                        "step": 1,
                                    },
                                }
                            ],
                        },
                    ],
                    "outputs": [
                        {
                            "kind": "Tensor",
                            "dtype": "float32",
                            "shape": [
                                {"kind": "constant", "value": 1},
                                {"kind": "constant", "value": lfm2.text_model_args.vocab_size},
                            ],
                        }
                    ],
                    "runtimeConstraints": [],
                },
            }),
        },
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    logging.info("Finalizing ExecuTorch program...")
    et_program = lowered.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=False,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
            sym_shape_eval_pass={
                "vision_encoder": ConstraintBasedSymShapeEvalPass(),
                "token_embedding": ConstraintBasedSymShapeEvalPass(),
                "text_decoder": ConstraintBasedSymShapeEvalPass(),
            },
        )
    )

    if _return_program:
        return et_program

    logging.info(f"Saving {output}...")
    with open(output, "wb") as f:  # type: ignore[arg-type]
        et_program.write_to_file(f)
    logging.info(f"Saved {output}. Methods: {et_program.methods}")


def main():
    parser = ArgumentParser(description="Export LFM2.5-VL to ExecuTorch (MLX)")
    parser.add_argument("--model_dir", default="LiquidAI/LFM2.5-VL-450M")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--vision_dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="compute dtype for the vision tower + projector (default fp32; "
        "bf16 mis-lowers on the iOS MLX delegate)",
    )
    parser.add_argument(
        "--vision_backend",
        default="mlx",
        choices=["mlx", "xnnpack"],
        help="delegate the vision_encoder method to this backend; decoder + "
        "token_embedding always stay MLX",
    )
    parser.add_argument(
        "--quantize",
        default=False,
        action=BooleanOptionalAction,
        help="int4 (4w) quantize decoder linears + token embedding",
    )
    parser.add_argument("--params", default=None)
    parser.add_argument("--output", default="lfm2_5_vl_mlx.pte")
    args = parser.parse_args()

    export_all(
        model_dir=args.model_dir,
        output=args.output,
        dtype=DType[args.dtype],
        vision_dtype=DType[args.vision_dtype],
        vision_backend=args.vision_backend,
        quantize=args.quantize,
        params_path=_resolve_params_path(args.model_dir, args.params),
    )


if __name__ == "__main__":
    main()
