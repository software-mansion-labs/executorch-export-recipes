# ROOT CAUSE (investigated 2026-08-19, shared by every Core ML model disabled in
# this repo except rfdetr_keypoint): Core ML's CPU cast kernel segfaults on
# dynamically-shaped tensors. macOS crash report from the failing execute():
#
#     libvDSP.dylib   vDSP_vflt32                     <- int32 -> float32 convert
#     Espresso        Espresso::cast_kernel_cpu::__launch(...)
#     Espresso        Espresso::net::__forward(...)
#     CoreML          -[MLNeuralNetworkEngine executePlan:error:]
#
# The decisive test is same model, same code, shape spec the only difference:
#     all_mpnet_base_v2            dynamic Dim -> SIGSEGV, static -> OK (1, 768)
#     clip_vit_base_patch32_text   dynamic Dim -> SIGSEGV, static -> OK (1, 512)
#     fsmn_vad                     dynamic Dim -> error 0x1, static -> OK (2e-6)
#
# It is NOT a length/buffer mismatch: dynamic mpnet crashes at EVERY length
# (1, 24, 200, 380 and the max 382), so the dynamic path itself is broken, not a
# size disagreement between the compiled plan and the actual input.
#
# It is also NOT "Core ML cannot do dynamic shapes". all_minilm_l6_v2 and
# multi_qa_minilm_l6_cos_v1 use the same Dim("tokens", ...) pattern and run
# correctly (cos 0.99998 / 0.99997). So it is dynamic shapes PLUS something these
# graphs emit that BERT does not -- mpnet's relative-position-bucket arithmetic and
# CLIP's causal-mask/argmax indexing both build integer tensors sized by the
# symbolic dim, which is what that cast is converting. The exact offending node has
# NOT been isolated; treat that last step as a hypothesis, not a finding.
#
# WORKAROUND, not applied here: export with static shapes. That costs the
# variable-length input contract (the runtime would have to pad to a fixed length),
# so it is a product decision rather than an exporter one.
#
# DISABLED: Core ML is not supported for this model, and this whole exporter is
# commented out. The export succeeds and produces a .pte, but the first forward
# SIGSEGVs (exit 139) at BOTH fp16 and fp32. Verified 2026-08-19 on ExecuTorch
# 1.4.1 / coremltools 9.0.
#
# Same crash signature as the two mpnet sentence transformers
# (all_mpnet_base_v2, multi_qa_mpnet_base_dot_v1). It is NOT simply "text encoder
# with a dynamic sequence Dim" -- the structurally similar BERT-based encoders
# export and run correctly on Core ML and match the float reference:
#     all_minilm_l6_v2                       cos 0.99998
#     multi_qa_minilm_l6_cos_v1              cos 0.99997
#     paraphrase_multilingual_minilm_l12_v2  cos 0.999998   (already shipped)
# The CLIP *vision* tower is also fine (cos 0.999998 fp16 / 1.000000 fp32), so
# only this text tower is affected.
#
# "coreml" is removed from this model's model.json backends so batch exports skip
# it. The code below is preserved verbatim, commented, so the finding can be
# re-tested against a future coremltools / ExecuTorch: uncomment it and re-add
# "coreml" to model.json.

# from pathlib import Path
#
# import torch
# import torch.nn.functional as F
# from transformers import AutoModel
# from torch.export import Dim, export as torch_export
#
# from exporter.schema import load_and_validate_schema
#
# _HERE = Path(__file__).resolve().parent
# _MODEL_DIR = _HERE.parent
#
# _SOURCE = "openai/clip-vit-base-patch32"
# _BOS = 49406
# _EOS = 49407
#
#
# class _ClipTextEncoder(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         model = AutoModel.from_pretrained(_SOURCE)
#         self.text_model = model.text_model
#         self.text_projection = model.text_projection
#
#     def forward(self, input_ids, attention_mask):
#         bos = torch.tensor([[_BOS]], dtype=input_ids.dtype)
#         eos = torch.tensor([[_EOS]], dtype=input_ids.dtype)
#         input_ids = torch.cat([bos, input_ids, eos], dim=1)
#         one = torch.tensor([[1]], dtype=attention_mask.dtype)
#         attention_mask = torch.cat([one, attention_mask, one], dim=1)
#
#         pooled = self.text_model(input_ids, attention_mask).pooler_output
#         projected = self.text_projection(pooled)
#         return F.normalize(projected, p=2, dim=1)
#
#
# def export(config: dict, out_dir: str) -> str:
#     import coremltools as ct
#     from executorch.backends.apple.coreml.compiler import CoreMLBackend
#     from executorch.backends.apple.coreml.partition import CoreMLPartitioner
#     from executorch.exir import to_edge_transform_and_lower
#
#     variant = config["variant"]
#     max_tokens = variant["methods"]["forward"]["inputs"][0]["shape"][-1]
#     precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
#         variant["precision"]
#     ]
#
#     module = _ClipTextEncoder().eval()
#     inputs = (
#         torch.zeros(1, max_tokens, dtype=torch.int64),
#         torch.ones(1, max_tokens, dtype=torch.int64),
#     )
#     dim = Dim("tokens", min=1, max=max_tokens)
#     dynamic = {"input_ids": {1: dim}, "attention_mask": {1: dim}}
#
#     ep = torch_export(module, inputs, dynamic_shapes=dynamic)
#     schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")
#
#     specs = CoreMLBackend.generate_compile_specs(
#         minimum_deployment_target=ct.target.iOS18, compute_precision=precision
#     )
#     program = to_edge_transform_and_lower(
#         {"forward": ep},
#         partitioner=[CoreMLPartitioner(compile_specs=specs)],
#         constant_methods={"get_model_schema": schema_json},
#     ).to_executorch()
#
#     out_path = Path(out_dir) / variant["file"]
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_bytes(program.buffer)
#     return str(out_path)
