# ROOT CAUSE (re-investigated 2026-08-20 -- this SUPERSEDES the earlier
# "CPU cast kernel" diagnosis, which was wrong).
#
# The fault is on the NON-CPU compute paths. Full matrix for all_mpnet_base_v2,
# dynamic Dim("tokens", 1..382), measured on a CLEARED Core ML compiled-model
# cache (~/Library/Caches/executorchcoreml -- a stale cache silently serves an
# older compilation and will fake a pass):
#
#     compute_unit     precision   result
#     CPU_ONLY         fp16        OK
#     CPU_ONLY         fp32        OK
#     CPU_AND_NE       fp16        SIGSEGV
#     CPU_AND_NE       fp32        OK  <- only because fp32 cannot run on the ANE,
#                                         so it silently falls back to the CPU
#     ALL              fp16        SIGSEGV
#     ALL              fp32        SIGSEGV
#
# So CPU_AND_NE is NOT a fix. Anything that actually reaches the ANE or the GPU
# crashes; only the CPU path is safe. A CPU_ONLY Core ML build has no accelerator
# advantage over the existing XNNPACK export, so shipping one buys nothing --
# which is why "coreml" stays out of this model's model.json backends.
#
# The trigger is the masked mean-pool over a dynamic sequence dim, bisected one
# wrapper piece at a time (encoder alone OK, + CLS/SEP prepend OK, + mean-pool
# SIGSEGV). Core ML infers the pooled output as data-dependent [?, H] even though
# the reduction removes the dynamic axis, and the non-CPU backends reject it:
#     Espresso exception: "Invalid blob shape": Data-dependent shapes were disabled
# Reproduces from plain coremltools with no ExecuTorch involved. Filed upstream as
# apple/coremltools#2825; see docs/coremltools-2825-repro.md.
#
# NOT universal: clip_vit_base_patch32_text runs correctly on CPU_AND_NE at both
# fp16 and fp32 and IS enabled. This failure is specific to the mpnet-style graph.
#
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
# _SOURCE = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# _CLS = 0
# _SEP = 2
#
#
# class _Embedder(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = AutoModel.from_pretrained(_SOURCE)
#
#     def _mean_pool(self, tokens, attention_mask):
#         mask = attention_mask.unsqueeze(-1).expand(tokens.size()).float()
#         return torch.sum(tokens * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
#
#     def forward(self, input_ids, attention_mask):
#         cls = torch.tensor([[_CLS]], dtype=input_ids.dtype)
#         sep = torch.tensor([[_SEP]], dtype=input_ids.dtype)
#         input_ids = torch.cat([cls, input_ids, sep], dim=1)
#         one = torch.tensor([[1]], dtype=attention_mask.dtype)
#         attention_mask = torch.cat([one, attention_mask, one], dim=1)
#         tokens = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
#         pooled = self._mean_pool(tokens, attention_mask)
#         return F.normalize(pooled, p=2, dim=1)
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
#     module = _Embedder().eval()
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
#         minimum_deployment_target=ct.target.iOS18,
#         compute_precision=precision,
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
