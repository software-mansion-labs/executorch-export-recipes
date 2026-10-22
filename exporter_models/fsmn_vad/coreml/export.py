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
# fsmn_vad SPECIFICS: this model does not segfault -- it fails with error 0x1 at
# execute under CPU_AND_NE in both fp16 and fp32 (re-tested 2026-08-20 on a clear
# cache). That is a different fault from the mpnet segfault above and is NOT fixed
# by changing compute_unit, so Core ML stays disabled here too.
#
# import importlib.util
# import sys
# from pathlib import Path
#
# import torch
#
# _HERE = Path(__file__).resolve().parent
# _MODEL_DIR = _HERE.parent
#
#
# def _load_vad_model():
#     spec = importlib.util.spec_from_file_location("fsmn_vad_coreml_model", _MODEL_DIR / "_vad_model.py")
#     mod = importlib.util.module_from_spec(spec)
#     sys.modules["fsmn_vad_coreml_model"] = mod
#     spec.loader.exec_module(mod)
#     return mod
#
#
# def export(config: dict, out_dir: str) -> str:
#     import coremltools as ct
#     from executorch.backends.apple.coreml.compiler import CoreMLBackend
#     from executorch.backends.apple.coreml.partition import CoreMLPartitioner
#     from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
#     from huggingface_hub import hf_hub_download
#     from torch.export import Dim, export as torch_export
#
#     vm = _load_vad_model()
#     fwd = config["variant"]["methods"]["forward"]
#     feat_shape = tuple(fwd["inputs"][0]["shape"])  # [1000, 512]
#
#     model = vm.VAD(
#         input_dim=400,
#         input_affine_dim=140,
#         fsmn_layers=4,
#         linear_dim=250,
#         proj_dim=128,
#         lorder=20,
#         rorder=0,
#         lstride=1,
#         rstride=1,
#         output_affine_dim=140,
#         output_dim=248,
#         use_softmax=True,
#     )
#
#     ckpt = hf_hub_download("funasr/fsmn-vad", "model.pt")
#     original_state_dict = torch.load(ckpt, map_location="cpu")
#
#     prefix = "encoder."
#     corrected_state_dict = {
#         k[len(prefix):]: v
#         for k, v in original_state_dict.items()
#         if k.startswith(prefix)
#     }
#
#     model.load_state_dict(corrected_state_dict)
#     model.eval()
#
#     # Core ML has no complex dtype, and coremltools 9.0 cannot convert
#     # aten.fft_rfft at all. The front-end only ever uses |rfft(x)|**2, so swap the
#     # complex FFT for an equivalent real-valued DFT written as two matmuls:
#     #   Re = x @ cos_basis, Im = -x @ sin_basis, |X|**2 = Re**2 + Im**2
#     # This drops the sqrt/square round-trip as well, since .abs() is immediately
#     # squared upstream. Mathematically identical, and every op lowers to Core ML.
#     import math
#
#     frame_len = dummy_frame_len = 512
#     n_bins = frame_len // 2 + 1
#     n = torch.arange(frame_len, dtype=torch.float64).unsqueeze(1)
#     k = torch.arange(n_bins, dtype=torch.float64).unsqueeze(0)
#     angle = 2.0 * math.pi * n * k / frame_len
#     cos_basis = angle.cos().to(torch.float32)
#     sin_basis = angle.sin().to(torch.float32)
#
#     def _fbank_dft(strided_input, mel_energies):
#         re = strided_input @ cos_basis
#         im = -(strided_input @ sin_basis)
#         power = re * re + im * im
#         energies = torch.mm(power, mel_energies.T)
#         eps = torch.tensor(torch.finfo(torch.float).eps, dtype=strided_input.dtype)
#         return torch.max(energies, eps).log()
#
#     vm.fbank = _fbank_dft
#
#     dummy = (torch.randn(*feat_shape, dtype=torch.float32),)
#     dynamic_shapes = ({0: Dim("T", min=100, max=1000)},)
#
#     from exporter.schema import load_and_validate_schema
#
#     schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")
#
#     with torch.no_grad():
#         ep = torch_export(model, dummy, dynamic_shapes=dynamic_shapes)
#         methods = {"forward": ep}
#         precision = {"fp16": ct.precision.FLOAT16, "fp32": ct.precision.FLOAT32}[
#             config["variant"]["precision"]
#         ]
#         specs = CoreMLBackend.generate_compile_specs(
#             minimum_deployment_target=ct.target.iOS18, compute_precision=precision
#         )
#         et_program = to_edge_transform_and_lower(
#             methods,
#             partitioner=[CoreMLPartitioner(compile_specs=specs, lower_full_graph=False)],
#             compile_config=EdgeCompileConfig(_check_ir_validity=False),
#             constant_methods={"get_model_schema": schema_json},
#         ).to_executorch()
#
#     out_path = Path(out_dir) / config["variant"]["file"]
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_bytes(et_program.buffer)
#     return str(out_path)
