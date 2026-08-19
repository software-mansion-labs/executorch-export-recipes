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
# commented out. Verified 2026-08-19 on ExecuTorch 1.4.1 / coremltools 9.0.
#
# The blocker is the DYNAMIC sequence dim. The model exports with
# Dim("T", min=100, max=1000) because VAD runs on variable-length audio; the Core
# ML .pte builds fine but every execute() fails with ExecuTorch error 0x1 at both
# fp16 and fp32. Re-exported with a STATIC [1000, 512] input it runs and is
# numerically excellent: max abs error 2e-6 vs the float PyTorch reference. So the
# graph and the weights are fine -- only the dynamic shape is not.
#
# Two other things had to be solved first, and both worked; keep them if this is
# revisited:
#   1. coremltools 9.0 cannot convert aten.fft_rfft ("got 1 input(s), expected
#      [4]"), and skipping it just moves the failure to torch.complex64, which
#      Core ML has no dtype for.
#   2. Fixed by rewriting the front-end: fbank only ever uses |rfft(x)|**2, so the
#      complex FFT is replaced by a real DFT as two matmuls (Re = x @ cos_basis,
#      Im = -x @ sin_basis, |X|**2 = Re**2 + Im**2). That also drops the
#      abs()/pow(2) round-trip. The commented code below contains that rewrite.
#
# aten.as_strided_copy.default is unsupported by Core ML and falls back to CPU;
# that is harmless (lower_full_graph=False handles it).
#
# To enable: needs dynamic-shape support from the Core ML delegate. Shipping a
# fixed-length variant instead would change this model's API contract, so it is
# not done here. "coreml" is removed from model.json backends so batches skip it.

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
