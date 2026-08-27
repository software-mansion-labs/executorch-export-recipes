# MLX: does not lower -- the MLX partitioner crashes on in-place clamp.
#
# Measured 2026-08-20 with the software-mansion-labs fork:
#
#     handler failed for <EdgeOpOverload: aten.clamp_.default>:
#         float() argument must be a string or a real number, not 'Slot'
#     TypeError: float() argument must be a string or a real number, not 'Slot'
#
# The export emits a .pte but it fails to LOAD at runtime. This is a defect in the
# MLX backend's in-place-passthrough handler, NOT the missing-compile-config issue
# that blocked efficientnet_v2_s / ssdlite320 / rfdetr_nano_* (those were fixed by
# adding EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True) and now
# export, execute and ship). Adding that config here does not help, because the
# failure is inside the partitioner's op handler rather than in IR validation.
#
# Core ML is separately disabled for this model (see coreml/export.py), so fsmn_vad
# has no working Apple accelerator backend today. "mlx" is removed from model.json.

import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent


def _load_vad_model():
    spec = importlib.util.spec_from_file_location("fsmn_vad_mlx_model", _MODEL_DIR / "_vad_model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fsmn_vad_mlx_model"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> str:
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
    from huggingface_hub import hf_hub_download
    from torch.export import Dim, export as torch_export

    vm = _load_vad_model()
    fwd = config["variant"]["methods"]["forward"]
    feat_shape = tuple(fwd["inputs"][0]["shape"])  # [1000, 512]

    model = vm.VAD(
        input_dim=400,
        input_affine_dim=140,
        fsmn_layers=4,
        linear_dim=250,
        proj_dim=128,
        lorder=20,
        rorder=0,
        lstride=1,
        rstride=1,
        output_affine_dim=140,
        output_dim=248,
        use_softmax=True,
    )

    ckpt = hf_hub_download("funasr/fsmn-vad", "model.pt")
    original_state_dict = torch.load(ckpt, map_location="cpu")

    prefix = "encoder."
    corrected_state_dict = {
        k[len(prefix):]: v
        for k, v in original_state_dict.items()
        if k.startswith(prefix)
    }

    model.load_state_dict(corrected_state_dict)
    model.eval()

    # aten._fft_r2c has no MLX handler AND no portable CPU kernel, so the .pte
    # fails to load with ExecuTorch 0x14 (OperatorMissing). The front-end only
    # ever uses |rfft(x)|**2, so swap the complex FFT for an equivalent real DFT
    # written as two matmuls:
    #   Re = x @ cos_basis, Im = -x @ sin_basis, |X|**2 = Re**2 + Im**2
    # That also drops the abs()/pow(2) round-trip. Mathematically identical; the
    # same rewrite measured 2e-6 max abs error against the float reference.
    import math

    frame_len = 512
    n_bins = frame_len // 2 + 1
    n = torch.arange(frame_len, dtype=torch.float64).unsqueeze(1)
    k = torch.arange(n_bins, dtype=torch.float64).unsqueeze(0)
    angle = 2.0 * math.pi * n * k / frame_len
    cos_basis = angle.cos().to(torch.float32)
    sin_basis = angle.sin().to(torch.float32)

    def _fbank_dft(strided_input, mel_energies):
        re = strided_input @ cos_basis
        im = -(strided_input @ sin_basis)
        power = re * re + im * im
        energies = torch.mm(power, mel_energies.T)
        eps = torch.tensor(torch.finfo(torch.float).eps, dtype=strided_input.dtype)
        return torch.max(energies, eps).log()

    vm.fbank = _fbank_dft

    dummy = (torch.randn(*feat_shape, dtype=torch.float32),)
    dynamic_shapes = ({0: Dim("T", min=100, max=1000)},)

    from exporter.schema import load_and_validate_schema

    schema_json = load_and_validate_schema(_MODEL_DIR / "schema.json")

    with torch.no_grad():
        ep = torch_export(model, dummy, dynamic_shapes=dynamic_shapes)
        methods = {"forward": ep}
        et_program = to_edge_transform_and_lower(
            methods,
            transform_passes=get_default_passes(),
            partitioner=[MLXPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
            constant_methods={"get_model_schema": schema_json},
        ).to_executorch()

    out_path = Path(out_dir) / config["variant"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(et_program.buffer)
    return str(out_path)
