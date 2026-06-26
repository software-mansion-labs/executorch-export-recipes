import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

MODEL_ID = "vivym/bk-sdm-tiny-vpred"


def _load_export_utils():
    p = _HERE / "_export_utils.py"
    spec = importlib.util.spec_from_file_location("bk_sdm_tiny__export_utils", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bk_sdm_tiny__export_utils"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> None:
    from transformers import CLIPTextModel
    from diffusers import AutoencoderKL, UNet2DConditionModel

    eu = _load_export_utils()

    out = Path(out_dir)
    variants = {v["file"]: v for v in config["variants"]}

    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    text_encoder.eval()
    eu.export_encoder(
        text_encoder,
        out / "bk_sdm_tiny_text_encoder_xnnpack_fp32.pte",
    )
    del text_encoder

    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
    unet.eval()
    eu.export_unet(unet, out / "bk_sdm_tiny_unet_xnnpack_fp32.pte", 512)
    eu.export_unet(unet, out / "bk_sdm_tiny_unet_256_xnnpack_fp32.pte", 256)
    del unet

    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae")
    vae.eval()
    eu.export_vae(vae, out / "bk_sdm_tiny_vae_xnnpack_fp32.pte", 512)
    eu.export_vae(vae, out / "bk_sdm_tiny_vae_256_xnnpack_fp32.pte", 256)
    del vae
