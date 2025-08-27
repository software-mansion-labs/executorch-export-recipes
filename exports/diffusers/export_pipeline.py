"""
This code follows the Stable Diffusion pipeline:  
https://huggingface.co/docs/diffusers/en/using-diffusers/write_own_pipeline  

It has been tested on the `vivym/bk-sdm-tiny-vpred` model.
"""

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from export_utils import export_encoder, export_scheduler, export_tokenizer, export_unet, export_vae
import argparse
import os


def export_pipeline(model_id, model_dir):
    os.makedirs(model_dir, exist_ok=True)

    device = "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id, subfolder="tokenizer"
    )
    export_tokenizer(tokenizer, model_dir)

    scheduler = PNDMScheduler.from_pretrained(
        model_id, subfolder="scheduler"
    )
    export_scheduler(scheduler, model_dir)

    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", use_safetensors=True
    ).to(device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", use_safetensors=True
    ).to(device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", use_safetensors=True
    ).to(device)
    print("\nPipeline components loaded successfully\n")

    input = (torch.randint(low=0, high=10000, size=(1, 77), dtype=torch.int64),)
    model_subdir = os.path.join(model_dir, "text_encoder")
    export_encoder(text_encoder, input, model_subdir)
    print("Text encoder exported")

    input = (
        torch.rand(2, 4, 64, 64, dtype=torch.float32),
        torch.tensor([999], dtype=torch.int64),
        torch.rand(2, 77, 768, dtype=torch.float32)
    )
    model_subdir = os.path.join(model_dir, "unet")
    export_unet(unet, input, model_subdir)
    print("UNet exported")

    input = (torch.rand(1, 4, 64, 64, dtype=torch.float32),)
    model_subdir = os.path.join(model_dir, "vae")
    export_vae(vae, input, model_subdir)
    print("VAE exported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", 
        required=True, 
        type=str,
    )
    parser.add_argument(
        "--model_dir", 
        required=False, 
        type=str,
        default="./",
    )
    args = parser.parse_args()

    export_pipeline(args.model_id, args.model_dir)
