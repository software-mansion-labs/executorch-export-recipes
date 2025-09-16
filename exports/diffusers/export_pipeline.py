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
from tqdm import tqdm
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity_error()

IMAGE_SIZE = 512

def export_pipeline(model_id, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    device = "cpu"

    with tqdm(total=5, unit="step") as pbar:
        # Tokenizer
        pbar.set_description("Export tokenizer")
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        export_tokenizer(tokenizer, model_dir)
        pbar.update(1)

        # Scheduler
        pbar.set_description("Export scheduler")
        scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")
        export_scheduler(scheduler, model_dir)
        pbar.update(1)

        # Text encoder
        pbar.set_description("Export text encoder")
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", use_safetensors=True
        ).to(device)

        input_text = (
            torch.randint(low=0, high=10000, size=(1, 77), dtype=torch.int64),
            torch.randint(low=0, high=10000, size=(1, 77), dtype=torch.int64),
        )
        model_subdir = os.path.join(model_dir, "text_encoder")
        export_encoder(text_encoder, model_subdir, input_text)
        pbar.update(1)

        # UNet
        pbar.set_description("Export UNet")
        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", use_safetensors=True
        ).to(device)
        input_unet = (
            torch.rand(2, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8, dtype=torch.float32),
            torch.tensor([999], dtype=torch.int64),
            torch.rand(2, 77, unet.config.cross_attention_dim, dtype=torch.float32)
        )
        model_subdir = os.path.join(model_dir, "unet")
        export_unet(unet, model_subdir, input_unet)
        pbar.update(1)

        # VAE
        pbar.set_description("Export VAE")
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", use_safetensors=True
        ).to(device)

        input_vae = (torch.rand(1, 4, IMAGE_SIZE // 8, IMAGE_SIZE // 8, dtype=torch.float32),)
        model_subdir = os.path.join(model_dir, "vae")
        export_vae(vae, model_subdir, input_vae)
        pbar.update(1)

    print("\nPipeline exported successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        required=True,
        type=str
    )
    parser.add_argument(
        "--model_dir",
        required=False,
        type=str,
        default="./"
    )
    args = parser.parse_args()

    export_pipeline(args.model_id, args.model_dir)
