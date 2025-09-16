"""
The model loading from .pte file is implemented according to:  
https://docs.pytorch.org/executorch/stable/using-executorch-export.html#export-and-lowering
"""

import torch
from tqdm.auto import tqdm
from executorch.runtime import Runtime, Verification
from transformers import PretrainedConfig, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from PIL import Image
import argparse
import os

IMAGE_PATH = "generated_image.png"

def get_model(ModelClass, model_path, config_path=None):
    class Model(ModelClass):
        def __init__(self, forward_fn, config=None):
            if config:
                super().__init__(config)
            else:
                super().__init__()
            self.forward_fn = forward_fn

        def forward(self, *args):
            return self.forward_fn.execute(args)

    runtime = Runtime.get()
    program = runtime.load_program(model_path, verification = Verification.Minimal)
    method = program.load_method("forward")
    config = PretrainedConfig.from_pretrained(config_path) if config_path else None
    return Model(method, config)


def run_pipeline(prompt, model_dir, num_steps, image_size):
    device="cpu"
    tokenizer_dir = os.path.join(model_dir, "tokenizer")
    if not os.path.exists(tokenizer_dir):
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}/")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)

    config_path = os.path.join(model_dir, "scheduler/scheduler_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found in {os.path.join(model_dir, "scheduler/")}")
    scheduler = PNDMScheduler.from_pretrained(config_path)

    model_path = os.path.join(model_dir, "text_encoder/model.pte")
    config_path = os.path.join(model_dir, "text_encoder/config.json")
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Model or config not found in {os.path.join(model_dir, "text_encoder/")}")
    text_encoder = get_model(
        CLIPTextModel, model_path, config_path
    ).to(device)

    model_path = os.path.join(model_dir, "unet/model.pte")
    config_path = os.path.join(model_dir, "unet/config.json")
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Model or config not found in {os.path.join(model_dir, "unet/")}")
    unet = get_model(
        UNet2DConditionModel, model_path, config_path
    ).to(device)

    model_path = os.path.join(model_dir, "vae/model.pte")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found in {os.path.join(model_dir, "vae/")}")
    vae = get_model(
        AutoencoderKL, model_path
    ).to(device)

    print("\nPipeline loaded successfully\n")


    guidance_scale = 7.5
    generator = torch.manual_seed(0)
    batch_size = 1

    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device), text_input.attention_mask.to(device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device), uncond_input.attention_mask.to(device))[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.config.in_channels, image_size // 8, image_size // 8),
        generator=generator,
    ).to(device)
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_inference_steps=num_steps)

    for t in tqdm(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
        with torch.no_grad():
            timestep = t.to(torch.int64).unsqueeze(0)
            noise_pred = unet.forward(latent_model_input, timestep, text_embeddings)[0]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.forward(latents)[0]

    image = image.to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    
    image.save(IMAGE_PATH)
    image.show()
    print(f"Image saved as {IMAGE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--model_dir",
        required=False,
        type=str,
        default="./"
    )
    parser.add_argument(
        "--num_steps",
        required=False,
        type=int,
        default=10,
    )
    parser.add_argument(
        "--image_size",
        required=False, 
        type=int,
        default=512,
    )
    args = parser.parse_args()

    run_pipeline(args.prompt, args.model_dir, args.num_steps, args.image_size)