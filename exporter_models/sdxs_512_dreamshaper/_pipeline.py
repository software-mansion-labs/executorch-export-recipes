"""SDXS-512-DreamShaper: the three exported methods and their example inputs.

SDXS is a distilled ONE-STEP, no-CFG Stable-Diffusion-1.5-style pipeline: a CLIP
text encoder, a UNet run exactly once at timestep 999, and a TAESD
(AutoencoderTiny) decoder. The whole pipeline ships as a SINGLE .pte exposing
three methods with fully static shapes:

    encode:  input_ids int64  [1, 77]           -> text_emb float32 [1, 77, 768]
    denoise: sample    float32[1, 4, 64, 64],
             timestep  int64  [1],
             emb       float32[1, 77, 768]      -> eps      float32 [1, 4, 64, 64]
    decode:  latents   float32[1, 4, 64, 64]    -> image    float32 [1, 3, 512, 512]

`decode` bakes the TAESD-to-image mapping (`image = clamp(raw / 2 + 0.5, 0, 1)`),
so it returns RGB already in [0, 1] and the runtime does no unscaling. The
scheduler is NOT exported: for the single distilled step the DEIS update is
exactly linear in (latents, eps), and those pinned coefficients live in the
react-native-executorch task instead.

Everything here is backend-independent; each `<backend>/export.py` builds these
same three modules and differs only in the lowering.
"""

import torch

# Upstream weights (the SWM HF repo publishes only the exported .pte).
SOURCE_REPO = "IDKiro/sdxs-512-dreamshaper"

# The single distilled denoising step. Traced as a real input so the method
# signature keeps a timestep argument, but this is the only value SDXS ever uses.
TIMESTEP = 999


class Encode(torch.nn.Module):
    """CLIP text encoder: token ids -> last hidden state."""

    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(input_ids)[0]


class Denoise(torch.nn.Module):
    """UNet: one epsilon prediction. No CFG, so the batch stays 1."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, emb: torch.Tensor
    ) -> torch.Tensor:
        return self.unet(
            sample, timestep, encoder_hidden_states=emb, return_dict=False
        )[0]


class Decode(torch.nn.Module):
    """TAESD decoder + the [0, 1] image mapping."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        image = self.vae.decode(latents, return_dict=False)[0]
        return (image / 2 + 0.5).clamp(0, 1)


def load_methods() -> dict[str, torch.nn.Module]:
    """Load the upstream pipeline and wrap it into the three exported methods."""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        SOURCE_REPO, torch_dtype=torch.float32
    ).to("cpu")
    for module in (pipe.text_encoder, pipe.unet, pipe.vae):
        module.eval()

    return {
        "encode": Encode(pipe.text_encoder).eval(),
        "denoise": Denoise(pipe.unet).eval(),
        "decode": Decode(pipe.vae).eval(),
    }


def example_inputs(methods_config: dict) -> dict[str, tuple[torch.Tensor, ...]]:
    """Build tracing inputs from the config's declared input shapes.

    Shapes come from the config (the contract), values are pinned here: token
    ids and latents only have to be valid, but the timestep must be the real
    distilled step.
    """
    torch.manual_seed(0)

    def shapes(method: str) -> list[list[int]]:
        return [tensor["shape"] for tensor in methods_config[method]["inputs"]]

    (ids_shape,) = shapes("encode")
    sample_shape, timestep_shape, emb_shape = shapes("denoise")
    (latents_shape,) = shapes("decode")

    return {
        "encode": (torch.zeros(ids_shape, dtype=torch.int64),),
        "denoise": (
            torch.randn(sample_shape, dtype=torch.float32),
            torch.full(timestep_shape, TIMESTEP, dtype=torch.int64),
            torch.randn(emb_shape, dtype=torch.float32),
        ),
        "decode": (torch.randn(latents_shape, dtype=torch.float32),),
    }


def exported_programs(
    methods: dict[str, torch.nn.Module],
    inputs: dict[str, tuple[torch.Tensor, ...]],
) -> dict[str, "torch.export.ExportedProgram"]:
    """torch.export every method with fully static shapes."""
    with torch.no_grad():
        return {
            name: torch.export.export(module, inputs[name], strict=True)
            for name, module in methods.items()
        }
