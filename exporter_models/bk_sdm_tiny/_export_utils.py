import torch
import torch.nn as nn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

MIN_IMAGE_SIZE = 128


def export_model_to_path(model, out_path, input, dynamic_shapes={}):
    with torch.no_grad():
        exported_program = export(model, input, dynamic_shapes=dynamic_shapes)
        executorch_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(executorch_program.buffer)


def export_encoder(model, out_path):
    input = (
        torch.randint(low=0, high=10000, size=(1, 77), dtype=torch.int64),
        torch.randint(low=0, high=10000, size=(1, 77), dtype=torch.int64),
    )
    export_model_to_path(model, out_path, input)


def export_unet(model, out_path, image_size):
    H_DIM = Dim('_h', min=MIN_IMAGE_SIZE // 32, max=image_size // 32)
    W_DIM = Dim('_w', min=MIN_IMAGE_SIZE // 32, max=image_size // 32)
    dynamic_shapes = {
        "sample": {
            2: 4 * H_DIM,
            3: 4 * W_DIM,
        },
        "timestep": {},
        "encoder_hidden_states": {}
    }
    input = (
        torch.rand(2, 4, image_size // 8, image_size // 8, dtype=torch.float32),
        torch.tensor([999], dtype=torch.int64),
        torch.rand(2, 77, model.config.cross_attention_dim, dtype=torch.float32),
    )
    export_model_to_path(model, out_path, input, dynamic_shapes)


def export_vae(vae, out_path, image_size):
    class VaeDecoderWrapper(nn.Module):
        def __init__(self, vae: nn.Module):
            super().__init__()
            self.vae = vae

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            output = self.vae.decode(input)[0]
            if output.shape[0] == 1:
                output = output.squeeze(0)
            output = output.permute(1, 2, 0)
            output = output / 2 + 0.5
            output = torch.clamp(output, 0.0, 1.0)
            output = output * 255
            return output

    vae_wrapper = VaeDecoderWrapper(vae)
    H_DIM = Dim('_h', min=MIN_IMAGE_SIZE // 32, max=image_size // 32)
    W_DIM = Dim('_w', min=MIN_IMAGE_SIZE // 32, max=image_size // 32)
    dynamic_shapes = {
        "input": {
            2: 4 * H_DIM,
            3: 4 * W_DIM,
        }
    }
    input = (
        torch.rand(1, 4, image_size // 8, image_size // 8, dtype=torch.float32),
    )
    export_model_to_path(vae_wrapper, out_path, input, dynamic_shapes)
