from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.export import export as torch_export

_SOURCE = "openai/clip-vit-base-patch32"


class _ClipVisionEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModel.from_pretrained(_SOURCE)
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values):
        pooled = self.vision_model(pixel_values).pooler_output
        projected = self.visual_projection(pooled)
        return F.normalize(projected, p=2, dim=1)


def export(config: dict, out_dir: str) -> str:
    from executorch import exir
    from executorch.backends.mlx import MLXPartitioner
    from executorch.backends.mlx.passes import get_default_passes
    from executorch.exir import EdgeCompileConfig

    variant = config["variant"]
    shape = tuple(variant["methods"]["forward"]["inputs"][0]["shape"])

    module = _ClipVisionEncoder().eval()

    if variant.get("quantized"):
        # 97% of this model's parameters live in nn.Linear, the highest share
        # of any model we ship, so weight-only quantization is unusually
        # effective here: fp32 351 MB -> int8 ~95 MB -> int4 ~53 MB.
        from executorch.backends.mlx.llm.quantization import quantize_model_

        qcfg = {"int8": {"qlinear_config": "8w"},
                "int4": {"qlinear_config": "4w", "qlinear_group_size": 64}}[
            variant["precision"]
        ]
        quantize_model_(module, **qcfg)

    inputs = (torch.zeros(shape, dtype=torch.float32),)

    ep = torch_export(module, inputs)
    program = exir.to_edge_transform_and_lower(
        {"forward": ep},
        transform_passes=get_default_passes(),
        partitioner=[MLXPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
    ).to_executorch()

    out_path = Path(out_dir) / variant["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(program.buffer)
    return str(out_path)
