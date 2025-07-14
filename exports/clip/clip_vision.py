"""
This script exports the text_model from the `openai/clip-vit-base-patch32` model
to Executorch format, partitions it for XNNPACK backend, and tests the exported model.

The model is wrapped in a custom PyTorch module that handles the postprocessing:
 - L2 normalization
"""

from transformers import AutoModel, AutoProcessor
import torch
import torch.nn.functional as F

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export

from executorch.runtime import Runtime

import time

torch.set_printoptions(precision=8)


# Input Preparation
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
encoded_image_input = processor(
    images=torch.randint(0, 256, (1, 3, 224, 224)),
    return_tensors="pt",
    padding=True,
    truncation=True,
)


class ClipVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = model.vision_model
        self.visual_projection = model.visual_projection

    def forward(self, pixel_values):
        model_output = self.vision_model(pixel_values)
        pooler_output = model_output.pooler_output
        projected_output = self.visual_projection(pooler_output)
        return F.normalize(projected_output, p=2, dim=1)


vision_model = ClipVisionModel()
vision_model.eval()

with torch.no_grad():
    image_embeddings = vision_model(encoded_image_input["pixel_values"])

model_name = "clip-vit-base-patch32-vision_xnnpack"

inputs = (encoded_image_input["pixel_values"],)
print("Inputs:", inputs)
print(inputs[0].dtype)
exported_program = export(vision_model, inputs)

executorch_program = to_edge_transform_and_lower(
    exported_program, partitioner=[XnnpackPartitioner()]
).to_executorch()

# Save the executorch program to a file
with open(f"{model_name}.pte", "wb") as file:
    file.write(executorch_program.buffer)

# Testing the Model
runtime = Runtime.get()

program = runtime.load_program(f"{model_name}.pte")
method = program.load_method("forward")
outputs = method.execute([encoded_image_input["pixel_values"]])

print("Outputs:", outputs)
print("Shape of outputs:", outputs[0].shape)
print("Type of outputs:", outputs[0].dtype)

assert torch.allclose(
    outputs[0], image_embeddings, atol=1e-6
), "Outputs do not match the expected clip image embeddings."
print("Outputs match the expected clip image embeddings.")

print("Image embeddings:", image_embeddings)
print("Outputs:", outputs[0])

# Benchmarks
tik = time.time()
for _ in range(100):
    outputs = method.execute(
        [
            encoded_image_input["pixel_values"],
        ]
    )
tok = time.time()

print("Execution time for 100 iterations:", tok - tik, "seconds")
