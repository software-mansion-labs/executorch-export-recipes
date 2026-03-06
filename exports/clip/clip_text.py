"""
This script exports the text_model from the `openai/clip-vit-base-patch32` model
to Executorch format, partitions it for XNNPACK backend, and tests the exported model.

The model is wrapped in a custom PyTorch module that handles the postprocessing:
 - L2 normalization

Max sequence length in the text encoder is adjusted to 74 to account for the cls (<|startoftext|>) [49406]  and sep (<|endoftext|>) [49407] tokens.
"""

from transformers import AutoModel, AutoProcessor
import torch
import torch.nn.functional as F

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

from executorch.runtime import Runtime

import time

torch.set_printoptions(precision=8)


# Input Preparation
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
input = "Hello, World!"
encoded_text_input = processor(
    text=input,
    return_tensors="pt",
    padding=True,
    truncation=True,
)


# Models Preparation
class ClipTextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = model.text_model
        self.text_projection = model.text_projection

    def forward(self, input_ids, attention_mask):
        input_ids = torch.cat(
            (
                torch.tensor([[49406]], dtype=input_ids.dtype),
                input_ids,
                torch.tensor([[49407]], dtype=input_ids.dtype),
            ),
            1,
        )
        attention_mask = torch.cat(
            (
                torch.tensor([[1]], dtype=attention_mask.dtype),
                attention_mask,
                torch.tensor([[1]], dtype=attention_mask.dtype),
            ),
            1,
        )

        model_output = self.text_model(input_ids, attention_mask)
        pooler_output = model_output.pooler_output
        projected_output = self.text_projection(pooler_output)
        return F.normalize(projected_output, p=2, dim=1)


text_model = ClipTextModel()
text_model.eval()

with torch.no_grad():
    text_embeddings = text_model(
        encoded_text_input["input_ids"], encoded_text_input["attention_mask"]
    )

model_name = "clip_vit_base_patch32_text_xnnpack_fp32"

dynamic_shapes = {
    "input_ids": {
        1: Dim(
            "tokens", min=1, max=74
        ),  # Adjusted max length to 74 = 76 - 2 (cls and sep tokens)
    },
    "attention_mask": {
        1: Dim(
            "tokens", min=1, max=74
        ),  # Adjusted max length to 74 = 76 - 2 (cls and sep tokens)
    },
}

inputs = (encoded_text_input["input_ids"], encoded_text_input["attention_mask"])
print("Inputs:", inputs)
print(inputs[0].dtype, inputs[1].dtype)
exported_program = export(text_model, inputs, dynamic_shapes=dynamic_shapes)

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
outputs = method.execute(
    [encoded_text_input["input_ids"], encoded_text_input["attention_mask"]]
)

print("Outputs:", outputs)
print("Shape of outputs:", outputs[0].shape)
print("Type of outputs:", outputs[0].dtype)

assert torch.allclose(
    outputs[0], text_embeddings, atol=1e-6
), "Outputs do not match the expected clip text embeddings."
print("Outputs match the expected clip text embeddings.")

print("Text embeddings:", text_embeddings)
print("Outputs:", outputs[0])

# Benchmarks
tik = time.time()
for _ in range(100):
    outputs = method.execute(
        [
            encoded_text_input["input_ids"],
            encoded_text_input["attention_mask"],
        ]
    )
tok = time.time()

print("Execution time for 100 iterations:", tok - tik, "seconds")
