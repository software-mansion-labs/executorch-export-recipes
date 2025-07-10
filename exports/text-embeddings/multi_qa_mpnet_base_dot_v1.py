"""
This script exports the `sentence-transformers/all-MiniLM-L6-v2` model
to Executorch format, partitions it for XNNPACK backend, and tests the exported model.

The model is wrapped in a custom PyTorch module that handles the postprocessing:
 - mean pooling
 - L2 normalization

Max sequence length is adjusted to 254 to account for the CLS and SEP tokens.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

from executorch.runtime import Runtime

import time

torch.set_printoptions(precision=8)

# Input Preparation
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
sentences = ["Hello, World!"]
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

print("Encoded input:", encoded_input)
print("Shape of input_ids:", encoded_input["input_ids"].shape)
print("Shape of attention_mask:", encoded_input["attention_mask"].shape)

print("Input IDs type:", encoded_input["input_ids"].dtype)
print("Attention mask type:", encoded_input["attention_mask"].dtype)

start_token = torch.tensor(
    tokenizer.cls_token_id, dtype=encoded_input["input_ids"].dtype
)  # <s> token ID for the start of the sequence
end_token = torch.tensor(
    tokenizer.sep_token_id, dtype=encoded_input["input_ids"].dtype
)  # </s> token ID for the end of the sequence
print("Start token:", start_token, tokenizer.decode(start_token))
print("End token:", end_token, tokenizer.decode(end_token))


# Model Preparation
class AllMiniLML6V2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        )
        print(self.model.forward.__doc__)

    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask):
        batch_size = input_ids.shape[0]

        start_tokens = torch.tensor(
            [[0]], dtype=input_ids.dtype
        )  # <s> token ID for the start of the sequence
        end_tokens = torch.tensor(
            [[2]], dtype=input_ids.dtype
        )  # </s> token ID for the end of the sequence

        input_ids = torch.cat([start_tokens, input_ids, end_tokens], dim=1)

        start_attention = torch.ones((batch_size, 1), dtype=attention_mask.dtype)
        end_attention = torch.ones((batch_size, 1), dtype=attention_mask.dtype)

        attention_mask = torch.cat(
            [start_attention, attention_mask, end_attention], dim=1
        )

        model_output = self.model(
            input_ids=input_ids,  # Exclude <s> and </s> tokens
            attention_mask=attention_mask,
        )

        sentence_embeddings = self.cls_pooling(model_output)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings


model = AllMiniLML6V2()

with torch.no_grad():
    sentence_embeddings = model(
        encoded_input["input_ids"], encoded_input["attention_mask"]
    )

print("Sentence embeddings:", sentence_embeddings)
print("Shape of sentence embeddings:", sentence_embeddings.shape)
print("Type of sentence embeddings:", sentence_embeddings.dtype)

# Export and Lowering
model_name = "multi-qa-mpnet-base-dot-v1_xnnpack.pte"

dynamic_shapes = {
    "input_ids": {
        1: Dim(
            "tokens", min=1, max=510
        ),  # Adjusted max length to 510 = 512 - 2 (<s> and </s> tokens)
    },
    "attention_mask": {
        1: Dim(
            "tokens", min=1, max=510
        ),  # Adjusted max length to 510 = 512 - 2 (<s> and </s> tokens)
    },
}

inputs = (encoded_input["input_ids"], encoded_input["attention_mask"])
print("Inputs:", inputs)
print(inputs[0].dtype, inputs[1].dtype)
exported_program = export(model, inputs, dynamic_shapes=dynamic_shapes)

executorch_program = to_edge_transform_and_lower(
    exported_program, partitioner=[XnnpackPartitioner()]
).to_executorch()

# Save the executorch program to a file
with open(f"{model_name}", "wb") as file:
    file.write(executorch_program.buffer)

# Testing the Model
runtime = Runtime.get()

program = runtime.load_program(f"{model_name}")
method = program.load_method("forward")
outputs = method.execute([encoded_input["input_ids"], encoded_input["attention_mask"]])

print("Outputs:", outputs)
print("Shape of outputs:", outputs[0].shape)
print("Type of outputs:", outputs[0].dtype)

assert torch.allclose(
    outputs[0], sentence_embeddings, atol=1e-6
), "Outputs do not match the expected sentence embeddings."
print("Outputs match the expected sentence embeddings.")

# Benchmarks
tik = time.time()
for _ in range(100):
    outputs = method.execute(
        [encoded_input["input_ids"], encoded_input["attention_mask"]]
    )
tok = time.time()

print("Execution time for 100 iterations:", tok - tik, "seconds")
