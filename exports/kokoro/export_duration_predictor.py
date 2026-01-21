from model.model import KModel
from model.temporal_scaling import scale
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from torch.nn import Module
from typing import Literal
import argparse
import torch
from torch.export import Dim

# ---------------------------------------
# Duration predictor - exported interface
# ---------------------------------------

class DurationPredictorWrapper(Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.bert = model.bert
        self.bert_encoder = model.bert_encoder
        self.text_encoder_module = model.predictor.text_encoder
        self.lstm = model.predictor.lstm
        self.duration_proj = model.predictor.duration_proj

        self.lstm_padding = model.lstm_padding
    def forward(self, input_ids: torch.LongTensor, text_mask: torch.BoolTensor,
                      s: torch.FloatTensor, speed: torch.Tensor):
        # Padding stage
        if self.lstm_padding:
            pad_len = self.lstm_padding - input_ids.shape[-1]
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
            text_mask = torch.nn.functional.pad(text_mask, (0, pad_len), value=False)

        bert_dur = self.bert(input_ids, attention_mask=text_mask.int())
        d_en = self.bert_encoder(bert_dur).transpose(-1, -2)
        d = self.text_encoder_module(d_en, s, torch.tensor(input_ids.shape[-1]), ~text_mask)    # A problem here (with dynamic shapes)
        # Probably impossible to fix since LSTM export with dynamic shapes is not supported

        x, _ = self.lstm(d)
        
        duration = self.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

        return pred_dur, d


# --------------------------------------
# Duration predictor - model adjustments
# --------------------------------------

# ...


# --------------------------------------------
# Duration predictor - input data definition
# --------------------------------------------

# Predefined inputs for export sizes
inputs_dict = {
    "inp-32": (
        torch.randint(0, 178, size=(1, 32)),
        torch.ones((1, 32), dtype=torch.bool),
        torch.randn(size=(1, 128)),
        torch.tensor([1.0], dtype=torch.float32),
    ),
    "inp-64": (
        torch.randint(0, 178, size=(1, 64)),
        torch.ones((1, 64), dtype=torch.bool),
        torch.randn(size=(1, 128)),
        torch.tensor([1.0], dtype=torch.float32),
    ),
    "inp-128": (
        torch.randint(0, 178, size=(1, 128)),
        torch.ones((1, 128), dtype=torch.bool),
        torch.randn(size=(1, 128)),
        torch.tensor([1.0], dtype=torch.float32),
    ),
}

# Keys: input size names -> Values: equivalent number of input tokens
input_name_mappings = {
    "inp-32": 32,
    "inp-64": 64,
    "inp-128": 128,
}

# ------------------------------------
# Duration predictor - export pipeline
# ------------------------------------

def convert_to_executorch_program(inputs, transform_and_lower: bool = True, dynamic: bool = False,
                                  max_tokens: int | None = None):
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True, lstm_padding=max_tokens)
    duration_predictor = DurationPredictorWrapper(model)
    duration_predictor.eval()

    # Apply modifications (none for duration predictor)
    # ...

    # Dynamic shapes definition
    dynamic_shapes = None
    if dynamic:
        # Let's say we avoid giving less then 16 tokens on the input, since
        # the response quality might be degraded in such case.
        t = Dim("t", min=8, max=max_tokens)   # Tokens (number)

        dynamic_shapes = (
            {1: t},   # input_ids: (1, t)
            {1: t},   # text_mask: (1, t)
            None,     # s (voice half-array, always static)
            None      # speed (a single float)
        )

    exported_program = torch.export.export(duration_predictor, inputs, dynamic_shapes=dynamic_shapes)

    if not transform_and_lower:
        return duration_predictor, exported_program

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )

    return duration_predictor, executorch_program

# -------------------------------
# Duration predictor - CLI entry
# -------------------------------

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", choices=["inp-32", "inp-64", "inp-128"], required=False)
    parser.add_argument("--bundled", type=lambda x: x.lower() == "true", default=False, required=False)
    parser.add_argument("--dynamic", type=lambda x: x.lower() == "true", default=False, required=False)
    parser.add_argument("--max-tokens", type=int, required=False, help="Maximum number of tokens (used only if --dynamic is set)")
    args = parser.parse_args()

    # Mode selection
    if args.bundled is False and args.input_size is None:
        raise RuntimeError("--input-size is required when bundled == false")
    
    if args.dynamic and args.max_tokens is None:
        raise RuntimeError("--max_tokens required when dynamic == true")

    bundled = args.bundled
    max_tokens = args.max_tokens
    input_mode = args.input_size if not bundled else "inp-64"

    ROOT_DESTINATION = "exported_models/tmp"

    if not bundled:
        # Single export
        print(f"Exporting duration predictor with {input_mode} input...")
        n_tokens = input_name_mappings[input_mode]
        inputs = inputs_dict[input_mode]

        if input_mode in ["inp-32", "inp-64", "inp-128"]:
            input_ids = inputs[0]
            input_ids[0][0] = 0
            input_ids[0][-1] = 0

        _, edge_program = convert_to_executorch_program(inputs, dynamic=args.dynamic, max_tokens=max_tokens)
        executorch_program = edge_program.to_executorch()

        print_delegation_info(executorch_program.exported_program().graph_module)

        inp_alias = str(n_tokens) if not args.dynamic else "dynamic"
        with open(f"{ROOT_DESTINATION}/duration_predictor_{inp_alias}.pte", "wb") as file:
            executorch_program.write_to_file(file)
    else:
        # Bundled export: multiple entry points
        print("Exporting bundled duration predictor...")

        predictors = {}
        for name, n_tokens in input_name_mappings.items():
            # if name == "inp-128":
            #     continue
            inputs = inputs_dict[name]
            input_ids = inputs[0]
            input_ids[0][0] = 0
            input_ids[0][-1] = 0
            _, exported_predictor = convert_to_executorch_program(inputs_dict[name], dynamic=args.dynamic, transform_and_lower=False, max_tokens=max_tokens)
            predictors[n_tokens] = exported_predictor

        executorch_program = to_edge_transform_and_lower(
            {f"forward_{n_tokens}": ep for n_tokens, ep in predictors.items()},
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        with open(f"{ROOT_DESTINATION}/duration_predictor.pte", "wb") as file:
            executorch_program.write_to_file(file)

    print("Finished!")