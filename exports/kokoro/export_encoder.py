from model.model import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from torch.nn import Module
from typing import Literal
import argparse
import torch
from torch.export import Dim

# ---------------------------
# Encoder - exported interface
# ---------------------------

class TextEncoderWrapper(Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.text_encoder = model.text_encoder

    def forward(self, input_ids: torch.LongTensor,
                      text_mask: torch.BoolTensor,
                      pred_aln_trg: torch.FloatTensor):
        t_en = self.text_encoder(input_ids, ~text_mask)

        return t_en @ pred_aln_trg

# ---------------------------------
# Encoder - model adjustments
# ---------------------------------

def remove_weight_norms(encoder: TextEncoderWrapper):
    for module in encoder.text_encoder.cnn:
        conv = module[0]
        _ = conv.weight  # forces parameter update
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)

# -------------------------------
# Encoder - input data definition
# -------------------------------

# Predefined inputs for different sizes
inputs_dict = {
    "inp-32": (
        torch.randint(0, 178, size=(1, 32)),
        torch.ones((1, 32), dtype=torch.bool),
        torch.randint(0, 2, (1, 32, 92)).float()
    ),
    "inp-64": (
        torch.randint(0, 178, size=(1, 64)),
        torch.ones((1, 64), dtype=torch.bool),
        torch.randint(0, 2, (1, 64, 164)).float()
    ),
    "inp-128": (
        torch.randint(0, 178, size=(1, 128)),
        torch.ones((1, 128), dtype=torch.bool),
        torch.randint(0, 2, (1, 128, 296)).float()
    ),
}

# Map input names to token counts for filenames
input_name_mappings = {
    "inp-32": "32",
    "inp-64": "64",
    "inp-128": "128",
}

# -------------------------
# Encoder - export pipeline
# -------------------------

def convert_to_executorch_program(inputs, transform_and_lower: bool = True, dynamic: bool = False, 
                                  max_tokens: int | None = None, max_duration: int | None = None):
    # Model setup
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True, lstm_padding=max_tokens)
    encoder = TextEncoderWrapper(model)
    encoder.eval()

    # Deparametrize for XNNPACK delegation
    remove_weight_norms(encoder)

    # Dynamic shapes definition
    dynamic_shapes = None
    if dynamic:
        t = Dim("t", min=16, max=max_tokens)   # Tokens (number)
        d = Dim("d", min=32, max=max_duration)   # Duration

        dynamic_shapes = (
            {1: t},         # input_ids: (1, t)
            {1: t},         # text_mask: (1, t)
            {1: t, 2: d},   # pred_aln_trg:   (1, t, d)
        )

    # Export to ExportedProgram
    exported_program = torch.export.export(encoder, inputs, dynamic_shapes=dynamic_shapes)

    # Optionally lower to edge and partition
    if not transform_and_lower:
        return encoder, exported_program

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    )
    return encoder, executorch_program

# -------------------------
# Encoder - CLI entry point
# -------------------------

if __name__ == "__main__":
    # Args: input size and bundled mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", choices=["inp-32", "inp-64", "inp-128"], required=False)
    parser.add_argument("--bundled", type=lambda x: x.lower() == "true", default=False, required=False)
    parser.add_argument("--dynamic", type=lambda x: x.lower() == "true", default=False, required=False)
    parser.add_argument("--max-tokens", type=int, required=False, help="Maximum number of tokens (used only if --dynamic is set)")
    parser.add_argument("--max-duration", type=int, required=False, help="Maximum duration (used only if --dynamic is set)")
    args = parser.parse_args()

    # Mode selection
    if args.bundled:
        input_mode = "inp-64"
    elif args.input_size is None:
        raise RuntimeError("--input-size is required when bundled == false")
    
    if args.dynamic and (args.max_duration is None or args.max_tokens is None):
        raise RuntimeError("--max_tokens and --max_duration required when dynamic == true")

    bundled = args.bundled
    max_tokens = args.max_tokens
    max_duration = args.max_duration
    input_mode = args.input_size if not bundled else "inp-64"

    root_destination = "exported_models/tmp"

    if not bundled:
        # Single export
        print(f"Exporting encoder with {input_mode} input...")
        input_alias = input_name_mappings[input_mode]
        inputs = inputs_dict[input_mode]

        # Ensure BOS/EOS for random inputs
        if input_mode in ["inp-32", "inp-64", "inp-128"]:
            input_ids = inputs[0]
            input_ids[0][0] = 0
            input_ids[0][-1] = 0

        _, edge_program = convert_to_executorch_program(inputs, dynamic=args.dynamic, max_tokens=max_tokens, max_duration=max_duration)
        executorch_program = edge_program.to_executorch()

        print_delegation_info(executorch_program.exported_program().graph_module)

        inp_alias = input_alias if not args.dynamic else "dynamic"
        with open(f"{root_destination}/text_encoder_{inp_alias}.pte", "wb") as file:
            executorch_program.write_to_file(file)
    else:
        # Bundled export: multiple entrypoints forward_{n_tokens}
        print("Exporting bundled encoder...")
        encoders = {}
        for name, n_tokens in input_name_mappings.items():
            inputs = inputs_dict[name]
            input_ids = inputs[0]
            input_ids[0][0] = 0
            input_ids[0][-1] = 0
            _, exported_enc = convert_to_executorch_program(inputs, transform_and_lower=False, dynamic=args.dynamic, max_tokens=max_tokens, max_duration=max_duration)
            encoders[n_tokens] = exported_enc

        edge_program = to_edge_transform_and_lower(
            {f"forward_{n_tokens}": ep for n_tokens, ep in encoders.items()},
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        with open(f"{root_destination}/text_encoder.pte", "wb") as file:
            edge_program.write_to_file(file)

    print("Finished!")