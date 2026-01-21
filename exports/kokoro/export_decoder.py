from model.model import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from huggingface_hub import hf_hub_download
from torch.nn import Module
from typing import Literal
import argparse
import torch
from torch.export import Dim

# ----------------------------
# Decoder - exported interface
# ----------------------------

class DecoderWrapper(Module):
    def __init__(self, model: KModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, asr: torch.FloatTensor, F0_pred: torch.FloatTensor,
                N_pred: torch.FloatTensor, ref: torch.FloatTensor):
        return self.decoder(asr, F0_pred, N_pred, ref)


# ---------------------------
# Decoder - model adjustments
# ---------------------------

# Get rid of all weight_norm decllarations to enable delegating the convolutions to XNNPACK backend
# NOTE: Speeds up the entire inference from ~62 seconds to 0.68 seconds
def remove_weight_norms(decoder):
    # decoder -> generator -> resblocks -> convs1, convs2
    for module in decoder.generator.resblocks:
        for conv in module.convs1:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
        for conv in module.convs2:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    # decoder -> generator -> noise_res -> convs1, convs2
    for module in decoder.generator.noise_res:
        for conv in module.convs1:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
        for conv in module.convs2:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    # decoder -> generator -> ups
    for conv_t in decoder.generator.ups:
        _ = conv_t.weight  # forces parameter update
        torch.nn.utils.parametrize.remove_parametrizations(conv_t, "weight", leave_parametrized=True)
    # decoder -> generator -> conv_post
    _ = decoder.generator.conv_post.weight
    torch.nn.utils.parametrize.remove_parametrizations(decoder.generator.conv_post, "weight", leave_parametrized=True)
    # decoder -> encode, decode -> pool, conv1, conv2, conv1x1
    modules = [module for module in decoder.decode] + [decoder.encode]
    for module in modules:
        module_weight_norms = [module.conv1, module.conv2]
        if hasattr(module.pool, "weight"):
            module_weight_norms.append(module.pool)
        if hasattr(module, "conv1x1") and hasattr(module.conv1x1, "weight"):
            module_weight_norms.append(module.conv1x1)
        for conv in module_weight_norms:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    # decoder -> F0_conv, N_conv, asr_res
    for conv in [decoder.F0_conv, decoder.N_conv, decoder.asr_res[0]]:
        _ = conv.weight  # forces parameter update
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)


# -------------------------------
# Decoder - input data definition
# -------------------------------

inputs_dict = {
    "inp-32": (
		torch.randn(size=(1, 512, 92)),
		torch.randn(size=(1, 184)),
		torch.randn(size=(1, 184)),
		torch.randn(size=(1, 128))
	),
	"inp-64": (
		torch.randn(size=(1, 512, 164)),
		torch.randn(size=(1, 328)),
		torch.randn(size=(1, 328)),
		torch.randn(size=(1, 128))
	),
	"inp-128": (
		torch.randn(size=(1, 512, 296)),
		torch.randn(size=(1, 592)),
		torch.randn(size=(1, 592)),
		torch.randn(size=(1, 128))
	)
}

# Keys: input size names
# Values: equivalent number of input tokens
input_name_mappings = {
    "inp-32": "32",
    "inp-64": "64",
    "inp-128": "128"
}


# -------------------------
# Decoder - export pipeline
# -------------------------

def convert_to_executorch_program(inputs, transform_and_lower: bool = True, dynamic: bool = False):
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True)
    decoder = DecoderWrapper(model)
    decoder.eval()

    # Apply deparametrization
    remove_weight_norms(decoder.decoder)

    # Dynamic shapes definition
    dynamic_shapes = None
    if dynamic:
        k = Dim("k", min=32, max=296)
        k2 = 2 * k                  # F0/N = 2 * asr_time

        dynamic_shapes = (
            {2: k},   # asr: (1, 512, k)
            {1: k2},  # F0_pred: (1, 2*k)
            {1: k2},  # N_pred:   (1, 2*k)
            None      # ref: (1,128) fixed
        )

    # Export
    exported_program = torch.export.export(decoder, inputs, dynamic_shapes=dynamic_shapes)

    if not transform_and_lower:
        return decoder, exported_program

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()]
    )

    return decoder, executorch_program

if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", choices=["inp-32", "inp-64", "inp-128"], required=False)
    parser.add_argument("--bundled", type=lambda x: x.lower() == "true", default=False, required=False)
    parser.add_argument("--dynamic", type=lambda x: x.lower() == "true", default=False, required=False)
    args = parser.parse_args()

    # Input mode selection
    if args.bundled:
        input_mode = "random-medium"
    elif args.input_size is None:
        raise RuntimeError("--input-size is required when bundled == false")

    bundled = args.bundled
    input_mode = args.input_size

    ROOT_DESTINATION = "exported_models/tmp"

    if not bundled:
        print(f"Expporting decoder with {input_mode} input...")

        input_alias = input_name_mappings[input_mode]
        inputs = inputs_dict[input_mode]

        _, executorch_program = convert_to_executorch_program(inputs, dynamic=args.dynamic)
        executorch_program = executorch_program.to_executorch()

        print_delegation_info(executorch_program.exported_program().graph_module)

        inp_alias = input_alias if not args.dynamic else "dynamic"
        with open(f"{ROOT_DESTINATION}/text_decoder_{inp_alias}.pte", "wb") as file:
            executorch_program.write_to_file(file)
    else:
        print(f"Expporting bundled decoder...")

        decoders = {}
        for input_mode, n_tokens in input_name_mappings.items():
            _, exported_decoder = convert_to_executorch_program(inputs_dict[input_mode], transform_and_lower=False)
            decoders[n_tokens] = exported_decoder
        
        executorch_program = to_edge_transform_and_lower({
            f"forward_{n_tokens}": ep for n_tokens, ep in decoders.items()
        }, partitioner=[XnnpackPartitioner()]).to_executorch()

        with open(f"{ROOT_DESTINATION}/text_decoder.pte", "wb") as file:
            executorch_program.write_to_file(file)

    print("Finished!")