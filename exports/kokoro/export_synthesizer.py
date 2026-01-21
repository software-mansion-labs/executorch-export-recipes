from model.model import KModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.devtools.backend_debug import print_delegation_info, get_delegation_info
from executorch.exir import to_edge_transform_and_lower
from torch.nn import Module
from typing import Literal
import argparse
import torch
from torch.export import Dim

# --------------------------------
# Synthesizer - exported interface
# --------------------------------

class Synthesizer(Module):
  def __init__(self, model: KModel, duration_padding: int | None = None):
    super().__init__()

    # F0N predictor modules
    self.shared = model.predictor.shared
    self.F0_blocks = model.predictor.F0
    self.F0_proj = model.predictor.F0_proj
    self.N_blocks = model.predictor.N
    self.N_proj = model.predictor.N_proj

    self.lstm_dur_padding = duration_padding

    # Text encoder modules
    self.text_encoder = model.text_encoder

    # Text decoder modules
    self.decoder = model.decoder
  
  def forward(self, input_ids: torch.LongTensor,
                    text_mask: torch.BoolTensor,
                    indices: torch.LongTensor,
                    d: torch.FloatTensor,
                    ref_s: torch.FloatTensor):
    ref = ref_s[:, :128]
    s = ref_s[:, 128:]

    # F0N predictor inference
    # -----------------------
    input_size = input_ids.shape[-1]

    pred_aln_trg = torch.zeros((input_size, indices.shape[0]), dtype=torch.float32)
    pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1.0
    pred_aln_trg = pred_aln_trg.unsqueeze(0)
    en = d.transpose(-1, -2) @ pred_aln_trg

    x = en.transpose(-1, -2)
    torch._check_is_size(x.shape[1])

    # Padd to the maximum duration
    if self.lstm_dur_padding is not None:
        original_seq_len = x.shape[1]
        torch._check_is_size(original_seq_len)
        pad_amount = self.lstm_dur_padding - original_seq_len
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount))

    # LSTM inference
    x, _ = self.shared(x)

    # Return back to the original shape
    if self.lstm_dur_padding is not None:
        x = x[:, :original_seq_len, :]

    F0 = x.transpose(-1, -2)
    for block in self.F0_blocks:
        F0 = block(F0, s)
    F0 = self.F0_proj(F0).squeeze(1)

    N = x.transpose(-1, -2)
    for block in self.N_blocks:
        N = block(N, s)
    N = self.N_proj(N).squeeze(1)

    # Text encoder inference
    t_en = self.text_encoder(input_ids, ~text_mask)
    asr = t_en @ pred_aln_trg

    # Text decoder inference
    return self.decoder(asr, F0, N, ref)
  

# -------------------------------
# Synthesizer - model adjustments
# -------------------------------

def remove_weight_norms(synthesizer: Synthesizer):
    # f0n_predictor -> F0_blocks, N_blocks -> pool, conv1, conv2, conv1x1
    modules = [module for module in synthesizer.F0_blocks] + [module for module in synthesizer.N_blocks]
    for module in modules:
        module_weight_norms = [module.conv1, module.conv2]
        if hasattr(module.pool, "weight"):
            module_weight_norms.append(module.pool)
        if hasattr(module, "conv1x1") and hasattr(module.conv1x1, "weight"):
            module_weight_norms.append(module.conv1x1)
        for conv in module_weight_norms:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
  
    # text_encoder -> cnn.conv
    for module in synthesizer.text_encoder.cnn:
        conv = module[0]
        _ = conv.weight  # forces parameter update
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)

    # decoder -> generator -> resblocks -> convs1, convs2
    for module in synthesizer.decoder.generator.resblocks:
        for conv in module.convs1:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
        for conv in module.convs2:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    # decoder -> generator -> noise_res -> convs1, convs2
    for module in synthesizer.decoder.generator.noise_res:
        for conv in module.convs1:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
        for conv in module.convs2:
            _ = conv.weight  # forces parameter update
            torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)
    # decoder -> generator -> ups
    for conv_t in synthesizer.decoder.generator.ups:
        _ = conv_t.weight  # forces parameter update
        torch.nn.utils.parametrize.remove_parametrizations(conv_t, "weight", leave_parametrized=True)
    # decoder -> generator -> conv_post
    _ = synthesizer.decoder.generator.conv_post.weight
    torch.nn.utils.parametrize.remove_parametrizations(synthesizer.decoder.generator.conv_post, "weight", leave_parametrized=True)
    # decoder -> encode, decode -> pool, conv1, conv2, conv1x1
    modules = [module for module in synthesizer.decoder.decode] + [synthesizer.decoder.encode]
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
    for conv in [synthesizer.decoder.F0_conv, synthesizer.decoder.N_conv, synthesizer.decoder.asr_res[0]]:
        _ = conv.weight  # forces parameter update
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight", leave_parametrized=True)


# -------------------------------
# Synthesizer - input data definition
# -------------------------------

inputs_dict = {
    "inp-32": (
		torch.randint(0, 178, size=(1, 32)),
    torch.ones((1, 32), dtype=torch.bool),
		torch.randint(0, 2, (92,), dtype=torch.long),
    torch.randn(size=(1, 32, 640)),
    torch.randn(size=(1, 256))
	),
	"inp-64": (
		torch.randint(0, 178, size=(1, 64)),
    torch.ones((1, 64), dtype=torch.bool),
		torch.randint(0, 2, (164,), dtype=torch.long),
    torch.randn(size=(1, 64, 640)),
    torch.randn(size=(1, 256))
	),
	"inp-128": (
		torch.randint(0, 178, size=(1, 128)),
    torch.ones((1, 128), dtype=torch.bool),
		torch.randint(0, 2, (296,), dtype=torch.long),
    torch.randn(size=(1, 128, 640)),
    torch.randn(size=(1, 256))
	)
}

input_name_mappings = {
    "inp-32": "32",
    "inp-64": "64",
    "inp-128": "128"
}


# -----------------------------
# Synthesizer - export pipeline
# -----------------------------

def convert_to_executorch_program(inputs, transform_and_lower: bool = True, dynamic: bool = False,
                                  max_tokens: int | None = None, max_duration: int | None = None):
    model = KModel(repo_id="hexgrad/Kokoro-82M", disable_complex=True, lstm_padding=max_tokens)
    synthesizer = Synthesizer(model, duration_padding=max_duration)
    synthesizer.eval()

    # Apply deparametrization
    remove_weight_norms(synthesizer)

    # Dynamic shapes definition
    dynamic_shapes = None
    if dynamic:
        t = Dim("t", min=8, max=max_tokens)
        d = Dim("k", min=16, max=max_duration)

        dynamic_shapes = (
            {1: t},
            {1: t},
            {0: d},
            {1: t},
            None
        )

    # Export
    exported_program = torch.export.export(synthesizer, inputs, dynamic_shapes=dynamic_shapes)

    if not transform_and_lower:
        return synthesizer, exported_program

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner = [XnnpackPartitioner()]
    )

    return synthesizer, executorch_program


# -----------------------------
# Synthesizer - main export API
# -----------------------------

if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", choices=["inp-32", "inp-64", "inp-128"], required=False)
    parser.add_argument("--bundled", type=lambda x: x.lower() == "true", default=False, required=False)
    parser.add_argument("--dynamic", type=lambda x: x.lower() == "true", default=False, required=False)
    parser.add_argument("--max-tokens", type=int, required=False, help="Maximum number of tokens (used only if --dynamic is set)")
    parser.add_argument("--max-duration", type=int, required=False, help="Maximum duration (used only if --dynamic is set)")
    args = parser.parse_args()

    # Input mode selection
    if args.bundled:
        input_mode = "random-medium"
    elif args.input_size is None:
        raise RuntimeError("--input-size is required when bundled == false")
    
    if args.dynamic and (args.max_duration is None or args.max_tokens is None):
        raise RuntimeError("--max_tokens and --max_duration required when dynamic == true")

    bundled = args.bundled
    max_tokens = args.max_tokens
    max_duration = args.max_duration
    input_mode = args.input_size

    ROOT_DESTINATION = "exported_models/tmp"

    if not bundled:
        print(f"Expporting synthesizer with {input_mode} input...")

        input_alias = input_name_mappings[input_mode]
        inputs = inputs_dict[input_mode]

        _, executorch_program = convert_to_executorch_program(inputs, dynamic=args.dynamic, max_tokens=max_tokens, max_duration=max_duration)
        executorch_program = executorch_program.to_executorch()

        print_delegation_info(executorch_program.exported_program().graph_module)

        inp_alias = input_alias if not args.dynamic else "dynamic"
        with open(f"{ROOT_DESTINATION}/synthesizer_{inp_alias}.pte", "wb") as file:
            executorch_program.write_to_file(file)
    else:
        print(f"Expporting bundled synthesizer...")

        decoders = {}
        for input_mode, n_tokens in input_name_mappings.items():
            _, exported_synthesizer = convert_to_executorch_program(inputs_dict[input_mode], transform_and_lower=False, dynamic=args.dynamic, max_tokens=max_tokens, max_duration=max_duration)
            decoders[n_tokens] = exported_synthesizer
        
        executorch_program = to_edge_transform_and_lower({
            f"forward_{n_tokens}": ep for n_tokens, ep in decoders.items()
        }, partitioner=[XnnpackPartitioner()]).to_executorch()

        with open(f"{ROOT_DESTINATION}/synthesizer.pte", "wb") as file:
            executorch_program.write_to_file(file)

    print("Finished!")