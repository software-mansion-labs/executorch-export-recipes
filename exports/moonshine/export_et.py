import executorch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, to_edge, ExecutorchBackendConfig, to_edge_transform_and_lower
from torch.export import Dim
import torch
from torch.export import export, ExportedProgram
from model_pt import MoonshineModel
from transcribe import load_audio
from convert_moonshine import convert_keras_to_pytorch
import moonshine
import lovely_tensors as lt
import torch.nn as nn
from einops import rearrange
lt.monkey_patch()
from functorch.experimental.control_flow import cond

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, t):
        freqs = torch.einsum("i , j -> i j", t.type_as(self.inv_freq), self.inv_freq)
        freqs = torch.stack((freqs, freqs), dim=-1)
        return rearrange(freqs, "... d r -> ... (d r)")
    
class PreprocessorEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.preprocessor = model.preprocessor
        self.encoder = model.encoder

    def forward(self, x):
        x = self.preprocessor(x)
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, model:MoonshineModel):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, seq, enc_src):
        logits = self.decoder.forward_cached(x=seq, enc_src=enc_src)
        logits = logits[:, -1]  # get last token
        sample = logits.argmax(dim=-1, keepdim=True)
        seq = torch.cat((seq, sample), dim=-1)
        return seq

def export_encoder(encoder):
    with torch.no_grad():
        
        example_args = (torch.rand((1,414*384)),)
        dim1_tmp = Dim("dim1_x", min=207, max=1250)
        dim1_x = dim1_tmp*384
        dynamic_shapes = {"x": {1: dim1_x}}
        aten_dialect: ExportedProgram = export(
                encoder,
                example_args,
                dynamic_shapes=dynamic_shapes
            )
        
        encoder_edge_program: EdgeProgramManager = to_edge_transform_and_lower(
            {
                "forward" : aten_dialect,
            },
            partitioner=[XnnpackPartitioner()]
        )
        executorch_program = encoder_edge_program.to_executorch(
            ExecutorchBackendConfig(
                passes=[],
            )
        )
        with open("encoder.pte", "wb") as file:
            file.write(executorch_program.buffer)

def export_decoder(decoder, encodings):
    with torch.no_grad():
        seq = torch.as_tensor([[1,2,3,4,5,6,7,8]]).to(dtype=torch.int64)
        dim1_x = Dim("dim1_seq", min=2, max=178)
        dim1_enc = Dim("dim1_enc", min=10, max=1248)
        dynamic_shapes = {"seq": {1: dim1_x}, "enc_src":{1: dim1_enc}}

        aten_dialect_forward: ExportedProgram = export(
                decoder,
                (seq, encodings.contiguous().cpu()),
                dynamic_shapes=dynamic_shapes
            )
        decoder_edge_program: EdgeProgramManager = to_edge_transform_and_lower(
            {
                "forward" : aten_dialect_forward,
            },
            partitioner=[XnnpackPartitioner()]
        )

        executorch_program = decoder_edge_program.to_executorch(
            ExecutorchBackendConfig(
                passes=[],  # User-defined passes
            )
        )
        with open("decoder.pte", "wb") as file:
            file.write(executorch_program.buffer)
            
            
def validate_export():
    from pathlib import Path
    from executorch.runtime import Verification, Runtime, Program, Method

    et_runtime: Runtime = Runtime.get()
    encoder: Program = et_runtime.load_program(
        Path("encoder.pte"),
        verification=Verification.Minimal,
    )
    decoder: Program = et_runtime.load_program(
        Path("decoder.pte"),
        verification=Verification.Minimal,
    )
    encoder_forward: Method = encoder.load_method("forward")
    decoder_forward: Method = decoder.load_method("forward")
    input = load_audio(moonshine.ASSETS_DIR / 'beckett.wav')
    encoder_output = encoder_forward.execute((input.cpu(),))[0]
    

    sot_token = 1
    eot_token = 2
    seq = torch.as_tensor([[sot_token]])
    sample = 1
    seq_len=20

    while sample != eot_token and len(seq.flatten()) <= seq_len:
        seq = decoder_forward.execute((seq, encoder_output.contiguous().cpu()))[0]
        sample = seq[:,-1]
    expected_output = torch.tensor([1, 18274, 1898, 3926, 5229, 29892, 694, 4383, 1018, 1449, 29892, 4418, 1449, 29892, 4418, 2253, 29889, 2])
    assert (torch.allclose(seq[0].data, expected_output))
if __name__ == '__main__':
    try:
        pt_model = convert_keras_to_pytorch(model_name='moonshine/tiny', device='cpu', verify=False)
    except Exception as e:
        print(f"Keras conversion failed: {e}")

    audio = load_audio(moonshine.ASSETS_DIR / 'beckett.wav').cpu()
    encoder_input = pt_model.preprocessor(audio).cpu()
    encoder_output = pt_model.encoder(encoder_input).cpu()

    try:
        encoder = PreprocessorEncoder(pt_model)
        export_encoder(encoder)
    except Exception as e:
        print('Failed exporting encoder', e)
        exit(0)

    try:
        decoder = Decoder(pt_model)
        export_decoder(decoder, encoder_output)
    except Exception as e:
        print('Failed exporting decoder', e)
        exit(0)
    
    validate_export()