import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent


def _load_model_pt():
    spec = importlib.util.spec_from_file_location("moonshine_tiny_model_pt", _HERE / "_model_pt.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moonshine_tiny_model_pt"] = mod
    spec.loader.exec_module(mod)
    return mod


def export(config: dict, out_dir: str) -> None:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir import ExecutorchBackendConfig, to_edge_transform_and_lower
    from torch.export import Dim
    from torch.export import export as torch_export

    mp = _load_model_pt()
    MoonshineModel = mp.MoonshineModel

    model = MoonshineModel(288, 288, 8, 6, 6, dec_ff_swiglu=True)
    model.load_state_dict(torch.load(_HERE / "moonshine.pth", map_location="cpu"))
    model.eval()

    class PreprocessorEncoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.preprocessor = model.preprocessor
            self.encoder = model.encoder
        def forward(self, x):
            return self.encoder(self.preprocessor(x))

    class Decoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.decoder = model.decoder
        def forward(self, seq, enc_src):
            logits = self.decoder.forward_cached(x=seq, enc_src=enc_src)
            logits = logits[:, -1]
            sample = logits.argmax(dim=-1, keepdim=True)
            return torch.cat((seq, sample), dim=-1)

    encoder = PreprocessorEncoder(model)
    decoder = Decoder(model)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        enc_example = (torch.rand((1, 414 * 384)),)
        dim1_tmp = Dim("dim1_x", min=207, max=1250)
        dim1_x = dim1_tmp * 384
        enc_shapes = {"x": {1: dim1_x}}
        enc_ep = torch_export(encoder, enc_example, dynamic_shapes=enc_shapes)
        enc_et = to_edge_transform_and_lower(
            {"forward": enc_ep},
            partitioner=[XnnpackPartitioner()],
        ).to_executorch(ExecutorchBackendConfig(passes=[]))
        (out / "moonshine_tiny_xnnpack_encoder.pte").write_bytes(enc_et.buffer)

    with torch.no_grad():
        encoder_output = encoder(*enc_example)
        seq = torch.as_tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int64)
        dim1_seq = Dim("dim1_seq", min=2, max=178)
        dim1_enc = Dim("dim1_enc", min=10, max=1248)
        dec_shapes = {"seq": {1: dim1_seq}, "enc_src": {1: dim1_enc}}
        dec_ep = torch_export(
            decoder,
            (seq, encoder_output.contiguous().cpu()),
            dynamic_shapes=dec_shapes,
        )
        dec_et = to_edge_transform_and_lower(
            {"forward": dec_ep},
            partitioner=[XnnpackPartitioner()],
        ).to_executorch(ExecutorchBackendConfig(passes=[]))
        (out / "moonshine_tiny_xnnpack_decoder.pte").write_bytes(dec_et.buffer)
