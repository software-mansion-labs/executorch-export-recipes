"""Full Fluid Speech Pro source loading and bounded inference export boundaries.

Source weights and loader: FluidVoice-Labs/fluid-speech-pro, CC BY 4.0.
See SOURCE_LICENSE/SOURCE_NOTICE. NeMo components retain their upstream license.
"""
from pathlib import Path
import importlib.util
import os
import sys
import torch
from torch import nn

SOURCE_REPO = 'FluidVoice-Labs/fluid-speech-pro'
SOURCE_REVISION = '46027a3bffc5817ce8f9b29add0a062610134a90'
ROOT = Path(__file__).resolve().parent
MAX_AUDIO = 160000


def load_source(local_dir=None):
    from huggingface_hub import hf_hub_download
    directory = Path(local_dir or os.environ.get('FLUID_SOURCE', ''))
    checkpoint = directory / 'fluid-speech-pro.nemo'
    if not checkpoint.is_file():
        checkpoint = Path(hf_hub_download(SOURCE_REPO, 'fluid-speech-pro.nemo', revision=SOURCE_REVISION, token=False))
    spec = importlib.util.spec_from_file_location('fluid_speech_original_loader', ROOT / '_source_loader.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.load(str(checkpoint), map_location='cpu', clean_mode=False).eval()
    model.preprocessor.featurizer.dither = 0.0
    return model


class Features(nn.Module):
    """Bounded capacity; preserve NeMo STFT, true-length normalization and mask."""
    def __init__(self, model):
        super().__init__()
        self.preprocessor = model.preprocessor

    def forward(self, audio, length):
        return self.preprocessor(input_signal=audio, length=length)


class Encode(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, features, length):
        return self.encoder.forward_for_export(features, length)


class Predict(nn.Module):
    """One explicit LSTM step with source gate order i,f,g,o and external state.

    EXPORT PATCH: express the source two-layer LSTM using existing tensor ops,
    avoiding an opaque recurrent operator. Verify against source decoder.predict.
    Blank-as-pad embedding supplies the zero SOS vector for initial prediction.
    """
    def __init__(self, model):
        super().__init__()
        self.embedding = model.decoder.prediction['embed']
        self.lstm = model.decoder.prediction['dec_rnn'].lstm

    def forward(self, token, hidden, cell):
        x = self.embedding(token).reshape(1, 640)
        hs, cs = [], []
        for layer in range(2):
            gates = (torch.nn.functional.linear(x, getattr(self.lstm, f'weight_ih_l{layer}'), getattr(self.lstm, f'bias_ih_l{layer}'))
                     + torch.nn.functional.linear(hidden[layer], getattr(self.lstm, f'weight_hh_l{layer}'), getattr(self.lstm, f'bias_hh_l{layer}')))
            i, f, g, o = gates.chunk(4, dim=1)
            c = f.sigmoid() * cell[layer] + i.sigmoid() * g.tanh()
            x = o.sigmoid() * c.tanh()
            hs.append(x)
            cs.append(c)
        return x.reshape(1, 1, 640), torch.stack(hs), torch.stack(cs)


class Joint(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enc = model.joint.enc
        self.pred = model.joint.pred
        self.net = model.joint.joint_net

    def forward(self, acoustic, prediction):
        # Raw logits: TDT normalizes token and duration distributions separately.
        return self.net(self.enc(acoustic).unsqueeze(2) + self.pred(prediction).unsqueeze(1))


def prepare_capture(model):
    """EXPORT PATCH: name inferred reshape dimensions explicitly.

    NeMo's -1 reshapes introduce spurious divisibility guards for a strided,
    dynamic time axis. These replacements perform identical source operations;
    preserve the source's length-aware convolution and original forward hooks.
    """
    import types

    def subsample(self, x, lengths):
        x, lengths = self.conv(x, lengths)
        batch, channels, frames, bins = x.shape
        return self.out(x.transpose(1, 2).reshape(batch, frames, self.out.in_features)), lengths

    def rel_shift(self, x):
        batch, heads, frames, positions = x.shape
        # Same padded-flattened sequence with the first `frames` entries removed.
        # index_select materializes contiguous data without an inferred view.
        x = torch.nn.functional.pad(x, (1, 0)).reshape(batch, heads, frames * (positions + 1))
        indices = torch.arange(frames * positions, device=x.device) + frames
        return x.index_select(2, indices).reshape(batch, heads, frames, positions)

    pre = model.encoder.pre_encode
    assert pre._subsampling == 'dw_striding' and pre.conv2d_subsampling
    pre.forward = types.MethodType(subsample, pre)
    for layer in model.encoder.layers:
        layer.self_attn.rel_shift = types.MethodType(rel_shift, layer.self_attn)
    model.encoder.set_max_audio_length(1001)
    return model


class Subsample(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.pre = model.encoder.pre_encode
        # Source hook applies the separately registered Fluid block.
        self.cleaner = model.encoder.fv_insert

    def forward(self, features, length):
        return self.pre(features.transpose(1, 2), length)


class Transformer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, embeddings, length):
        return self.encoder.forward_internal(embeddings, length, bypass_pre_encode=True)
