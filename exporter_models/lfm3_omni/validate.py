"""Compare Omni components with upstream or a loaded artifact.

Use the recipe's export environment for --pte, or an isolated transformers
5.20.2 environment for --reference. Run separately from an export on small hosts.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pte', type=Path)
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--native-dir', type=Path, help='Saved exact-runtime outputs from the fixed seed-42 fixtures')
    args = parser.parse_args()
    if not args.pte and not args.reference and not args.native_dir:
        parser.error('choose --pte, --reference, or --native-dir')
    root = Path(__file__).parent / '_lfm3'
    spec = importlib.util.spec_from_file_location('lfm3_omni_validation', root / '__init__.py', submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from lfm3_omni_validation.speech import DepthDecoder, Codec
    from lfm3_omni_validation.audio import AudioEncoder, EncoderConfig, convert, mel_filters, COMPUTED_BUFFERS
    from safetensors.torch import load_file
    source = args.source.resolve()
    hf = json.loads((source / 'config.json').read_text())
    state = load_file(str(source / 'model.safetensors'))
    depth = DepthDecoder(hf).eval()
    depth.load_state_dict({k.removeprefix('model.depth_decoder.'): v.float() for k, v in state.items() if k.startswith('model.depth_decoder.')})
    audio = AudioEncoder(hf, mel_filters()).eval()
    missing, extra = audio.load_state_dict({k:v.float() for k,v in convert(state, EncoderConfig.from_hf(hf)).items()},strict=False)
    assert set(missing) == set(COMPUTED_BUFFERS) and not extra, (missing, extra)
    codec = Codec().eval()
    prefixes = {'quantizer.project_out.':'project_out.', 'from_quantizer.':'from_quantizer.', 'decoder.lfm.':'trunk.', 'decoder.lin.':'lin.'}
    converted = {}
    for k,v in load_file(str(source / 'audio_tokenizer/model.safetensors')).items():
        for prefix,target in prefixes.items():
            if k.startswith(prefix): converted[target + k[len(prefix):]] = v.float()
    missing, extra = codec.load_state_dict(converted,strict=False)
    assert set(missing) == {'inverse_dft','envelope_window'} and not extra, (missing,extra)
    del state, converted
    if args.reference:
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)
        cls = get_class_from_dynamic_module('modeling_lfm3_omni.Lfm3OmniForConditionalGeneration',str(source))
        module = sys.modules[cls.__module__]
        weights = load_file(str(source / 'model.safetensors'))
        reference_depth = module.Lfm3OmniDepthDecoder(config).float().eval()
        reference_depth.load_state_dict({k.removeprefix('model.depth_decoder.'):v.float() for k,v in weights.items() if k.startswith('model.depth_decoder.')})
        reference_audio = module.Lfm3OmniEncoder(config.audio_encoder_config).float().eval()
        reference_audio.load_state_dict({k.removeprefix('model.audio_encoder.'):v.float() for k,v in weights.items() if k.startswith('model.audio_encoder.')})
        projector = module.Lfm3OmniProjector(config).float().eval()
        projector.load_state_dict({k.removeprefix('model.audio_projector.'):v.float() for k,v in weights.items() if k.startswith('model.audio_projector.')})
        codec_cls = get_class_from_dynamic_module('modeling_liquid_codec.LiquidCodecModel',str(source))
        cfg_cls = get_class_from_dynamic_module('configuration_liquid_codec.LiquidCodecConfig',str(source))
        reference_codec = codec_cls(cfg_cls.from_pretrained(source/'audio_tokenizer')).float().eval()
        reference_codec.load_state_dict(load_file(str(source/'audio_tokenizer/model.safetensors')))
        del weights
    if args.pte:
        from executorch.extension.pybindings.portable_lib import _load_for_executorch
        program = _load_for_executorch(str(args.pte))
    if args.native_dir:
        import numpy as np
        class SavedNativeOutputs:
            def run_method(self, method, inputs):
                if method == 'depth_decoder': name, shape = 'depth', (1, 12, 2017)
                elif method == 'audio_decoder':
                    frames = inputs[0].shape[-1]
                    name, shape = f'codec-{frames}', (1, frames * 1920)
                elif method == 'audio_encoder':
                    samples = inputs[0].shape[-1]
                    frames = ((samples - 400) // 160 + 1 + 7) // 8
                    name, shape = f'audio-{samples}', (1, frames, 2048)
                else: raise ValueError(method)
                path = args.native_dir / f'actual-{name}.bin'
                return [torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(shape))]
        program = SavedNativeOutputs()
    results = {}
    def compare(name, actual, expected):
        actual,expected = actual.float(),expected.float()
        error = (actual-expected).abs()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(),expected.flatten(),dim=0).item()
        passed = actual.shape == expected.shape and torch.isfinite(actual).all().item() and error.max().item() <= .0005 and cosine >= .99999
        results[name] = {'shape':list(actual.shape),'max_abs':error.max().item(),'mean_abs':error.mean().item(),'cosine':cosine,'pass':passed}
        print(name, results[name], flush=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
    torch.manual_seed(42)
    with torch.no_grad():
        hidden = torch.randn(1,2048); codes = torch.randint(0,2016,(1,12)); masked = torch.arange(12)[None] % 2 == 0
        expected = depth(hidden,codes,masked)
        if args.reference: compare('reference_depth',reference_depth(hidden,codes,masked),expected)
        if args.pte or args.native_dir: compare('runtime_depth',program.run_method('depth_decoder',(hidden,codes,masked))[0].clone(),expected)
        for frames in [1,4,32]:
            codes = torch.randint(0,2016,(1,12,frames)); expected = codec(codes)
            if args.reference: compare(f'reference_codec_{frames}',reference_codec.decode(codes).audio_values,expected)
            if args.pte or args.native_dir: compare(f'runtime_codec_{frames}',program.run_method('audio_decoder',(codes,))[0].clone(),expected)
        for samples in [400,16400,80400]:
            wave = torch.randn(1,samples)*.1; expected = audio(wave)
            if args.reference:
                mel = audio.log_mel(wave)
                encoded,_ = reference_audio(mel,torch.tensor([mel.shape[-1]]))
                compare(f'reference_audio_{samples}',projector(encoded),expected)
            if args.pte or args.native_dir: compare(f'runtime_audio_{samples}',program.run_method('audio_encoder',(wave,))[0].clone(),expected)
    if not all(result['pass'] for result in results.values()):
        raise SystemExit('Parity failed. See the output report.')


if __name__ == '__main__':
    main()
