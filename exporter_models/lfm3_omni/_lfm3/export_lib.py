"""LFM3 Omni multimethod export, including speech generation and synthesis."""
import json
import os
import shutil
from pathlib import Path

import torch
from torch import nn

SOURCE_REPO = 'LiquidAI/LFM3-Omni-2B'
SOURCE_REVISION = '21d75123f86c1f7ffb4bef1ea202710c4e7ebe67'


def resolve_source():
    local = os.environ.get('LFM3_OMNI_SOURCE')
    if local:
        return Path(local).expanduser().resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(SOURCE_REPO, revision=SOURCE_REVISION,
        allow_patterns=['*.json', '*.jinja', '*.safetensors', 'LICENSE']))


class TokenEmbedding(nn.Module):
    def __init__(self, embedding):
        super().__init__(); self.embedding = embedding

    def forward(self, tokens):
        return self.embedding(tokens).float()


class TextDecoder(nn.Module):
    def __init__(self, lm):
        super().__init__(); self.lm = lm

    def forward(self, embeddings, positions):
        hidden = self.lm.decode(embeddings.to(self.lm.embed_tokens.weight.dtype), positions)[:, -1]
        # Speech heads need the same final-normalized hidden state as lm_head.
        return self.lm.lm_head(hidden).float(), hidden.float()


def build_all(source, backend, context):
    from safetensors.torch import load_file
    from .load import build, read_safetensors
    from .audio import AudioEncoder, EncoderConfig, convert as audio_state, mel_filters, COMPUTED_BUFFERS
    from .vision import VisionEncoder, VisionConfig, convert as vision_state
    from .speech import DepthDecoder, AudioEmbedding, Codec
    hf = json.loads((source / 'config.json').read_text())
    state = read_safetensors(source)
    lm = build(source, prefix='model.language_model.', hf_state=state, dtype=None,
        max_context_len=context, attn_impl='mlx' if backend == 'mlx' else 'llama',
        window_slice=backend == 'xnnpack')
    vision = VisionEncoder(hf, (512, 512))
    vision.load_state_dict({k:v.float() for k,v in vision_state(state, VisionConfig.from_hf(hf)).items()})
    audio = AudioEncoder(hf, mel_filters())
    missing, extra = audio.load_state_dict({k:v.float() for k,v in audio_state(state, EncoderConfig.from_hf(hf)).items()}, strict=False)
    if set(missing) != set(COMPUTED_BUFFERS) or extra:
        raise RuntimeError(f'audio mismatch: {missing}, {extra}')
    depth = DepthDecoder(hf)
    depth.load_state_dict({k.removeprefix('model.depth_decoder.'):v.float() for k,v in state.items() if k.startswith('model.depth_decoder.')})
    embedding = AudioEmbedding(hf)
    embedding.codebook_embedding.weight = nn.Parameter(state['model.codebook_embedding.weight'].float())
    codec = Codec()
    weights = load_file(str(source / 'audio_tokenizer/model.safetensors'))
    prefixes = {'quantizer.project_out.':'project_out.', 'from_quantizer.':'from_quantizer.',
        'decoder.lfm.':'trunk.', 'decoder.lin.':'lin.'}
    converted = {}
    for k, value in weights.items():
        for prefix, target in prefixes.items():
            if k.startswith(prefix): converted[target + k[len(prefix):]] = value.float()
    missing, extra = codec.load_state_dict(converted, strict=False)
    if set(missing) != {'inverse_dft', 'envelope_window'} or extra:
        raise RuntimeError(f'codec mismatch: {missing}, {extra}')
    return {'vision_encoder':vision.eval(), 'audio_encoder':audio.eval(),
        'token_embedding':TokenEmbedding(lm.embed_tokens), 'text_decoder':TextDecoder(lm),
        'depth_decoder':depth.eval(), 'audio_embedding':embedding.eval(), 'audio_decoder':codec.eval()}, lm


def schema(variant):
    result = {}
    dynamic = {'token_embedding': [('inputs',0,1,1,256),('outputs',0,1,1,256)],
        'text_decoder':[('inputs',0,1,1,256),('inputs',1,0,1,256)],
        'audio_encoder':[('inputs',0,1,400,479920),('outputs',0,1,1,375)],
        'audio_decoder':[('inputs',0,2,1,256),('outputs',0,1,1920,491520)]}
    for name, method in variant['methods'].items():
        sides = {side:[{'kind':'Tensor','dtype':t['dtype'],'shape':[{'kind':'constant','value':d} for d in t['shape']]} for t in method[side]] for side in ['inputs','outputs']}
        for side, i, axis, lo, hi in dynamic.get(name, []):
            step = 160 if name=='audio_encoder' and side=='inputs' else 1920 if name=='audio_decoder' and side=='outputs' else 1
            sides[side][i]['shape'][axis] = {'kind':'range','range':{'min':lo,'max':hi,'step':step}}
        result[name] = {**sides, 'runtimeConstraints':[]}
    return json.dumps(result)


def export(config, out_dir, backend):
    from torch.export import Dim
    from .lowering import lower, quantize_layerwise
    if backend == 'mlx':
        from executorch.backends.mlx import custom_ops
    else:
        from executorch.extension.llm.custom_ops import custom_ops
    variant = config['variant']; consts = variant['constants'].copy()
    source = resolve_source()
    modules, lm = build_all(source, backend, consts['get_max_context_len'])
    if backend == 'mlx':
        from executorch.backends.mlx.llm.quantization import quantize_model_
        for m in modules.values(): m.to(torch.bfloat16)
        modules['audio_encoder'].log_mel.float()
        # Codec math/DFT and FSQ constants require original dtypes/precision.
        modules['audio_decoder'].float()
        bits = {'int4':'4w','int8':'8w'}[variant['precision']]
        quantize_model_(lm, qlinear_config=bits, qlinear_group_size=64, qembedding_config=bits, qembedding_group_size=64)
    else:
        quantize_layerwise(lm, qlinear_config='8da4w', qlinear_group_size=32, qembedding_config='8w', qembedding_group_size=0)
    eps = {}
    with torch.no_grad():
        for name, module in modules.items():
            inputs = tuple(torch.zeros(t['shape'], dtype=getattr(torch, t['dtype'])) for t in variant['methods'][name]['inputs'])
            dynamic = None
            if name in ('token_embedding','text_decoder'):
                s = Dim(name+'_seq', min=1, max=256)
                dynamic = ({1:s},) if name=='token_embedding' else ({1:s},{0:s})
                if name=='text_decoder': inputs = (inputs[0], torch.arange(256))
            elif name=='audio_encoder':
                k = Dim('hops', min=0, max=2997); dynamic = ({1:160*k+400},)
            elif name=='audio_decoder':
                f = Dim('codec_frames',min=1,max=256); dynamic = ({2:f},)
            eps[name] = torch.export.export(module, inputs, dynamic_shapes=dynamic, strict=True)
            print('captured',name,flush=True)
    consts['get_model_schema'] = schema(variant)
    program = lower(eps, backend, variant, consts)
    out = Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    path = out / variant['file']; path.write_bytes(program.buffer)
    for asset in ['tokenizer.json','tokenizer_config.json','chat_template.jinja','LICENSE']:
        shutil.copyfile(source/asset,out/asset)
    (out/'provenance.json').write_text(json.dumps({'repo':SOURCE_REPO,'revision':SOURCE_REVISION,'backend':backend,'precision':variant['precision']},indent=2))
    print('wrote',path,flush=True)
    return str(path)
