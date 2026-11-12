"""Component and streaming parity against the original pinned HF eager model.

Example: uv run --locked --project exporter_models/nemotron3_diarization python
exporter_models/nemotron3_diarization/validate.py --pte /path/model.pte
--audio /path/IB4001.Mix-Headset.wav --start 210 --report /path/report.json
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import soundfile as sf
import torch


def compare(name, actual, expected, records, atol=1e-4, rtol=1e-3):
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (name, actual.dtype, expected.dtype)
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    difference = (actual - expected).abs()
    failures = (~torch.isclose(actual, expected, atol=atol, rtol=rtol)).sum().item()
    item = dict(name=name, shape=list(actual.shape), max_abs=difference.max().item() if difference.numel() else 0.,
                max_relative=(difference / expected.abs().clamp(min=1e-6)).max().item() if difference.numel() else 0.,
                outside_budget=failures, atol=atol, rtol=rtol)
    records.append(item)
    print(json.dumps(item), flush=True)
    assert failures == 0, item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pte', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--chunks', type=int, default=6)
    parser.add_argument('--checkpoint')
    parser.add_argument('--report', required=True)
    parser.add_argument('--skip-boundaries', action='store_true')
    args = parser.parse_args()
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    root = Path(__file__).parent
    spec = importlib.util.spec_from_file_location('nemotron_validation_model', root / '_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from executorch.runtime import Runtime
    from transformers.models.nemotron3_diarization.modeling_nemotron3_diarization import Nemotron3DiarizationSpeakerCache

    model, processor = module.load_source(args.checkpoint)
    original_policy = Nemotron3DiarizationSpeakerCache(model.config.streaming_config)
    program = Runtime.get().load_program(args.pte)
    methods = {name: program.load_method(name) for name in ['preprocess', 'embed', 'encode', 'compress']}
    execute = lambda name, inputs: [x.clone() for x in methods[name].execute([x.contiguous() for x in inputs])]
    records = []
    report = dict(source_revision=module.SOURCE_REVISION, pte=str(Path(args.pte).resolve()),
                  method_names=sorted(program.method_names), records=records)
    for name, value in [('sample_rate',16000),('max_num_of_spks',8),('spkcache_len',264),('fifo_len',264),('cache_update_period',222)]:
        assert program.load_method(name).execute([]) == [value], name
    report['metadata'] = {name: str(method.metadata) for name,method in methods.items()}
    report['schema'] = json.loads(program.load_method('get_model_schema').execute([])[0])
    audio, sr = sf.read(args.audio, start=round(args.start*16000), frames=args.chunks*160000, dtype='float32')
    assert sr == 16000, sr
    waveform = torch.from_numpy(audio)
    if waveform.ndim > 1:
        waveform = waveform.mean(-1)
    assert len(waveform) >= 1600
    try:
        with torch.inference_mode():
            if not args.skip_boundaries:
                for samples in [1600,5120,159999,160000]:
                    clip = waveform[:samples]
                    if len(clip) != samples: continue
                    padded = torch.nn.functional.pad(clip, (0,160000-samples))
                    features = execute('preprocess', [padded,torch.tensor([samples])])[0]
                    reference = processor.feature_extractor(clip.numpy(),sampling_rate=sr,return_tensors='pt')
                    frames = reference.input_features.shape[1]
                    compare(f'features-{samples}',features[:,:frames],reference.input_features,records,atol=1e-4,rtol=1e-4)
                for groups in [1,2,126]:
                    features = torch.randn(1,groups*8,128)
                    compare(f'embed-{groups}', execute('embed',[features])[0], module.Embed(model)(features), records,rtol=1e-4)
                for length in [2,126,800]:
                    embeddings = torch.randn(1,length,512)
                    valid = torch.ones(1,length,dtype=torch.int64)
                    valid[:,max(1,length-3):] = 0
                    actual=execute('encode',[embeddings,valid]); expected=module.Encode(model)(embeddings,valid)
                    for i in range(2):compare(f'encode-{length}-{i}',actual[i],expected[i],records)
                for length in [265,266,400,800]:
                    embeddings=torch.randn(1,length,512); probabilities=torch.rand(1,length,8)
                    actual=execute('compress',[embeddings,probabilities])
                    expected=original_policy._compress(embeddings,probabilities,model.silence_embeds)
                    for i in range(2):compare(f'compress-{length}-{i}',actual[i],expected[i],records,atol=0,rtol=0)

            reference_cache = None
            cache = torch.zeros(1,0,512); stored_probs=torch.zeros(1,0,8)
            fifo = torch.zeros(1,0,512); compressed=False
            all_probs=[]; started=time.perf_counter(); compression_count=0
            for index,clip in enumerate(waveform.split(160000)):
                if len(clip)<1600:break
                features = execute('preprocess',[torch.nn.functional.pad(clip,(0,160000-len(clip))),torch.tensor([len(clip)])])[0]
                source = processor.feature_extractor(clip.numpy(),sampling_rate=sr,return_tensors='pt')
                frames=source.input_features.shape[1]
                compare(f'stream-features-{index}',features[:,:frames],source.input_features,records,rtol=1e-4)
                mel=torch.nn.functional.pad(features[:,:frames],(0,0,0,(-frames)%8)).contiguous()
                chunk=execute('embed',[mel])[0]
                num_chunk=chunk.shape[1]; cached=cache.shape[1]+fifo.shape[1]
                combined=torch.cat([cache,fifo,chunk],dim=1).contiguous()
                valid=torch.cat([torch.ones(1,cached,dtype=torch.int64),torch.nn.functional.pad(source.attention_mask,(0,(-frames)%8))[:,::8].long()],dim=1)
                predictions,pooled=execute('encode',[combined,valid])
                actual=predictions[:,cached*8:cached*8+frames]
                reference=model(**source,speaker_cache=reference_cache,num_lookahead_frames=0)
                reference_cache=reference.speaker_cache
                expected=reference.logits.sigmoid()
                compare(f'stream-output-{index}',actual,expected,records)
                all_probs.append(actual[:,:len(clip)//160].clone())
                fifo_and_chunk=torch.cat([fifo,chunk],dim=1)
                popped=0 if fifo_and_chunk.shape[1]<=264 else min(max(222,fifo_and_chunk.shape[1]-264),fifo_and_chunk.shape[1])
                if popped:
                    old_probs=stored_probs if compressed else pooled[:,:cache.shape[1]]
                    new_cache=torch.cat([cache,fifo_and_chunk[:,:popped]],dim=1)
                    new_probs=torch.cat([old_probs,pooled[:,cache.shape[1]:cache.shape[1]+popped]],dim=1)
                    if new_cache.shape[1]>264:
                        new_cache,new_probs=execute('compress',[new_cache,new_probs]); compressed=True;compression_count+=1
                    cache,stored_probs=new_cache,new_probs
                fifo=fifo_and_chunk[:,popped:].contiguous()
                compare(f'stream-cache-{index}',cache,reference_cache.embeds[:,:reference_cache.num_cache_frames],records)
                compare(f'stream-fifo-{index}',fifo,reference_cache.fifo[:,:reference_cache.num_fifo_frames],records)
            probabilities=torch.cat(all_probs,dim=1)
            report.update(stream_seconds=len(waveform)/sr,comparison_elapsed_seconds=time.perf_counter()-started,
                          cache_compressions=compression_count,active_speaker_frames=(probabilities[0]>=.5).sum(0).tolist())
            torch.save(probabilities,Path(args.report).with_suffix('.probabilities.pt'))
            report['passed']=True
    finally:
        Path(args.report).parent.mkdir(parents=True,exist_ok=True)
        Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['records','metadata','schema']}),flush=True)

if __name__ == '__main__':
    main()
