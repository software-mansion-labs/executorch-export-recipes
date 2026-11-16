"""Benchmark exported TDT components on labeled, fixed audio crops.

Fixture JSON contains audio_path, start_seconds, and cohorts with reference_text,
turns (start/end, optional baseline_text, source_tokens), and a label. No audio is
included in this repository. Use licensed audio and retain its attribution.
"""
import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import time
from pathlib import Path
import re

import sentencepiece as spm
import soundfile as sf
import torch
import numpy as np
from validate import greedy
from _model import SOURCE_REVISION


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def wer(reference, hypothesis):
    a, b = words(reference), words(hypothesis)
    costs = list(range(len(b)+1))
    for i, x in enumerate(a):
        row = [i+1] + [0]*len(b)
        for j, y in enumerate(b):
            row[j+1] = min(row[j]+1, costs[j+1]+1, costs[j]+(x != y))
        costs = row
    return dict(reference_words=len(a), hypothesis_words=len(b), edits=costs[-1], wer=costs[-1]/max(1,len(a)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--pte-dir', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    assert args.iterations >= 3
    torch.set_num_threads(4)
    fixture = json.loads(Path(args.fixture).read_text())
    from executorch.runtime import Runtime
    from executorch.extension.pybindings.portable_lib import _unsafe_reset_threadpool, _threadpool_get_thread_count
    _unsafe_reset_threadpool(4)
    assert _threadpool_get_thread_count()==4
    runtime = Runtime.get()
    methods, programs = {}, {}
    started = time.perf_counter()
    for name in ['preprocess','subsample','encoder','predictor','joint']:
        program=runtime.load_program(str(Path(args.pte_dir)/f'fluid_speech_pro_{name}_xnnpack_fp32.pte'))
        programs[name]=program
        methods[name]=({r:program.load_method(f'forward_{r}') for r in range(8)} if name=='subsample' else program.load_method('forward'))
    load_ms=(time.perf_counter()-started)*1000
    tokenizer=spm.SentencePieceProcessor(model_file=args.tokenizer)
    audio,sr=sf.read(fixture['audio_path'],start=round(fixture.get('start_seconds',0)*16000),dtype='float32')
    assert sr==16000 and audio.ndim==1
    def run(name,*inputs):
        method=methods[name][inputs[0].shape[-1]%8] if name=='subsample' else methods[name]
        return [x.clone() for x in method.execute([x.contiguous() for x in inputs])]
    def transcribe(crop):
        assert 2560<=len(crop)<=160000,len(crop)
        # NeMo TranscriptionTensorDataset pads short tensor inputs to one
        # second, symmetrically. Preserve this source pipeline behavior.
        if len(crop)<16000:
            missing=16000-len(crop)
            crop=np.pad(crop,(missing//2,missing-missing//2))
        x=torch.from_numpy(crop.copy()).unsqueeze(0)
        f,fl=run('preprocess',torch.nn.functional.pad(x,(0,160000-len(crop))),torch.tensor([len(crop)]))
        f=f[:,:,:len(crop)//160+1].contiguous()
        return greedy(lambda *v:run('encoder',*run('subsample',*v)),lambda *v:run('predictor',*v),lambda *v:run('joint',*v),f,fl)
    report=dict(source_revision=SOURCE_REVISION,runtime='ExecuTorch 1.4.1 XNNPACK host CPU',torch=torch.__version__,platform=platform.platform(),threads=4,load_ms=load_ms,short_audio_policy='NeMo3 tensor transcription: symmetric zero padding to1second',cohorts=[],latency=[],limits=fixture.get('limits',''))
    with torch.inference_mode():
        transcribe(audio[:16000])
        for cohort in fixture['cohorts']:
            results=[]
            for i,turn in enumerate(cohort['turns']):
                crop=audio[math.floor(turn['start']*16000):math.ceil(turn['end']*16000)].copy()
                start=time.perf_counter();ids,capped=transcribe(crop);elapsed=(time.perf_counter()-start)*1000
                row=dict(start=turn['start'],end=turn['end'],text=tokenizer.decode(ids),token_ids=ids,elapsed_ms=elapsed,capped=capped,audio_sha256=hashlib.sha256(crop.tobytes()).hexdigest())
                if 'source_tokens' in turn:row['exact_source_tokens']=ids==turn['source_tokens']
                results.append(row);print(cohort['label'],i,round(elapsed,2),row['text'],flush=True)
            item=dict(label=cohort['label'],turns=results,score=wer(cohort['reference_text'],' '.join(x['text'] for x in results)),total_asr_ms=sum(x['elapsed_ms'] for x in results),exact_source_token_crops=sum(x.get('exact_source_tokens',False) for x in results))
            if all('baseline_text' in t for t in cohort['turns']):
                item['historical_lfm_score']=wer(cohort['reference_text'],' '.join(t['baseline_text'] for t in cohort['turns']))
            report['cohorts'].append(item)
            print('SUMMARY',cohort['label'],item['score'],flush=True)
        for label,crop in [('one-second',audio[:16000]),('five-seconds',audio[:80000]),('ten-seconds',audio[:160000])]:
            for _ in range(3):transcribe(crop)
            measurements=[]
            for _ in range(args.iterations):
                t=time.perf_counter();transcribe(crop);measurements.append((time.perf_counter()-t)*1000)
            p95=float(torch.quantile(torch.tensor(measurements,dtype=torch.float64),.95))
            report['latency'].append(dict(label=label,samples=len(crop),warmups=3,iterations=args.iterations,median_ms=statistics.median(measurements),p95_ms=p95,raw_ms=measurements))
        # Silence is observed and retained; no assumed VAD result hides model output.
        ids,capped=transcribe(np.zeros(16000,dtype='float32'))
        report['silence']=dict(text=tokenizer.decode(ids),token_ids=ids,capped=capped)
    report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    Path(args.report).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
