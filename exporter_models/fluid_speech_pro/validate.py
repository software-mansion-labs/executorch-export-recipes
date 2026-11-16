"""Verify full FP32 components against the original NeMo model and greedy text.

Run in this model's locked environment. Host CPU validation does not validate iOS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import resource

import torch
import soundfile as sf
from _model import load_source, Features, Encode, Predict, Joint, prepare_capture, SOURCE_REVISION


def greedy(encode, predict, joint, features, length):
    encoded, lengths = encode(features, length)
    hidden = torch.zeros(2, 1, 640)
    cell = torch.zeros_like(hidden)
    token = torch.tensor([[8192]])
    prediction, next_h, next_c = predict(token, hidden, cell)
    tokens, frame, symbols, capped = [], 0, 0, 0
    while frame < int(lengths[0]):
        logits = joint(encoded[:, :, frame:frame+1].transpose(1, 2).contiguous(), prediction)[0].reshape(-1)
        label = int(logits[:8193].argmax())
        skip = int(logits[8193:].argmax())
        if label != 8192:
            tokens.append(label)
            hidden, cell = next_h, next_c
            prediction, next_h, next_c = predict(torch.tensor([[label]]), hidden, cell)
        symbols += 1
        frame += skip
        if skip > 0:
            symbols = 0
        elif label == 8192 or symbols >= 10:
            frame += 1
            symbols = 0
            capped += label != 8192
        if len(tokens) > 1024:
            raise RuntimeError('TDT exceeded bounded token budget')
    return tokens, capped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pte-dir', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--start', type=float, default=210)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    from nemo.core.classes import typecheck
    from executorch.runtime import Runtime
    from executorch.extension.pybindings.portable_lib import _unsafe_reset_threadpool, _threadpool_get_thread_count
    _unsafe_reset_threadpool(4)
    assert _threadpool_get_thread_count()==4
    model = load_source(args.source)
    runtime = Runtime.get()
    programs, methods = {}, {}
    records = []
    report = dict(source_revision=SOURCE_REVISION, threads=4, checks=records, artifacts={}, runtime='ExecuTorch 1.4.1 host CPU',passed=False)
    for name in ['preprocess','predictor','joint','subsample','encoder']:
        path = Path(args.pte_dir) / f'fluid_speech_pro_{name}_xnnpack_fp32.pte'
        programs[name] = runtime.load_program(str(path))
        if name=='subsample':
            methods[name]={r:programs[name].load_method(f'forward_{r}') for r in range(8)}
        else:
            methods[name] = programs[name].load_method('forward')
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8*1024*1024), b''): hasher.update(block)
        digest = hasher.hexdigest()
        report['artifacts'][name] = dict(bytes=path.stat().st_size,sha256=digest,methods=sorted(programs[name].method_names),metadata=str({r:str(m.metadata) for r,m in methods[name].items()}) if name=='subsample' else str(methods[name].metadata))
    def run(name, *inputs):
        method=methods[name][inputs[0].shape[-1]%8] if name=='subsample' else methods[name]
        return [x.clone() if isinstance(x,torch.Tensor) else x for x in method.execute([x.contiguous() for x in inputs])]
    def compare(name, actual, expected, atol=1e-3,rtol=1e-3):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype,(name,actual.shape,expected.shape,actual.dtype,expected.dtype)
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all(),name
        if not actual.is_floating_point():
            assert torch.equal(actual,expected),name
            records.append(dict(name=name,exact=True));return
        diff=(actual-expected).abs()
        bad=int((~torch.isclose(actual,expected,atol=atol,rtol=rtol)).sum())
        item=dict(name=name,max_abs=float(diff.max()),max_relative=float((diff/expected.abs().clamp_min(1e-6)).max()),outside_budget=bad,atol=atol,rtol=rtol,shape=list(actual.shape))
        records.append(item);print(json.dumps(item),flush=True)
        assert bad==0,item
    audio,sr=sf.read(args.audio,start=round(args.start*16000),frames=160000,dtype='float32')
    assert sr==16000 and audio.ndim==1
    try:
        with torch.inference_mode(),typecheck.disable_checks():
            for samples in [5120,16000,80001,160000]:
                x=torch.from_numpy(audio[:samples].copy()).unsqueeze(0)
                lengths=torch.tensor([samples])
                padded=torch.nn.functional.pad(x,(0,160000-samples))
                actual,flen=run('preprocess',padded,lengths)
                expected,expected_len=Features(model)(x,lengths)
                frames=expected.shape[-1]
                compare(f'features-{samples}',actual[:,:,:frames],expected,atol=1e-4,rtol=1e-4)
                compare(f'length-{samples}',flen,expected_len)
                reference,ref_len=Encode(model)(expected,expected_len)
                embeddings,sub_len=run('subsample',actual[:,:,:frames],flen)
                source_sub,source_sub_len=model.encoder.pre_encode(expected.transpose(1,2),expected_len)
                compare(f'subsample-{samples}',embeddings,source_sub)
                compare(f'subsample-length-{samples}',sub_len,source_sub_len)
                encoded,encoded_len=run('encoder',embeddings,sub_len)
                compare(f'encoder-{samples}',encoded,reference)
                compare(f'encoder-length-{samples}',encoded_len,ref_len)
            # Every residue method, at its declared lower and upper time bound.
            full_features,_=Features(model)(torch.from_numpy(audio.copy()).unsqueeze(0),torch.tensor([160000]))
            for frames in [*range(17,25),*range(994,1002)]:
                feature=full_features[:,:,:frames].contiguous()
                valid=torch.tensor([frames-1])
                source_sub,source_len=model.encoder.pre_encode(feature.transpose(1,2),valid)
                actual_sub,actual_len=run('subsample',feature,valid)
                compare(f'boundary-subsample-{frames}',actual_sub,source_sub)
                compare(f'boundary-subsample-length-{frames}',actual_len,source_len)
                original,original_len=model.encoder.forward_internal(source_sub,source_len,bypass_pre_encode=True)
                actual,actual_encoder_len=run('encoder',actual_sub,actual_len)
                compare(f'boundary-encoder-{frames}',actual,original)
                compare(f'boundary-encoder-length-{frames}',actual_encoder_len,original_len)
            for label in [8192,2060,297]:
                token=torch.tensor([[label]])
                h,c=torch.randn(2,1,640)*.1,torch.randn(2,1,640)*.1
                expected,state=model.decoder.predict(token,state=(h,c),add_sos=False)
                got=run('predictor',token,h,c)
                for i,ref in enumerate([expected,*state]):compare(f'predictor-{label}-{i}',got[i],ref)
                acoustic=reference[:,:,:1].transpose(1,2).contiguous()
                before=model.joint.log_softmax
                model.joint.log_softmax=False
                try:ref=model.joint.joint(acoustic,expected)
                finally:model.joint.log_softmax=before
                compare(f'joint-{label}',run('joint',acoustic,expected)[0],ref)
            for samples in [16000,80000,160000]:
                x=torch.from_numpy(audio[:samples].copy()).unsqueeze(0);length=torch.tensor([samples])
                f,fl=run('preprocess',torch.nn.functional.pad(x,(0,160000-samples)),length)
                f=f[:,:,:samples//160+1].contiguous()
                tokens,capped=greedy(lambda *v:run('encoder',*run('subsample',*v)),lambda *v:run('predictor',*v),lambda *v:run('joint',*v),f,fl)
                original=model.transcribe([x[0].numpy()],batch_size=1,verbose=False)[0]
                exact=tokens==original.y_sequence.tolist()
                item=dict(name=f'greedy-{samples}',exact=exact,actual_tokens=tokens,reference_tokens=original.y_sequence.tolist(),actual_text=model.tokenizer.ids_to_text(tokens),reference_text=original.text,capped=capped)
                records.append(item);print(json.dumps(item),flush=True)
                assert exact,item
            report['passed']=True
    finally:
        report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        Path(args.report).write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
