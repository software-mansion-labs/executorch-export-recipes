"""Device bench log -> {clip: text} JSON + timing summary. usage: device_wer.py bench.log out.json"""
import json, re, statistics, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")
res, enc, pre, dec, prefill = {}, [], [], [], []
for line in open(sys.argv[1]):
    m = re.match(r"(\S+)\.f32 pre ([\d.]+) enc ([\d.]+) prefill ([\d.]+) \((\d+) tok\) decode ([\d.]+) ms/tok \((\d+)\) ids ?(.*)", line)
    if not m: continue
    ids = [int(x) for x in m[8].split(",") if x]
    raw = tok.decode(ids)
    res[m[1]] = (raw.split("<asr_text>", 1)[1] if "<asr_text>" in raw else raw).strip()
    pre.append(float(m[2])); enc.append(float(m[3])); prefill.append(float(m[4])); dec.append(float(m[6]))
json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
med = statistics.median
print(f"{len(res)} clips | median preprocess {med(pre):.1f} ms, encoder {med(enc):.1f} ms, prefill {med(prefill):.1f} ms, decode {med(dec):.2f} ms/tok")
