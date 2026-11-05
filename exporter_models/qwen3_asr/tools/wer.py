"""WER of {clip: text} JSONs against references (lowercase, alnum + apostrophe).

usage: wer.py refs.json hyp.json [hyp.json ...]   (refs.json = {clip: reference text})
A clip whose word errors exceed half its reference length counts as derailed.
"""
import json
import re
import sys


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", s.lower())).strip().split()


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1]


refs = json.load(open(sys.argv[1]))
for path in sys.argv[2:]:
    hyp = json.load(open(path))
    errors = words = 0
    derailed = []
    for k, h in hyp.items():
        r = norm(refs[k])
        e = edit_distance(r, norm(h))
        errors += e
        words += len(r)
        if e > len(r) // 2:
            derailed.append(k)
    print(f"{path.split('/')[-1]}: {len(hyp)} clips WER {100 * errors / words:.2f}%  derailed clips: {derailed}")
