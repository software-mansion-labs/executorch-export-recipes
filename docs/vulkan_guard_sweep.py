"""Which published Vulkan models would gain from the two AOT fixes?

Reads the op-name strings the Vulkan serializer writes into each delegate blob,
so this needs no device and no re-export. Downloads one representative per model
and deletes it immediately; a whole-fleet checkout would not fit on disk.

  guard  = eq.Scalar + any.dim + full_like + where.self around a softmax, i.e.
           PyTorch's _safe_softmax masked-row guard. Dead without an attention
           mask, and it runs over the full attention matrix.
  views  = aten.view_copy.default count. Only the same-shape ones are removable,
           which this cannot tell apart, so it is an upper bound.
"""
import json, os, subprocess, sys

S = os.path.dirname(os.path.abspath(__file__))
T = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
GUARD = ['aten.eq.Scalar', 'aten.any.dim', 'aten.full_like.default', 'aten.where.self']

def counts(path, ops):
    out = subprocess.run(['strings', '-a', path], capture_output=True, text=True).stdout
    lines = out.split('\n')
    return {op: sum(1 for l in lines if l == op) for op in ops}

plan = json.load(open(f'{S}/vk_sweep_plan.json'))
extra = json.load(open(f'{S}/vk_sweep_extra.json')) if os.path.exists(f'{S}/vk_sweep_extra.json') else []
jobs = [r for r in plan if not r.get('error')] + extra

tmp = f'{S}/sweep_tmp.pte'
results = []
for i, r in enumerate(jobs, 1):
    url = f"https://huggingface.co/{r['repo']}/resolve/{r['rev']}/{r['pick']}"
    rc = subprocess.run(['curl', '-sL', '-H', f'Authorization: Bearer {T}', '-o', tmp, url])
    size = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    # A truncated or error-page download still parses as "no ops found", which
    # would read as a clean model. Compare against the size the tree reported.
    if rc.returncode != 0 or (r.get('bytes') and abs(size - r['bytes']) > 4096):
        print(f"[{i}/{len(jobs)}] {r['model']:34} DOWNLOAD BAD ({size} vs {r.get('bytes')})", flush=True)
        results.append({**r, 'error': f'download {size} vs {r.get("bytes")}'})
        continue
    c = counts(tmp, GUARD + ['aten.view_copy.default', 'aten._softmax.default'])
    guard = min(c[g] for g in GUARD)
    row = {'model': r['model'], 'pick': r['pick'], 'rev': r['rev'], 'mb': round(size/1e6, 1),
           'guard': guard, 'softmax': c['aten._softmax.default'],
           'views': c['aten.view_copy.default']}
    results.append(row)
    print(f"[{i}/{len(jobs)}] {r['model']:34} guard={guard:3} softmax={row['softmax']:3} "
          f"views={row['views']:4}  {row['mb']:7.1f} MB", flush=True)
    os.remove(tmp)

json.dump(results, open(f'{S}/vk_sweep_results.json', 'w'), indent=1)
print('\nSWEEPDONE')
