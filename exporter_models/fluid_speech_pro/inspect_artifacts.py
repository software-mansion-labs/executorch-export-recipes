"""Inspect serialized delegate/operator tables without reading multi-GB weights."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pte-dir',required=True)
    parser.add_argument('--report',required=True)
    args=parser.parse_args()
    from executorch.exir._serialize._program import _get_extended_header, _flatbuffer_to_program
    result={}
    for path in sorted(Path(args.pte_dir).glob('*.pte')):
        with path.open('rb') as f:
            header=_get_extended_header(f.read(64))
            assert header and header.is_valid()
            f.seek(0);program=_flatbuffer_to_program(f.read(header.program_size))
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        result[path.name]=dict(bytes=path.stat().st_size,sha256=digest.hexdigest(),methods={m.name:dict(delegates=[d.id for d in m.delegates],portable_operators=[o.name+'.'+o.overload for o in m.operators]) for m in program.execution_plan})
    Path(args.report).write_text(json.dumps(result,indent=2))
    print(json.dumps({n:{k:len(v['delegates']) for k,v in d['methods'].items()} for n,d in result.items()},indent=2))

if __name__=='__main__':main()
