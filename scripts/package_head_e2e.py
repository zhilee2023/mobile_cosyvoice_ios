"""Copy only the two optional speech-head assets; never overwrite originals."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[1]/'speech-head-manifest.json').read_text())
    entries=[manifest[k] for k in ('candidateGraph','candidateWeight')]
    for entry in entries:
        source=args.source/entry['file']
        if source.stat().st_size!=entry['bytes'] or sha(source)!=entry['sha256']:raise ValueError('Candidate source identity mismatch')
    args.output.mkdir(parents=True,exist_ok=False)
    for entry in entries:
        target=args.output/entry['file'];shutil.copyfile(args.source/entry['file'],target)
        if sha(target)!=entry['sha256']:raise ValueError('Copied asset hash mismatch')
    print(json.dumps({'directory':str(args.output.resolve()),'files':entries,'total_bytes':sum(e['bytes'] for e in entries)},indent=2))
