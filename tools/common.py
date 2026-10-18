from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()
def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def verify_sources():
    manifest=read_json(ROOT/'data/source_manifest.json')
    for entry in manifest['files']:
        p=ROOT/entry['path']
        if not p.is_file() or sha256(p)!=entry['sha256']:raise ValueError('Frozen source mismatch: '+entry['path'])
    return manifest
