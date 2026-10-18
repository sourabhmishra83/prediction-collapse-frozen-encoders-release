from pathlib import Path
import argparse,hashlib,json,shutil

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def materialize(suite,root,dest,local=False):
 suite,root,dest=map(Path,(suite,root,dest));plan=json.loads((suite/'prepared/plan.json').read_text(encoding='utf-8'));receipt=[]
 def put(src,rel,expected):
  assert src.is_file() and sha(src)==expected,f'Missing or mismatched frozen asset: {src}'
  target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert sha(target)==expected,f'Existing destination differs: {target}'
  else:shutil.copyfile(src,target)
  assert sha(target)==expected;receipt.append({'source':str(src),'destination':str(rel),'sha256':expected})
 for meta in plan['inputs'].values():
  matches=[p for p in (root/'full_study/inputs').glob(meta['filename'])] if local else list(root.rglob(meta['filename']))
  matches=[p for p in matches if p.is_file() and p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256']]
  assert matches,f"Exact input missing: {meta['filename']}";put(matches[0],Path('inputs')/meta['filename'],meta['sha256'])
 for c in plan['caches']:
  src=root/'experiments/score_audit_private/cache'/c['encoder']/c['source'] if local else root/c['drive_relative']
  for name,key in [('vectors.pt','sha256'),('metadata.json','metadata_sha256')]:put(src/name,Path('cache')/c['encoder']/c['source']/name,c[key])
 for j in plan['existing_heads']:
  src=(root/('experiments/second_encoder/completed_audit/downloaded/results' if j['encoder']=='BERT' else 'experiments/score_audit_private/roberta_heads')/j['job_id']) if local else root/j['drive_relative']
  for name,key in [('selected_model.pt','head_sha256'),('result.json','result_sha256')]:put(src/name,Path('existing')/j['job_id']/name,j[key])
 (dest/'asset_provenance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(f'Copied and verified {len(receipt)} frozen assets; zero fitting.')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--suite',required=True);a.add_argument('--root',required=True);a.add_argument('--dest',required=True);a.add_argument('--local',action='store_true');v=a.parse_args();materialize(v.suite,v.root,v.dest,v.local)
