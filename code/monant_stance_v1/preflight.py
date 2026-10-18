"""Validate frozen private input package and describe a prospective plan; no training."""
import argparse,json
from pathlib import Path
from run_stance import load_inputs,SPEC
from stance_kernels import fingerprint
ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);args=ap.parse_args();d,m=load_inputs(args.input_dir)
assert set(d.split)=={'train','validation','test'}
assert len(d[d.split=='test'])==308
for part in ['train','validation','test']:assert set(d[d.split==part].label_id)=={0,1,2}
print(json.dumps({'input_sha256':m['sha256'],'rows':len(d),'split_counts':m['split_counts'],'prospective_training_jobs':len(SPEC['models'])*len(SPEC['losses'])*len(SPEC['seeds']),'spec_sha256':fingerprint(SPEC),'training_executed':False},indent=2))
