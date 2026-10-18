"""Run only the exact, explicitly approved 36-job replication plan."""
import argparse
from pathlib import Path
from study_common import read,digest
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--input-dir',required=True);ap.add_argument('--execute',action='store_true');ap.add_argument('--approved-plan-sha256',required=True);a=ap.parse_args()
    out=Path(a.out)
    if not a.execute:ap.error('Training requires explicit approval and --execute')
    if digest(out/'run_plan.json')!=a.approved_plan_sha256:ap.error('Approval does not match this frozen plan')
    plan=read(out/'run_plan.json');spec=read(out/'contract.json')['spec']
    assert len(plan['jobs'])==36 and all(j['model']=='tbase' and j['split_seed']==42 and j['updates']==30 for j in plan['jobs'])
    assert spec['encoder']['name']=='google-bert/bert-base-uncased' and spec['encoder']['revision']=='86b5e0934494bd15c9632b12f734a8a67f723594'
    import torch
    if not torch.cuda.is_available():raise SystemExit('Select a GPU runtime in Colab. Training has not started.')
    from engine import Runner
    runner=Runner(a.input_dir,out,'cuda')
    for job in plan['jobs']:runner.execute(job,resume=True)
    from verify_replication import verify_replication
    verify_replication(a.input_dir,out)
if __name__=='__main__':main()
