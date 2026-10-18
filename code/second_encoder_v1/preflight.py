"""Validate preparation and numerical fixtures without any optimizer step."""
from pathlib import Path
import argparse,json,hashlib
import pandas as pd
from study_common import read,digest,fingerprint,load_inputs,measures,core_lr
def check(input_dir,prepared):
    p=Path(prepared);contract=read(p/'contract.json');plan=read(p/'run_plan.json');data=load_inputs(input_dir,contract['inputs'])
    assert plan['contract_sha256']==fingerprint(contract) and len(plan['jobs'])==36
    assert contract['spec']['encoder']['revision']=='86b5e0934494bd15c9632b12f734a8a67f723594'
    pairs={}
    for j in plan['jobs']:
        assert j['job_sha256']==fingerprint({k:v for k,v in j.items() if k!='job_sha256'})
        assert j['model']=='tbase' and j['split_seed']==42 and j['updates']==30 and j['family']=='architecture'
        for name in ['split','cohort']:assert digest(p/j[name])==j[name+'_sha256']
        split=pd.read_csv(p/j['split']);cohort=pd.read_csv(p/j['cohort']).row_index
        assert split.groupby('text_group').split.nunique().max()==1 and split.row_index.is_unique
        assert split.label.tolist()==data[j['source']].label.tolist() and split.text_group.tolist()==data[j['source']].text_group.tolist()
        pool=split[split.split=='train'].set_index('row_index');chosen=pool.loc[cohort]
        assert cohort.is_unique and len(chosen)==j['fixed_n'] and int(chosen.label.sum())==round(j['ratio']*j['fixed_n'])
        key=(j['source'],j['ratio'],j['seed']);pairs.setdefault(key,[]).append(j['cohort_sha256'])
    assert len(pairs)==18 and all(len(v)==2 and len(set(v))==1 for v in pairs.values())
    baseline=measures([0,0,0,1],[0,0,0,0]);assert baseline['collapsed'] and abs(baseline['macro_f1']-3/7)<1e-12
    assert not measures([0,1],[0,1])['collapsed'] and measures([0,1],[0,1])['macro_f1']==1
    assert core_lr(0,30)==1/3 and core_lr(2,30)==1
    import torch
    from study_common import make_model,seed_everything
    seed_everything(0);a=make_model('tbase',768,256,.3).eval();seed_everything(0);b=make_model('tbase',768,256,.3).eval()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    with torch.no_grad():assert a(torch.zeros(4,768)).shape==(4,2)
    return {'status':'PASS','prepared_jobs':36,'paired_cohorts':18,'optimizer_steps':0,'encoder_downloaded_or_executed':False,'training_executed':False,'checks':['input and split/cohort hashes','row/label/text-group identity','cohort balance and pairing','metric fixtures','learning-rate fixtures','head initialization pairing and forward shape']}
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--prepared',required=True);a=ap.parse_args();print(json.dumps(check(a.input_dir,a.prepared),indent=2))
