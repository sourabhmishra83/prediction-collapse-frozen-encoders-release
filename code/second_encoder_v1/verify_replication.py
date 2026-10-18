"""Check planned rows, per-example predictions, provenance, endpoints and comparisons."""
import argparse
from pathlib import Path
import pandas as pd
from study_common import read,digest,load_inputs,write
from verify import verify
def verify_replication(input_dir,out):
    out=Path(out);contract=read(out/'contract.json');plan=read(out/'run_plan.json');data=load_inputs(input_dir,contract['inputs'])
    assert len(plan['jobs'])==36
    env=read(out/'execution_environment.json');assert not env['synthetic_smoke_only']
    for name,h in env['code_sha256'].items():assert digest(Path(__file__).parent/name)==h
    for job in plan['jobs']:
        split=pd.read_csv(out/job['split']);groups=set(split[split.split=='train'].text_group)
        for target in contract['spec']['sources']:
            targetsplit=pd.read_csv(out/f'splits/{target}_s42.csv');expected=targetsplit[(targetsplit.split=='test')&~targetsplit.text_group.isin(groups)]
            p=pd.read_csv(out/'results'/job['job_id']/f'test_{target}.csv')
            assert p.row_index.tolist()==expected.row_index.tolist() and p.label.tolist()==expected.label.tolist()
            assert p.label.tolist()==data[target].label.iloc[p.row_index].tolist()
            assert all(set(p[c])<={0,1} for c in ['initial_prediction','selected_prediction','final_prediction'])
        folder=out/'results'/job['job_id'];assert all((folder/name).exists() for name in ['initial_model.pt','selected_model.pt','final_model.pt'])
    report=verify(out)
    frame=pd.read_csv(out/'analysis/verified_endpoints.csv');assert len(frame)==324 and frame.job_id.nunique()==36
    old=pd.read_csv(out/'roberta_reference_endpoints.csv');old=old[(old.family=='architecture')&(old.model=='tbase')&(old.split_seed==42)]
    mapping={j['job_id']:j['reference_roberta_job_id'] for j in plan['jobs']};frame['reference_job_id']=frame.job_id.map(mapping)
    paired=frame.merge(old[['job_id','target','endpoint','macro_f1','collapsed']],left_on=['reference_job_id','target','endpoint'],right_on=['job_id','target','endpoint'],suffixes=('_bert','_roberta'),validate='one_to_one')
    assert len(paired)==324;paired['bert_minus_roberta_macro_f1']=paired.macro_f1_bert-paired.macro_f1_roberta
    paired.to_csv(out/'analysis/paired_encoder_endpoints.csv',index=False)
    write(out/'completion.json',{**report,'encoder':'BERT','reference_encoder':'RoBERTa','comparison_is_descriptive':True,'one_split_only':True,'new_training_executed_by_verifier':False})
    return report
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();print(verify_replication(a.input_dir,a.out))
