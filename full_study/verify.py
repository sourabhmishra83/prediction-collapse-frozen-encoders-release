"""Recompute stored predictions and check plan, graph and selection provenance."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from study_common import read,write,digest,fingerprint,measures,HERE

def verify(out,allow_partial=False,allow_smoke=False):
    out=Path(out);contract=read(out/'contract.json');plan=read(out/'run_plan.json');spec=contract['spec']
    assert plan['contract_sha256']==fingerprint(contract)
    rows=[];missing=[];initials={};pair_cohorts={}
    for job in plan['jobs']:
        assert job['job_sha256']==fingerprint({k:v for k,v in job.items() if k!='job_sha256'})
        split=pd.read_csv(out/job['split']);cohort=pd.read_csv(out/job['cohort']).row_index
        assert digest(out/job['split'])==job['split_sha256'] and digest(out/job['cohort'])==job['cohort_sha256']
        assert split.row_index.is_unique and split.groupby('text_group').split.nunique().max()==1
        pool=split[split.split=='train'];chosen=pool.set_index('row_index').loc[cohort]
        assert cohort.is_unique and len(chosen)==job['fixed_n']
        assert int(chosen.label.sum())==int(round(job['ratio']*job['fixed_n']))
        folder=out/'results'/job['job_id']
        if not (folder/'result.json').exists():missing.append(job['job_id']);continue
        result=read(folder/'result.json');assert result['job']==job
        if result['synthetic_smoke_only'] and not allow_smoke:raise ValueError('Synthetic smoke outputs are not study results')
        for name,h in result['files_sha256'].items():assert digest(folder/name)==h
        trajectory=read(folder/'trajectory.json')['updates'];assert len(trajectory)==job['updates']
        assert [r['optimizer_update'] for r in trajectory]==list(range(1,job['updates']+1))
        best=max(trajectory,key=lambda r:r['source_validation']['macro_f1'])
        assert best['optimizer_update']==result['selected_optimizer_update']
        assert best['source_validation']==result['selected_validation']
        assert trajectory[-1]['source_validation']==result['final_validation']
        pair=(job['family'],job['source'],job['split_seed'],job['ratio'],job['seed'],job['model'])
        if pair in initials:assert initials[pair]==result['initial_parameter_sha256']
        initials[pair]=result['initial_parameter_sha256']
        pair_cohorts.setdefault(pair,set()).add(job['cohort_sha256']);assert len(pair_cohorts[pair])==1
        source_groups=set(pool.text_group)
        for target in spec['sources']:
            target_split=pd.read_csv(out/f'splits/{target}_s{job["split_seed"]}.csv')
            expected=target_split[(target_split.split=='test')&(~target_split.text_group.isin(source_groups))]
            pred=pd.read_csv(folder/f'test_{target}.csv')
            assert pred.row_index.tolist()==expected.row_index.tolist() and pred.label.tolist()==expected.label.tolist()
            assert not set(expected.text_group)&source_groups
            for endpoint in ['initial','selected','final']:
                assert set(pred[endpoint+'_prediction'])<={0,1}
                m=measures(pred.label.to_numpy(),pred[endpoint+'_prediction'].to_numpy(),spec['collapse_threshold'])
                saved=result['test_targets'][target][endpoint]
                for key,value in m.items():
                    if isinstance(value,(float,int,bool)):assert abs(float(value)-float(saved[key]))<1e-10
                    else:assert value==saved[key]
                rows.append({'job_id':job['job_id'],'family':job['family'],'source':job['source'],'target':target,
                    'split_seed':job['split_seed'],'ratio':job['ratio'],'seed':job['seed'],'model':job['model'],
                    'loss':job['loss'],'batch_size':str(job['batch_size']),'updates':job['updates'],'endpoint':endpoint,
                    'macro_f1':m['macro_f1'],'mcc':m['mcc'],'class0_recall':m['class0_recall'],'class1_recall':m['class1_recall'],
                    'collapsed':m['collapsed'],'in_domain':target==job['source']})
    for folder in (out/'graph_records').glob('*') if (out/'graph_records').exists() else []:
        record=read(folder/'source_reference.json');n=record['reference_rows'];edges=np.load(folder/'training_edges.npz')['edges']
        import hashlib
        assert hashlib.sha256(edges.tobytes()).hexdigest()==record['training_edges_sha256']
        assert not edges.size or (edges.min()>=0 and edges.max()<n)
        for p in folder.glob('*_query_edges.npz'):
            edges=np.load(p)['edges']
            assert not edges.size or (edges[0].min()>=n and edges[1].max()<n and edges[1].min()>=0)
    report={'planned_jobs':len(plan['jobs']),'completed_jobs':len(plan['jobs'])-len(missing),'complete':not missing,
            'synthetic_smoke_only':allow_smoke,'new_training_executed_by_verifier':False,'all_completed_predictions_verified':True}
    write(out/'analysis/verification.json',report)
    if rows:
        frame=pd.DataFrame(rows);frame.to_csv(out/'analysis/verified_endpoints.csv',index=False)
        # Keep split-specific seed summaries. Reusing seeds across splits/ratios does
        # not turn all rows into independent population-level observations.
        keys=['family','source','target','split_seed','ratio','model','loss','batch_size','updates','endpoint']
        summary=frame.groupby(keys).agg(n=('job_id','size'),macro_f1_mean=('macro_f1','mean'),
            macro_f1_sd=('macro_f1','std'),collapsed_count=('collapsed','sum')).reset_index()
        summary.to_csv(out/'analysis/condition_by_split.csv',index=False)
    if missing and not allow_partial:raise ValueError(f'{len(missing)} jobs missing; partial results must not be described as complete')
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args()
    print(verify(args.out,allow_partial=args.allow_partial))
