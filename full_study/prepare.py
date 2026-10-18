"""Create the prospective 432-job plan; no graph encoding or training."""
import argparse
from pathlib import Path
import pandas as pd
from study_common import read,write,digest,fingerprint,load_inputs,allocate_groups,cohort_rows,HERE

def prepare(input_dir,out,spec,manifest):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    contract={'spec':spec,'inputs':manifest}
    if (out/'contract.json').exists() and read(out/'contract.json')!=contract:
        raise ValueError('Existing output has another specification/input snapshot; choose a new output directory')
    data=load_inputs(input_dir,manifest)
    if (out/'run_plan.json').exists():
        plan=read(out/'run_plan.json')
        if plan['contract_sha256']!=fingerprint(contract):raise ValueError('Plan contract mismatch')
        for job in plan['jobs']:
            if fingerprint({k:v for k,v in job.items() if k!='job_sha256'})!=job['job_sha256']:raise ValueError('Job changed')
            for kind in ['split','cohort']:
                if digest(out/job[kind])!=job[kind+'_sha256']:raise ValueError('Frozen split/cohort changed')
        return plan
    if (out/'results').exists():raise ValueError('Results exist without a plan; refusing to recreate')
    write(out/'contract.json',contract);splits={};sizes={};report={}
    for name in spec['sources']:
        feasible=[]
        for split_seed in spec['split_seeds']:
            frame=allocate_groups(data[name],split_seed);path=out/'splits'/f'{name}_s{split_seed}.csv';path.parent.mkdir(exist_ok=True);frame.to_csv(path,index=False)
            splits[name,split_seed]=frame;pool=frame[frame.split=='train'];n0=int((pool.label==0).sum());n1=int((pool.label==1).sum())
            feasible.extend(min(n1/p,n0/(1-p)) for p in spec['ratios'])
            report[f'{name}_s{split_seed}']={'rows':len(frame),'text_groups':frame.text_group.nunique(),'cross_split_groups':int((frame.groupby('text_group').split.nunique()>1).sum()),'row_counts':frame.groupby(['split','label']).size().to_dict()}
            report[f'{name}_s{split_seed}']['row_counts']={f'{a}_class{b}':int(v) for (a,b),v in report[f'{name}_s{split_seed}']['row_counts'].items()}
        sizes[name]=min(spec['preferred_fixed_n'][name],int(.9*min(feasible)))
    jobs=[]
    for name in spec['sources']:
        for split_seed in spec['split_seeds']:
            frame=splits[name,split_seed];split_path=f'splits/{name}_s{split_seed}.csv'
            for ratio in spec['ratios']:
                for seed in spec['training_seeds']:
                    cohort=f'cohorts/{name}_s{split_seed}_p{ratio:.2f}_r{seed}.csv';path=out/cohort;path.parent.mkdir(exist_ok=True)
                    pd.DataFrame({'row_index':cohort_rows(frame,sizes[name],ratio,seed)}).to_csv(path,index=False)
                    shared={'source':name,'split_seed':split_seed,'ratio':ratio,'seed':seed,'fixed_n':sizes[name],
                        'input_sha256':manifest[name]['sha256'],'split':split_path,'split_sha256':digest(out/split_path),
                        'cohort':cohort,'cohort_sha256':digest(path)}
                    for model in spec['models']:
                        for loss in spec['losses']:
                            jobs.append({**shared,'family':'architecture','model':model,'loss':loss,'updates':spec['head']['epochs'],
                                'batch_size':'full','job_id':f'arch_{name}_s{split_seed}_p{ratio:.2f}_{model}_{loss}_r{seed}'})
                    if ratio in spec['matched_updates']['ratios']:
                        for budget in spec['matched_updates']['budgets']:
                            for batch in spec['matched_updates']['batch_sizes']:
                                jobs.append({**shared,'family':'matched_updates','model':'tbase','loss':'unweighted_ce','updates':budget,
                                    'batch_size':batch,'job_id':f'updates_{name}_s{split_seed}_p{ratio:.2f}_b{batch}_u{budget}_r{seed}'})
    for job in jobs:job['job_sha256']=fingerprint(job)
    plan={'contract_sha256':fingerprint(contract),'fixed_n':sizes,'jobs':jobs,'execution_status':'prepared_not_executed'}
    write(out/'run_plan.json',plan);write(out/'split_report.json',report)
    counts={family:sum(j['family']==family for j in jobs) for family in ['architecture','matched_updates']}
    write(out/'preparation_summary.json',{'job_counts':counts,'total_jobs':len(jobs),'fixed_n':sizes,'training_executed':False})
    return plan

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir',default=str(HERE/'inputs'));ap.add_argument('--out',required=True);args=ap.parse_args()
    plan=prepare(args.input_dir,args.out,read(HERE/'spec.json'),read(HERE/'input_manifest.json'))
    print('Prepared',len(plan['jobs']),'jobs; no training executed. Sizes:',plan['fixed_n'])
