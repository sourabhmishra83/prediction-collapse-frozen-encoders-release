"""Small synthetic execution checks. Never counts as empirical paper evidence."""
import copy, json, tempfile, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from study_common import HERE,read,write,digest,SourceReferences,make_model,matched_lr,seed_everything
from prepare import prepare
from run import Runner
from verify import verify

def main():
    torch.set_num_threads(2);checks=[]
    graph=read(HERE/'spec.json')['graph'];graph={**graph,'k':3,'min_df':1}
    refs=SourceReferences(['shared alpha token','shared beta token','shared gamma token','shared delta token'],graph)
    refs.query_edges(['HELDOUT_ONLY_TOKEN gamma'])
    assert 'heldout_only_token' not in refs.vectorizer.vocabulary_;checks.append('TFIDF vocabulary excludes held-out tokens')
    seed_everything(17);x=torch.randn(4,8);query=torch.randn(2,8)
    edges=torch.as_tensor(refs.query_edges(['shared alpha','shared beta']))
    altered=query.clone();altered[1]=altered[1]*100
    for kind in ['gcn','gat']:
        seed_everything(2);m=make_model(kind,8,12,0).eval()
        with torch.no_grad():
            train=m(x,torch.as_tensor(refs.train_edges));all_rows=m(torch.cat([x,query]),edges)
            changed=m(torch.cat([x,altered]),edges)
        torch.testing.assert_close(train,all_rows[:4],atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(all_rows[:5],changed[:5],atol=1e-6,rtol=1e-5)
    checks.append('GCN/GAT query features cannot influence references or other queries')
    with tempfile.TemporaryDirectory(prefix='discover_full_study_smoke_') as temporary:
        root=Path(temporary);inputs=root/'inputs';inputs.mkdir();manifest={};vectors={}
        spec=copy.deepcopy(read(HERE/'spec.json'));spec['split_seeds']=[42];spec['training_seeds']=[0];spec['head'].update(hidden=12,dropout=.1,epochs=3)
        spec['graph']=graph;spec['preferred_fixed_n']={name:20 for name in spec['sources']}
        spec['matched_updates'].update(budgets=[2,5],batch_sizes=['full',4],scheduler_horizon=5,warmup_updates=1)
        for source_index,name in enumerate(spec['sources']):
            rows=[]
            for i in range(100):
                label=i%2;text=f'{name} shared topic {"alpha" if label else "beta"} example code{i}'
                if i==1:text=f'{name} shared topic beta example code0'
                rows.append({'text':text,'label':label})
            p=inputs/f'{name}_clean.csv';pd.DataFrame(rows).to_csv(p,index=False)
            manifest[name]={'filename':p.name,'bytes':p.stat().st_size,'sha256':digest(p)}
            generator=torch.Generator().manual_seed(20+source_index);vectors[name]=torch.randn(100,8,generator=generator)
        out=root/'run';plan=prepare(inputs,out,spec,manifest)
        # Execute one representative source at p=.8 for six architecture/loss jobs
        # and all four matched-budget/mode jobs. Other jobs remain intentionally absent.
        selected=[j for j in plan['jobs'] if j['source']=='coaid' and j['ratio']==.8]
        runner=Runner(inputs,out,'cpu',synthetic_vectors=vectors)
        for job in selected:runner.execute(job)
        runner.execute(selected[0],resume=True)
        report=verify(out,allow_partial=True,allow_smoke=True)
        assert report['completed_jobs']==10
        for mode in ['full',4]:
            trajectories=[]
            for budget in [2,5]:
                job=next(j for j in selected if j['family']=='matched_updates' and j['batch_size']==mode and j['updates']==budget)
                trajectories.append(read(out/'results'/job['job_id']/'trajectory.json')['updates'])
            assert trajectories[0]==trajectories[1][:2]
        checks.extend(['10 end-to-end synthetic jobs: all three models, both losses and matched-update budgets',
            'saved prediction metrics, paired initialization and cohort checks pass',
            'short matched-update trajectory exactly equals long-run prefix for each batch mode',
            'completed-job resume validates hashes and does not retrain'])
        for p in (out/'splits').glob('*.csv'):
            s=pd.read_csv(p);assert s.groupby('text_group').split.nunique().max()==1 and len(s)==100
        checks.append('mixed-label normalized-text groups retained together; no source rows removed')
        before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
        prepare(inputs,out,spec,manifest)
        assert all(Path(p).read_bytes()==v for p,v in before.items())
        checks.append('preparation resume preserves all existing plan, split, cohort and result bytes')
    report={'checks':checks,'synthetic_only':True,'real_experiments_executed':False,'gpu_tested':False}
    write(HERE/'selftest_report.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
