"""Verify saved stance predictions, coverage, selection and paired seeds; never train."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from stance_kernels import read,write,digest,fingerprint
from run_stance import metrics

def verify(out,allow_partial=False,allow_synthetic=False):
 out=Path(out);contract=read(out/'execution_contract.json');plan=read(out/'run_plan.json');assert plan['contract_sha256']==fingerprint(contract)
 if contract['synthetic_only'] and not allow_synthetic:raise ValueError('Synthetic fixtures are not empirical results')
 index=pd.read_csv(out/'evaluation_index.csv');assert index.row_id.is_unique
 for field in ['article_group','claim_group']:
  groups=[set(index[index.split==p][field]) for p in ['train','validation','test']];assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
 rows=[];missing=[];initial={};spec=contract['spec']
 for job in plan['jobs']:
  assert fingerprint({k:v for k,v in job.items() if k!='job_sha256'})==job['job_sha256']
  folder=out/'results'/job['job_id']
  if not (folder/'result.json').exists():missing.append(job['job_id']);continue
  result=read(folder/'result.json');assert result['job']==job and result['synthetic_only']==contract['synthetic_only']
  for name,h in result['files_sha256'].items():assert digest(folder/name)==h
  key=(job['model'],job['seed']);initial.setdefault(key,result['initial_parameter_sha256']);assert initial[key]==result['initial_parameter_sha256']
  trajectory=read(folder/'trajectory.json')['updates'];assert len(trajectory)==spec['updates'];assert [r['optimizer_update'] for r in trajectory]==list(range(1,spec['updates']+1))
  best=max(trajectory,key=lambda r:r['validation']['macro_f1']);assert best['optimizer_update']==result['selected_optimizer_update']
  assert best['validation']==result['endpoints']['selected']['validation'];assert trajectory[-1]['validation']==result['endpoints']['final']['validation']
  for part in ['validation','test']:
   pred=pd.read_csv(folder/(part+'_predictions.csv'));expected=index[index.split==part]
   assert pred.row_id.tolist()==expected.row_id.tolist() and pred.label_id.tolist()==expected.label_id.tolist()
   for endpoint in ['initial','selected','final']:
    values=pred[endpoint+'_prediction'].to_numpy();assert set(values)<={0,1,2};m=metrics(pred.label_id.to_numpy(),values);saved=result['endpoints'][endpoint][part]
    for name,value in m.items():assert np.allclose(value,saved[name],atol=1e-12)
    rows.append({'job_id':job['job_id'],'model':job['model'],'loss':job['loss'],'seed':job['seed'],'population':part,'endpoint':endpoint,**{k:v for k,v in m.items() if k not in ['recall','confusion_matrix','prediction_counts']},**{f'recall_{c}':v for c,v in enumerate(m['recall'])}})
 if (out/'graph_edges.npz').exists():
  graph=np.load(out/'graph_edges.npz');n=int((index.split=='train').sum());train=graph['train'];assert train.min()>=0 and train.max()<n
  for part in ['validation','test']:
   edges=graph[part];assert np.array_equal(edges[:,:train.shape[1]],train);query=edges[:,train.shape[1]:]
   if query.size:assert query[0].min()>=n and query[0].max()<n+int((index.split==part).sum()) and query[1].min()>=0 and query[1].max()<n
 frame=pd.DataFrame(rows);report={'planned_jobs':len(plan['jobs']),'completed_jobs':len(plan['jobs'])-len(missing),'missing_jobs':missing,'complete':not missing,'synthetic_only':contract['synthetic_only'],'new_training_executed':False,'all_completed_prediction_metrics_verified':True,'test_expert_pairs':int((index.split=='test').sum())};write(out/'analysis/verification.json',report)
 if len(frame):
  frame.to_csv(out/'analysis/verified_endpoints.csv',index=False);summary=frame.groupby(['model','loss','population','endpoint']).agg(n=('job_id','size'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),collapsed_count=('collapsed','sum')).reset_index();summary.to_csv(out/'analysis/condition_summary.csv',index=False)
 if missing and not allow_partial:raise ValueError('Incomplete panel; see verification.json')
 return report

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--allow-partial',action='store_true');args=ap.parse_args();print(verify(args.out,args.allow_partial))
