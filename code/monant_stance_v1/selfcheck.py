"""Numerical checks and verifier fixtures; no optimizer steps or model training."""
import copy,json,tempfile,hashlib
from pathlib import Path
import torch,numpy as np,pandas as pd
from stance_kernels import make_model,SourceReferences,seed_everything,read,write,digest,fingerprint
from run_stance import metrics,SPEC
from verify_stance import verify
HERE=Path(__file__).resolve().parent
checks=[];torch.set_num_threads(2);cfg=copy.deepcopy(SPEC['graph']);cfg.update(k=2,min_df=1)
refs=SourceReferences(['shared train alpha','shared train beta','shared train gamma','shared train delta'],cfg);edges=torch.as_tensor(refs.query_edges(['HELDOUT_ONLY_TOKEN shared alpha','shared beta']))
assert 'heldout_only_token' not in refs.vectorizer.vocabulary_;checks.append('TF-IDF vocabulary fitted to training only')
for kind in ['tbase','gcn','gat']:
 seed_everything(10);model=make_model(kind,8,12,0,classes=3).eval();x=torch.randn(4,8);q=torch.randn(2,8);before={k:v.clone() for k,v in model.state_dict().items()}
 if kind=='tbase':out=model(x);assert out.shape==(4,3)
 else:
  base=model(x,torch.as_tensor(refs.train_edges));out=model(torch.cat([x,q]),edges);other=q.clone();other[1]*=100;perturbed=model(torch.cat([x,other]),edges);torch.testing.assert_close(base,out[:4],atol=1e-6,rtol=1e-5);torch.testing.assert_close(out[:5],perturbed[:5],atol=1e-6,rtol=1e-5)
 loss=torch.nn.CrossEntropyLoss()(out[:4],torch.tensor([0,1,2,0]));loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
checks.append('All three heads produce three-class finite gradients without modifying weights; receive-only graph invariance holds')
assert metrics(np.array([0,1,2]),np.array([0,1,2]))['macro_f1']==1.;assert metrics(np.array([0,1,2]),np.zeros(3,dtype=int))['collapsed'];checks.append('Three-class metrics and constant-prediction collapse checked')
with tempfile.TemporaryDirectory() as temp:
 out=Path(temp);spec=copy.deepcopy(SPEC);spec['updates']=2;spec['models']=['tbase'];spec['seeds']=[0];contract={'spec':spec,'synthetic_only':True};write(out/'execution_contract.json',contract)
 index=pd.DataFrame({'row_id':['train0','train1','train2','val0','val1','val2','test0','test1','test2'],'label_id':[0,1,2]*3,'split':['train']*3+['validation']*3+['test']*3,'article_group':list('abcdefghi'),'claim_group':list('abcdefghi')});index.to_csv(out/'evaluation_index.csv',index=False)
 jobs=[]
 for loss in spec['losses']:
  job={'model':'tbase','loss':loss,'seed':0,'job_id':loss};job['job_sha256']=fingerprint(job);jobs.append(job);folder=out/'results'/loss;folder.mkdir(parents=True)
  m=metrics(np.array([0,1,2]),np.array([0,1,2]));trajectory={'updates':[{'optimizer_update':i,'validation':m} for i in [1,2]]};write(folder/'trajectory.json',trajectory)
  for part in ['validation','test']:
   pred=index[index.split==part][['row_id','label_id']].copy()
   for endpoint in ['initial','selected','final']:pred[endpoint+'_prediction']=[0,1,2]
   pred.to_csv(folder/(part+'_predictions.csv'),index=False)
  write(folder/'result.json',{'job':job,'synthetic_only':True,'initial_parameter_sha256':'fixture_same_seed','selected_optimizer_update':1,'endpoints':{e:{p:m for p in ['validation','test']} for e in ['initial','selected','final']},'files_sha256':{p.name:digest(p) for p in folder.iterdir()}})
 write(out/'run_plan.json',{'contract_sha256':fingerprint(contract),'jobs':jobs});assert verify(out,allow_synthetic=True)['complete']
 bad=out/'results'/spec['losses'][0]/'test_predictions.csv';bad.write_text(bad.read_text().replace('test0,0,0,0,0','test0,0,1,0,0'))
 try:verify(out,allow_synthetic=True);raise RuntimeError('Corruption was not detected')
 except AssertionError:pass
checks.append('Synthetic fixtures exercise paired seeds, metric recomputation, earliest-tie selection and corrupt-file rejection')
report={'checks':checks,'synthetic_only':True,'optimizer_steps_executed':0,'real_training_executed':False,'gpu_tested':False};write(HERE/'selfcheck_report.json',report);print(json.dumps(report,indent=2))
