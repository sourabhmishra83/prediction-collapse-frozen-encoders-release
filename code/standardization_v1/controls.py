"""Approved-plan-only controls. Importing this module never trains."""
from pathlib import Path
import argparse,json,hashlib,sys,warnings,uuid,platform
import numpy as np,pandas as pd,torch,scipy,sklearn
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,average_precision_score
from study_common import digest,read,write,fingerprint,load_inputs,measures,make_model,seed_everything,core_lr

def weights(y,loss):
 return np.ones(len(y)) if loss=='unweighted_ce' else len(y)/(2*np.bincount(y,minlength=2)[y])

def logistic_objective_gradient(theta,x,y,sw,C=1.):
 w,b=theta[:-1],theta[-1];z=x@w+b;s=sw.sum();err=sw*(expit(z)-y)/s
 value=np.sum(sw*(np.logaddexp(0,z)-y*z))/s+np.dot(w,w)/(2*C*s)
 return value,np.r_[x.T@err+w/(C*s),err.sum()]

def best_threshold(y,margin):
 """All distinct source-val decisions; ties closest to zero then lower cutoff."""
 y=np.asarray(y,int);s=np.asarray(margin,float);assert len(y)>0 and np.isfinite(s).all() and set(y)<={0,1}
 order=np.argsort(s,kind='stable');v=s[order];ys=y[order];cs=np.r_[0,np.cumsum(ys)]
 candidates=np.unique(np.r_[v,np.nextafter(v[-1],np.inf),0.]);idx=np.searchsorted(v,candidates,side='left')
 tp=ys.sum()-cs[idx];fp=len(y)-idx-tp;fn=ys.sum()-tp;tn=(len(y)-ys.sum())-fp
 a=2*tp+fp+fn;b=2*tn+fp+fn
 f=.5*(np.divide(2*tp,a,out=np.zeros_like(a,float),where=a!=0)+np.divide(2*tn,b,out=np.zeros_like(b,float),where=b!=0))
 ties=np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12));pick=min(ties,key=lambda i:(abs(candidates[i]),candidates[i]))
 return float(candidates[pick]),float(f[pick])

def metrics(y,margin,pred):
 m=measures(np.asarray(y),np.asarray(pred));m.update(auroc=float(roc_auc_score(y,margin)),average_precision=float(average_precision_score(y,margin)),target_prevalence=float(np.mean(y)),collapse_direction='class1' if m['pred_class1_fraction']>=.98 else 'class0' if m['pred_class1_fraction']<=.02 else 'none');return m

def read_plan(suite):
 suite=Path(suite);summary=read(suite/'plan_summary.json');assert digest(suite/'prepared/plan.json')==summary['plan_sha256'];return read(suite/'prepared/plan.json'),summary['plan_sha256']

def authorize(suite,plan_hash,execute):
 if not execute:raise PermissionError('Model fitting requires --execute and a recorded approved exact plan')
 a=read(Path(suite)/'approval_record.json')
 if a.get('status')!='approved' or a.get('approved_plan_sha256')!=plan_hash or not a.get('approval_text'):raise PermissionError('Human approval of this exact 144-fit plan has not been recorded. Stop and obtain approval in the research chat.')
 return a

def preflight(suite,input_dir,assets):
 suite=Path(suite);assets=Path(assets);plan,h=read_plan(suite);data=load_inputs(input_dir,plan['inputs'])
 for j in plan['jobs']:
  assert j['job_sha256']==fingerprint({k:v for k,v in j.items() if k!='job_sha256'})
  for k in ['split','cohort']:assert digest(suite/'prepared'/j[k])==j[k+'_sha256']
  split=pd.read_csv(suite/'prepared'/j['split']);cohort=pd.read_csv(suite/'prepared'/j['cohort']).row_index
  assert split.row_index.tolist()==list(range(len(data[j['source']]))) and split.label.tolist()==data[j['source']].label.tolist()
  assert split.text_group.tolist()==data[j['source']].text_group.tolist() and split.groupby('text_group').split.nunique().max()==1
  selected=split.set_index('row_index').loc[cohort];assert cohort.is_unique and len(cohort)==j['fixed_n'] and set(selected.split)=={'train'}
  assert selected.label.sum()==round(j['ratio']*j['fixed_n'])
 for c in plan['caches']:
  folder=assets/'cache'/c['encoder']/c['source'];assert digest(folder/'vectors.pt')==c['sha256'] and digest(folder/'metadata.json')==c['metadata_sha256']
  x=torch.load(folder/'vectors.pt',map_location='cpu',weights_only=True);assert x.shape==(len(data[c['source']]),768) and torch.isfinite(x).all()
 for j in plan['existing_heads']:
  folder=assets/'existing'/j['job_id'];assert digest(folder/'selected_model.pt')==j['head_sha256'] and digest(folder/'result.json')==j['result_sha256']
 assert len(plan['jobs'])==144 and len(plan['existing_heads'])==72
 return {'status':'PASS','plan_sha256':h,'new_fits_planned':144,'existing_heads':72,'new_training_executed':False}

class Study:
 def __init__(self,suite,input_dir,assets,out,device):
  self.suite=Path(suite);self.plan,self.h=read_plan(suite);self.data=load_inputs(input_dir,self.plan['inputs']);self.assets=Path(assets);self.out=Path(out);self.device=torch.device(device);self.out.mkdir(parents=True,exist_ok=True)
  self.x={};self.splits={}
  for c in self.plan['caches']:self.x[c['encoder'],c['source']]=torch.load(self.assets/'cache'/c['encoder']/c['source']/'vectors.pt',map_location='cpu',weights_only=True)
  for s in self.plan['spec']['sources']:self.splits[s]=pd.read_csv(self.suite/f'prepared/splits/{s}_s42.csv')
  env={'plan_sha256':self.h,'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'torch':torch.__version__,'device':str(self.device),'gpu':torch.cuda.get_device_name(0) if self.device.type=='cuda' else None,'code_sha256':{p.name:digest(p) for p in (self.suite/'code').glob('*.py')}}
  ep=self.out/'execution_environment.json'
  if ep.exists():assert read(ep)==env,'Runtime or code changed; use a separately versioned output folder'
  else:write(ep,env)

 def rows(self,job,target,part):
  split=self.splits[target];q=split[split.split==part]
  groups=set(self.splits[job['source']].query("split == 'train'").text_group)
  if target!=job['source'] or part=='test':q=q[~q.text_group.isin(groups)]
  ids=q.row_index.to_numpy();return ids,q.label.to_numpy(int),self.x[job['encoder'],target][ids]

 def score(self,model,x):
  if isinstance(model,dict):return x.numpy().astype(np.float64)@model['coef']+float(model['intercept'])
  with torch.no_grad():z=model(x.to(self.device)).detach().cpu().numpy()
  return z[:,1].astype(np.float64)-z[:,0].astype(np.float64)

 def val(self,model,job):
  ids,y,x=self.rows(job,job['source'],'val');s=self.score(model,x);return measures(y,(s>0).astype(int))

 def decision_evaluations(self,model,job,folder):
  ids,y,x=self.rows(job,job['source'],'val');s=self.score(model,x);threshold,valf=best_threshold(y,s)
  cohort=pd.read_csv(self.suite/'prepared'/job['cohort']).row_index.to_numpy();cy=self.data[job['source']].label.to_numpy(int)[cohort]
  effective_prior=.5 if job['loss']=='weighted_ce' else float(cy.mean());source_val_prior=float(y.mean());offset=float(logit(source_val_prior)-logit(effective_prior))
  rules={'raw':{'kind':'raw','threshold':0.},'source_prior':{'kind':'offset','offset':offset},'source_val_threshold':{'kind':'threshold','threshold':threshold,'source_val_macro_f1':valf}}
  pd.DataFrame({'row_index':ids,'label':y,'margin':s}).to_csv(folder/'source_validation_scores.csv',index=False)
  rows=[]
  for target in self.plan['spec']['sources']:
   ids,labels,features=self.rows(job,target,'test');scores=self.score(model,features);preds={}
   for name,cfg in rules.items():
    pred=(scores>=threshold).astype(int) if name=='source_val_threshold' else ((scores+offset)>0).astype(int) if name=='source_prior' else (scores>0).astype(int)
    preds[name]=pred;rows.append({'job_id':job['job_id'],'family':job['family'],'encoder':job['encoder'],'source':job['source'],'target':target,'ratio':job['ratio'],'seed':job['seed'],'loss':job['loss'],'rule':name,'endpoint':'converged' if job['family']=='linear' else 'selected',**metrics(labels,scores,pred)})
   pd.DataFrame({'row_index':ids,'label':labels,'margin':scores,**{k+'_prediction':v for k,v in preds.items()}}).to_csv(folder/f'test_{target}_scores.csv',index=False)
  return rows,{'effective_training_prior':effective_prior,'source_validation_prior':source_val_prior,'source_validation_rows_sha256':fingerprint(self.rows(job,job['source'],'val')[0].tolist()),'rules':rules}

 def execute(self,job):
  dest=self.out/'results'/job['job_id']
  if (dest/'result.json').exists():
   saved=read(dest/'result.json');assert saved['job']==job
   for n,h in saved['files_sha256'].items():assert digest(dest/n)==h
   return
  if dest.exists():raise FileExistsError('Incomplete directory retained; investigate '+str(dest))
  temp=dest.parent/(dest.name+'.temporary_'+uuid.uuid4().hex);temp.mkdir(parents=True)
  cohort=pd.read_csv(self.suite/'prepared'/job['cohort']).row_index.to_numpy();x=self.x[job['encoder'],job['source']][cohort];y=self.data[job['source']].label.to_numpy(int)[cohort];record={'job':job,'plan_sha256':self.h,'new_training_executed':job['family']!='existing_mlp'}
  if job['family']=='linear':
   sw=weights(y,job['loss']);cfg=self.plan['spec']['linear'];fit=LogisticRegression(solver='lbfgs',penalty='l2',C=cfg['C'],tol=cfg['tol'],max_iter=cfg['max_iter'],fit_intercept=True)
   with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always');fit.fit(x.numpy().astype(np.float64),y,sample_weight=sw)
   theta=np.r_[fit.coef_[0],fit.intercept_[0]];objective,grad=logistic_objective_gradient(theta,x.numpy().astype(np.float64),y,sw,cfg['C'])
   converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught) and int(fit.n_iter_[0])<cfg['max_iter'] and np.max(np.abs(grad))<=1e-5
   status={'converged':bool(converged),'iterations':int(fit.n_iter_[0]),'objective':float(objective),'max_abs_gradient':float(np.max(np.abs(grad))),'warnings':[str(w.message) for w in caught]};write(temp/'convergence.json',status)
   if not converged:raise RuntimeError('Convergence criterion failed; diagnostics preserved at '+str(temp))
   model={'coef':fit.coef_[0],'intercept':float(fit.intercept_[0])};np.savez(temp/'linear_model.npz',**model);record.update(convergence=status,initialization='not measured',training_ever='not measured; L-BFGS internal iterates are not monitored',selected_final='one converged state, not separate checkpoints')
  else:
   cfg=self.plan['spec']['mirror_mlp'];seed_everything(job['seed']);model=make_model('tbase',768,cfg['width'],cfg['dropout']).to(self.device).eval()
   if job['family']=='existing_mlp':
    original=read(self.assets/'existing'/job['job_id']/'result.json');model.load_state_dict(torch.load(self.assets/'existing'/job['job_id']/'selected_model.pt',map_location=self.device,weights_only=True));model.eval()
    assert self.val(model,job)==original['selected_validation'],'Existing selected validation replay differs; preserve and investigate'
    record['original_job_id']=original['job']['job_id'];record['selected_optimizer_update']=original['selected_optimizer_update']
   else:
    init={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};torch.save(init,temp/'initial_model.pt');record['initial_parameter_sha256']=hashlib.sha256(b''.join(v.numpy().tobytes() for v in init.values())).hexdigest();record['initialization_validation']=self.val(model,job)
    initial_scores={t:self.score(model,self.rows(job,t,'test')[2]) for t in self.plan['spec']['sources']}
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay']);criterion=torch.nn.CrossEntropyLoss(weight=None if job['loss']=='unweighted_ce' else torch.as_tensor(len(y)/(2*np.bincount(y,minlength=2)),dtype=torch.float32,device=self.device));xt=x.to(self.device);yt=torch.as_tensor(y,device=self.device)
    trajectory=[];best=-1;beststate=None
    for u in range(30):
     lr=cfg['lr']*core_lr(u,30)
     for g in opt.param_groups:g['lr']=lr
     model.train();opt.zero_grad();loss=criterion(model(xt),yt);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();model.eval();v=self.val(model,job)
     trajectory.append({'optimizer_update':u+1,'learning_rate':lr,'training_loss':float(loss.detach().cpu()),'source_validation':v})
     if v['macro_f1']>best:best=v['macro_f1'];beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};record['selected_optimizer_update']=u+1
    torch.save(model.state_dict(),temp/'final_model.pt');record['final_validation']=self.val(model,job);final_scores={t:self.score(model,self.rows(job,t,'test')[2]) for t in self.plan['spec']['sources']};model.load_state_dict(beststate);model.eval();torch.save(beststate,temp/'selected_model.pt');record['selected_validation']=self.val(model,job);record['training_ever_validation_collapsed']=any(v['source_validation']['collapsed'] for v in trajectory);write(temp/'trajectory.json',trajectory)
    endpoint=[]
    for target in self.plan['spec']['sources']:
     ids,labels,features=self.rows(job,target,'test');ss=self.score(model,features)
     pd.DataFrame({'row_index':ids,'label':labels,'initial_margin':initial_scores[target],'selected_margin':ss,'final_margin':final_scores[target]}).to_csv(temp/f'checkpoints_{target}.csv',index=False)
     for name,s in [('initial',initial_scores[target]),('selected',ss),('final',final_scores[target])]:endpoint.append({'target':target,'endpoint':name,**metrics(labels,s,(s>0).astype(int))})
    record['checkpoint_test_metrics']=endpoint
  evaluations,rules=self.decision_evaluations(model,job,temp);record.update(evaluations=evaluations,decision_record=rules);record['files_sha256']={p.name:digest(p) for p in temp.iterdir()};write(temp/'result.json',record);temp.rename(dest);print('Verified completion:',job['job_id'],flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--suite',required=True);ap.add_argument('--input-dir',required=True);ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True);ap.add_argument('--stage',choices=['linear','mirror'],required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 plan,h=read_plan(a.suite);approval=authorize(a.suite,h,a.execute);preflight(a.suite,a.input_dir,a.assets);torch.set_num_threads(2)
 device='cuda' if torch.cuda.is_available() and a.stage=='mirror' else 'cpu';study=Study(a.suite,a.input_dir,a.assets,a.out,device)
 # Keep one runtime per output stage. Stage environments can differ between CPU/GPU.
 write(Path(a.out)/'approval_record.json',approval)
 for j in plan['jobs']:
  if j['stage']==a.stage:study.execute(j)
 if a.stage=='mirror':
  for j in plan['existing_heads']:study.execute(j)
 write(Path(a.out)/'stage_finished.json',{'stage':a.stage,'plan_sha256':h,'complete':True})

if __name__=='__main__':main()
