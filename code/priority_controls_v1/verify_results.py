"""Recompute private prediction metrics, source-only decisions and checkpoints."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.special import logit
from controls import read_plan,best_threshold,metrics
from study_common import read,write,digest

def verify(suite,out,stage):
 suite=Path(suite);out=Path(out);plan,h=read_plan(suite);jobs=[j for j in plan['jobs'] if j['stage']==stage]+(plan['existing_heads'] if stage=='mirror' else [])
 expected=72 if stage=='linear' else 144;assert len(jobs)==expected
 allrows=[];missing=[]
 for j in jobs:
  folder=out/'results'/j['job_id']
  if not (folder/'result.json').exists():missing.append(j['job_id']);continue
  result=read(folder/'result.json');assert result['job']==j and result['plan_sha256']==h
  for name,sha in result['files_sha256'].items():assert digest(folder/name)==sha
  val=pd.read_csv(folder/'source_validation_scores.csv');split=pd.read_csv(suite/'prepared'/j['split']);expectedval=split[split.split=='val'];assert val.row_index.tolist()==expectedval.row_index.tolist() and val.label.tolist()==expectedval.label.tolist()
  threshold,valf=best_threshold(val.label.to_numpy(),val.margin.to_numpy());d=result['decision_record'];assert abs(threshold-d['rules']['source_val_threshold']['threshold'])<1e-10 and abs(valf-d['rules']['source_val_threshold']['source_val_macro_f1'])<1e-12
  cohort=pd.read_csv(suite/'prepared'/j['cohort']).row_index.to_numpy();labels=split.set_index('row_index').label.loc[cohort];effective=.5 if j['loss']=='weighted_ce' else labels.mean();offset=logit(val.label.mean())-logit(effective);assert abs(offset-d['rules']['source_prior']['offset'])<1e-12
  groups=set(split[split.split=='train'].text_group)
  for target in plan['spec']['sources']:
   targetsplit=pd.read_csv(suite/f'prepared/splits/{target}_s42.csv');test=targetsplit[(targetsplit.split=='test')&~targetsplit.text_group.isin(groups)];p=pd.read_csv(folder/f'test_{target}_scores.csv');assert p.row_index.tolist()==test.row_index.tolist() and p.label.tolist()==test.label.tolist()
   for rule in ['raw','source_prior','source_val_threshold']:
    pred=(p.margin.to_numpy()>=threshold).astype(int) if rule=='source_val_threshold' else ((p.margin.to_numpy()+offset)>0).astype(int) if rule=='source_prior' else (p.margin.to_numpy()>0).astype(int)
    assert np.array_equal(pred,p[rule+'_prediction']);m=metrics(p.label,p.margin,pred);saved=next(r for r in result['evaluations'] if r['target']==target and r['rule']==rule)
    for key,v in m.items():
     if isinstance(v,(float,int,bool)):assert abs(float(v)-float(saved[key]))<1e-10
     else:assert v==saved[key]
    allrows.append(saved)
  if j['family']=='linear':assert result['convergence']['converged'] and result['convergence']['iterations']<10000 and result['convergence']['max_abs_gradient']<=1e-5
  if j['family']=='mirror_mlp':
   trajectory=read(folder/'trajectory.json');assert len(trajectory)==30 and [t['optimizer_update'] for t in trajectory]==list(range(1,31))
   best=max(trajectory,key=lambda t:t['source_validation']['macro_f1']);assert best['optimizer_update']==result['selected_optimizer_update'] and best['source_validation']==result['selected_validation']
   assert trajectory[-1]['source_validation']==result['final_validation'];assert result['training_ever_validation_collapsed']==any(t['source_validation']['collapsed'] for t in trajectory)
   assert all((folder/n).is_file() for n in ['initial_model.pt','selected_model.pt','final_model.pt'])
   for target in plan['spec']['sources']:
    p=pd.read_csv(folder/f'checkpoints_{target}.csv');test=pd.read_csv(folder/f'test_{target}_scores.csv');assert p.row_index.tolist()==test.row_index.tolist() and p.label.tolist()==test.label.tolist()
    for endpoint in ['initial','selected','final']:
     m=metrics(p.label,p[endpoint+'_margin'],(p[endpoint+'_margin']>0).astype(int));saved=next(x for x in result['checkpoint_test_metrics'] if x['target']==target and x['endpoint']==endpoint)
     for k,v in m.items():
      if isinstance(v,(float,int,bool)):assert abs(float(v)-float(saved[k]))<1e-10
      else:assert v==saved[k]
 assert not missing,'Incomplete/failing jobs retained: '+str(missing)
 frame=pd.DataFrame(allrows);assert len(frame)==expected*9;frame.to_csv(out/'verified_decision_evaluations.csv',index=False)
 frame.groupby(['encoder','family','source','target','ratio','loss','rule']).agg(evaluations=('macro_f1','size'),collapse=('collapsed','sum'),macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),auroc=('auroc','mean'),average_precision=('average_precision','mean'),mcc=('mcc','mean'),prevalence=('target_prevalence','mean'),prediction_share=('pred_class1_fraction','mean')).reset_index().to_csv(out/'condition_summary.csv',index=False)
 report={'status':'PASS','complete':True,'stage':stage,'plan_sha256':h,'new_fits':72,'existing_head_replays':72 if stage=='mirror' else 0,'verified_classifier_states':expected,'decision_rule_target_evaluations':len(frame),'new_training_executed_by_verifier':False,'restriction':'Private scores/labels and learned weights are not public release artifacts'};write(out/'completion.json',report);return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--suite',required=True);ap.add_argument('--out',required=True);ap.add_argument('--stage',choices=['linear','mirror'],required=True);a=ap.parse_args();print(verify(a.suite,a.out,a.stage))
