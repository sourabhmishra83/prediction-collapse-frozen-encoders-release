"""Fixtures and frozen preparation checks; zero model fitting/optimizer steps."""
from pathlib import Path
import argparse,json
import torch
torch.set_num_threads(2)
import numpy as np
from sklearn.metrics import f1_score
from controls import best_threshold,logistic_objective_gradient,weights,preflight,authorize,read_plan

def fixtures():
 rng=np.random.default_rng(42)
 for n in [4,13,30]:
  y=np.r_[0,1,rng.integers(0,2,n-2)];s=np.round(rng.normal(size=n),1);t,f=best_threshold(y,s);candidates=np.unique(np.r_[s,np.nextafter(s.max(),np.inf),0.]);values=np.array([f1_score(y,s>=c,average='macro',labels=[0,1],zero_division=0) for c in candidates]);best=values.max();tied=[c for c,v in zip(candidates,values) if abs(v-best)<=1e-12];assert abs(f-best)<1e-12 and t==min(tied,key=lambda x:(abs(x),x))
 x=rng.normal(size=(8,3));y=np.array([0,0,0,0,0,0,1,1]);sw=weights(y,'weighted_ce');assert abs(sw[y==0].sum()-sw[y==1].sum())<1e-12
 theta=rng.normal(size=4);value,g=logistic_objective_gradient(theta,x,y,sw)
 for k in range(4):
  delta=np.zeros(4);delta[k]=1e-6;a,_=logistic_objective_gradient(theta+delta,x,y,sw);b,_=logistic_objective_gradient(theta-delta,x,y,sw);assert abs((a-b)/2e-6-g[k])<1e-7
 return {'status':'PASS','fixture_checks':['all distinct validation threshold decisions and deterministic ties','balanced loss mass','penalized logistic gradient finite differences'],'model_fits':0,'optimizer_steps':0}
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--suite');ap.add_argument('--input-dir');ap.add_argument('--assets');a=ap.parse_args();report=fixtures()
 if a.suite:
  report['frozen_checks']=preflight(a.suite,a.input_dir,a.assets)
  _,h=read_plan(a.suite)
  try:authorize(a.suite,h,False)
  except PermissionError:report['execution_guard']='PASS: --execute required'
  else:raise AssertionError('Execution guard missing')
 print(json.dumps(report,indent=2))
