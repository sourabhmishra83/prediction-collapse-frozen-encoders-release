from pathlib import Path
import hashlib,json,shutil,sys,copy
import pandas as pd
R=Path(__file__).resolve().parents[1];E=R/'experiments/priority_controls';P=E/'prepared';C=E/'code';P.mkdir(parents=True,exist_ok=True);C.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'full_study'));from study_common import cohort_rows,digest
B=R/'experiments/second_encoder/prepared';contract=json.loads((B/'contract.json').read_text());oldplan=json.loads((B/'run_plan.json').read_text())
for src in ['coaid','fakenewsnet','liar']:
 for ss in [42]:
  srcpath=B/f'splits/{src}_s{ss}.csv';dst=P/f'splits/{src}_s{ss}.csv';dst.parent.mkdir(exist_ok=True);shutil.copy2(srcpath,dst)
for p in (B/'cohorts').glob('*'):
 dst=P/'cohorts'/p.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(p,dst)
sizes={'coaid':783,'fakenewsnet':4874,'liar':4000}
for src,n in sizes.items():
 split=pd.read_csv(P/f'splits/{src}_s42.csv')
 for seed in [0,1,2]:
  p=P/f'cohorts/{src}_s42_p0.20_r{seed}.csv';pd.DataFrame({'row_index':cohort_rows(split,n,.2,seed)}).to_csv(p,index=False,lineterminator='\n')
cache=[]
for enc in ['RoBERTa','BERT']:
 for src in sizes:
  folder=R/'experiments/score_audit_private/cache'/enc/src;meta=json.loads((folder/'metadata.json').read_text());assert digest(folder/'vectors.pt')==meta['sha256']
  cache.append({'encoder':enc,'source':src,'bytes':(folder/'vectors.pt').stat().st_size,'sha256':meta['sha256'],'metadata_sha256':digest(folder/'metadata.json'),'metadata':meta,'drive_relative':('New_Experiments/Full_Study_Runs/full_study_v1/embedding_cache' if enc=='RoBERTa' else 'New_Experiments/second_encoder_results/bert_v1/embedding_cache')+'/'+src})
jobs=[];existing=[]
for enc in ['RoBERTa','BERT']:
 for src,n in sizes.items():
  for ratio in [.2,.5,.8]:
   for seed in [0,1,2]:
    for loss in ['unweighted_ce','weighted_ce']:
     shared={'encoder':enc,'source':src,'ratio':ratio,'seed':seed,'loss':loss,'fixed_n':n,'split':f'splits/{src}_s42.csv','cohort':f'cohorts/{src}_s42_p{ratio:.2f}_r{seed}.csv'}
     for k in ['split','cohort']:shared[k+'_sha256']=digest(P/shared[k])
     jobs.append({**shared,'family':'linear','stage':'linear' if ratio!=.2 else 'mirror','job_id':f'linear_{enc}_{src}_p{ratio:.2f}_{loss}_r{seed}'})
     if ratio==.2:jobs.append({**shared,'family':'mirror_mlp','stage':'mirror','job_id':f'mirror_mlp_{enc}_{src}_p0.20_{loss}_r{seed}'})
     else:
      oldid=f'arch_{src}_s42_p{ratio:.2f}_tbase_{loss}_r{seed}';oldid=('bert_'+oldid) if enc=='BERT' else oldid
      folder=(R/'experiments/second_encoder/completed_audit/downloaded/results'/oldid) if enc=='BERT' else (R/'experiments/score_audit_private/roberta_heads'/oldid)
      result=json.loads((folder/'result.json').read_text());assert digest(folder/'selected_model.pt')==result['files_sha256']['selected_model.pt']
      existing.append({**shared,'family':'existing_mlp','job_id':oldid,'head_sha256':digest(folder/'selected_model.pt'),'result_sha256':digest(folder/'result.json'),'drive_relative':('New_Experiments/second_encoder_results/bert_v1/results/' if enc=='BERT' else 'New_Experiments/Full_Study_Runs/full_study_v1/results/')+oldid,'saved_validation':result['selected_validation'],'original_job':result['job']})
spec={'version':1,'sources':list(sizes),'split_seed':42,'training_cohort_seeds':[0,1,2],'shares':[.2,.5,.8],'encoders':['RoBERTa','BERT'],'features':'Exact precomputed 768-dimensional final first-token hidden states; no re-encoding, fine-tuning, pooling change or standardization','linear':{'solver':'lbfgs','penalty':'l2','C':1.0,'tol':1e-8,'max_iter':10000,'fit_intercept':True,'dtype':'float64','convergence':'No ConvergenceWarning, n_iter < 10000, max absolute gradient of mean weighted logistic loss plus L2/(C*sum_weights) <= 1e-5; otherwise fail and retain diagnostic, no automatic relaxation'},'mirror_mlp':{'width':256,'dropout':.3,'lr':.001,'weight_decay':.0001,'clip_norm':1.0,'updates':30,'warmup_updates':3,'schedule':'Original 30-update cosine schedule','selection':'earliest best post-update source-validation macro-F1; initialization excluded'},'decision_rules':['raw margin > 0','source-validation-prior logit offset, using effective training prior .5 for weighted CE or realized cohort prior for unweighted','source-validation-only macro-F1-optimal threshold; ties closest to zero then lower threshold'],'threshold_target_policy':'Freeze one source-validation rule and apply unchanged to all three target tests; no target prevalence, labels or performance used to select rules','endpoint_policy':'MLP: initial, training-ever validation, selected and final separated. Linear: converged fitted state only; initial/training-ever not measured; selected/final not independent checkpoints. Decision rules apply only to selected/converged states.','statistics':'Descriptive paired seed/cohort comparisons on one fixed split; no post-result tuning or automatic new fits','excluded':['Monant','graph training','alternative pooling','standardization','exposure-matched arms','hyperparameter search','target-test calibration']}
plan={'spec':spec,'inputs':contract['inputs'],'caches':cache,'jobs':jobs,'existing_heads':existing,'counts':{'stage1_linear_fits':72,'stage2_mirror_linear_fits':36,'stage2_mirror_mlp_fits':36,'total_new_fits':144,'existing_heads_replayed':72,'selected_or_converged_states':216,'decision_rule_target_evaluations':1944}}
for j in jobs:j['job_sha256']=hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest()
(P/'plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n',encoding='utf-8');h=digest(P/'plan.json')
(E/'plan_summary.json').write_text(json.dumps({'plan_sha256':h,'counts':plan['counts'],'new_training_executed':False},indent=2)+'\n')
(E/'approval_record.json').write_text(json.dumps({'status':'pending','approved_plan_sha256':h,'scope':'144 new fits and source-validation decision controls in prepared/plan.json','approval_text':None},indent=2)+'\n')
shutil.copy2(R/'full_study/study_common.py',C/'study_common.py')
req=(R/'experiments/second_encoder/code/requirements.txt').read_text().replace('transformers==4.52.4\n','');(C/'requirements.txt').write_text(req)
print('Prepared 144 new-fit plan and 72 existing-head references; zero fitting. Plan SHA256:',h)
