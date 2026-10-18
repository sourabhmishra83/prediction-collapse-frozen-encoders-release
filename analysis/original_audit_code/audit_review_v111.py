from pathlib import Path
import json,hashlib,pandas as pd,numpy as np,torch
R=Path.cwd();O=R/'deliverables/paper_v1_11';O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
verified={}
for base,names in [('paper_v1_10',['Discover_AI_submission_manuscript_v1_10.md','feature_geometry.csv']),('paper_v1_7',['endpoint_shares_and_direction.csv'])]:
 b=R/'deliverables'/base;h=json.loads((b/'deliverable_hashes.json').read_text(encoding='utf-8'))
 for name in names:assert sha(b/name)==h[name];verified[str((b/name).relative_to(R))]=h[name]
e=pd.read_csv(R/'deliverables/paper_v1_7/endpoint_shares_and_direction.csv');e=e[(e.family=='architecture')&e.in_domain&(e.encoder=='RoBERTa')];e['direction']=np.where(e.collapse_class0,'class0',np.where(e.collapse_class1,'class1','none'));q=e.pivot(index=['job_id','ratio','loss','model','source','split_seed','seed'],columns='endpoint',values='direction').reset_index();assert len(q)==324
q.to_csv(O/'architecture_endpoint_directions.csv',index=False)
q.groupby(['ratio','loss','initial','selected','final']).size().reset_index(name='jobs').to_csv(O/'direction_transitions.csv',index=False)
planp=R/'experiments/priority_controls/prepared/plan.json';plan=json.loads(planp.read_text(encoding='utf-8'));assert sha(planp)==json.loads((planp.parents[1]/'plan_summary.json').read_text())['plan_sha256'];torch.set_num_threads(2)
rows=[];selected=[];baselines=[]
for c in plan['caches']:
 folder=R/'experiments/score_audit_private/cache'/c['encoder']/c['source'];p=folder/'vectors.pt';assert sha(p)==c['sha256'];sp=planp.parent/f"splits/{c['source']}_s42.csv";assert sha(sp)==next(j['split_sha256'] for j in plan['jobs'] if j['source']==c['source']);split=pd.read_csv(sp);ids=split.query("split=='train'").row_index.to_numpy();x=torch.load(p,map_location='cpu',weights_only=True).numpy()[ids].astype(np.float64);x-=x.mean(axis=0);norm=np.linalg.norm(x,axis=1);assert (norm>0).all();u=x/norm[:,None];n=len(u);cos=float((np.square(u.sum(axis=0)).sum()-n)/(n*(n-1)))
 rows.append({'encoder':c['encoder'],'source':c['source'],'training_rows':n,'mean_centered_distinct_pair_cosine':cos,'centering':'full source training mean, no evaluation rows','cache_sha256':c['sha256']});verified[str(p.relative_to(R))]=c['sha256']
 if c['encoder']=='RoBERTa':
  v=split.query("split=='val'");p1=v.label.mean();baselines.append({'source':c['source'],'split':42,'validation_rows':len(v),'class1_prevalence':p1,'constant0_macro_f1':(1-p1)/(2-p1),'constant1_macro_f1':p1/(1+p1)})
for j in plan['existing_heads']:
 if j['encoder']!='RoBERTa':continue
 p=R/'experiments/score_audit_private/roberta_heads'/j['job_id']/'result.json';assert sha(p)==j['result_sha256'];a=json.loads(p.read_text(encoding='utf-8'));selected.append({k:j[k] for k in ['job_id','source','ratio','loss','seed']}|{'selected_update':a['selected_optimizer_update']});verified[str(p.relative_to(R))]=j['result_sha256']
pd.DataFrame(rows).to_csv(O/'centered_feature_geometry.csv',index=False);pd.DataFrame(selected).to_csv(O/'selected_updates_split42_text.csv',index=False);pd.DataFrame(baselines).to_csv(O/'constant_validation_baselines.csv',index=False)
(O/'evidence_audit.json').write_text(json.dumps({'status':'PASS','verified_sha256':verified,'model_fits':0,'optimizer_steps':0,'encoder_inference':0,'all_architecture_jobs':324,'selected_update_subset_jobs':36,'scope':'Saved endpoint directions for all architecture jobs; saved selected update numbers for split42 text-only subset only; no inference of unseen intermediate trajectories.'},indent=2),encoding='utf-8')
print(q.groupby(['ratio','loss','initial','selected','final']).size().to_string());print(pd.DataFrame(selected).groupby(['ratio','loss']).selected_update.agg(['min','median','max']).to_string());print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(baselines).to_string(index=False))
