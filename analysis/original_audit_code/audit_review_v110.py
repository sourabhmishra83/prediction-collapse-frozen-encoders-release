from pathlib import Path
import json,hashlib,pandas as pd,numpy as np,torch
R=Path.cwd();O=R/'deliverables/paper_v1_10';O.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
verified={}
for base,names in [('deliverables/paper_v1_9',['Discover_AI_submission_manuscript_v1_9.md']),('deliverables/paper_v1_7',['endpoint_shares_and_direction.csv']),('deliverables/paper_v1_8',['linear_decision_evaluations.csv','mirror_decision_evaluations.csv'])]:
 b=R/base;h=json.loads((b/'deliverable_hashes.json').read_text(encoding='utf-8'))
 for n in names:assert sha(b/n)==h[n];verified[str((b/n).relative_to(R))]=h[n]
planp=R/'experiments/priority_controls/prepared/plan.json';plan=json.loads(planp.read_text(encoding='utf-8'));assert sha(planp)==json.loads((planp.parents[1]/'plan_summary.json').read_text())['plan_sha256'];torch.set_num_threads(2)
rows=[]
for c in plan['caches']:
 folder=R/'experiments/score_audit_private/cache'/c['encoder']/c['source'];p=folder/'vectors.pt';assert sha(p)==c['sha256'] and sha(folder/'metadata.json')==c['metadata_sha256'];sp=planp.parent/f"splits/{c['source']}_s42.csv";expected=next(j['split_sha256'] for j in plan['jobs'] if j['source']==c['source']);assert sha(sp)==expected
 split=pd.read_csv(sp);ids=split.query("split=='train'").row_index.to_numpy();x=torch.load(p,map_location='cpu',weights_only=True).numpy()[ids].astype(np.float64);assert np.isfinite(x).all();norm=np.linalg.norm(x,axis=1);assert (norm>0).all();u=x/norm[:,None];n=len(u);mean_cos=float((np.square(u.sum(axis=0)).sum()-n)/(n*(n-1)))
 # Check the efficient exact estimator against explicit pair products on a fixture subset.
 v=u[:20];explicit=float((v@v.T)[~np.eye(len(v),dtype=bool)].mean());fast=float((np.square(v.sum(axis=0)).sum()-len(v))/(len(v)*(len(v)-1)));assert abs(explicit-fast)<1e-12
 rows.append({'encoder':c['encoder'],'source':c['source'],'split_seed':42,'population':'full source training reference pool, not resampled cohorts','training_rows':n,'dimensions':x.shape[1],'mean_pairwise_cosine':mean_cos,'median_vector_norm':float(np.median(norm)),'median_coordinate_sd':float(np.median(x.std(axis=0,ddof=0))),'cache_sha256':c['sha256'],'split_sha256':expected});verified[str(p.relative_to(R))]=c['sha256'];del x,u
pd.DataFrame(rows).to_csv(O/'feature_geometry.csv',index=False)
e=pd.read_csv(R/'deliverables/paper_v1_7/endpoint_shares_and_direction.csv');e=e[(e.family=='architecture')&e.in_domain&(e.encoder=='RoBERTa')];e['direction']=np.where(e.collapse_class0,'class0',np.where(e.collapse_class1,'class1','none'));q=e.pivot(index=['job_id','ratio','loss','model','source','split_seed','seed'],columns='endpoint',values='direction').reset_index();assert len(q)==324;q.groupby(['ratio','loss','model','initial','selected']).size().reset_index(name='jobs').to_csv(O/'initial_selected_direction_counts.csv',index=False);q.groupby(['ratio','loss','initial','selected']).size().reset_index(name='jobs').to_csv(O/'initial_selected_direction_pooled.csv',index=False)
f=pd.concat([pd.read_csv(R/'deliverables/paper_v1_8'/f'{stage}_decision_evaluations.csv') for stage in ['linear','mirror']],ignore_index=True);assert len(f)==1944 and f.job_id.nunique()==216
f.groupby(['family','encoder','source','target','loss','ratio','rule','collapse_direction']).size().reset_index(name='evaluations').to_csv(O/'decision_directions_by_condition.csv',index=False)
p=f.pivot(index=['job_id','encoder','source','target','family','loss','ratio','seed'],columns='rule',values='collapse_direction').reset_index();p.groupby(['family','raw','source_prior']).size().reset_index(name='evaluations').to_csv(O/'raw_to_offset_direction.csv',index=False)
# Source-validation-only offset; no target test priors used.
offsets=[]
for j in plan['jobs']:
 if j['family']!='linear':continue
 split=pd.read_csv(planp.parent/j['split']);cohort=pd.read_csv(planp.parent/j['cohort']).row_index;prior=float(split.query("split=='val'").label.mean());train=float(split.set_index('row_index').loc[cohort].label.mean());eff=.5 if j['loss']=='weighted_ce' else train;offset=np.log(prior/(1-prior))-np.log(eff/(1-eff));offsets.append({'encoder':j['encoder'],'source':j['source'],'ratio':j['ratio'],'loss':j['loss'],'seed':j['seed'],'source_validation_prevalence':prior,'effective_training_prior':eff,'offset':offset})
pd.DataFrame(offsets).to_csv(O/'source_prior_offsets.csv',index=False)
summary={'status':'PASS','source_hashes_verified':verified,'model_fits':0,'optimizer_steps':0,'encoder_inference':0,'cached_feature_matrices':6,'architecture_jobs_cross_tabulated':324,'control_evaluations':1944,'p80_unweighted_direction_retained':int(q.query("ratio==.8 and loss=='unweighted_ce'").eval('initial==selected').sum()),'scope':'Geometry measured on source training reference rows only. Saved aggregate endpoint and prediction-direction tables reanalyzed; no causal mechanism inferred.'};(O/'evidence_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(pd.DataFrame(rows)[['encoder','source','training_rows','mean_pairwise_cosine','median_vector_norm','median_coordinate_sd']].to_string(index=False));print('PASS: hashes and no-training diagnostics.')
