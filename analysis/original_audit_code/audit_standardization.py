from pathlib import Path
import json, hashlib, ast
import numpy as np
import pandas as pd

R=Path.cwd(); S=R/'experiments/standardization'; A=S/'completed_audit/aggregates'; O=R/'deliverables/paper_v1_15'; O.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((S/'prepared/plan.json').read_text()); summary=json.loads((S/'plan_summary.json').read_text())
assert sha(S/'prepared/plan.json')==summary['plan_sha256']
for n,h in plan['frozen_files'].items(): assert sha(S/n)==h,n
c=json.loads((A/'completion.json').read_text()); env=json.loads((A/'execution_environment.json').read_text())
assert c['status']=='PASS' and c['complete'] and c['checkpoint_weight_replay']
assert c['plan_sha256']==summary['plan_sha256']==env['plan_sha256']
assert c['planned_new_fits']==c['verified_complete_job_records']==72 and c['verified_post_update_trajectory_records']==2160
for n,h in env['code_sha256'].items(): assert sha(S/'code'/n)==h
d=pd.read_csv(A/'decision_evaluations.csv'); e=pd.read_csv(A/'endpoint_evaluations.csv'); j=pd.read_csv(A/'job_diagnostics.csv'); pairs=pd.read_csv(A/'paired_comparison.csv')
assert len(d)==len(e)==len(pairs)==648 and len(j)==72
assert not d.duplicated(['job_id','target','rule']).any() and not e.duplicated(['job_id','target','endpoint']).any()
assert set(j.job_id)=={x['job_id'] for x in plan['jobs']} and not j.job_id.duplicated().any()
expected=pd.DataFrame(plan['jobs'])
for _,row in j.iterrows():
    p=expected[expected.job_id.eq(row.job_id)].iloc[0]
    for k in ['baseline_job_id','encoder','source','ratio','loss','seed']: assert row[k]==p[k]
assert j.selected_update.between(1,30).all()
for frame in [d,e]:
    for _,row in frame.iterrows():
        cm=np.array(ast.literal_eval(row.confusion_matrix)); tn,fp,fn,tp=map(int,cm.ravel()); n=int(cm.sum())
        close=lambda a,b:np.isclose(a,b,atol=1e-12,rtol=1e-10)
        assert n==row.rows and close((tp+fn)/n,row.target_prevalence)
        assert close((tp+tn)/n,row.accuracy) and close((tp+fp)/n,row.pred_class1_fraction)
        recalls=[tn/(tn+fp),tp/(tp+fn)]
        assert close(recalls[0],row.class0_recall) and close(recalls[1],row.class1_recall)
        f1=(2*tn/(2*tn+fp+fn)+2*tp/(2*tp+fp+fn))/2
        assert close(f1,row.macro_f1)
        den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        assert close((tp*tn-fp*fn)/den if den else 0,row.mcc)
        direction='class1' if (tp+fp)/n>=.98 else 'class0' if (tn+fn)/n>=.98 else 'none'
        assert bool(row.collapsed)==(direction!='none')
        # Source implementation uses class_0/class_1 labels; compare collapse flag above.
base=pd.read_csv(S/'prepared/baseline_decisions.csv')
rebuilt=d.merge(base,left_on=['baseline_job_id','target','rule'],right_on=['job_id','target','rule'],suffixes=('_standardized','_raw_features'),validate='one_to_one')
assert len(rebuilt)==648
for col in pairs.columns:
    if col.startswith('delta_'):
        k=col[6:];assert np.allclose(pairs[col],rebuilt[k+'_standardized']-rebuilt[k+'_raw_features'],atol=1e-12,equal_nan=True)
    elif pd.api.types.is_numeric_dtype(pairs[col]): assert np.allclose(pairs[col],rebuilt[col],atol=1e-12,equal_nan=True)
    else: assert pairs[col].fillna('NA').equals(rebuilt[col].fillna('NA')),col
for col in ['encoder','source','ratio','loss','seed','rows','target_prevalence']:
    assert (pairs[col+'_standardized']==pairs[col+'_raw_features']).all(),col
sel=e[e.endpoint.eq('selected')].merge(d[d.rule.eq('raw')],on=['job_id','target'],suffixes=('_e','_d'),validate='one_to_one')
for k in ['macro_f1','mcc','pred_class1_fraction','auroc','average_precision']:
    assert np.allclose(sel[k+'_e'],sel[k+'_d'],atol=1e-12,equal_nan=True)
pairs['domain']=np.where(pairs.source_standardized.eq(pairs.target),'In domain','Cross domain')
tot=pairs.groupby(['encoder_standardized','domain','rule']).agg(n=('rule','size'),raw_collapse=('collapsed_raw_features','sum'),standardized_collapse=('collapsed_standardized','sum'),raw_f1=('macro_f1_raw_features','mean'),standardized_f1=('macro_f1_standardized','mean'),raw_auroc=('auroc_raw_features','mean'),standardized_auroc=('auroc_standardized','mean')).reset_index()
tot.to_csv(O/'standardization_domain_summary.csv',index=False)
t=pairs[pairs.rule.eq('raw')&pairs.domain.eq('In domain')&pairs.ratio_standardized.eq(.8)&pairs.loss_standardized.eq('unweighted_ce')].groupby(['encoder_standardized','source_standardized']).agg(n=('rule','size'),raw_collapse=('collapsed_raw_features','sum'),standardized_collapse=('collapsed_standardized','sum'),raw_f1=('macro_f1_raw_features','mean'),standardized_f1=('macro_f1_standardized','mean'),standardized_f1_sd=('macro_f1_standardized','std'),raw_auroc=('auroc_raw_features','mean'),standardized_auroc=('auroc_standardized','mean')).reset_index()
t.to_csv(O/'standardization_table14.csv',index=False)
report={'status':'PASS','plan_sha256':summary['plan_sha256'],'aggregate_zip_sha256':sha(S/'completed_audit/Discover_standardization_v1_aggregates.zip'),'aggregate_hashes':{p.name:sha(p) for p in A.iterdir()},'verified_job_records':72,'decision_records':648,'endpoint_records':648,'local_checks':['frozen file/code/plan hashes','exact approved grid','unique keys','confusion-matrix metrics and collapse flags','paired baseline reconstruction and deltas','paired source/target row counts and prevalence','selected endpoint consistency'],'colab_report':'PASS including checkpoint replay; 2160 post-update validation records','local_limit':'Per-example scores, checkpoints and intermediate trajectories were checked by the Colab verifier; not independently downloaded/replayed locally. AUROC/AP are carried from that verifier, not recoverable from confusion matrices.','training_executed_locally':False,'device':env['device'],'torch':env['torch']}
(O/'standardization_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('PASS: frozen provenance, 72 records, aggregate metrics and all 648 baseline pairs. No training.')
