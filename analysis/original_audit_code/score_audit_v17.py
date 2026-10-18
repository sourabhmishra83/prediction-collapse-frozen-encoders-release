from pathlib import Path
import json,hashlib,sys
import torch,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,matthews_corrcoef,recall_score
R=Path(__file__).resolve().parents[1];A=R/'experiments/score_audit_private';O=R/'deliverables/paper_v1_7';B=R/'experiments/second_encoder';F=R/'full_study'
sys.path.insert(0,str(F));from study_common import make_model,digest,load_inputs
torch.set_num_threads(1);data=load_inputs(F/'inputs',json.loads((F/'input_manifest.json').read_text()))
plan=json.loads((B/'prepared/run_plan.json').read_text());vectors={};cachemeta=[]
for enc in ['RoBERTa','BERT']:
 for source in ['coaid','fakenewsnet','liar']:
  folder=A/'cache'/enc/source;meta=json.loads((folder/'metadata.json').read_text());assert digest(folder/'vectors.pt')==meta['sha256']
  vectors[enc,source]=torch.load(folder/'vectors.pt',map_location='cpu',weights_only=True)
  assert len(vectors[enc,source])==len(data[source]);cachemeta.append({'encoder':enc,'source':source,'metadata':meta})
rows=[];private=A/'replayed_scores';private.mkdir(exist_ok=True)
for enc in ['RoBERTa','BERT']:
 for job in plan['jobs']:
  jobid=job['job_id'] if enc=='BERT' else job['reference_roberta_job_id']
  folder=B/'completed_audit/downloaded/results'/jobid if enc=='BERT' else A/'roberta_heads'/jobid
  result=json.loads((folder/'result.json').read_text());assert digest(folder/'selected_model.pt')==result['files_sha256']['selected_model.pt']
  model=make_model('tbase',768,256,.3).eval();model.load_state_dict(torch.load(folder/'selected_model.pt',map_location='cpu',weights_only=True))
  src=pd.read_csv(B/'prepared'/job['split']);groups=set(src[src.split=='train'].text_group)
  for target in ['coaid','fakenewsnet','liar']:
   split=pd.read_csv(B/f'prepared/splits/{target}_s42.csv');test=split[(split.split=='test')&~split.text_group.isin(groups)];ids=test.row_index.to_numpy();y=test.label.to_numpy()
   with torch.no_grad():logits=model(vectors[enc,target][ids]).numpy()
   pred=logits.argmax(1);m=result['test_targets'][target]['selected']
   assert abs(f1_score(y,pred,average='macro',zero_division=0)-m['macro_f1'])<1e-12,(enc,jobid,target)
   assert abs(matthews_corrcoef(y,pred)-m['mcc'])<1e-12
   assert abs(pred.mean()-m['pred_class1_fraction'])<1e-12
   if enc=='BERT':assert np.array_equal(pred,pd.read_csv(folder/f'test_{target}.csv').selected_prediction)
   margin=logits[:,1].astype(np.float64)-logits[:,0].astype(np.float64)
   pd.DataFrame({'row_index':ids,'label':y,'logit_margin_class1':margin}).to_csv(private/f'{enc}_{jobid}_{target}.csv',index=False)
   rows.append({'encoder':enc,'job_id':jobid,'source':job['source'],'target':target,'ratio':job['ratio'],'loss':job['loss'],'seed':job['seed'],'in_domain':target==job['source'],'endpoint':'selected','rows':len(y),'prevalence':y.mean(),'pred_share':pred.mean(),'collapsed':m['collapsed'],'auroc':roc_auc_score(y,margin),'average_precision':average_precision_score(y,margin),'macro_f1':m['macro_f1'],'mcc':m['mcc']})
df=pd.DataFrame(rows);df.to_csv(O/'selected_score_metrics.csv',index=False)
df.groupby(['encoder','source','target','ratio','loss']).agg(evaluations=('collapsed','size'),collapsed=('collapsed','sum'),prevalence=('prevalence','mean'),pred_share=('pred_share','mean'),auroc=('auroc','mean'),average_precision=('average_precision','mean'),macro_f1=('macro_f1','mean')).reset_index().to_csv(O/'selected_score_summary.csv',index=False)
(O/'score_replay_audit.json').write_text(json.dumps({'status':'PASS','trained_heads_replayed':72,'source_target_evaluations':len(rows),'cache_sha256_verified':6,'selected_head_file_hashes_verified':72,'hard_metric_replay_matches':len(rows),'BERT_per_example_predictions_match':108,'score':'class-1 logit minus class-0 logit; larger means class 1','AUROC_AP_positive_class':1,'new_training_executed':False,'scope':'Selected text-only checkpoints; split 42; both shares and losses; three seeds; no final/initial RoBERTa or graph score claims'},indent=2))
(A/'verified_cache_metadata.json').write_text(json.dumps(cachemeta,indent=2))
print(df[(df.ratio==.8)&df.in_domain].groupby(['encoder','source','loss']).agg(collapse=('collapsed','sum'),auroc=('auroc','mean'),AP=('average_precision','mean'),f1=('macro_f1','mean')).to_string());print('PASS',len(rows),'score evaluations, zero optimizer steps')
