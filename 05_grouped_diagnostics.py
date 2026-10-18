"""Create aggregate confusion, recall, threshold and constant-prediction tables."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, f1_score

base=Path(__file__).resolve().parent/'experiments/grouped_control'
rows=[];thresholds=[];baselines=[]
for p in sorted((base/'results').glob('*/result.json')):
    result=json.loads(p.read_text());job=result['job']
    pred=pd.read_csv(p.parent/'test_predictions.csv')
    row={k:job[k] for k in ['job_id','source','target_fake_ratio','loss','seed']}
    for endpoint in ['selected','final']:
        y=pred.label.to_numpy();h=pred[endpoint+'_prediction'].to_numpy()
        cm=confusion_matrix(y,h,labels=[0,1]);recall=recall_score(y,h,labels=[0,1],average=None)
        for key,value in zip(['tn','fp','fn','tp'],cm.ravel()):row[endpoint+'_'+key]=int(value)
        row[endpoint+'_real_recall']=float(recall[0]);row[endpoint+'_fake_recall']=float(recall[1])
        q=float(h.mean())
        for t in [.90,.95,.98,.99]:
            thresholds.append({**{k:job[k] for k in ['job_id','source','target_fake_ratio','loss','seed']},'endpoint':endpoint,'threshold':t,'collapsed':q>=t or q<=1-t})
    rows.append(row)
    for c in [0,1]:baselines.append({'source':job['source'],'constant_prediction':c,'test_macro_f1':float(f1_score(y,np.full(len(y),c),average='macro')),'test_accuracy':float(np.mean(y==c))})
out=base/'analysis'
pd.DataFrame(rows).to_csv(out/'class_diagnostics.csv',index=False)
pd.DataFrame(thresholds).groupby(['source','target_fake_ratio','loss','endpoint','threshold']).collapsed.agg(['sum','count']).reset_index().to_csv(out/'threshold_sensitivity.csv',index=False)
pd.DataFrame(baselines).drop_duplicates().to_csv(out/'constant_prediction_baselines.csv',index=False)
print('Aggregate diagnostics saved for',len(rows),'completed jobs')
