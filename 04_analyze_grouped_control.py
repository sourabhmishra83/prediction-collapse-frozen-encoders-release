"""Verify per-example outcomes before summarizing completed grouped controls."""
import sys, json, math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from common import ROOT, read_json, write_json, verify_sources, sha256

def main():
    verify_sources()
    base=ROOT/'experiments/grouped_control'
    plan=read_json(base/'run_plan.json'); rows=[]
    for job in plan['jobs']:
        folder=base/'results'/job['job_id']
        if not (folder/'result.json').exists(): continue
        result=read_json(folder/'result.json')
        if result['job']!=job: raise ValueError('Configuration mismatch')
        for key,digest in [('input','input_sha256'),('split_manifest','split_sha256'),('cohort_manifest','cohort_sha256')]:
            assert sha256(ROOT/job[key])==job[digest]
        pred=pd.read_csv(folder/'test_predictions.csv')
        split=pd.read_csv(ROOT/job['split_manifest']); test=split[split.split=='test'].row_index.to_numpy()
        truth=pd.read_csv(ROOT/job['input'],keep_default_na=False).label.to_numpy()[test]
        assert np.array_equal(pred.row_index,test) and np.array_equal(pred.label,truth)
        trajectory=read_json(folder/'trajectory.json')
        assert len(trajectory['epochs'])==30
        best=max(trajectory['epochs'],key=lambda r:r['macro_f1'])
        assert best['epoch_index']==result['selected_epoch_index']
        row={k:job[k] for k in ['job_id','source','target_fake_ratio','seed','loss','fixed_n','cohort_sha256']}
        row['initialization_collapsed']=result['initialization_validation']['collapsed']
        row['training_ever_collapsed']=result['training_ever_collapsed']
        row['selected_epoch_index']=result['selected_epoch_index']
        for endpoint in ['selected','final']:
            yhat=pred[endpoint+'_prediction'].to_numpy(); assert set(yhat)<= {0,1}
            q=float(yhat.mean())
            measured={'macro_f1':float(f1_score(truth,yhat,average='macro')),'accuracy':float(accuracy_score(truth,yhat)),'mcc':float(matthews_corrcoef(truth,yhat)),'pred_fake_frac':q,'collapsed':q>=.98 or q<=.02}
            for key,value in measured.items():
                assert abs(float(value)-float(result[endpoint+'_test'][key]))<1e-10
                row[endpoint+'_'+key]=value
        rows.append(row)
    out=base/'analysis';out.mkdir(exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/'verified_runs.csv',index=False)
    report={'completed_jobs':len(rows),'planned_jobs':len(plan['jobs']),'all_predictions_verified':True,'complete':len(rows)==len(plan['jobs'])}
    if rows:
        groups=[]
        for (source,ratio,loss),g in df.groupby(['source','target_fake_ratio','loss']):
            r={'source':source,'target_fake_ratio':ratio,'loss':loss,'n':len(g),'selected_macro_f1_mean':g.selected_macro_f1.mean(),'selected_macro_f1_sd':g.selected_macro_f1.std(ddof=1)}
            for endpoint in ['selected','final']:
                k=int(g[endpoint+'_collapsed'].sum());n=len(g);z=1.959963984540054;p=k/n;den=1+z*z/n
                mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
                r.update({endpoint+'_collapsed':k,endpoint+'_collapse_wilson_low':mid-half,endpoint+'_collapse_wilson_high':mid+half})
            groups.append(r)
        pd.DataFrame(groups).to_csv(out/'condition_summary.csv',index=False)
        paired=[]
        for key,g in df.groupby(['source','target_fake_ratio','seed']):
            if len(g)!=2:continue
            a=g[g.loss=='unweighted_ce'].iloc[0];b=g[g.loss=='weighted_ce'].iloc[0]
            assert a.cohort_sha256==b.cohort_sha256
            paired.append({'source':key[0],'target_fake_ratio':key[1],'seed':int(key[2]),'selected_macro_f1_delta_weighted_minus_unweighted':b.selected_macro_f1-a.selected_macro_f1,'unweighted_selected_collapsed':bool(a.selected_collapsed),'weighted_selected_collapsed':bool(b.selected_collapsed)})
        pd.DataFrame(paired).to_csv(out/'paired_differences.csv',index=False)
        report['verified_pairs']=len(paired)
    write_json(out/'verification.json',report)
    text='# Grouped-text control execution\n\n'+f"Verified {len(rows)} of {len(plan['jobs'])} planned training jobs.\n\n"
    if report['complete']:
        text+='All test predictions, labels, selected-epoch rules, and paired cohort hashes passed verification. See condition_summary.csv and paired_differences.csv for results. Five training seeds share one grouped split; uncertainty across seeds does not establish robustness across alternative splits. A zero count is not proof of zero population risk.\n'
    else:text+='Partial results only; do not use these as the completed experiment in the manuscript.\n'
    (out/'RESULTS.md').write_text(text,encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
