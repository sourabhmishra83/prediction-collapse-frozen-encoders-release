"""Read-only public evidence checks, requiring Python's standard library only."""
from pathlib import Path
import ast, csv, hashlib, json, math

ROOT=Path(__file__).resolve().parents[1]
def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-11)
def flag(s):return str(s).lower() in ['true','1']
def coverage(name,n,jobs,keys,endpoints=None):
    data=rows(name)
    assert len(data)==n,(name,len(data))
    assert len({r['job_id'] for r in data})==jobs,name
    assert len({tuple(r[k] for k in keys) for r in data})==n,name
    if endpoints:assert {r['endpoint'] for r in data}==set(endpoints),name
    return data
def hard_metrics(r):
    cm=ast.literal_eval(r['confusion_matrix']);tn,fp=map(int,cm[0]);fn,tp=map(int,cm[1]);n=tn+fp+fn+tp
    assert n==int(r['rows']) and n>0
    p=(tp+fp)/n;assert close(p,r['pred_class1_fraction'])
    assert close((tp+fn)/n,r['target_prevalence'])
    assert close((tn+tp)/n,r['accuracy'])
    f0=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
    assert close((f0+f1)/2,r['macro_f1'])
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    assert close((tp*tn-fp*fn)/den if den else 0,r['mcc'])
    assert close(tn/(tn+fp) if tn+fp else 0,r['class0_recall'])
    assert close(tp/(tp+fn) if tp+fn else 0,r['class1_recall'])
    assert flag(r['collapsed'])==(p<=.02 or p>=.98)

manifest=json.loads((ROOT/'release_manifest.json').read_text())
for name,expected in manifest.items():
    path=ROOT/name;assert path.is_file(),name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
coverage('results/full_study/verified_endpoints.csv',3888,432,['job_id','target','endpoint'],['initial','selected','final'])
coverage('results/monant_stance_v1/Monant_verified_endpoint_metrics.csv',108,18,['job_id','population','endpoint'],['initial','selected','final'])
coverage('results/second_encoder/verified_endpoints.csv',324,36,['job_id','target','endpoint'],['initial','selected','final'])
linear=coverage('results/analysis/v1_8/linear_decision_evaluations.csv',648,72,['job_id','target','rule'])
other=coverage('results/analysis/v1_8/mirror_decision_evaluations.csv',1296,144,['job_id','target','rule'])
assert len({r['job_id'] for r in linear+other})==216
standard=coverage('results/standardization/decision_evaluations.csv',648,72,['job_id','target','rule'])
endpoints=coverage('results/standardization/endpoint_evaluations.csv',648,72,['job_id','target','endpoint'],['initial','selected','final'])
assert {r['job_id'] for r in standard}=={r['job_id'] for r in endpoints}
for r in linear+other+standard+endpoints:hard_metrics(r)
lookup={(r['job_id'],r['target'],r['rule']):r for r in linear+other}
raw_lookup={(r['job_id'],r['target']):r for r in endpoints if r['endpoint']=='selected'}
for r in standard:
    assert r['rows']==lookup[r['baseline_job_id'],r['target'],r['rule']]['rows']
    if r['rule']=='raw':
        for field in ['macro_f1','auroc','average_precision','mcc','pred_class1_fraction']:
            assert close(r[field],raw_lookup[r['job_id'],r['target']][field])
inside=[r for r in standard if r['rule']=='raw' and r['source']==r['target']]
outside=[r for r in standard if r['rule']=='raw' and r['source']!=r['target']]
assert len(inside)==72 and sum(flag(r['collapsed']) for r in inside)==0
assert len(outside)==144 and sum(flag(r['collapsed']) for r in outside)==12
assert sum(flag(lookup[r['baseline_job_id'],r['target'],'raw']['collapsed']) for r in inside)==14
assert sum(flag(lookup[r['baseline_job_id'],r['target'],'raw']['collapsed']) for r in outside)==50
diagnostics=rows('results/standardization/job_diagnostics.csv')
assert len(diagnostics)==72 and len({r['job_id'] for r in diagnostics})==72
assert sum(flag(r['training_ever_validation_collapse']) for r in diagnostics)==3
assert all(r['encoder']=='BERT' and float(r['ratio'])==.8 and r['loss']=='unweighted_ce' for r in diagnostics if flag(r['training_ever_validation_collapse']))
assert all(not flag(r['collapsed']) for r in endpoints if r['source']==r['target'])
completion=json.loads((ROOT/'results/standardization/completion.json').read_text())
assert completion['status']=='PASS' and completion['checkpoint_weight_replay'] and completion['verified_complete_job_records']==72
assert not list(ROOT.rglob('*.pt')) and not list(ROOT.rglob('*.docx'))
print(f'PASS: {len(manifest)} release hashes; complete unique aggregate keys for all declared panels.')
print('PASS: 3240 aggregate confusion-matrix metric records; paired baseline/endpoint consistency.')
print('PASS: standardized selected in-domain collapse 14/72 -> 0/72; cross-domain 50/144 -> 12/144; three transient validation cases.')
print('No fitting, encoder inference, source-data download or private per-example/checkpoint replay occurred.')
print('AUROC/AP and trajectory/replay claims rely on the archived private-data verifiers; record counts are not independent replications.')
