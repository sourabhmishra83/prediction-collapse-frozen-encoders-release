"""Prepare new split manifests only. Do not train or alter frozen inputs."""
import sys,re,hashlib,json
from pathlib import Path
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from common import ROOT,verify_sources,sha256,write_json
def main():
    verify_sources();cfg=json.loads((ROOT/'experiments/grouped_control/spec.json').read_text());report={'split_seed':cfg['split_seed'],'training_executed':False,'datasets':{}}
    dest=ROOT/'experiments/grouped_control/splits';dest.mkdir(parents=True,exist_ok=True)
    for name in cfg['sources']:
        src=ROOT/f'data/inputs/{name}_clean.csv';df=pd.read_csv(src,keep_default_na=False)
        normal=df.text.astype(str).map(lambda t:re.sub(r'\s+',' ',t).strip().casefold())
        key=normal.map(lambda t:hashlib.sha256(t.encode('utf-8')).hexdigest())
        frame=pd.DataFrame({'row_index':np.arange(len(df)),'source_id':df.id.astype(str),'label':df.label.astype(int),'original_split':df.split,'text_group':key})
        groups=frame.groupby('text_group').label.agg(['min','max','size'])
        groups['stratum']=np.where(groups['min']!=groups['max'],'mixed',groups['min'].astype(str))
        assigned={};rng=np.random.default_rng(cfg['split_seed'])
        for stratum in sorted(groups.stratum.unique()):
            keys=groups[groups.stratum==stratum].index.to_numpy();keys=keys[rng.permutation(len(keys))]
            n=len(keys);ntrain=int(n*.8);nval=int(n*.1)
            for i,g in enumerate(keys):assigned[g]='train' if i<ntrain else 'val' if i<ntrain+nval else 'test'
        frame['split']=frame.text_group.map(assigned)
        overlap=int((frame.groupby('text_group').split.nunique()>1).sum())
        if overlap:raise ValueError('Group allocation failed')
        for split in ['train','val','test']:
            if set(frame[frame.split==split].label)!={0,1}:raise ValueError('Missing class in '+name+'/'+split)
        target=dest/f'{name}_split.csv';frame.to_csv(target,index=False)
        counts={split:{str(label):int((part.label==label).sum()) for label in [0,1]} for split,part in frame.groupby('split')}
        report['datasets'][name]={'input_sha256':sha256(src),'rows':len(frame),'text_groups':len(groups),'mixed_label_groups':int((groups.stratum=='mixed').sum()),'split_label_counts':counts,'cross_split_normalized_text_groups':overlap,'manifest_sha256':sha256(target),'allocation_unit':'normalized-text group; all original rows and labels retained','note':'80/10/10 allocation is by group count within pure-label and mixed-label strata; row fractions are approximate.'}
    write_json(ROOT/'experiments/grouped_control/preparation_report.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
