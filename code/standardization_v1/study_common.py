"""Shared, explicit scientific operations for the prospective full-study controls."""
from pathlib import Path
import hashlib, json, random, math
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    temporary.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()

def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def text_key(value):
    return hashlib.sha256(' '.join(str(value).casefold().split()).encode()).hexdigest()

def load_inputs(input_dir,manifest):
    result={}
    for name,entry in manifest.items():
        path=Path(input_dir)/entry['filename']
        if not path.is_file() or digest(path)!=entry['sha256']:
            raise ValueError('Missing or different input: '+str(path))
        df=pd.read_csv(path,keep_default_na=False)
        if not {'text','label'}.issubset(df) or not set(df.label.unique())<={0,1}:
            raise ValueError('Input schema/labels invalid: '+name)
        df['text_group']=df.text.map(text_key);result[name]=df
    return result

def allocate_groups(df,seed):
    groups=df.groupby('text_group').label.agg(['min','max'])
    groups['stratum']=np.where(groups['min']!=groups['max'],'mixed',groups['min'].astype(str))
    rng=np.random.default_rng(seed);assignment={}
    for stratum in sorted(groups.stratum.unique()):
        keys=groups[groups.stratum==stratum].index.to_numpy()
        keys=keys[rng.permutation(len(keys))];n=len(keys);ntrain=int(.8*n);nval=int(.1*n)
        for i,key in enumerate(keys):assignment[key]='train' if i<ntrain else 'val' if i<ntrain+nval else 'test'
    frame=pd.DataFrame({'row_index':np.arange(len(df)),'label':df.label.astype(int),'text_group':df.text_group,'split':df.text_group.map(assignment)})
    if frame.groupby('text_group').split.nunique().max()!=1:raise ValueError('Split group leakage')
    for part in ['train','val','test']:
        if set(frame[frame.split==part].label)!={0,1}:raise ValueError('A split lacks a class; review the allocation')
    return frame

def cohort_rows(split,n,ratio,seed):
    rng=np.random.RandomState(seed);pool=split[split.split=='train'];n1=int(round(n*ratio));n0=n-n1
    chosen=np.concatenate([rng.choice(pool[pool.label==1].row_index.to_numpy(),n1,replace=False),rng.choice(pool[pool.label==0].row_index.to_numpy(),n0,replace=False)])
    return np.sort(chosen)

def core_lr(update,epochs):
    warm=max(1,int(.1*epochs))
    if update<warm:return (update+1)/warm
    return .5*(1+math.cos(math.pi*(update-warm)/max(1,epochs-warm)))

def matched_lr(update,spec):
    warm=spec['warmup_updates'];horizon=spec['scheduler_horizon']
    if update<warm:return (update+1)/warm
    return .5*(1+math.cos(math.pi*(update-warm)/max(1,horizon-warm)))

def measures(y,pred,threshold=.98):
    from sklearn.metrics import f1_score,accuracy_score,matthews_corrcoef,confusion_matrix,recall_score
    q=float(np.mean(pred));recall=recall_score(y,pred,labels=[0,1],average=None,zero_division=0)
    return {'macro_f1':float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(y,pred)),'mcc':float(matthews_corrcoef(y,pred)),
            'pred_class1_fraction':q,'collapsed':q>=threshold or q<=1-threshold,
            'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),
            'class0_recall':float(recall[0]),'class1_recall':float(recall[1]),'rows':len(y)}

def seed_everything(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

class SourceReferences:
    """Fit only source training text; receive-only source neighbors for new queries."""
    def __init__(self,texts,config):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neighbors import NearestNeighbors
        self.config=config
        self.vectorizer=TfidfVectorizer(max_features=config['max_features'],ngram_range=tuple(config['ngram_range']),
            min_df=config['min_df'],sublinear_tf=config['sublinear_tf'],norm='l2',dtype=np.float32)
        self.features=self.vectorizer.fit_transform(texts)
        self.neighbors=NearestNeighbors(metric='cosine',algorithm='brute',n_jobs=1).fit(self.features)
        self.n=len(texts);self.k=min(config['k'],max(1,self.n-1))
        row=[];col=[]
        for start in range(0,self.n,128):
            distances,indices=self.neighbors.kneighbors(self.features[start:start+128],n_neighbors=min(self.k+1,self.n))
            for offset,(ds,ns) in enumerate(zip(distances,indices)):
                receiver=start+offset;kept=0
                for distance,sender in zip(ds,ns):
                    if sender==receiver or 1-float(distance)<=config['minimum_cosine_similarity']:continue
                    row.extend([receiver,int(sender)]);col.extend([int(sender),receiver]);kept+=1
                    if kept==self.k:break
        pairs=sorted(set(zip(row,col)))
        self.train_edges=np.array(pairs,dtype=np.int64).T if pairs else np.empty((2,0),dtype=np.int64)
        self.vocabulary_sha256=fingerprint(self.vectorizer.get_feature_names_out().tolist())

    def query_edges(self,texts):
        features=self.vectorizer.transform(texts);row=[];col=[]
        for start in range(0,len(texts),128):
            ds,ns=self.neighbors.kneighbors(features[start:start+128],n_neighbors=self.k)
            for offset,(distances,neighbors) in enumerate(zip(ds,ns)):
                for distance,sender in zip(distances,neighbors):
                    if 1-float(distance)>self.config['minimum_cosine_similarity']:
                        row.append(self.n+start+offset);col.append(int(sender))
        extra=np.array([row,col],dtype=np.int64)
        if extra.size and (extra[0].min()<self.n or extra[1].max()>=self.n):raise ValueError('Query-edge leakage')
        return np.concatenate([self.train_edges,extra],axis=1)

def make_model(kind,dim,hidden,dropout):
    import torch
    from torch import nn
    class Layer(nn.Module):
        def __init__(self,in_dim,out_dim,attention):
            super().__init__();self.attention=attention
            self.linear=nn.Linear(in_dim,out_dim,bias=not attention);self.drop=nn.Dropout(dropout)
            if attention:self.a_src=nn.Linear(out_dim,1,bias=False);self.a_dst=nn.Linear(out_dim,1,bias=False)
        def forward(self,x,edge):
            n=len(x);loops=torch.arange(n,device=x.device)
            row=torch.cat([edge[0],loops]);col=torch.cat([edge[1],loops]);h=self.linear(x)
            if self.attention:
                score=torch.nn.functional.leaky_relu(self.a_src(h).squeeze(-1)[row]+self.a_dst(h).squeeze(-1)[col],negative_slope=.2)
                maximum=torch.full((n,),-torch.inf,device=x.device).scatter_reduce_(0,row,score,reduce='amax',include_self=True)
                numerator=torch.exp(score-maximum[row]);denom=torch.zeros(n,device=x.device).index_add_(0,row,numerator)
                weight=self.drop(numerator/denom[row].clamp(min=1e-12));messages=h[col]*weight[:,None]
            else:
                degree=torch.zeros(n,device=x.device).index_add_(0,row,torch.ones(len(row),device=x.device))
                inverse=degree.clamp(min=1).pow(-.5);h=self.drop(h);messages=h[col]*(inverse[row]*inverse[col])[:,None]
            return torch.zeros_like(h).index_add_(0,row,messages)
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.kind=kind;self.drop=nn.Dropout(dropout)
            if kind=='tbase':self.fc1=nn.Linear(dim,hidden);self.fc2=nn.Linear(hidden,2)
            else:
                self.l1=Layer(dim,hidden,kind=='gat');self.l2=Layer(hidden,hidden,kind=='gat')
                self.fuse=nn.Linear(dim+hidden,hidden);self.out=nn.Linear(hidden,2)
        def forward(self,x,edge=None):
            if self.kind=='tbase':return self.fc2(self.drop(torch.relu(self.fc1(x))))
            act=torch.nn.functional.elu if self.kind=='gat' else torch.relu
            graph=act(self.l1(x,edge));graph=act(self.l2(graph,edge))
            return self.out(self.drop(torch.relu(self.fuse(torch.cat([x,graph],dim=1)))))
    return Model()
