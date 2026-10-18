"""Prospective three-class weakly supervised stance experiment; CUDA by default."""
import argparse,copy,hashlib,json,platform,time,uuid
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import f1_score,recall_score,confusion_matrix,accuracy_score,matthews_corrcoef
from stance_kernels import read,write,digest,fingerprint,seed_everything,make_model,SourceReferences,core_lr
HERE=Path(__file__).resolve().parent
SPEC={'version':'monant_stance_v1','seeds':[0,1,2],'models':['tbase','gcn','gat'],'losses':['unweighted_ce','weighted_ce'],'updates':30,'hidden':256,'dropout':.3,'lr':.001,'weight_decay':.0001,'clip_norm':1.,'collapse_threshold':.98,'encoder':{'name':'roberta-base','revision':'e2da8e2f811d1448a5b465c236feacd80ffbac7b','max_length':256,'truncation':'longest_first','batch_size':16},'graph':{'k':10,'max_features':20000,'ngram_range':[1,2],'min_df':2,'sublinear_tf':True,'minimum_cosine_similarity':1e-8},'selection':'earliest best automatic-validation macro-F1 after updates; initialization excluded','test':'all 308 completed unique expert stance pairs; no tuning on test','labels':{'supporting':0,'contradicting':1,'neutral':2}}

def metrics(y,p):
 count=np.bincount(p,minlength=3);return {'macro_f1':float(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,p)),'mcc':float(matthews_corrcoef(y,p)),'recall':recall_score(y,p,labels=[0,1,2],average=None,zero_division=0).tolist(),'confusion_matrix':confusion_matrix(y,p,labels=[0,1,2]).tolist(),'prediction_counts':count.tolist(),'collapsed':bool(count.max()/len(p)>=.98),'rows':len(y)}

def load_inputs(folder):
 m=read(Path(folder)/'input_manifest.json');p=Path(folder)/m['file'];assert digest(p)==m['sha256'];d=pd.read_csv(p,keep_default_na=False)
 assert d.row_id.is_unique and set(d.label_id)=={0,1,2}
 for field in ['article_id','claim_id','article_group','claim_group']:
  groups=[set(d[d.split==part][field]) for part in ['train','validation','test']]
  assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
 return d,m

def execute(input_dir,out,spec=None,synthetic_vectors=None):
 spec=copy.deepcopy(spec or SPEC);out=Path(out);out.mkdir(parents=True,exist_ok=True);d,manifest=load_inputs(input_dir)
 device=torch.device('cpu' if synthetic_vectors is not None else 'cuda');torch.set_num_threads(4)
 if device.type=='cuda':
  if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required. Select GPU runtime before executing.')
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 import transformers,sklearn,scipy
 record={'spec':spec,'input_sha256':manifest['sha256'],'code_sha256':{p.name:digest(p) for p in HERE.glob('*.py')},'environment':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'tf32':False,'sparse_scatter_bitwise_determinism_not_guaranteed':True},'synthetic_only':synthetic_vectors is not None}
 if (out/'execution_contract.json').exists() and read(out/'execution_contract.json')!=record:raise ValueError('Frozen runtime/code/input changed; choose a new RUN_LABEL')
 write(out/'execution_contract.json',record)
 # Frozen row identities/labels support result verification without restricted texts.
 index=d[['row_id','label_id','split','article_group','claim_group']];index.to_csv(out/'evaluation_index.csv',index=False)
 plan=[{'model':model,'loss':loss,'seed':seed,'job_id':f'{model}_{loss}_s{seed}'} for model in spec['models'] for loss in spec['losses'] for seed in spec['seeds']]
 for job in plan:job['job_sha256']=fingerprint(job)
 write(out/'run_plan.json',{'jobs':plan,'contract_sha256':fingerprint(record),'planned_training_jobs':len(plan)})
 # Populate all embeddings before resetting any paired head/training RNG seed.
 cache=out/'features.pt'
 if synthetic_vectors is not None:vectors=synthetic_vectors.float()
 elif cache.exists():
  assert digest(cache)==read(out/'features.json')['sha256'];vectors=torch.load(cache,map_location='cpu',weights_only=True)
 else:
  from transformers import AutoTokenizer,AutoModel
  cfg=spec['encoder'];tok=AutoTokenizer.from_pretrained(cfg['name'],revision=cfg['revision']);enc=AutoModel.from_pretrained(cfg['name'],revision=cfg['revision'],add_pooling_layer=False).to(device).eval();chunks=[]
  with torch.no_grad():
   for start in range(0,len(d),cfg['batch_size']):
    part=d.iloc[start:start+cfg['batch_size']];t=tok(part.claim_text.tolist(),part.article_text.tolist(),truncation=cfg['truncation'],padding=True,max_length=cfg['max_length'],return_tensors='pt');t={k:v.to(device) for k,v in t.items()};chunks.append(enc(**t).last_hidden_state[:,0,:].cpu())
    if start%(cfg['batch_size']*25)==0:print(f'Encoding claim/article pairs {start}/{len(d)}',flush=True)
  vectors=torch.cat(chunks);torch.save(vectors,cache);write(out/'features.json',{'sha256':digest(cache),'rows':len(d),'dimension':vectors.shape[1]});del enc
  torch.cuda.empty_cache()
 assert len(vectors)==len(d) and torch.isfinite(vectors).all()
 parts={part:d.index[d.split==part].to_numpy() for part in ['train','validation','test']};x={p:vectors[rows].to(device) for p,rows in parts.items()};y={p:torch.as_tensor(d.iloc[rows].label_id.to_numpy(),device=device,dtype=torch.long) for p,rows in parts.items()};refs=None;edges={}
 def graph_edges():
  nonlocal refs,edges
  if refs is not None:return
  texts={p:(d.iloc[rows].claim_text+' [ARTICLE] '+d.iloc[rows].article_text).tolist() for p,rows in parts.items()}
  refs=SourceReferences(texts['train'],spec['graph']);edges['train']=torch.as_tensor(refs.train_edges,device=device)
  for part in ['validation','test']:edges[part]=torch.as_tensor(refs.query_edges(texts[part]),device=device)
  np.savez_compressed(out/'graph_edges.npz',**{p:e.cpu().numpy() for p,e in edges.items()})
  write(out/'graph_provenance.json',{'fit':'training pair text only','k':spec['graph']['k'],'vocabulary_sha256':refs.vocabulary_sha256,'query_to_training_messages':False,'query_to_query_messages':False,'edge_orientation':'receiver,sender','training_rows':len(x['train'])})
 def evaluate(model,part):
  model.eval()
  with torch.no_grad():
   logits=model(x[part]) if model.kind=='tbase' else model(torch.cat([x['train'],x[part]]),edges[part])[len(x['train']):]
   assert torch.isfinite(logits).all();pred=logits.argmax(-1).cpu().numpy();return metrics(y[part].cpu().numpy(),pred),pred
 majority=int(torch.bincount(y['train'],minlength=3).argmax());write(out/'constant_baseline.json',{'class_chosen_from_training':majority,'test':metrics(y['test'].cpu().numpy(),np.full(len(y['test']),majority))})
 for job in plan:
  dest=out/'results'/job['job_id']
  if (dest/'result.json').exists():
   result=read(dest/'result.json');assert result['job']==job
   for name,h in result['files_sha256'].items():assert digest(dest/name)==h
   print('Verified completed',job['job_id'],flush=True);continue
  if dest.exists():raise FileExistsError('Unexpected incomplete final directory; preserve and investigate')
  if job['model']!='tbase':graph_edges()
  seed_everything(job['seed']);model=make_model(job['model'],vectors.shape[1],spec['hidden'],spec['dropout'],classes=3).to(device)
  initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};initial_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in initial.values())).hexdigest();initval,_=evaluate(model,'validation')
  weights=None
  if job['loss']=='weighted_ce':weights=len(y['train'])/(3*torch.bincount(y['train'],minlength=3).float())
  criterion=torch.nn.CrossEntropyLoss(weight=weights);optimizer=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay']);trajectory=[];best=-1;best_state=None;best_update=None
  for update in range(spec['updates']):
   model.train();optimizer.zero_grad();lr=spec['lr']*core_lr(update,spec['updates'])
   for group in optimizer.param_groups:group['lr']=lr
   logits=model(x['train']) if job['model']=='tbase' else model(x['train'],edges['train']);loss=criterion(logits,y['train']);assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),spec['clip_norm']);optimizer.step();m,_=evaluate(model,'validation');trajectory.append({'optimizer_update':update+1,'learning_rate':lr,'training_loss':float(loss.item()),'validation':m})
   if m['macro_f1']>best:best=m['macro_f1'];best_update=update+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   if (update+1)%10==0:print(job['job_id'],update+1,'/',spec['updates'],flush=True)
  final_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};temporary=dest.parent/(dest.name+'.temporary_'+uuid.uuid4().hex);temporary.mkdir(parents=True)
  endpoints={};predictions={part:pd.DataFrame({'row_id':d.iloc[parts[part]].row_id.to_numpy(),'label_id':y[part].cpu().numpy()}) for part in ['validation','test']}
  for name,state in [('initial',initial),('selected',best_state),('final',final_state)]:
   model.load_state_dict(state);endpoints[name]={}
   for part in ['validation','test']:
    m,pred=evaluate(model,part);endpoints[name][part]=m;predictions[part][name+'_prediction']=pred
  for part,frame in predictions.items():frame.to_csv(temporary/(part+'_predictions.csv'),index=False)
  torch.save(best_state,temporary/'selected_model.pt');write(temporary/'trajectory.json',{'initial_validation':initval,'updates':trajectory})
  write(temporary/'result.json',{'job':job,'selected_optimizer_update':best_update,'initial_parameter_sha256':initial_hash,'endpoints':endpoints,'training_ever_validation_collapsed':any(t['validation']['collapsed'] for t in trajectory),'synthetic_only':synthetic_vectors is not None,'files_sha256':{p.name:digest(p) for p in temporary.iterdir()}});temporary.rename(dest);print('Completed',job['job_id'],flush=True)
 write(out/'completion.json',{'completed_selected_jobs':len(plan),'planned_jobs':len(plan),'synthetic_only':synthetic_vectors is not None})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--input-dir',required=True);ap.add_argument('--out',required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 if not args.execute:ap.error('--execute is required; preparation is not execution authorization')
 execute(args.input_dir,args.out)
