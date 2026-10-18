"""Execute the new controls only with --execute; never overwrite completed jobs."""
import argparse, copy, hashlib, json, os, platform, sys, time, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from study_common import HERE,read,write,digest,fingerprint,load_inputs,SourceReferences,make_model,measures,seed_everything,core_lr,matched_lr

def environment(device):
    import sklearn, scipy, transformers
    return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'pandas':pd.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,
        'transformers':transformers.__version__,'device':str(device),
        'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        'cuda':torch.version.cuda,'tf32':False,'sparse_scatter_bitwise_determinism_not_guaranteed':True}

class Runner:
    def __init__(self,input_dir,out,device,synthetic_vectors=None):
        self.out=Path(out);self.contract=read(self.out/'contract.json');self.spec=self.contract['spec']
        self.plan=read(self.out/'run_plan.json');self.device=torch.device(device)
        if self.plan['contract_sha256']!=fingerprint(self.contract):raise ValueError('Plan/contract mismatch')
        self.data=load_inputs(input_dir,self.contract['inputs']);self.vectors={};self.contexts={};self.queries={}
        self.synthetic=synthetic_vectors
        torch.set_num_threads(4)
        if self.device.type=='cuda':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        code={p.name:digest(p) for p in HERE.glob('*.py')}
        record={'environment':environment(self.device),'code_sha256':code,'contract_sha256':fingerprint(self.contract),
                'synthetic_smoke_only':synthetic_vectors is not None}
        p=self.out/'execution_environment.json'
        if p.exists() and read(p)!=record:raise ValueError('Code/runtime/contract differs; use a new output directory')
        write(p,record)

    def encode(self,name):
        if name in self.vectors:return self.vectors[name]
        if self.synthetic is not None:
            self.vectors[name]=self.synthetic[name].float().cpu();return self.vectors[name]
        cfg=self.spec['encoder'];key=fingerprint({'encoder':cfg,'input':self.contract['inputs'][name],
                                                'environment':environment(self.device)})
        folder=self.out/'embedding_cache'/name;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'metadata.json').exists():
            meta=read(folder/'metadata.json')
            if meta['key']!=key or digest(folder/'vectors.pt')!=meta['sha256']:raise ValueError('Embedding cache mismatch')
            value=torch.load(folder/'vectors.pt',map_location='cpu',weights_only=True)
        else:
            from transformers import AutoTokenizer,AutoModel
            tokenizer=AutoTokenizer.from_pretrained(cfg['name'],revision=cfg['revision'])
            encoder=AutoModel.from_pretrained(cfg['name'],revision=cfg['revision'],add_pooling_layer=False).to(self.device).eval()
            chunks=[];texts=self.data[name].text.astype(str).tolist()
            with torch.no_grad():
                for start in range(0,len(texts),cfg['batch_size']):
                    tokens=tokenizer(texts[start:start+cfg['batch_size']],padding=True,truncation=True,
                        max_length=cfg['maximum_tokens'],return_tensors='pt')
                    tokens={k:v.to(self.device) for k,v in tokens.items()}
                    chunks.append(encoder(**tokens).last_hidden_state[:,0,:].cpu())
                    if start%(cfg['batch_size']*10)==0:
                        print(f'Encoding {name}: {min(start+cfg["batch_size"],len(texts))}/{len(texts)}',flush=True)
            value=torch.cat(chunks);torch.save(value,folder/'vectors.pt')
            write(folder/'metadata.json',{'key':key,'sha256':digest(folder/'vectors.pt'),'rows':len(value),'dimension':value.shape[1]})
            del encoder
            if self.device.type=='cuda':torch.cuda.empty_cache()
        if len(value)!=len(self.data[name]):raise ValueError('Embedding/input row mismatch')
        self.vectors[name]=value;return value

    def context(self,source,split_seed,graph):
        key=(source,split_seed)
        if key not in self.contexts:
            split=pd.read_csv(self.out/f'splits/{source}_s{split_seed}.csv')
            rows=split[split.split=='train'].row_index.to_numpy()
            self.contexts[key]={'split':split,'rows':rows,'x':self.encode(source)[rows].to(self.device),
                'groups':set(split[split.split=='train'].text_group),'references':None}
        context=self.contexts[key]
        if graph and context['references'] is None:
            refs=SourceReferences(self.data[source].text.iloc[context['rows']].astype(str).tolist(),self.spec['graph'])
            context['references']=refs;context['edge']=torch.as_tensor(refs.train_edges,device=self.device)
            folder=self.out/'graph_records'/f'{source}_s{split_seed}';folder.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(folder/'training_edges.npz',edges=refs.train_edges)
            write(folder/'source_reference.json',{'source':source,'split_seed':split_seed,'reference_rows':len(context['rows']),
                'reference_row_order_sha256':fingerprint(context['rows'].tolist()),'vocabulary_sha256':refs.vocabulary_sha256,
                'training_edges_sha256':hashlib.sha256(refs.train_edges.tobytes()).hexdigest(),
                'fitting_rows':'source training pool only','query_to_training_messages':False,'graph':self.spec['graph']})
        return context

    def query(self,source,split_seed,target,part,graph):
        key=(source,split_seed,target,part,graph)
        if key not in self.queries:
            context=self.context(source,split_seed,graph)
            split=pd.read_csv(self.out/f'splits/{target}_s{split_seed}.csv');original=split[split.split==part]
            selected=original[~original.text_group.isin(context['groups'])]
            rows=selected.row_index.to_numpy()
            if not len(rows):raise ValueError('No novel held-out rows in '+str(key))
            item={'rows':rows,'labels':selected.label.to_numpy(),'excluded_overlap_rows':len(original)-len(selected),
                  'x':self.encode(target)[rows].to(self.device)}
            if graph:
                edges=context['references'].query_edges(self.data[target].text.iloc[rows].astype(str).tolist())
                item['edge']=torch.as_tensor(edges,device=self.device)
                folder=self.out/'graph_records'/f'{source}_s{split_seed}'
                extra=edges[:,context['references'].train_edges.shape[1]:]
                np.savez_compressed(folder/f'{target}_{part}_query_edges.npz',edges=extra)
                write(folder/f'{target}_{part}.json',{'target':target,'part':part,'query_rows':rows.tolist(),
                    'excluded_text_matches':item['excluded_overlap_rows'],'source_reference_only':True,
                    'combined_edges_sha256':hashlib.sha256(edges.tobytes()).hexdigest()})
            self.queries[key]=item
        return self.queries[key]

    def evaluate(self,model,source,split_seed,target,part):
        graph=model.kind!='tbase';context=self.context(source,split_seed,graph)
        query=self.query(source,split_seed,target,part,graph);model.eval()
        with torch.no_grad():
            if graph:
                x=torch.cat([context['x'],query['x']]);logits=model(x,query['edge'])[len(context['x']):]
            else:logits=model(query['x'])
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite evaluation logits')
            pred=logits.argmax(-1)
        pred=pred.cpu().numpy()
        return measures(query['labels'],pred,self.spec['collapse_threshold']),pred,query

    def execute(self,job,resume=False):
        dest=self.out/'results'/job['job_id']
        if (dest/'result.json').exists():
            if not resume:raise FileExistsError('Completed result exists: '+str(dest))
            saved=read(dest/'result.json')
            if saved['job']!=job:raise ValueError('Completed job configuration differs')
            for name,h in saved['files_sha256'].items():
                if digest(dest/name)!=h:raise ValueError('Completed job file differs')
            print('Verified completed:',job['job_id'],flush=True);return
        if dest.exists():raise FileExistsError('Unrecognized incomplete final directory; preserve and investigate: '+str(dest))
        for name,key in [('split','split_sha256'),('cohort','cohort_sha256')]:
            if digest(self.out/job[name])!=job[key]:raise ValueError('Job manifest mismatch')
        if fingerprint({k:v for k,v in job.items() if k!='job_sha256'})!=job['job_sha256']:raise ValueError('Job hash mismatch')
        if job['input_sha256']!=self.contract['inputs'][job['source']]['sha256']:raise ValueError('Job/input contract mismatch')
        write(self.out/'live_status.json',{'phase':'running','job':job['job_id'],'started_unix_seconds':time.time()})
        # Encoder construction consumes RNG state. Populate all caches BEFORE resetting
        # the paired head/training seed, independent of job order or cache warmness.
        for target in self.spec['sources']:self.encode(target)
        source=job['source'];ss=job['split_seed'];graph=job['model']!='tbase';context=self.context(source,ss,graph)
        global_rows=pd.read_csv(self.out/job['cohort']).row_index.to_numpy()
        lookup={int(row):i for i,row in enumerate(context['rows'])};local=np.array([lookup[int(row)] for row in global_rows])
        if len(set(local))!=job['fixed_n']:raise ValueError('Cohort mismatch')
        y=torch.as_tensor(self.data[source].label.to_numpy()[context['rows']],dtype=torch.long,device=self.device)
        cfg=self.spec['head'];seed_everything(job['seed'])
        model=make_model(job['model'],context['x'].shape[1],cfg['hidden'],cfg['dropout']).to(self.device)
        initial_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        init,_,_=self.evaluate(model,source,ss,source,'val')
        initial_test={target:self.evaluate(model,source,ss,target,'test')[1] for target in self.spec['sources']}
        # Evaluation is deterministic and consumes no training dropout random numbers.
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        weights=None
        if job['loss']=='weighted_ce':
            counts=torch.bincount(y[local],minlength=2).float();weights=len(local)/(2*counts)
        criterion=torch.nn.CrossEntropyLoss(weight=weights);best=-1.;best_state=None;best_update=None;trajectory=[]
        rng=np.random.RandomState(job['seed']);order=rng.permutation(len(local));cursor=0;passes=0
        for update in range(job['updates']):
            factor=core_lr(update,job['updates']) if job['family']=='architecture' else matched_lr(update,self.spec['matched_updates'])
            for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
            if job['batch_size']=='full':chosen=local;passes+=1
            else:
                if cursor==len(order):order=rng.permutation(len(local));cursor=0;passes+=1
                chosen=local[order[cursor:cursor+int(job['batch_size'])]];cursor+=len(chosen)
            model.train();optimizer.zero_grad()
            if graph:logits=model(context['x'],context['edge'])[chosen]
            else:logits=model(context['x'][chosen])
            loss=criterion(logits,y[chosen])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_norm']);optimizer.step()
            m,_,_=self.evaluate(model,source,ss,source,'val')
            trajectory.append({'optimizer_update':update+1,'learning_rate_used':cfg['lr']*factor,'training_loss':float(loss.item()),
                               'examples_this_update':len(chosen),'source_validation':m})
            if m['macro_f1']>best:
                best=m['macro_f1'];best_update=update+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if (update+1)%10==0:write(self.out/'live_status.json',{'phase':'running','job':job['job_id'],'updates_completed':update+1,'updates_planned':job['updates']})
        final_val,_,_=self.evaluate(model,source,ss,source,'val')
        final={target:self.evaluate(model,source,ss,target,'test') for target in self.spec['sources']}
        final_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state);selected_val,_,_=self.evaluate(model,source,ss,source,'val')
        selected={target:self.evaluate(model,source,ss,target,'test') for target in self.spec['sources']}
        temporary=dest.parent/(dest.name+'.temporary_'+uuid.uuid4().hex);temporary.mkdir(parents=True)
        torch.save(initial_state,temporary/'initial_model.pt');torch.save(final_state,temporary/'final_model.pt');torch.save(best_state,temporary/'selected_model.pt');write(temporary/'trajectory.json',{'initialization_validation':init,'updates':trajectory})
        test_results={}
        for target in self.spec['sources']:
            sm,sp,query=selected[target];fm,fp,_=final[target]
            pd.DataFrame({'row_index':query['rows'],'label':query['labels'],'initial_prediction':initial_test[target],
                          'selected_prediction':sp,'final_prediction':fp}).to_csv(temporary/f'test_{target}.csv',index=False)
            test_results[target]={'selected':sm,'final':fm,'initial':measures(query['labels'],initial_test[target],self.spec['collapse_threshold']),
                                  'excluded_text_matches':query['excluded_overlap_rows']}
        result={'job':job,'initial_parameter_sha256':initial_hash,'selected_optimizer_update':best_update,
            'initialization_validation':init,'selected_validation':selected_val,'final_validation':final_val,
            'training_ever_validation_collapsed':any(r['source_validation']['collapsed'] for r in trajectory),
            'examples_processed':sum(r['examples_this_update'] for r in trajectory),'test_targets':test_results,
            'selection_policy':self.spec['checkpoint_policy'],'synthetic_smoke_only':self.synthetic is not None,
            'files_sha256':{p.name:digest(p) for p in temporary.iterdir()}}
        write(temporary/'result.json',result);temporary.rename(dest)
        print(json.dumps({'completed':job['job_id'],'selected_source_f1':test_results[source]['selected']['macro_f1']}),flush=True)
        write(self.out/'live_status.json',{'phase':'job_completed','job':job['job_id']})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--out',required=True)
    ap.add_argument('--input-dir',default=str(HERE/'inputs'));ap.add_argument('--family',choices=['architecture','matched_updates','all'],default='all')
    ap.add_argument('--source');ap.add_argument('--model');ap.add_argument('--split-seed',type=int);ap.add_argument('--limit',type=int)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--allow-cpu',action='store_true');args=ap.parse_args()
    if not args.execute:ap.error('Training requires --execute; preparation is not a result')
    if not torch.cuda.is_available() and not args.allow_cpu:ap.error('GPU unavailable. Select GPU in Colab; --allow-cpu is an explicit slow alternative')
    runner=Runner(args.input_dir,args.out,'cuda' if torch.cuda.is_available() else 'cpu')
    jobs=[j for j in runner.plan['jobs'] if (args.family=='all' or j['family']==args.family) and
          (not args.source or j['source']==args.source) and (not args.model or j['model']==args.model) and
          (args.split_seed is None or j['split_seed']==args.split_seed)]
    if args.limit is not None:
        if args.limit<1:ap.error('--limit must be positive')
        jobs=jobs[:args.limit]
    if not jobs:ap.error('No selected jobs')
    for job in jobs:runner.execute(job,resume=args.resume)
    write(Path(args.out)/'live_status.json',{'phase':'selected_job_set_completed','selected_jobs':len(jobs),'total_planned_jobs':len(runner.plan['jobs'])})

if __name__=='__main__':raise SystemExit('Use execute.py with the approved concrete plan hash.')
