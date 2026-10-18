"""Explicitly gated training runner. Preparation does not invoke this file."""
import argparse,sys,json,random,copy,math,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'tools'))
from common import ROOT,read_json,write_json,sha256,verify_sources
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--limit',type=int,default=None);ap.add_argument('--job-id');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    if not args.execute:ap.error('Training is disabled. Explicit experiment authorization is required before --execute.')
    verify_sources();plan=read_json(ROOT/'experiments/grouped_control/run_plan.json');cfg=plan['specification'];jobs=plan['jobs']
    if args.job_id:jobs=[j for j in jobs if j['job_id']==args.job_id]
    if args.limit is not None:
        if args.limit<1:ap.error('--limit must be positive')
        jobs=jobs[:args.limit]
    if not jobs:ap.error('No selected jobs')
    import numpy as np,pandas as pd,torch
    from torch import nn
    from transformers import AutoTokenizer,AutoModel
    from sklearn.metrics import f1_score,accuracy_score,matthews_corrcoef
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');encoded={};rows=[]
    dest=ROOT/'experiments/grouped_control/results';dest.mkdir(parents=True,exist_ok=True)
    class Head(nn.Module):
        def __init__(self,dim):
            super().__init__();self.fc1=nn.Linear(dim,256);self.fc2=nn.Linear(256,2);self.drop=nn.Dropout(.3)
        def forward(self,x):return self.fc2(self.drop(torch.relu(self.fc1(x))))
    def measures(labels,pred):
        q=float(pred.mean());return {'macro_f1':float(f1_score(labels,pred,average='macro')),'accuracy':float(accuracy_score(labels,pred)),'mcc':float(matthews_corrcoef(labels,pred)),'pred_fake_frac':q,'collapsed':q>=.98 or q<=.02}
    for job in jobs:
        for key,digest in [('input','input_sha256'),('split_manifest','split_sha256'),('cohort_manifest','cohort_sha256')]:
            if sha256(ROOT/job[key])!=job[digest]:raise ValueError('Job manifest mismatch')
        jobdir=dest/job['job_id']
        if args.resume and (jobdir/'result.json').exists():
            saved=read_json(jobdir/'result.json')
            if saved['job']!=job:raise ValueError('Existing job configuration differs')
            print('Verified completed job: '+job['job_id'],flush=True);continue
        if jobdir.exists():raise FileExistsError('Refusing to overwrite existing job '+job['job_id'])
        # Create the result directory only after encoding succeeds.
        if job['source'] not in encoded:
            df=pd.read_csv(ROOT/job['input'],keep_default_na=False)
            cache=ROOT/'experiments/grouped_control/embedding_cache'/job['source'];cache.mkdir(parents=True,exist_ok=True)
            signature={'input_sha256':job['input_sha256'],'encoder':'roberta-base','maximum_tokens':128,'pooling':'last_hidden_state[:,0,:]','frozen':True}
            if (cache/'vectors.pt').exists() and (cache/'metadata.json').exists():
                metadata=read_json(cache/'metadata.json')
                if metadata['signature']!=signature or sha256(cache/'vectors.pt')!=metadata['sha256']:raise ValueError('Embedding cache mismatch')
                vectors=torch.load(cache/'vectors.pt',map_location='cpu',weights_only=True)
                print('Loaded verified embeddings: '+job['source'],flush=True)
            else:
                tokenizer=AutoTokenizer.from_pretrained('roberta-base',local_files_only=True);encoder=AutoModel.from_pretrained('roberta-base',local_files_only=True).to(device).eval();chunks=[]
                with torch.no_grad():
                    for start in range(0,len(df),32):
                        tokens=tokenizer(df.text.astype(str).iloc[start:start+32].tolist(),padding=True,truncation=True,max_length=128,return_tensors='pt');tokens={k:v.to(device) for k,v in tokens.items()};chunks.append(encoder(**tokens).last_hidden_state[:,0,:].cpu())
                        if start%320==0:print(f'Encoding {job["source"]}: {min(start+32,len(df))}/{len(df)}',flush=True)
                vectors=torch.cat(chunks);torch.save(vectors,cache/'vectors.pt')
                write_json(cache/'metadata.json',{'signature':signature,'sha256':sha256(cache/'vectors.pt'),'rows':len(df),'dimensions':int(vectors.shape[1])})
                del encoder
                if device.type=='cuda':torch.cuda.empty_cache()
            if len(vectors)!=len(df):raise ValueError('Embedding row mismatch')
            encoded[job['source']]=(vectors.to(device),df.label.to_numpy())
        jobdir.mkdir()
        x,y_np=encoded[job['source']];y=torch.as_tensor(y_np,dtype=torch.long,device=device);split=pd.read_csv(ROOT/job['split_manifest']);idx=pd.read_csv(ROOT/job['cohort_manifest']).row_index.to_numpy()
        random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed'])
        if torch.cuda.is_available():torch.cuda.manual_seed_all(job['seed'])
        model=Head(x.shape[1]).to(device);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        def schedule(ep):
            return (ep+1)/3 if ep<3 else .5*(1+np.cos(np.pi*(ep-3)/27))
        sched=torch.optim.lr_scheduler.LambdaLR(optim,schedule)
        weights=None
        if job['loss']=='weighted_ce':
            counts=torch.bincount(y[idx],minlength=2).float();weights=len(idx)/(2*counts)
        criterion=nn.CrossEntropyLoss(weight=weights)
        val=split[split.split=='val'].row_index.to_numpy();test=split[split.split=='test'].row_index.to_numpy();trajectory=[]
        def evaluate(which):
            model.eval()
            with torch.no_grad():pred=model(x[which]).argmax(-1).cpu().numpy()
            return measures(y_np[which],pred),pred
        init,_=evaluate(val);best=-1.;best_state=None;best_epoch=None
        for epoch in range(30):
            lr=optim.param_groups[0]['lr'];model.train();optim.zero_grad();loss=criterion(model(x[idx]),y[idx]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();sched.step();m,_=evaluate(val)
            trajectory.append({'epoch_index':epoch,'optimizer_updates':epoch+1,'learning_rate_used':lr,'train_loss':float(loss.item()),**m})
            if m['macro_f1']>best:best=m['macro_f1'];best_state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()});best_epoch=epoch
        final_val,_=evaluate(val);final_test,final_pred=evaluate(test)
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},jobdir/'final_head.pt')
        model.load_state_dict(best_state);selected_val,_=evaluate(val);selected_test,selected_pred=evaluate(test)
        torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},jobdir/'selected_head.pt')
        pd.DataFrame({'row_index':test,'label':y_np[test],'selected_prediction':selected_pred,'final_prediction':final_pred}).to_csv(jobdir/'test_predictions.csv',index=False)
        write_json(jobdir/'trajectory.json',{'initialization_validation':init,'epochs':trajectory})
        result={'job':job,'device':str(device),'selected_epoch_index':best_epoch,'selection_candidates':'post-update epochs only; initialization diagnostic excluded','initialization_validation':init,'training_ever_collapsed':any(r['collapsed'] for r in trajectory),'selected_validation':selected_val,'selected_test':selected_test,'final_validation':final_val,'final_test':final_test}
        write_json(jobdir/'result.json',result);rows.append({'job_id':job['job_id'],'selected_test_f1':selected_test['macro_f1'],'selected_test_collapsed':selected_test['collapsed']});print(json.dumps(rows[-1]),flush=True)
    write_json(dest/'last_invocation.json',{'completed_jobs':rows,'training_executed':True})
if __name__=='__main__':main()
