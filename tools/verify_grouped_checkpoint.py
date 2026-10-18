"""Replay a saved selected checkpoint against cached test embeddings."""
import sys
import torch
import pandas as pd
import numpy as np
from common import ROOT, read_json, sha256, write_json
torch.set_num_threads(4)
base=ROOT/'experiments/grouped_control';folder=base/'results'/sys.argv[1]
result=read_json(folder/'result.json');job=result['job'];cache=base/'embedding_cache'/job['source']
assert sha256(cache/'vectors.pt')==read_json(cache/'metadata.json')['sha256']
x=torch.load(cache/'vectors.pt',weights_only=True)
state=torch.load(folder/'selected_head.pt',weights_only=True)
head=torch.nn.Sequential(torch.nn.Linear(768,256),torch.nn.ReLU(),torch.nn.Dropout(.3),torch.nn.Linear(256,2)).eval()
head.load_state_dict({'0.weight':state['fc1.weight'],'0.bias':state['fc1.bias'],'3.weight':state['fc2.weight'],'3.bias':state['fc2.bias']})
pred=pd.read_csv(folder/'test_predictions.csv')
with torch.no_grad():actual=head(x[pred.row_index.to_numpy()]).argmax(-1).numpy()
assert np.array_equal(actual,pred.selected_prediction.to_numpy())
write_json(folder/'checkpoint_replay_verification.json',{'selected_predictions_exact_match':True,'test_rows':len(pred)})
print('Checkpoint replay verified: '+job['job_id'],flush=True)
