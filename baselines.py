"""Training-only popularity and a causal two-layer SASRec baseline."""
import collections
import json
from data import CODE,ROOT,config,input_view,load_split,load_catalog,write_json,write_jsonl
from history import select_history


def baseline(seed=42,profile='main'):
    out=ROOT/'checkpoints'/profile/f'sasrec_{seed}'
    if (out/'complete.json').exists():return
    import numpy as np
    import torch
    from torch import nn
    from transformers import set_seed
    from model import tokenizer
    from evaluate import metrics
    cfg=config();sc=cfg['sasrec'];set_seed(seed);tok=tokenizer();catalog=load_catalog()
    torch.cuda.set_per_process_memory_fraction(cfg['torch_memory_fraction'])
    rows={s:load_split(profile,s) for s in ('train','validation','test')}
    popular=collections.Counter(r['train_target'] for r in rows['train'])
    popularity=[s for s,_ in sorted(popular.items(),key=lambda x:(-x[1],x[0]))[:64]]
    for split in ('validation','test'):
        predictions=[dict(sample_id=r['sample_id'],outputs=[dict(text=s) for s in popularity]) for r in rows[split]]
        result,per=metrics(rows[split],predictions,catalog)
        result.update(experiment=f'popularity_{seed}_{split}',protocol='deterministic ranked top-K; not a sampling budget comparison')
        result['raw_predictions']=str(ROOT/'predictions'/profile/f'popularity_{seed}_{split}.jsonl')
        write_json(CODE/'reports/results_v2'/f'{profile}_popularity_{seed}_{split}.json',result)
        write_jsonl(ROOT/'predictions'/profile/f'popularity_{seed}_{split}.jsonl',predictions)
    # Include the full independent product catalog and training history vocabulary.
    # No validation/test history or target contributes new learned embedding rows.
    vocabulary=sorted(catalog | {e['sid'] for r in rows['train'] for e in r['history']})
    mapping={s:i+2 for i,s in enumerate(vocabulary)}
    candidates=torch.tensor([mapping[s] for s in sorted(catalog)],device='cuda')
    def encode(row):
        view=input_view(row);chosen=select_history(view,tok,cfg['input_budget'],'recent')[-sc['max_events']:]
        seq=[mapping.get(row['history'][i]['sid'],1) for i in chosen] or [1]
        return seq
    sequences={s:[encode(r) for r in rr] for s,rr in rows.items()}
    def batch(seqs):
        x=torch.zeros((len(seqs),max(map(len,seqs))),dtype=torch.long,device='cuda')
        for i,s in enumerate(seqs):x[i,:len(s)]=torch.tensor(s,device='cuda')
        return x
    class SASRec(nn.Module):
        def __init__(self):
            super().__init__();d=sc['hidden_size']
            self.items=nn.Embedding(len(mapping)+2,d,padding_idx=0,sparse=True)
            self.positions=nn.Embedding(sc['max_events'],d)
            self.dropout=nn.Dropout(sc['dropout'])
            layer=nn.TransformerEncoderLayer(d,sc['heads'],d*4,sc['dropout'],batch_first=True,norm_first=True,activation='gelu')
            self.layers=nn.TransformerEncoder(layer,sc['layers'],enable_nested_tensor=False)
            self.norm=nn.LayerNorm(d)
            nn.init.normal_(self.items.weight,std=.02)
            with torch.no_grad():self.items.weight[0].zero_()
        def forward(self,x):
            z=self.dropout(self.items(x)*math_sqrt+self.positions(torch.arange(x.shape[1],device=x.device)))
            causal=torch.triu(torch.ones((x.shape[1],x.shape[1]),device=x.device,dtype=torch.bool),diagonal=1)
            z=self.norm(self.layers(z,mask=causal,src_key_padding_mask=x==0))
            return z[torch.arange(len(x),device=x.device),(x!=0).sum(1)-1]
    math_sqrt=sc['hidden_size']**.5
    model=SASRec().cuda()
    dense=[p for name,p in model.named_parameters() if name!='items.weight']
    opt=torch.optim.AdamW(dense,lr=sc['learning_rate'],weight_decay=.01)
    item_opt=torch.optim.SparseAdam([model.items.weight],lr=sc['learning_rate'])
    def predict(split):
        model.eval();result=[]
        with torch.inference_mode():
            for start in range(0,len(rows[split]),sc['batch_size']):
                z=model(batch(sequences[split][start:start+sc['batch_size']]))
                best_scores=torch.full((len(z),64),-float('inf'),device='cuda');best_ids=torch.zeros((len(z),64),dtype=torch.long,device='cuda')
                for offset in range(0,len(candidates),8192):
                    ids=candidates[offset:offset+8192];scores=z@model.items(ids).T
                    joined=torch.cat([best_scores,scores],1);item_ids=torch.cat([best_ids,ids.expand(len(z),-1)],1)
                    best_scores,index=joined.topk(64,dim=1);best_ids=item_ids.gather(1,index)
                for row,ids in zip(rows[split][start:start+len(z)],best_ids.tolist()):
                    result.append(dict(sample_id=row['sample_id'],outputs=[dict(text=vocabulary[i-2]) for i in ids]))
        return result
    out=ROOT/'checkpoints'/profile/f'sasrec_{seed}';out.mkdir(parents=True,exist_ok=True)
    best=-1;history=[];epochs=2 if profile=='smoke' else sc['epochs']
    for epoch in range(epochs):
        model.train();order=np.random.default_rng(seed+epoch).permutation(len(rows['train']));losses=[]
        for start in range(0,len(order),sc['batch_size']):
            idx=order[start:start+sc['batch_size']]
            x=batch([sequences['train'][i] for i in idx]);y=torch.tensor([mapping[rows['train'][i]['train_target']] for i in idx],device='cuda')
            neg=candidates[torch.randint(len(candidates),(len(idx),sc['negatives']),device='cuda')]
            equal=neg==y[:,None]
            while equal.any():
                neg[equal]=candidates[torch.randint(len(candidates),(int(equal.sum()),),device='cuda')];equal=neg==y[:,None]
            z=model(x);pos=(z*model.items(y)).sum(-1);negative=torch.einsum('bd,bnd->bn',z,model.items(neg))
            loss=torch.nn.functional.softplus(-pos).mean()+torch.nn.functional.softplus(negative).mean()
            opt.zero_grad();item_opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(dense,1)
            opt.step();item_opt.step();losses.append(float(loss.detach()))
        prediction=predict('validation');score,_=metrics(rows['validation'],prediction,catalog)
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),recall32=score['recall@32']))
        if score['recall@32']>best:
            best=score['recall@32'];torch.save(model.state_dict(),out/'selected.pt')
        print('SASRec',history[-1],flush=True)
    model.load_state_dict(torch.load(out/'selected.pt',weights_only=True))
    write_json(out/'vocabulary.json',vocabulary)
    write_json(CODE/'reports/training'/f'{profile}_sasrec_{seed}.json',dict(epochs=history,vocabulary=len(vocabulary),
               note='SASRec architecture; next-target supervision; source presentation order, not known global chronology; no text pretraining',peak_bytes=torch.cuda.max_memory_allocated()))
    for split in ('validation','test'):
        prediction=predict(split);score,per=metrics(rows[split],prediction,catalog)
        score.update(experiment=f'sasrec_{seed}_{split}',protocol='full product-catalog ranked retrieval; next-target supervision')
        score['raw_predictions']=str(ROOT/'predictions'/profile/f'sasrec_{seed}_{split}.jsonl')
        write_json(CODE/'reports/results_v2'/f'{profile}_sasrec_{seed}_{split}.json',score)
        write_jsonl(ROOT/'predictions'/profile/f'sasrec_{seed}_{split}.jsonl',prediction)
    write_json(out/'complete.json',dict(seed=seed,epochs=epochs,validation_recall32=best))
