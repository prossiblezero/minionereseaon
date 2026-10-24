"""PEFT/Transformers SFT; TRL standard GRPO; explicit branched credit assignment."""
import collections
import gc
import json
import math
import time
from pathlib import Path
from data import CODE,ROOT,config,input_view,load_split,load_catalog,parse_answer,codes,write_json,write_jsonl
from history import select_history,prompt_text


def training_records(rows,tok,variant,budget):
    records=[];dropped=0
    strategy='balanced' if variant=='mixed_balanced' else 'recent'
    for row in rows:
        view=input_view(row);chosen=select_history(view,tok,budget,strategy)
        kept={row['history'][i]['sid'] for i in chosen}
        all_history={e['sid'] for e in row['history']}
        if (set(row['metadata']['reasoning_sids']) & all_history)-kept:
            dropped+=1;continue
        for thinking in (False,variant in ('mixed','mixed_balanced','mixed_recent_history')):
            prompt=prompt_text(view,tok,chosen,thinking)
            completion=(row['reasoning']+'\n</think>\n\n' if thinking else '')+row['train_target']+tok.eos_token
            p=tok.encode(prompt,add_special_tokens=False);a=tok.encode(completion,add_special_tokens=False)
            records.append(dict(input_ids=p+a,attention_mask=[1]*(len(p)+len(a)),labels=[-100]*len(p)+a))
    return records,dropped


def sft(variant,seed,profile='main',max_steps=None):
    import torch
    from transformers import Trainer,TrainingArguments,TrainerCallback,set_seed,DataCollatorForSeq2Seq
    from datasets import Dataset
    from model import load_model,tokenizer,Generator
    from evaluate import metrics
    if variant not in ('direct','mixed','direct_tokenmatched','mixed_balanced','mixed_recent_history'):
        raise ValueError(variant)
    cfg=config();set_seed(seed);tok=tokenizer()
    out=ROOT/'checkpoints'/profile/f'{variant}_{seed}'
    if (out/'complete.json').exists():
        print('Already complete',out,flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    rows=load_split(profile,'train')
    training_budget=cfg['input_budget'];paired_removed=0
    if variant in ('mixed_balanced','mixed_recent_history'):
        from history import history_gate
        training_budget=history_gate(json.loads((CODE/'reports'/f'data_report_{profile}.json').read_text()))
        paired=[]
        for row in rows:
            view=input_view(row);refs=set(row['metadata']['reasoning_sids']) & {e['sid'] for e in row['history']}
            retained=[{row['history'][i]['sid'] for i in select_history(view,tok,training_budget,s)} for s in ('recent','balanced')]
            if all(refs<=ids for ids in retained):paired.append(row)
        paired_removed=len(rows)-len(paired);rows=paired
    records,dropped=training_records(rows,tok,variant,training_budget)
    if not records:raise RuntimeError('No valid training records')
    token_target=None
    if variant=='direct_tokenmatched':
        selected=ROOT/'checkpoints'/profile/f'mixed_{seed}'/'selected'/'training.json'
        token_target=json.loads(selected.read_text())['processed_tokens']
    model=load_model(trainable=True)
    trainable,total=model.get_nb_trainable_parameters()
    ledger=dict(processed_tokens=0,supervised_tokens=0,base_samples=len(rows),training_records=len(records),dropped_history_references=dropped,
                trainable_parameters=trainable,total_parameters=total,monitor_mode='non-thinking for all SFT variants',variant=variant,seed=seed,
                input_budget=training_budget,paired_history_removed=paired_removed)
    checkpoints=sorted(out.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    if checkpoints:
        ledger.update(json.loads((checkpoints[-1]/'token_ledger.json').read_text()))
    validation=load_split(profile,'validation')
    monitor_path=CODE/'artifacts'/f'monitor_samples_{profile}.json'
    if monitor_path.exists():
        by_id={r['sample_id']:r for r in validation};validation=[by_id[sid] for sid in json.loads(monitor_path.read_text())]
    else:
        validation=validation[:4 if profile=='smoke' else cfg['sft']['monitor_examples']]
        write_json(monitor_path,[r['sample_id'] for r in validation])
    catalog=load_catalog()
    class CountTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            ledger['processed_tokens']+=int(inputs['attention_mask'].sum())
            ledger['supervised_tokens']+=int((inputs['labels']!=-100).sum())
            return super().compute_loss(model,inputs,return_outputs=return_outputs,num_items_in_batch=num_items_in_batch)
        def create_scheduler(self,num_training_steps,optimizer=None):
            if token_target is None:return super().create_scheduler(num_training_steps,optimizer)
            if self.lr_scheduler is None:
                def scale(_):
                    progress=min(1.0,ledger['processed_tokens']/token_target);warmup=cfg['sft']['warmup_ratio']
                    return progress/warmup if progress<warmup else (1-progress)/(1-warmup)
                self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer,scale)
                self._created_lr_scheduler=True
            return self.lr_scheduler
    class Monitor(TrainerCallback):
        best=-1
        last_step=-1
        def __init__(self):
            path=out/'selected/training.json'
            if path.exists():
                prior=json.loads(path.read_text());self.best=prior['monitor']['recall@4' if profile=='smoke' else 'recall@32']
        def on_save(self,args,state,control,**kwargs):
            write_json(out/f'checkpoint-{state.global_step}'/'token_ledger.json',ledger)
        def check(self,state):
            if self.last_step==state.global_step:return
            self.last_step=state.global_step
            predictions=[];gen=Generator(backend='transformers',existing_model=model)
            for row in validation:
                view=input_view(row);chosen=select_history(view,tok,training_budget,'recent')
                outputs=gen.generate(prompt_text(view,tok,chosen,False),4 if profile=='smoke' else 32,seed+int(row['sample_id'][:6],16))
                predictions.append(dict(sample_id=row['sample_id'],outputs=outputs))
            n=4 if profile=='smoke' else 32
            score,_=metrics(validation,predictions,catalog,ks=(n,))
            score.update(step=state.global_step,processed_tokens=ledger['processed_tokens'],supervised_tokens=ledger['supervised_tokens'])
            write_json(CODE/'reports/training'/f'{profile}_{variant}_{seed}_step{state.global_step}.json',score)
            write_jsonl(ROOT/'predictions'/profile/f'monitor_{variant}_{seed}_step{state.global_step}.jsonl',predictions)
            if score[f'recall@{n}']>self.best or token_target is not None:
                self.best=score[f'recall@{n}'];dest=out/'selected';model.save_pretrained(dest);tok.save_pretrained(dest)
                write_json(dest/'training.json',dict(ledger,step=state.global_step,monitor=score))
            model.train();model.config.use_cache=False
            del gen;torch.cuda.empty_cache()
        def on_step_end(self,args,state,control,**kwargs):
            if token_target is not None and ledger['processed_tokens']>=token_target:
                control.should_training_stop=True
            if token_target is None and state.global_step%cfg['sft']['eval_steps']==0:
                self.check(state)
        def on_train_end(self,args,state,control,**kwargs):
            self.check(state)
    epochs=cfg['sft']['epochs']
    if token_target is not None:
        epochs=math.ceil(token_target/sum(len(x['input_ids']) for x in records))+1
    args=TrainingArguments(output_dir=str(out),per_device_train_batch_size=1,gradient_accumulation_steps=cfg['sft']['gradient_accumulation_steps'],
                           num_train_epochs=epochs,max_steps=max_steps or -1,learning_rate=cfg['sft']['learning_rate'],weight_decay=cfg['sft']['weight_decay'],
                           warmup_ratio=cfg['sft']['warmup_ratio'],bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                           logging_steps=5,save_strategy='steps',save_steps=cfg['sft']['eval_steps'],save_total_limit=2,report_to='none',
                           seed=seed,data_seed=seed,dataloader_num_workers=0,remove_unused_columns=False,optim='adamw_torch',disable_tqdm=True)
    trainer=CountTrainer(model=model,args=args,train_dataset=Dataset.from_list(records),
                         data_collator=DataCollatorForSeq2Seq(tok,padding=True,label_pad_token_id=-100),callbacks=[Monitor()])
    started=time.perf_counter()
    trainer.train(resume_from_checkpoint=str(checkpoints[-1]) if checkpoints else None)
    if token_target is not None and ledger['processed_tokens']<token_target:
        raise RuntimeError('Token-matched run stopped before reaching its required budget')
    ledger.update(seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_steps=trainer.state.global_step,
                  token_target=token_target,selected=str(out/'selected'))
    write_json(out/'complete.json',ledger);write_json(CODE/'reports/training'/f'{profile}_{variant}_{seed}_complete.json',ledger)
    print(json.dumps(ledger),flush=True)


def reward(text,targets,catalog,prefix=False):
    sid,_=parse_answer(text,catalog)
    if sid is None:return 0.0
    if sid in targets:return 1.0
    if not prefix:return 0.0
    a=codes(sid)
    return max((.5 if a[:2]==codes(t)[:2] else .2 if a[0]==codes(t)[0] else 0.0) for t in targets)


def branch_advantages(rewards):
    import numpy as np
    r=np.asarray(rewards,dtype=float)
    if r.ndim!=2:raise ValueError('One prompt must be represented as N x K rewards')
    a=(r-r.mean())/(r.std()+1e-4)
    return a.mean(axis=1).tolist(),a.tolist()


def sample_branches(gen,prompt,n,k,seed):
    cfg=config();prefixes=gen.raw(prompt,n,cfg['answer_budget']-cfg['generation']['direct_tokens'],seed,stop=['</think>'])
    prompts=[prompt+p['text']+'\n</think>\n\n' for p in prefixes]
    answers=gen.raw_many(prompts,k,cfg['generation']['direct_tokens'],[seed+1009+i for i in range(len(prefixes))])
    groups=[]
    for p,branch_prompt,children in zip(prefixes,prompts,answers):
        groups.append(dict(prefix=p,branch_prompt=branch_prompt,answers=children))
    return groups


def sample_branch_batch(gen,requests,n,k):
    cfg=config()
    prefixes=gen.raw_many([r['prompt'] for r in requests],n,cfg['answer_budget']-cfg['generation']['direct_tokens'],
                          [r['seed'] for r in requests],stop=['</think>'])
    prompts=[];seeds=[]
    for request,group in zip(requests,prefixes):
        for i,prefix in enumerate(group):
            prompts.append(request['prompt']+prefix['text']+'\n</think>\n\n');seeds.append(request['seed']+1009+i)
    answers=gen.raw_many(prompts,k,cfg['generation']['direct_tokens'],seeds)
    records=[];index=0
    for group in prefixes:
        branches=[]
        for prefix in group:
            branches.append(dict(prefix=prefix,branch_prompt=prompts[index],answers=answers[index]));index+=1
        records.append(dict(branches=branches))
    return records


def branch_segments(prompt,branches,rewards,tok):
    shared,individual=branch_advantages(rewards);segments=[]
    for i,b in enumerate(branches):
        segments.append((prompt,b['prefix']['text'],shared[i],'reasoning'))
        for j,a in enumerate(b['answers']):
            segments.append((b['branch_prompt'],tok.decode(a['token_ids'],skip_special_tokens=False),individual[i][j],'answer'))
    return segments


def smoke():
    import subprocess,sys
    from model import tokenizer,load_model
    import torch
    tok=tokenizer();model=load_model(trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    sample=training_records(load_split('smoke','train')[:1],tok,'mixed',config()['input_budget'])[0][-1]
    inputs={k:torch.tensor([v],device='cuda') for k,v in sample.items()}
    out=model(**inputs);out.loss.backward()
    assert torch.isfinite(out.loss)
    gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    write_json(CODE/'reports/smoke_forward_backward.json',dict(loss=float(out.loss.detach()),lora_gradient_tensors=len(gradients),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    del out,inputs,model;gc.collect();torch.cuda.empty_cache()
    subprocess.run([sys.executable,'run.py','sft','--profile','smoke','--variant','mixed','--max-steps','2'],cwd=CODE,check=True)


def rollout_worker(request,response,generator=None):
    from model import Generator
    from data import digest,read_jsonl
    request=Path(request);response=Path(response);spec=json.loads(request.read_text())
    signature=digest(request.read_text());sidecar=response.with_suffix('.request_sha256')
    if sidecar.exists() and sidecar.read_text()!=signature:
        raise RuntimeError('Existing rollout belongs to a different request')
    response.parent.mkdir(parents=True,exist_ok=True);sidecar.write_text(signature)
    done={x['sample_id'] for x in read_jsonl(response)} if response.exists() else set()
    pending=[r for r in spec['requests'] if r['sample_id'] not in done]
    if not pending:return
    gen=generator if generator is not None else Generator(spec['adapter'])
    if generator is not None:
        from vllm.lora.request import LoRARequest
        gen.engine.wake_up()
        if gen.adapter:gen.engine.llm_engine.remove_lora(gen.adapter.lora_int_id)
        gen.adapter=LoRARequest(Path(spec['adapter']).name,int(digest(signature)[:7],16)+1,spec['adapter'])
    try:
        with response.open('a',encoding='utf-8') as f:
            count=32 if spec['mode']=='std32' else 8
            batch_size=max(1,64//count) if spec['mode'] in ('std','std32') else 16
            for start in range(0,len(pending),batch_size):
                batch=pending[start:start+batch_size]
                if spec['mode'] in ('std','std32'):
                    groups=gen.raw_many([r['prompt'] for r in batch],count,config()['generation']['thinking_tokens'],[r['seed'] for r in batch])
                    for group in groups:
                        for output in group:
                            output['text']='<think>\n'+output['text']
                            output['reasoning_tokens']=len(gen.tok.encode(output['text'].split('</think>',1)[0].removeprefix('<think>\n'),add_special_tokens=False))
                            output['forced_close']=False
                    values=[dict(outputs=group) for group in groups]
                else:
                    values=sample_branch_batch(gen,batch,spec['n'],spec['k'])
                for request,value in zip(batch,values):
                    f.write(json.dumps(dict(sample_id=request['sample_id'],**value),ensure_ascii=False)+'\n');f.flush()
                completed=len(done)+min(start+batch_size,len(pending))
                print('rollout',spec['mode'],completed,len(spec['requests']),flush=True)
                state_path=CODE/'reports/pipeline_status.json'
                if state_path.exists():
                    import datetime
                    state=json.loads(state_path.read_text())
                    state.update(rollout_progress=dict(file=response.name,mode=spec['mode'],completed=completed,total=len(spec['requests'])),
                                 updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    write_json(state_path,state)
    finally:
        if generator is not None:gen.engine.sleep(level=1)


def run_rollout(adapter,mode,requests,label,generator=None):
    import subprocess,sys,hashlib
    from data import read_jsonl
    cfg=config();directory=ROOT/'predictions/rl_rollouts';directory.mkdir(parents=True,exist_ok=True)
    request=directory/(label+'.request.json');response=directory/(label+'.jsonl')
    adapter_hash=hashlib.sha256((Path(adapter)/'adapter_model.safetensors').read_bytes()).hexdigest()
    write_json(request,dict(adapter=str(adapter),adapter_sha256=adapter_hash,mode=mode,n=cfg['rl']['n'],k=cfg['rl']['k'],requests=requests))
    if generator is None:
        subprocess.run([sys.executable,'run.py','rollout','--request',str(request),'--response',str(response)],cwd=CODE,check=True)
    else:rollout_worker(request,response,generator)
    return read_jsonl(response)


def sleeping_rollout_generator():
    import os
    from model import Generator,MODEL,tokenizer
    from vllm import LLM
    gen=Generator.__new__(Generator);gen.backend='vllm';gen.cfg=config();gen.tok=tokenizer();gen.adapter=None
    # vLLM 0.11 CuMemAllocator rejects expandable segments in its worker.
    allocator_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
    try:
        gen.engine=LLM(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',max_model_len=4096,
                       gpu_memory_utilization=gen.cfg['vllm_memory_fraction'],enforce_eager=False,
                       enable_lora=True,max_lora_rank=gen.cfg['lora']['r'],max_num_seqs=64,
                       max_num_batched_tokens=4096,enable_prefix_caching=True,seed=42,enable_sleep_mode=True)
    finally:
        if allocator_conf is None:os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        else:os.environ['PYTORCH_CUDA_ALLOC_CONF']=allocator_conf
    gen.engine.sleep(level=1)
    return gen


def branch_rewards(branches,targets,catalog,variant):
    values=[]
    for branch in branches:
        texts=[a['text'] for a in branch['answers']]
        rewards=[reward(t,targets,catalog,prefix='prefix' in variant) for t in texts]
        if variant=='nk_prefix_div':
            sids=[parse_answer(t,catalog)[0] for t in texts]
            m=len({codes(s)[0] for s in sids if s})
            diversity=max(0,m-1)/max(1,len(texts)-1)
            rewards=[r*diversity for r in rewards]
        values.append(rewards)
    return values


def popular_first_codes(rows):
    counts=collections.Counter(codes(r['train_target'])[0] for r in rows)
    return {k for k,_ in counts.most_common(max(1,math.ceil(len(counts)*.1)))}


def rl_prompt_schedule(rows,seed,profile):
    import numpy as np
    from data import digest
    indices=np.random.default_rng(seed).integers(len(rows),size=config()['rl']['steps']).tolist()
    value=dict(seed=seed,profile=profile,training_ids_sha256=digest('\n'.join(r['sample_id'] for r in rows)),
               sample_ids=[rows[i]['sample_id'] for i in indices],sampling='with replacement; shared across all RL variants')
    path=CODE/'artifacts'/f'rl_prompt_schedule_{profile}_{seed}.json'
    if path.exists() and json.loads(path.read_text())!=value:raise RuntimeError('Frozen RL prompt schedule changed')
    write_json(path,value)
    return indices


def reward_diagnostics(records,rows,catalog,variant,popular_heads=None):
    import numpy as np
    rowmap={r['sample_id']:r for r in rows};groups=[];heads=collections.Counter();exact=valid=attempts=tokens=prefix_only=0;seconds=0
    for record in records:
        targets=rowmap[record['sample_id']]['targets']
        if variant in ('std','std32'):
            outputs=record['outputs'];r=[reward(o['text'],targets,catalog) for o in outputs]
            tokens+=sum(o['generated_tokens'] for o in outputs);seconds+=sum(o['seconds'] for o in outputs)
        else:
            branches=record['branches'];outputs=[a for b in branches for a in b['answers']]
            r=np.asarray(branch_rewards(branches,targets,catalog,variant)).flatten().tolist()
            tokens+=sum(b['prefix']['generated_tokens'] for b in branches)+sum(o['generated_tokens'] for o in outputs)
            seconds+=sum(b['prefix']['seconds'] for b in branches)+sum(o['seconds'] for o in outputs)
        groups.append(dict(sample_id=record['sample_id'],mean_reward=float(np.mean(r)),reward_std=float(np.std(r))))
        for o in outputs:
            attempts+=1;sid,_=parse_answer(o['text'],catalog)
            if sid:
                valid+=1;heads[codes(sid)[0]]+=1;exact+=sid in targets
                prefix_only+=sid not in targets and any(codes(sid)[0]==codes(t)[0] for t in targets)
    probabilities=np.asarray(list(heads.values()),dtype=float)/max(1,valid)
    top=popular_first_codes(rows) if popular_heads is None else popular_heads
    return dict(variant=variant,groups=len(groups),active_group_fraction=sum(g['reward_std']>1e-8 for g in groups)/max(1,len(groups)),
                mean_reward=float(np.mean([g['mean_reward'] for g in groups])),exact_hit_rate=exact/max(1,attempts),prefix_only_hit_rate=prefix_only/max(1,attempts),invalid_rate=1-valid/max(1,attempts),
                first_code_entropy=float(-(probabilities*np.log(probabilities)).sum()),first_code_coverage=len(heads)/8192,
                popular_first_code_share=sum(v for k,v in heads.items() if k in top)/max(1,valid),generated_tokens=tokens,generation_seconds=seconds,
                first_code_counts=dict(heads),per_group=groups)


def probe_rewards(seed,profile):
    from model import tokenizer
    from evaluate import checkpoint
    cfg=config();tok=tokenizer();all_train=load_split(profile,'train');heads=popular_first_codes(all_train)
    rows=all_train[:8 if profile=='smoke' else cfg['rl']['probe_examples']]
    catalog=load_catalog();requests=[]
    for row in rows:
        view=input_view(row);selected=select_history(view,tok,cfg['input_budget'],'recent')
        requests.append(dict(sample_id=row['sample_id'],prompt=prompt_text(view,tok,selected,True),seed=seed+int(row['sample_id'][:6],16)))
    adapter=checkpoint('mixed',seed,profile);summaries=[]
    for mode,variants in [('std',['std']),('std32',['std32']),('nk',['nk','nk_prefix','nk_prefix_div'])]:
        records=run_rollout(adapter,mode,requests,f'{profile}_probe_{mode}_{seed}')
        for variant in variants:
            report=reward_diagnostics(records,rows,catalog,variant,heads)
            report['audit']=dict(target_catalog_coverage=sum(all(t in catalog for t in row['targets']) for row in rows)/len(rows),
                parser_unit_tests='tests/test_pipeline.py',truncated_outputs=sum(
                o.get('finish_reason')=='length' for r in records for o in (r.get('outputs') or [a for b in r['branches'] for a in b['answers']])),
                forced_prefix_closures=sum(b['prefix'].get('stop_reason')!='</think>' for r in records for b in r.get('branches',[])))
            report['eligible_for_training']=report['active_group_fraction']>=cfg['rl']['min_active_group_fraction']
            write_json(CODE/'reports/rl'/f'{profile}_probe_{variant}_{seed}.json',report)
            summaries.append({k:v for k,v in report.items() if k not in ('per_group','first_code_counts')})
    write_json(CODE/'reports/rl'/f'{profile}_probe_summary_{seed}.json',summaries)
    print(json.dumps(summaries),flush=True)


def monitor_rl(adapter,variant,seed,profile,step):
    import subprocess,sys,shutil
    tag=f'{profile}_rl_{variant}_{seed}_monitor_step{step}'
    subprocess.run([sys.executable,'run.py','evaluate','--profile',profile,'--variant','mixed','--adapter',str(adapter),
                    '--seed',str(seed),'--split','validation','--samples','32','--thinking','--limit',str(config()['sft']['monitor_examples']),
                    '--tag',tag,'--monitor'],cwd=CODE,check=True)
    report=json.loads((CODE/'reports/results_v2'/f'{tag}.json').read_text())
    out=ROOT/'checkpoints'/profile/f'rl_{variant}_{seed}'
    old=json.loads((out/'selection.json').read_text()) if (out/'selection.json').exists() else {'recall@32':-1}
    if report['recall@32']>old['recall@32']:
        shutil.copytree(adapter,out/'selected',dirs_exist_ok=True)
        write_json(out/'selection.json',dict(step=step,**report))


def branched_update(policy,optimizer,prompt,branches,rewards,tok,cfg):
    import numpy as np
    import torch
    # The vLLM worker is asleep during updates. Allow the same training cap as
    # standard GRPO, including full-vocabulary logits for long completions.
    torch.cuda.set_per_process_memory_fraction(.8)
    policy.cuda();policy.train();optimizer.zero_grad()
    def logps(p,completion,reference=False):
        ids=tok.encode(p,add_special_tokens=False);answer=tok.encode(completion,add_special_tokens=False)
        policy.set_adapter('reference' if reference else 'default')
        x=torch.tensor([ids+answer],device='cuda')
        logits=policy(input_ids=x,use_cache=False).logits[0,len(ids)-1:-1].float()/cfg['generation']['temperature']
        return logits.log_softmax(-1).gather(1,torch.tensor(answer,device='cuda')[:,None]).squeeze(1)
    segments=branch_segments(prompt,branches,rewards,tok)
    denominator=sum(len(tok.encode(c,add_special_tokens=False)) for _,c,_,_ in segments)
    kls=[];losses=[]
    for p,c,adv,kind in segments:
        if not c:continue
        with torch.no_grad():
            old=logps(p,c).detach();reference=logps(p,c,True).detach()
        current=logps(p,c);ratio=(current-old).exp()
        objective=torch.minimum(ratio*adv,ratio.clamp(1-cfg['rl']['clip'],1+cfg['rl']['clip'])*adv)
        delta=reference-current;kl=delta.exp()-delta-1
        loss=(-objective+cfg['rl']['beta']*kl).sum()/max(1,denominator)
        if not torch.isfinite(loss):raise FloatingPointError('Non-finite branched GRPO loss')
        loss.backward();kls.append(float(kl.detach().mean()));losses.append(float(loss.detach()))
    grad_norm=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],1)
    if not torch.isfinite(grad_norm):raise FloatingPointError('Non-finite branched GRPO gradient')
    optimizer.step()
    return dict(loss=sum(losses),kl=float(np.mean(kls)) if kls else 0,grad_norm=float(grad_norm),loss_tokens=denominator,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())


def load_reference_adapter(policy,adapter):
    import torch
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    policy.load_adapter(str(adapter),adapter_name='reference',is_trainable=False)
    initial=get_peft_model_state_dict(policy,adapter_name='default')
    # PEFT inserts the second adapter in BF16 before promoting it to FP32.
    # Copy again after promotion so the frozen reference preserves the SFT weights.
    set_peft_model_state_dict(policy,initial,adapter_name='reference')
    reference=get_peft_model_state_dict(policy,adapter_name='reference')
    if initial.keys()!=reference.keys() or not all(torch.equal(initial[k],reference[k]) for k in initial):
        raise RuntimeError('Policy and frozen reference must start with identical adapter weights')
    policy.set_adapter('default')


def rl(variant,seed,profile='main',probe=False,max_steps=None):
    import torch
    from model import load_model,tokenizer
    from evaluate import checkpoint
    if probe:
        return probe_rewards(seed,profile)
    if variant not in ('std','std32','nk','nk_prefix','nk_prefix_div'):raise ValueError(variant)
    cfg=config();probe_path=CODE/'reports/rl'/f'{profile}_probe_{variant}_{seed}.json'
    if not probe_path.exists():raise RuntimeError('Run --probe before RL; do not skip the sparse-reward gate')
    prior=json.loads(probe_path.read_text());out=ROOT/'checkpoints'/profile/f'rl_{variant}_{seed}';out.mkdir(parents=True,exist_ok=True)
    if not prior['eligible_for_training']:
        write_json(CODE/'reports/rl'/f'{profile}_{variant}_{seed}_skipped.json',dict(status='skipped',reason='active reward groups below 1% after parser/catalog/truncation audit',probe=str(probe_path)))
        return
    if (out/'complete.json').exists():return
    if variant in ('std','std32'):return standard_grpo(seed,profile,max_steps,variant)
    import numpy as np
    tok=tokenizer();catalog=load_catalog();rows=load_split(profile,'train');adapter=checkpoint('mixed',seed,profile);heads=popular_first_codes(rows)
    policy=load_model(adapter,trainable=True)
    load_reference_adapter(policy,adapter)
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    policy.set_adapter('default');policy.train();policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    for module in policy.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0
    optimizer=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=cfg['rl']['learning_rate'],weight_decay=0.0)
    rng=np.random.default_rng(seed);steps=max_steps or cfg['rl']['steps'];log=[];start_step=0
    schedule=rl_prompt_schedule(rows,seed,profile)
    resume=out/'training_state.pt'
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True)
        set_peft_model_state_dict(policy,state['policy'],adapter_name='default')
        optimizer.load_state_dict(state['optimizer']);rng.bit_generator.state=state['rng'];start_step=state['step'];log=state['log']
        del state
    policy.cpu();gc.collect();torch.cuda.empty_cache()
    if start_step and (start_step%cfg['rl']['eval_steps']==0 or start_step==steps):
        # The optimizer state is saved before monitoring. Recover a monitor that
        # was interrupted after a successfully saved update, including step 500.
        saved=out/f'step-{start_step}';policy.set_adapter('default');policy.save_pretrained(saved,selected_adapters=['default'])
        monitor_rl(saved,variant,seed,profile,start_step)
    if start_step>=steps:
        write_json(out/'complete.json',dict(steps=steps,variant=variant,reference=str(adapter)))
        return
    rollout_gen=sleeping_rollout_generator()
    for step in range(start_step+1,steps+1):
        started=time.perf_counter();index=int(rng.integers(len(rows)))
        assert index==schedule[step-1], 'RL training prompt order differs from the shared schedule'
        row=rows[index];view=input_view(row)
        selected=select_history(view,tok,cfg['input_budget'],'recent');prompt=prompt_text(view,tok,selected,True)
        snapshot=out/f'step-{step-1}';policy.set_adapter('default');policy.save_pretrained(snapshot,selected_adapters=['default'])
        policy.cpu();gc.collect();torch.cuda.empty_cache()
        record=run_rollout(snapshot,'nk',[dict(sample_id=row['sample_id'],prompt=prompt,seed=seed+step)],f'{profile}_{variant}_{seed}_step{step}',rollout_gen)[0]
        branches=record['branches'];rewards=branch_rewards(branches,row['targets'],catalog,variant)
        active=float(np.std(rewards))>1e-8
        update=branched_update(policy,optimizer,prompt,branches,rewards,tok,cfg)
        diagnostic=reward_diagnostics([record],[row],catalog,variant,heads)
        diagnostic.update(step=step,sample_id=row['sample_id'],active_reward_group=active,**update,seconds=time.perf_counter()-started,
                          credit_rule='shared reasoning once with mean child advantage')
        log.append({k:v for k,v in diagnostic.items() if k not in ('per_group','first_code_counts')})
        write_json(CODE/'reports/rl'/f'{profile}_{variant}_{seed}_training.json',log)
        saved_state=dict(step=step,policy={k:v.detach().cpu() for k,v in get_peft_model_state_dict(policy,adapter_name='default').items()},
                         optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,log=log)
        torch.save(saved_state,resume.with_suffix('.tmp'));resume.with_suffix('.tmp').replace(resume);del saved_state
        if step%cfg['rl']['eval_steps']==0 or step==steps:
            saved=out/f'step-{step}';policy.set_adapter('default');policy.save_pretrained(saved,selected_adapters=['default'])
            policy.cpu();gc.collect();torch.cuda.empty_cache();monitor_rl(saved,variant,seed,profile,step);policy.cuda()
        print('RL',log[-1],flush=True)
    write_json(out/'complete.json',dict(steps=steps,variant=variant,reference=str(adapter)))


def standard_grpo(seed,profile,max_steps,variant='std'):
    import torch
    from trl import GRPOConfig,GRPOTrainer
    from transformers import TrainerCallback,set_seed
    from datasets import Dataset
    from model import load_model,tokenizer
    from evaluate import checkpoint
    cfg=config();set_seed(seed);tok=tokenizer();catalog=load_catalog();adapter=checkpoint('mixed',seed,profile)
    group_size=32 if variant=='std32' else 8
    original_decode=tok.batch_decode
    def sid_decode(*args,**kwargs):
        kwargs['skip_special_tokens']=False
        return original_decode(*args,**kwargs)
    tok.batch_decode=sid_decode
    records=[];all_train=load_split(profile,'train');heads=popular_first_codes(all_train)
    for index in rl_prompt_schedule(all_train,seed,profile):
        row=all_train[index]
        view=input_view(row);chosen=select_history(view,tok,cfg['input_budget'],'recent')
        records.append(dict(prompt=prompt_text(view,tok,chosen,True),targets=row['targets'],sample_id=row['sample_id']))
    out=ROOT/'checkpoints'/profile/f'rl_{variant}_{seed}';out.mkdir(parents=True,exist_ok=True)
    from transformers.trainer_utils import get_last_checkpoint
    resume_checkpoint=get_last_checkpoint(str(out))
    diagnostics_path=CODE/'reports/rl'/f'{profile}_{variant}_{seed}_diagnostics.json'
    resume_step=json.loads((Path(resume_checkpoint)/'trainer_state.json').read_text())['global_step'] if resume_checkpoint else 0
    diagnostics=[r for r in json.loads(diagnostics_path.read_text()) if r.get('optimizer_step',0)<=resume_step] if resume_checkpoint and diagnostics_path.exists() else []
    def reward_fn(completions,targets,sample_id,**kwargs):
        values=[reward('<think>\n'+c,t,catalog) for c,t in zip(completions,targets)]
        assert len(values)==group_size and all(t==targets[0] for t in targets), 'GRPO requires one prompt and the configured number of complete trajectories per group'
        assert len(sample_id)==group_size and set(sample_id)=={records[len(diagnostics)]['sample_id']}, 'TRL prompt order differs from the shared RL schedule'
        outputs=[dict(text='<think>\n'+c,generated_tokens=len(tok.encode(c,add_special_tokens=False)),seconds=0) for c in completions]
        d=reward_diagnostics([dict(sample_id='group',outputs=outputs)],[dict(sample_id='group',targets=targets[0],train_target=targets[0][0])],catalog,variant,heads)
        diagnostics.append(dict(sample_id=sample_id[0],**{k:v for k,v in d.items() if k not in ('per_group','first_code_counts','generation_seconds')}))
        return values
    model=load_model(adapter,trainable=True)
    args=GRPOConfig(output_dir=str(out),per_device_train_batch_size=1,gradient_accumulation_steps=group_size,num_generations=group_size,generation_batch_size=group_size,
                    learning_rate=cfg['rl']['learning_rate'],beta=cfg['rl']['beta'],epsilon=cfg['rl']['clip'],max_steps=max_steps or cfg['rl']['steps'],
                    max_prompt_length=cfg['input_budget'],max_completion_length=cfg['answer_budget'],temperature=.8,top_p=.95,top_k=0,
                    bf16=True,disable_dropout=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                    use_vllm=False,report_to='none',save_steps=cfg['rl']['eval_steps'],logging_steps=1,seed=seed,disable_tqdm=True,loss_type='grpo',lr_scheduler_type='constant',
                    generation_kwargs={'top_k':0},save_total_limit=2,shuffle_dataset=False)
    trainer=GRPOTrainer(model=model,args=args,processing_class=tok,reward_funcs=reward_fn,train_dataset=Dataset.from_list(records))
    # TRL 0.24 disables LoRA by default for its reference. Explicitly replace that
    # path with a frozen copy of the actual SFT policy, never the raw base model.
    reference=load_model(adapter,trainable=False);reference.eval()
    from peft import get_peft_model_state_dict
    current=get_peft_model_state_dict(model);frozen=get_peft_model_state_dict(reference)
    assert current.keys()==frozen.keys() and all(torch.equal(current[k],frozen[k]) for k in current)
    del current,frozen
    for p in reference.parameters():p.requires_grad_(False)
    trainer.ref_model=trainer.accelerator.prepare_model(reference,evaluation_mode=True)
    # Generation keeps 32 full trajectories plus a separate frozen SFT reference.
    # The SFT default cap (45%) is insufficient even when the GPU has free memory.
    torch.cuda.set_per_process_memory_fraction(.8)
    class Monitor(TrainerCallback):
        started=0
        def on_step_begin(self,args,state,control,**kwargs):
            self.started=time.perf_counter()
        def on_step_end(self,args,state,control,**kwargs):
            if diagnostics:
                diagnostics[-1].update(optimizer_step=state.global_step,step_seconds=time.perf_counter()-self.started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                write_json(CODE/'reports/rl'/f'{profile}_{variant}_{seed}_diagnostics.json',diagnostics)
            if state.global_step%cfg['rl']['eval_steps']!=0 and state.global_step!=state.max_steps:return
            dest=out/f'step-{state.global_step}';model.save_pretrained(dest)
            model.cpu();reference.cpu();gc.collect();torch.cuda.empty_cache()
            monitor_rl(dest,variant,seed,profile,state.global_step)
            model.cuda();reference.cuda()
            write_json(CODE/'reports/rl'/f'{profile}_{variant}_{seed}_diagnostics.json',diagnostics)
        def on_log(self,args,state,control,logs=None,**kwargs):
            if diagnostics and logs and 'kl' in logs:diagnostics[-1]['kl']=logs['kl']
    trainer.add_callback(Monitor());trainer.train(resume_from_checkpoint=resume_checkpoint)
    write_json(diagnostics_path,diagnostics)
    write_json(out/'complete.json',dict(steps=trainer.state.global_step,reference=str(adapter),trainer_log=trainer.state.log_history))
