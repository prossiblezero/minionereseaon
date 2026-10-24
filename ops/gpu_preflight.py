"""Exercise real LoRA updates and vLLM sleep/wake; never count as an RL result."""
import gc
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    from transformers import set_seed
    from data import CODE,ROOT,config,input_view,load_split,load_catalog,write_json
    from evaluate import checkpoint
    from history import select_history,prompt_text
    from model import load_model,tokenizer
    from train import sleeping_rollout_generator,run_rollout,training_records,branched_update,branch_rewards,load_reference_adapter
    set_seed(42);tok=tokenizer();row=load_split('main','train')[0]
    policy=load_model(checkpoint('mixed',42),trainable=True)
    load_reference_adapter(policy,checkpoint('mixed',42))
    policy.set_adapter('default')
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    policy.train()
    for module in policy.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0
    optimizer=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=5e-6)
    directory=ROOT/'checkpoints/smoke/engine_preflight';directory.mkdir(parents=True,exist_ok=True)
    record=training_records([row],tok,'direct',config()['input_budget'])[0][0]
    view=input_view(row);chosen=select_history(view,tok,config()['input_budget'],'recent')
    request=dict(sample_id=row['sample_id'],prompt=prompt_text(view,tok,chosen,True),seed=42)
    policy.cpu();gc.collect();torch.cuda.empty_cache();gen=sleeping_rollout_generator()
    checks=[];run_id=time.time_ns()
    for step in range(2):
        snapshot=directory/f'step-{step}';policy.save_pretrained(snapshot,selected_adapters=['default'])
        weight_hash=hashlib.sha256((snapshot/'adapter_model.safetensors').read_bytes()).hexdigest()
        rows=run_rollout(snapshot,'nk',[request],f'smoke_engine_preflight_{run_id}_{step}_{weight_hash[:12]}',gen)
        assert len(rows)==1 and len(rows[0]['branches'])==4
        assert all(len(branch['answers'])==8 for branch in rows[0]['branches'])
        policy.cuda();policy.train();optimizer.zero_grad()
        loss=policy(**{key:torch.tensor([value],device='cuda') for key,value in record.items()}).loss
        assert torch.isfinite(loss);loss.backward()
        gradients=[p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(torch.count_nonzero(g) for g in gradients)
        optimizer.step()
        checks.append(dict(step=step,adapter_sha256=weight_hash,loss=float(loss.detach()),gradient_tensors=len(gradients),lora_id=gen.adapter.lora_int_id))
        del loss,gradients;policy.cpu();gc.collect();torch.cuda.empty_cache()
    assert checks[0]['adapter_sha256']!=checks[1]['adapter_sha256']
    assert checks[0]['lora_id']!=checks[1]['lora_id']
    branches=rows[0]['branches'];rewards=branch_rewards(branches,row['targets'],load_catalog(),'nk_prefix')
    update=branched_update(policy,optimizer,request['prompt'],branches,rewards,tok,config())
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    state=dict(policy={k:v.detach().cpu() for k,v in get_peft_model_state_dict(policy).items()},optimizer=optimizer.state_dict())
    torch.save(state,directory/'training_state.pt')
    restored=torch.load(directory/'training_state.pt',map_location='cpu',weights_only=True)
    set_peft_model_state_dict(policy,restored['policy']);optimizer.load_state_dict(restored['optimizer'])
    assert all(torch.equal(value.detach().cpu(),restored['policy'][key]) for key,value in get_peft_model_state_dict(policy).items())
    report=dict(passed=True,checks=checks,branched_update=update,checkpoint_reload=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                scope='Native sleep/wake, distinct LoRAs, real-label SFT gradients, actual branch-update function and checkpoint reload; not an RL performance result')
    write_json(CODE/'reports/rl/native_engine_preflight.json',report)
    print(report,flush=True)


if __name__=='__main__':main()
