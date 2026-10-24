"""Local model loading and the two explicitly identified generation backends."""
import time
from pathlib import Path
from data import ROOT, config, digest

MODEL = ROOT / 'models/onereason-0.8b'


def tokenizer():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    tok.padding_side='left'
    for text in ('<|prod_begin|>','<s_a_0>','<s_b_8191>','<s_c_4>'):
        ids=tok.encode(text,add_special_tokens=False)
        assert len(ids)==1 and tok.decode(ids,skip_special_tokens=False)==text, text
    assert '</think>' in tok.apply_chat_template([dict(role='user',content='test')],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return tok


def load_model(adapter=None, trainable=False):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model
    cfg=config()
    torch.cuda.set_per_process_memory_fraction(cfg['torch_memory_fraction'])
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda()
    if adapter:
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=trainable)
    elif trainable:
        lc=cfg['lora']
        model=get_peft_model(model,LoraConfig(**lc,bias='none',task_type='CAUSAL_LM'))
    if trainable:
        model.config.use_cache=False
        model.enable_input_require_grads()
    return model


class Generator:
    def __init__(self,adapter=None,backend='vllm',existing_model=None):
        self.backend=backend
        self.tok=tokenizer()
        self.cfg=config()
        self.adapter=None
        if backend=='vllm':
            from vllm import LLM
            from vllm.lora.request import LoRARequest
            self.engine=LLM(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',max_model_len=4096,
                            gpu_memory_utilization=self.cfg['vllm_memory_fraction'],enforce_eager=False,
                            enable_lora=True,max_lora_rank=self.cfg['lora']['r'],max_num_seqs=64,
                            max_num_batched_tokens=4096,enable_prefix_caching=True,seed=42)
            if adapter:
                path=str(Path(adapter).resolve())
                self.adapter=LoRARequest(Path(path).name,int(digest(path)[:7],16)+1,path)
        elif backend=='transformers':
            self.engine=existing_model if existing_model is not None else load_model(adapter)
            self.engine.eval()
        else:
            raise ValueError(backend)

    def raw(self,prompt,n,max_tokens,seed,temperature=None,stop=None):
        temperature=self.cfg['generation']['temperature'] if temperature is None else temperature
        started=time.perf_counter()
        if self.backend=='vllm':
            from vllm import SamplingParams
            params=SamplingParams(n=n,max_tokens=max_tokens,temperature=temperature,top_p=self.cfg['generation']['top_p'] if temperature else 1,
                                  seed=seed,skip_special_tokens=False,stop=stop,include_stop_str_in_output=False)
            output=self.engine.generate([prompt],params,lora_request=self.adapter,use_tqdm=False)[0]
            result=[dict(text=o.text,token_ids=list(o.token_ids),finish_reason=o.finish_reason,stop_reason=o.stop_reason) for o in output.outputs]
        else:
            import torch
            from transformers import set_seed
            result=[]
            # Chunking changes throughput but preserves the total attempt budget.
            for start in range(0,n,4):
                count=min(4,n-start)
                set_seed(seed+start)
                inputs=self.tok([prompt]*count,return_tensors='pt',padding=True).to('cuda')
                kwargs=dict(max_new_tokens=max_tokens,do_sample=bool(temperature),pad_token_id=self.tok.pad_token_id,
                            eos_token_id=self.tok.eos_token_id,use_cache=True)
                if temperature:
                    kwargs.update(temperature=temperature,top_p=self.cfg['generation']['top_p'],top_k=0)
                if stop:
                    kwargs.update(stop_strings=stop,tokenizer=self.tok)
                with torch.inference_mode():
                    output=self.engine.generate(**inputs,**kwargs)
                for ids in output[:,inputs.input_ids.shape[1]:].tolist():
                    if self.tok.eos_token_id in ids:
                        ids=ids[:ids.index(self.tok.eos_token_id)+1]
                    while ids and ids[-1]==self.tok.pad_token_id:
                        ids.pop()
                    text=self.tok.decode(ids,skip_special_tokens=False)
                    stop_reason=next((marker for marker in (stop or []) if marker in text),None)
                    if stop:
                        for marker in stop:
                            text=text.split(marker)[0]
                    result.append(dict(text=text,token_ids=ids,finish_reason='stop' if len(ids)<max_tokens else 'length',stop_reason=stop_reason))
        elapsed=time.perf_counter()-started
        for item in result:
            item.update(generated_tokens=len(item['token_ids']),seconds=elapsed/n)
        return result

    def raw_many(self,prompts,n,max_tokens,seeds,temperature=None,stop=None):
        if len(prompts)!=len(seeds):raise ValueError('Each prompt requires its own seed')
        if self.backend!='vllm':
            return [self.raw(p,n,max_tokens,s,temperature,stop) for p,s in zip(prompts,seeds)]
        from vllm import SamplingParams
        temperature=self.cfg['generation']['temperature'] if temperature is None else temperature
        params=[SamplingParams(n=n,max_tokens=max_tokens,temperature=temperature,
                 top_p=self.cfg['generation']['top_p'] if temperature else 1,seed=s,
                 skip_special_tokens=False,stop=stop,include_stop_str_in_output=False) for s in seeds]
        started=time.perf_counter()
        outputs=self.engine.generate(prompts,params,lora_request=self.adapter,use_tqdm=False)
        seconds=(time.perf_counter()-started)/max(1,len(prompts)*n)
        return [[dict(text=o.text,token_ids=list(o.token_ids),finish_reason=o.finish_reason,
                      stop_reason=o.stop_reason,generated_tokens=len(o.token_ids),seconds=seconds)
                 for o in output.outputs] for output in outputs]

    def generate(self,prompt,n,seed,thinking=False,budget=None,temperature=None):
        if budget is None:
            limit=self.cfg['generation']['thinking_tokens' if thinking else 'direct_tokens']
            outputs=self.raw(prompt,n,limit,seed,temperature)
            for o in outputs:
                if thinking:
                    o['text']='<think>\n'+o['text']
                    reason=o['text'].split('</think>',1)[0].removeprefix('<think>\n')
                    o['reasoning_tokens']=len(self.tok.encode(reason,add_special_tokens=False))
                else:
                    o['reasoning_tokens']=0
                o['forced_close']=False
            return outputs
        if budget==0:
            return self.generate(prompt,n,seed,False,None,temperature)
        prefixes=self.raw(prompt,n,budget,seed,temperature,stop=['</think>'])
        outputs=[]
        answer_prompts=[prompt+p['text']+'\n</think>\n\n' for p in prefixes]
        answers=self.raw_many(answer_prompts,1,self.cfg['generation']['direct_tokens'],
                              [seed+100003+i for i in range(len(prefixes))],temperature)
        for prefix,answer_group in zip(prefixes,answers):
            answer=answer_group[0]
            answer.update(text='<think>\n'+prefix['text']+'\n</think>\n\n'+answer['text'],
                          generated_tokens=prefix['generated_tokens']+answer['generated_tokens'],
                          reasoning_tokens=prefix['generated_tokens'],answer_tokens=answer['generated_tokens'],
                          seconds=prefix['seconds']+answer['seconds'],forced_close=prefix.get('stop_reason')!='</think>')
            outputs.append(answer)
        return outputs
