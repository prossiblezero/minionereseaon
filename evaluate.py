"""Fixed-attempt, full-catalog evaluation and paired cluster uncertainty."""
import collections
import hashlib
import json
from pathlib import Path
from data import CODE, ROOT, config, digest, input_view, load_catalog, load_split, parse_answer, read_jsonl, write_json, write_jsonl
from history import prompt_text, select_history


def metrics(rows,predictions,catalog,ks=(8,32,64)):
    by_id={r['sample_id']:r for r in rows}
    per=[]; covered=set(); invalid=collections.Counter(); tokens=seconds=attempts=duplicates=0;reasoning_lengths=[]
    for pred in predictions:
        row=by_id[pred['sample_id']]
        tokens+=pred.get('overhead_tokens',0);seconds+=pred.get('overhead_seconds',0)
        items=[]
        for o in pred['outputs']:
            sid,reason=parse_answer(o['text'],catalog)
            items.append(sid)
            invalid[reason]+=1
            if sid: covered.add(sid)
            tokens+=o.get('generated_tokens',0);seconds+=o.get('seconds',0);attempts+=1
            reasoning_lengths.append(o.get('reasoning_tokens',0))
        valid=[s for s in items if s]
        duplicates+=len(valid)-len(set(valid))
        record=dict(sample_id=row['sample_id'],group_id=row['group_id'])
        target=set(row['targets'])
        for k in ks:
            if len(items)<k:
                continue
            hit=len(set(items[:k]) & target)
            record[f'recall@{k}']=hit/len(target)
            record[f'pass@{k}']=float(hit>0)
        per.append(record)
    summary={key:sum(p[key] for p in per)/len(per) for key in per[0] if key.startswith(('recall@','pass@'))} if per else {}
    summary.update(examples=len(per),attempts=attempts,invalid_rate=1-invalid['valid']/max(1,attempts),
                   duplicate_rate=duplicates/max(1,attempts),catalog_coverage=len(covered)/max(1,len(catalog)),
                   unique_recommended_sids=len(covered),generated_tokens=tokens,generation_seconds=seconds,
                   tokens_per_example=tokens/max(1,len(per)),seconds_per_example=seconds/max(1,len(per)),parser_counts=dict(invalid))
    if reasoning_lengths:
        summary.update(mean_reasoning_tokens=sum(reasoning_lengths)/len(reasoning_lengths),p95_reasoning_tokens=sorted(reasoning_lengths)[min(len(reasoning_lengths)-1,int(len(reasoning_lengths)*.95))])
    return summary,per


def paired_bootstrap(left,right,key,resamples=1000,seed=42,with_se=False):
    import numpy as np
    a={r['sample_id']:r for r in left}; b={r['sample_id']:r for r in right}
    if a.keys()!=b.keys():
        raise ValueError('paired comparison requires identical sample IDs')
    groups=collections.defaultdict(list)
    for sid in a:
        groups[a[sid]['group_id']].append(b[sid][key]-a[sid][key])
    g=list(groups.values());rng=np.random.default_rng(seed)
    sums=np.array([sum(x) for x in g]); sizes=np.array([len(x) for x in g])
    estimates=[]
    for _ in range(resamples):
        idx=rng.integers(len(g),size=len(g));estimates.append(float(sums[idx].sum()/sizes[idx].sum()))
    result=dict(delta=float(sums.sum()/sizes.sum()),ci95=[float(x) for x in np.percentile(estimates,[2.5,97.5])],clusters=len(g),resamples=resamples)
    if with_se:result['bootstrap_standard_error']=float(np.std(estimates,ddof=1))
    return result


def checkpoint(variant,seed,profile='main'):
    if variant=='base': return None
    path=ROOT/'checkpoints'/profile/f'{variant}_{seed}'/'selected'
    if not path.exists(): raise FileNotFoundError(path)
    return path


def identical_input_cache(directory,settings,destination):
    fields=('adapter_sha256','seed','seed_offset','n','thinking','budget','backend','profile','split',
            'protocol_version','protocol_source_sha256','runtime_id')
    signature=tuple(settings.get(k) for k in fields);index={}
    for sidecar in sorted(directory.glob('*.settings.json')):
        source=sidecar.with_name(sidecar.name.removesuffix('.settings.json')+'.jsonl')
        if source==destination or not source.exists():continue
        old=json.loads(sidecar.read_text())
        if tuple(old.get(k) for k in fields)!=signature:continue
        with source.open('rb') as stream:
            while True:
                offset=stream.tell();line=stream.readline()
                if not line:break
                row=json.loads(line)
                if len(row['outputs'])!=settings['n']:raise ValueError('Cached attempt count differs from protocol')
                index.setdefault((row['sample_id'],row['prompt_sha256']),(source,offset))
    return index


def evaluate_one(variant,seed,split,profile='main',thinking=False,budget=None,strategy='recent',input_budget=None,n=64,backend='vllm',limit=None,tag=None,adapter_override=None,seed_offset=0,monitor=False,prediction_directory=None,result_directory=None,runtime_identity_path=None):
    from model import Generator,tokenizer
    cfg=config();input_budget=input_budget or cfg['input_budget']
    protocol=CODE/'artifacts/execution_protocol_v2.json'
    if protocol.exists() and hashlib.sha256((CODE/'config.json').read_bytes()).hexdigest()!=json.loads(protocol.read_text())['config_sha256']:
        raise RuntimeError('Sampling configuration differs from the frozen v2 run')
    adapter=Path(adapter_override) if adapter_override else checkpoint(variant,seed,profile)
    label=tag or f'{variant}_{seed}_{split}_{strategy}_in{input_budget}_think{thinking}_budget{budget}_n{n}_{backend}'
    prediction_directory=Path(prediction_directory) if prediction_directory else ROOT/'predictions_v2'/profile
    result_directory=Path(result_directory) if result_directory else CODE/'reports/results_v2'
    dest=prediction_directory/(label+'.jsonl')
    rows=load_split(profile,split)
    if monitor:
        if split!='validation':raise ValueError('Monitor selection only applies to validation')
        by_id={r['sample_id']:r for r in rows};ids=json.loads((CODE/'artifacts'/f'monitor_samples_{profile}.json').read_text());rows=[by_id[sid] for sid in ids]
    rows=rows[:limit]
    settings=dict(variant=variant,seed=seed,split=split,profile=profile,thinking=thinking,budget=budget,strategy=strategy,input_budget=input_budget,n=n,backend=backend,
                  sample_ids=[r['sample_id'] for r in rows],adapter=str(adapter),seed_offset=seed_offset,monitor=monitor,gpu_condition='shared; wall-time includes contention')
    if adapter:
        settings['adapter_sha256']=hashlib.sha256((adapter/'adapter_model.safetensors').read_bytes()).hexdigest()
    settings['protocol_version']='2.0-runtime-isolated'
    settings['protocol_source_sha256']=hashlib.sha256(b''.join((CODE/name).read_bytes() for name in ('data.py','history.py','model.py'))).hexdigest()
    runtime_identity=Path(runtime_identity_path) if runtime_identity_path else CODE/'artifacts/runtime_identity.json'
    if runtime_identity.exists():
        settings['runtime_id']=json.loads(runtime_identity.read_text())['runtime_id']
    sidecar=dest.with_suffix('.settings.json')
    if sidecar.exists() and json.loads(sidecar.read_text())!=settings:
        raise RuntimeError(f'Refusing to overwrite different experiment: {label}')
    write_json(sidecar,settings)
    previous=read_jsonl(dest) if dest.exists() else []
    done={p['sample_id'] for p in previous}
    remaining=[r for r in rows if r['sample_id'] not in done]
    if remaining:
        cached=identical_input_cache(dest.parent,settings,dest);tok=tokenizer();gen=None
        with dest.open('a',encoding='utf-8') as f:
            for i,row in enumerate(remaining):
                view=input_view(row); chosen=select_history(view,tok,input_budget,strategy)
                prompt=prompt_text(view,tok,chosen,thinking)
                sample_seed=(seed+seed_offset+int(digest(row['sample_id'])[:8],16))%(2**31-1)
                reuse=cached.get((row['sample_id'],digest(prompt)))
                if reuse:
                    with reuse[0].open('rb') as source:
                        source.seek(reuse[1]);outputs=json.loads(source.readline())['outputs']
                else:
                    if gen is None:gen=Generator(adapter,backend)
                    outputs=gen.generate(prompt,n,sample_seed,thinking,budget)
                record=dict(sample_id=row['sample_id'],group_id=row['group_id'],outputs=outputs,prompt_sha256=digest(prompt),
                            input_tokens=len(tok.encode(prompt,add_special_tokens=False)),retained_events=len(chosen),
                            reused_from=str(reuse[0]) if reuse else None)
                f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
                if i%10==0:print(label,i+1,len(remaining),flush=True)
    summary,per=metrics(rows,read_jsonl(dest),load_catalog())
    summary.update(experiment=label,raw_predictions=str(dest),settings=settings)
    write_json(result_directory/f'{label}.json',summary)
    write_jsonl(prediction_directory/(label+'.metrics.jsonl'),per)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def compare_backends(profile='smoke'):
    # Run each backend in its own process so allocator state cannot overlap.
    import subprocess,sys
    rows=load_split(profile,'validation')[:32]
    for backend in ('transformers','vllm'):
        subprocess.run([sys.executable,'run.py','backend-check','--profile',profile,'--backend',backend],cwd=CODE,check=True)
    left=read_jsonl(ROOT/'predictions_v2'/profile/'backend_transformers.jsonl')
    right=read_jsonl(ROOT/'predictions_v2'/profile/'backend_vllm.jsonl')
    assert [r['sample_id'] for r in left]==[r['sample_id'] for r in right]==[r['sample_id'] for r in rows]
    mismatches=[a['sample_id'] for a,b in zip(left,right) if a['outputs'][0]['text']!=b['outputs'][0]['text']]
    report=dict(examples=len(rows),exact_text_matches=len(rows)-len(mismatches),mismatches=mismatches,
                identical_token_sequences=sum(a['outputs'][0]['token_ids']==b['outputs'][0]['token_ids'] for a,b in zip(left,right)),
                protocol_version='2.0-runtime-isolated',note='vLLM omits EOS from text but retains its token ID. Compare token IDs; this is a greedy sanity check, not a claim of exact backend parity.')
    write_json(CODE/'reports/backend_parity_v2.json',report)
    return report


def backend_check(profile,backend):
    from model import Generator
    gen=Generator(backend=backend);records=[]
    for row in load_split(profile,'validation')[:32]:
        view=input_view(row);chosen=select_history(view,gen.tok,config()['input_budget'],'recent')
        prompt=prompt_text(view,gen.tok,chosen,False)
        records.append(dict(sample_id=row['sample_id'],outputs=gen.generate(prompt,1,42,temperature=0)))
    write_jsonl(ROOT/'predictions_v2'/profile/f'backend_{backend}.jsonl',records)


def run_eval_process(**kwargs):
    import subprocess,sys
    tag=kwargs.pop('tag')
    command=[sys.executable,'run.py','evaluate','--tag',tag]
    for key,value in kwargs.items():
        if value is None:continue
        option='--'+key.replace('_','-')
        if isinstance(value,bool):
            if value:command.append(option)
        else:command.extend([option,str(value)])
    subprocess.run(command,cwd=CODE,check=True)
    return json.loads((CODE/'reports/results_v2'/f'{tag}.json').read_text())


def experiments(suite,seed,split,profile,variant='mixed'):
    if suite=='thinkcost':return thinking_experiments(seed,split,profile,variant)
    from history import history_gate
    report=json.loads((CODE/'reports'/f'data_report_{profile}.json').read_text())
    budget=history_gate(report)
    write_json(CODE/'reports'/f'history_gate_{profile}.json',dict(main_budget=budget,criterion='at least 30% of test inputs require trimming',fractions=report['length_gate']['test']['trimmed_fraction']))
    selection_path=CODE/'artifacts'/f'history_selection_{profile}_{seed}.json'
    if split=='validation':
        candidates={'mixed':-1.0}
        for f in (CODE/'reports/results_v2').glob('*.json'):
            r=json.loads(f.read_text());s=r.get('settings',{});v=s.get('variant','')
            if s.get('profile')==profile and s.get('seed')==seed and s.get('split')=='validation' and s.get('thinking') and s.get('budget') is None and s.get('n')==64 and s.get('input_budget')==config()['input_budget'] and s.get('strategy')=='recent' and (v=='mixed' or v.startswith('rl_')):
                candidates[v]=r['recall@32']
        best=max(candidates,key=lambda v:(candidates[v],v=='mixed'))
        write_json(selection_path,dict(variant=best,validation_recall=candidates,main_budget=budget))
    else:
        best=json.loads(selection_path.read_text())['variant']
    variants=[best]+[v for v in ('mixed_recent_history','mixed_balanced') if (ROOT/'checkpoints'/profile/f'{v}_{seed}/selected').exists()]
    results=[]
    for variant in variants:
        for size in config()['history']['budgets'] if variant==best else [budget]:
            for strategy in ('recent','domain_recent','balanced') if variant==best else ('recent','balanced'):
                tag=f'{profile}_history_{variant}_{seed}_{split}_{strategy}_{size}'
                results.append(run_eval_process(tag=tag,profile=profile,variant=variant,seed=seed,split=split,samples=64,thinking=True,strategy=strategy,input_budget=size))
    rows=load_split(profile,split)
    for variant in variants:
        base_tag=f'{profile}_history_{variant}_{seed}_{split}_recent_{budget}'
        base=read_jsonl(ROOT/'predictions_v2'/profile/f'{base_tag}.metrics.jsonl')
        comparisons=[]
        for strategy in ('domain_recent','balanced') if variant==best else ('balanced',):
            tag=f'{profile}_history_{variant}_{seed}_{split}_{strategy}_{budget}'
            alternative=read_jsonl(ROOT/'predictions_v2'/profile/f'{tag}.metrics.jsonl')
            comparisons.append(dict(strategy=strategy,**paired_bootstrap(base,alternative,'recall@32')))
        write_json(CODE/'reports'/f'history_comparison_{profile}_{variant}_{seed}_{split}.json',comparisons)
    return results


def thinking_experiments(seed,split,profile,variant='mixed'):
    import numpy as np
    from data import codes
    namespace=profile if variant=='mixed' else f'{profile}_{variant}'
    selection_path=CODE/'artifacts'/f'thinking_selection_{namespace}_{seed}.json'
    if split=='test' and not selection_path.exists():
        raise RuntimeError('Freeze validation routing thresholds before test')
    cfg=config();rows=load_split(profile,split);catalog=load_catalog();summaries={};raw={}
    for budget in cfg['thinkcost']['budgets']:
        tag=f'{namespace}_thinkcost_{seed}_{split}_{budget}'
        summaries[budget]=run_eval_process(tag=tag,profile=profile,variant=variant,seed=seed,split=split,samples=32,thinking=budget>0,budget=budget)
        raw[budget]=read_jsonl(ROOT/'predictions_v2'/profile/(tag+'.jsonl'))
    # Route replay uses real 8- and 24-attempt calls; discarded probes count toward
    # cost, while each user still has exactly 32 final recommendation attempts.
    for name,n,offset in [('probe',8,0),('remainder',24,900001)]:
        tag=f'{namespace}_route_{seed}_{split}_{name}'
        run_eval_process(tag=tag,profile=profile,variant=variant,seed=seed,split=split,samples=n,seed_offset=offset)
        raw[name]=read_jsonl(ROOT/'predictions_v2'/profile/(tag+'.jsonl'))
    signals={}
    for row,pred in zip(rows,raw['probe']):
        sids=[parse_answer(o['text'],catalog)[0] for o in pred['outputs'][:8]]
        signals[row['sample_id']]={'uncertainty':len({codes(s)[0] for s in sids if s})/8,'history_length':len(row['history'])}
    if split=='validation':
        best=max([b for b in summaries if b>0],key=lambda b:(summaries[b]['recall@32'],-summaries[b]['tokens_per_example']))
        thresholds={}
        for signal in ('uncertainty','history_length'):
            ordered=sorted((signals[r['sample_id']][signal],digest(r['sample_id']+'route')) for r in rows)
            for fraction in cfg['thinkcost']['routing_fractions']:
                count=max(1,round(len(rows)*fraction))
                thresholds[f'{signal}_{fraction}']=list(ordered[-count])
        selection=dict(thinking_budget=best,thresholds=thresholds,selected_on='validation',seed=seed)
        if variant!='mixed':selection['variant']=variant
        write_json(selection_path,selection)
    else:
        if not selection_path.exists():raise RuntimeError('Freeze validation routing thresholds before test')
        selection=json.loads(selection_path.read_text());best=selection['thinking_budget']
    direct={p['sample_id']:dict(p) for p in raw['probe']}
    for p in raw['remainder']:direct[p['sample_id']]['outputs']=direct[p['sample_id']]['outputs']+p['outputs']
    thinking={p['sample_id']:p for p in raw[best]};results=[]
    baseline_per=metrics(rows,list(direct.values()),catalog,ks=(8,32))[1]
    def save(name,chosen,reference=False):
        predictions=[]
        for row in rows:
            sid=row['sample_id'];d=direct[sid];t=thinking[sid]
            if reference:
                outputs=d['outputs'][:16]+t['outputs'][:16];overhead=[]
            elif sid in chosen:
                outputs=t['outputs'];overhead=d['outputs'][:8]
            else:
                outputs=d['outputs'];overhead=[]
            predictions.append(dict(sample_id=sid,group_id=row['group_id'],outputs=outputs,
                                    overhead_tokens=sum(o['generated_tokens'] for o in overhead),overhead_seconds=sum(o['seconds'] for o in overhead)))
        destination=ROOT/'predictions_v2'/profile/f'{name}.jsonl'
        source_settings=dict(summaries[best]['settings'])
        source_settings.update(thinking=None,mode='route_replay',seed_offset=None,
                               source_predictions=[str(ROOT/'predictions_v2'/profile/f'{namespace}_route_{seed}_{split}_{part}.jsonl') for part in ('probe','remainder')]+[summaries[best]['raw_predictions']])
        summary,per=metrics(rows,predictions,catalog,ks=(8,32));summary.update(experiment=name,thinking_fraction=len(chosen)/len(rows),
            settings=source_settings,raw_predictions=str(destination),
            paired_vs_non_thinking=paired_bootstrap(baseline_per,per,'recall@32'),cost_method='route replay: sum actual 8/24/32-attempt call costs; discarded probe charged; controller overhead excluded')
        write_json(CODE/'reports/results_v2'/f'{name}.json',summary);write_jsonl(destination,predictions)
        write_jsonl(destination.with_suffix('.metrics.jsonl'),per);results.append(summary)
        return per
    save(f'{namespace}_route_{seed}_{split}_all_non_thinking',set())
    for key,threshold in selection['thresholds'].items():
        signal=key.rsplit('_',1)[0]
        chosen={r['sample_id'] for r in rows if (signals[r['sample_id']][signal],digest(r['sample_id']+'route'))>=tuple(threshold)}
        random_order=sorted([r['sample_id'] for r in rows],key=lambda sid:digest(sid+str(seed)+'random-route'))
        random_set=set(random_order[:len(chosen)])
        label=f'{namespace}_route_{seed}_{split}_{key}'
        per=save(label,chosen);control=save(label+'_random',random_set)
        write_json(CODE/'reports'/f'{label}_vs_random.json',paired_bootstrap(control,per,'recall@32'))
    save(f'{namespace}_thinkcost_{seed}_{split}_mixed16',set(),reference=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    x=[summaries[b]['tokens_per_example'] for b in sorted(summaries)];y=[summaries[b]['recall@32'] for b in sorted(summaries)]
    ax.plot(x,y,'o-')
    for b,xx,yy in zip(sorted(summaries),x,y):ax.annotate(str(b),(xx,yy),xytext=(4,4),textcoords='offset points')
    ax.set(xlabel='Generated tokens per user (32 attempts)',ylabel='Recall@32',title=f'{variant}: thinking budget and generation cost')
    fig.tight_layout();directory=CODE/'reports/figures';directory.mkdir(parents=True,exist_ok=True);fig.savefig(directory/f'{namespace}_thinking_{seed}_{split}.png',dpi=180);plt.close(fig)
    write_json(CODE/'reports'/f'{namespace}_thinking_{seed}_{split}_summary.json',dict(budgets=summaries,routes=results,selection=selection))


def report():
    import csv
    import statistics
    directory=CODE/'reports';results=[]
    for path in sorted((directory/'results_v2').glob('*.json')):
        result=json.loads(path.read_text())
        if result.get('settings',{}).get('profile')=='smoke' or path.name.startswith('smoke_'):continue
        if result.get('settings',{}).get('monitor'):continue
        if result.get('settings') and result['settings'].get('n') not in (32,64):continue
        result['status']='evaluated'
        settings=result.get('settings',{})
        result.update({k:settings.get(k) for k in ('seed','variant','split','thinking','budget','strategy','input_budget','n','adapter','adapter_sha256','backend','protocol_version')})
        training=directory/'training'/f'main_{settings.get("variant")}_{settings.get("seed")}_complete.json'
        if training.exists():result['training_peak_allocated_bytes']=json.loads(training.read_text()).get('peak_allocated_bytes')
        variant=settings.get('variant','')
        if variant.startswith('rl_'):
            suffix='diagnostics' if variant in ('rl_std','rl_std32') else 'training'
            path=directory/'rl'/f'main_{variant.removeprefix("rl_")}_{settings["seed"]}_{suffix}.json'
            if path.exists():result['training_peak_allocated_bytes']=max((row.get('peak_allocated_bytes',0) for row in json.loads(path.read_text())),default=None)
        result['source_result']=str(directory/'results_v2'/f"{result['experiment']}.json")
        results.append(result)
    for path in sorted((directory/'rl').glob('main_*_skipped.json')):
        skipped=json.loads(path.read_text());results.append(dict(experiment=path.stem.removesuffix('_skipped'),status='skipped',reason=skipped['reason']))
    fields=['experiment','status','reason','seed','variant','split','thinking','budget','strategy','input_budget','n','adapter','adapter_sha256','backend','protocol_version','source_result','raw_predictions',
            'examples','recall@8','recall@32','recall@64','invalid_rate','duplicate_rate','catalog_coverage','tokens_per_example','seconds_per_example','training_peak_allocated_bytes']
    with (directory/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    families=collections.defaultdict(dict)
    for r in results:
        s=r.get('settings',{})
        if not s or s.get('monitor') or s.get('mode')=='route_replay' or '_history_' in r['experiment']:continue
        family=tuple(s.get(k) for k in ('variant','split','thinking','budget','strategy','input_budget','n','backend'))
        families[family][s['seed']]=r
    seed_summary=[]
    for family,values in families.items():
        scores=[r['recall@32'] for r in values.values() if 'recall@32' in r]
        if scores:seed_summary.append(dict(configuration=family,seeds=sorted(values),mean_recall32=statistics.mean(scores),sample_std=statistics.stdev(scores) if len(scores)>1 else None))
    write_json(directory/'seed_summary.json',seed_summary)
    comparisons=[];cases=[]
    by_setting={}
    for r in results:
        s=r.get('settings',{})
        if s.get('split')=='test' and s.get('strategy')=='recent' and s.get('input_budget')==config()['input_budget'] and s.get('n')==64 and s.get('budget') is None and '_history_' not in r['experiment']:
            by_setting[(s['variant'],s['seed'],s['thinking'])]=r
    for seed in config()['seeds']:
        baseline=by_setting.get(('direct',seed,False))
        if not baseline:continue
        left=read_jsonl(Path(baseline['raw_predictions']).with_suffix('.metrics.jsonl'))
        for thinking in (False,True):
            alternative=by_setting.get(('mixed',seed,thinking))
            if not alternative:continue
            right=read_jsonl(Path(alternative['raw_predictions']).with_suffix('.metrics.jsonl'))
            comparisons.append(dict(seed=seed,thinking=thinking,**paired_bootstrap(left,right,'recall@32')))
            if seed==42 and thinking:
                rows={r['sample_id']:r for r in load_split('main','test')}
                lm={r['sample_id']:r for r in left};rm={r['sample_id']:r for r in right}
                lp={r['sample_id']:r for r in read_jsonl(baseline['raw_predictions'])};rp={r['sample_id']:r for r in read_jsonl(alternative['raw_predictions'])}
                counts=collections.Counter()
                for sid in sorted(lm):
                    a,b=lm[sid]['recall@32']>0,rm[sid]['recall@32']>0
                    category='both_hit' if a and b else 'both_miss' if not a and not b else 'thinking_win' if b else 'thinking_loss'
                    if counts[category]>=10:continue
                    counts[category]+=1;row=rows[sid]
                    def representative_outputs(prediction):
                        indices=list(range(4))
                        hit=next((i for i,o in enumerate(prediction['outputs'][:32]) if parse_answer(o['text'])[0] in row['targets']),None)
                        if hit is not None and hit not in indices:indices.append(hit)
                        return [dict(attempt=i+1,text=prediction['outputs'][i]['text'],hits_target=parse_answer(prediction['outputs'][i]['text'])[0] in row['targets']) for i in indices]
                    cases.append(dict(sample_id=sid,group_id=row['group_id'],category=category,source=row['source'],context=row['context_text'],history=row['history'],targets=row['targets'],
                                      direct_outputs=representative_outputs(lp[sid]),thinking_outputs=representative_outputs(rp[sid]),
                                      displayed_attempts='First four and first target hit among the scored 32, if outside those four',scored_attempts=32,raw_predictions=[baseline['raw_predictions'],alternative['raw_predictions']],
                                      retained_events=rp[sid]['retained_events'],input_tokens=rp[sid]['input_tokens'],
                                      direct_recall32=lm[sid]['recall@32'],thinking_recall32=rm[sid]['recall@32']))
    write_json(directory/'paired_comparisons.json',comparisons)
    write_jsonl(directory/'cases.jsonl',cases)
    history_results=[r for r in results if '_history_' in r['experiment']]
    if history_results:
        import numpy as np
        thresholds=np.quantile([len(r['history']) for r in load_split('main','train')],[1/3,2/3]).tolist()
        breakdown=[]
        for result in history_results:
            split=result['settings']['split'];rows={r['sample_id']:r for r in load_split('main',split)}
            grouped=collections.defaultdict(list)
            for m in read_jsonl(Path(result['raw_predictions']).with_suffix('.metrics.jsonl')):
                length=len(rows[m['sample_id']]['history']);group='short' if length<=thresholds[0] else 'medium' if length<=thresholds[1] else 'long'
                grouped[group].append(m['recall@32'])
            breakdown.append(dict(experiment=result['experiment'],groups={k:dict(examples=len(v),recall32=statistics.mean(v)) for k,v in grouped.items()}))
        write_json(directory/'history_length_groups.json',dict(training_tertiles=thresholds,results=breakdown))
    lines=['# MiniOneReason 实验状态与结果','',f'已生成正式汇总：{len(results)} 项。数值来自 reports/results_v2 原始汇总，未完成的实验不填数。','',
           '数据边界：按历史输入分组留出；不能证明用户级互斥或与预训练集无重叠。主指标在商品 SID 层面计算。',
           '机器条件：主模型实验来自共享 RTX 4090，迁移后的成本与历史实验来自 RTX 5090；各结果保留 runtime_id。跨显卡或共享负载下的耗时不能直接解释为方法收益。','',
           '| 实验 | 样本 | Recall@32 | 无效比例 | token/用户 |','| --- | ---: | ---: | ---: | ---: |']
    for r in results:
        if r['status']=='skipped':lines.append(f"| {r['experiment']} | — | 未训练：有效奖励组不足 | — | — |")
        else:lines.append(f"| {r['experiment']} | {r.get('examples',0)} | {r.get('recall@32',0):.6f} | {r.get('invalid_rate',0):.4f} | {r.get('tokens_per_example',0):.1f} |")
    lines+=['','不能根据训练 loss 下降或脚本退出成功声明推荐效果提升。收益结论需配对比较及区间；全零奖励跳过与训练失败分别记录。']
    if comparisons:
        lines+=['','配对差值（mixed 减 direct，测试 Recall@32）：']
        for c in comparisons:lines.append(f"- seed={c['seed']}，thinking={c['thinking']}：{c['delta']:.6f}，95% 区间 {c['ci95']}。")
    (directory/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    render_rl_reports()
    from ops.final_report import write_final_report
    audit=write_final_report()
    state_path=directory/'pipeline_status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get('stage') in ('F','complete','RL_cost','official') and not audit['complete']:
        raise RuntimeError('Final audit found missing or invalid experiment artifacts; see reports/final_audit.json')


def render_rl_reports():
    import csv
    folder=CODE/'reports/rl';probe=folder/'main_probe_summary_42.json'
    logs={}
    for path in folder.glob('main_*_42_training.json'):
        logs[path.stem.removeprefix('main_').removesuffix('_42_training')]=json.loads(path.read_text())
    for variant in ('std','std32'):
        standard=folder/f'main_{variant}_42_diagnostics.json'
        if standard.exists():logs[variant]=json.loads(standard.read_text())
    if not logs and not probe.exists():return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=CODE/'reports/figures';figures.mkdir(parents=True,exist_ok=True)
    if probe.exists():
        rows=json.loads(probe.read_text());fig,ax=plt.subplots(figsize=(7,4))
        ax.bar([r['variant'] for r in rows],[r['active_group_fraction'] for r in rows])
        ax.axhline(config()['rl']['min_active_group_fraction'],color='red',linestyle='--',label='Training gate (1%)')
        ax.set(ylabel='Fraction of reward groups with nonzero variance',title='Sparse-reward probe on fixed training inputs',
               ylim=(0,min(1,max(.05,max(r['active_group_fraction'] for r in rows)*1.25))))
        ax.legend();fig.tight_layout();fig.savefig(figures/'rl_reward_probe.png',dpi=180);plt.close(fig)
    if not logs:return
    flat=[];points={}
    for variant,rows in logs.items():
        cumulative=0;tokens_at={}
        for i,row in enumerate(rows):
            step=row.get('step',row.get('optimizer_step',i+1));cumulative+=row['generated_tokens'];tokens_at[step]=cumulative
            flat.append(dict(variant=variant,step=step,cumulative_generated_tokens=cumulative,**{k:row.get(k) for k in (
                'mean_reward','exact_hit_rate','prefix_only_hit_rate','active_group_fraction','invalid_rate','first_code_entropy','popular_first_code_share','kl')}))
        points[variant]=[]
        for path in (CODE/'reports/results_v2').glob(f'main_rl_{variant}_42_monitor_step*.json'):
            step=int(path.stem.rsplit('step',1)[1]);score=json.loads(path.read_text())
            if step in tokens_at:points[variant].append((step,tokens_at[step],score['recall@32']))
        points[variant].sort()
    if flat:
        with (CODE/'reports/rl_training_curves.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    if any(points.values()):
        fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
        for variant,values in points.items():
            if not values:continue
            axes[0].plot([v[0] for v in values],[v[2] for v in values],'o-',label=variant)
            axes[1].plot([v[1] for v in values],[v[2] for v in values],'o-',label=variant)
        axes[0].set(xlabel='Optimization steps',ylabel='Fixed validation monitor Recall@32')
        axes[1].set(xlabel='Cumulative generated tokens')
        axes[0].legend();axes[1].ticklabel_format(axis='x',style='sci',scilimits=(0,0));fig.tight_layout()
        fig.savefig(figures/'rl_step_and_token_comparison.png',dpi=180);plt.close(fig)
