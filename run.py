"""Small CLI; heavyweight imports happen only in the selected command."""
import argparse
import json
from pathlib import Path
from data import CODE, ROOT, write_json


def main():
    p=argparse.ArgumentParser(description='MiniOneReason real-data experiments')
    p.add_argument('command',choices=['download','prepare','validate','manifest','catalog-audit','smoke','sft','baseline','evaluate','thinkcost','rl','history','report','backend-check','parity','rollout','pipeline','supplement','official'])
    p.add_argument('--profile',choices=['smoke','main'],default='main')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--variant',default='mixed')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--suite',default='main')
    p.add_argument('--backend',choices=['vllm','transformers'],default='vllm')
    p.add_argument('--thinking',action='store_true')
    p.add_argument('--budget',type=int)
    p.add_argument('--input-budget',type=int)
    p.add_argument('--strategy',choices=['recent','domain_recent','balanced'],default='recent')
    p.add_argument('--samples',type=int,default=64)
    p.add_argument('--limit',type=int)
    p.add_argument('--max-steps',type=int)
    p.add_argument('--adapter')
    p.add_argument('--tag')
    p.add_argument('--request')
    p.add_argument('--response')
    p.add_argument('--seed-offset',type=int,default=0)
    p.add_argument('--wait-pid',type=int)
    p.add_argument('--monitor',action='store_true')
    p.add_argument('--probe',action='store_true')
    args=p.parse_args()
    (CODE/'reports').mkdir(exist_ok=True)
    if args.command=='pipeline':
        pipeline(args.wait_pid)
    elif args.command=='supplement':
        from ops.supplement import run
        run()
    elif args.command=='official':
        from ops.official import run
        run(args.wait_pid)
    elif args.command in ('download','prepare','validate','manifest','catalog-audit'):
        import data
        if args.command=='manifest': data.manifest()
        elif args.command=='download': data.download()
        elif args.command=='catalog-audit':data.audit_catalog_collisions()
        else: getattr(data,args.command)(args.profile)
    elif args.command in ('sft','smoke','rl'):
        import train
        if args.command=='sft': train.sft(args.variant,args.seed,args.profile,args.max_steps)
        elif args.command=='smoke': train.smoke()
        else: train.rl(args.variant,args.seed,args.profile,args.probe,args.max_steps)
    elif args.command=='baseline':
        from baselines import baseline
        baseline(args.seed,args.profile)
    elif args.command=='evaluate':
        from evaluate import evaluate_one
        evaluate_one(args.variant,args.seed,args.split,args.profile,args.thinking,args.budget,args.strategy,args.input_budget,args.samples,args.backend,args.limit,args.tag,args.adapter,args.seed_offset,args.monitor)
    elif args.command=='rollout':
        from train import rollout_worker
        rollout_worker(args.request,args.response)
    elif args.command=='backend-check':
        from evaluate import backend_check
        backend_check(args.profile,args.backend)
    elif args.command=='parity':
        from evaluate import compare_backends
        print(json.dumps(compare_backends(args.profile)))
    elif args.command in ('thinkcost','history','report'):
        from evaluate import experiments, report
        if args.command=='report': report()
        else: experiments(args.command,args.seed,args.split,args.profile,args.variant)


def scheduled_jobs(jobs):
    # Keep original IDs; additional seeds require a later user decision.
    stages={'B':0,'D':1,'C':2,'E':3,'F':4}
    active=[(index,job) for index,job in enumerate(jobs) if job['stage']!='B_repeats']
    return sorted(active,key=lambda item:(stages[item[1]['stage']],item[0]))


def pipeline(wait_pid=None):
    import datetime,fcntl,hashlib,os,subprocess,sys,time
    from data import config
    lock=(CODE/'reports/pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (CODE/'reports/pipeline.pid').write_text(str(os.getpid()))
    state_path=CODE/'reports/pipeline_status.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {'done':[],'skipped':[]}
    state.update(scope='seed_42',active_seeds=[42],deferred_seeds=[43,44],scheduling_policy='B -> D -> C -> E -> F')
    state.pop('next_after_current',None)
    def status(stage,message):
        state.pop('rollout_progress',None)
        state.update(stage=stage,current=message,pid=os.getpid(),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(state_path,state);(CODE/'reports/active_stage.txt').write_text(stage+': '+message+'\nPID '+str(os.getpid()))
    child=state.get('child_pid')
    if child:
        path=Path('/proc')/str(child)/'cmdline'
        while path.exists() and Path('/proc',str(child),'cwd').resolve()==CODE and b'run.py' in path.read_bytes():
            status(state.get('stage','resume'),'waiting for previously launched child '+str(child))
            time.sleep(5)
        state.pop('child_pid',None)
    if wait_pid:
        status('B','waiting for existing stage_b_main.sh process')
        while True:
            p=Path('/proc')/str(wait_pid)/'cmdline'
            if not p.exists() or b'stage_b_main.sh' not in p.read_bytes():break
            time.sleep(5)
    jobs=[]
    def add(stage,*args,when=None):jobs.append(dict(stage=stage,args=list(args),when=when))
    for variant in ('direct','mixed'):add('B','sft','--variant',variant,'--seed','42')
    add('B','evaluate','--profile','smoke','--variant','mixed','--split','validation','--limit','4','--samples','8','--tag','smoke_vllm_lora')
    add('B','baseline','--seed','42')
    def evaluations(stage,seed,variants):
        for split in ('validation','test'):
            for variant,modes in variants:
                for thinking in modes:
                    args=['evaluate','--variant',variant,'--seed',str(seed),'--split',split]
                    if thinking:args.append('--thinking')
                    add(stage,*args,when=f'{variant}_{seed}' if variant.startswith('rl_') else None)
    evaluations('B',42,[('base',[False]),('direct',[False]),('mixed',[False,True])])
    for split in ('validation','test'):add('C','thinkcost','--split',split)
    for seed in (43,44):
        for variant in ('direct','mixed'):add('B_repeats','sft','--variant',variant,'--seed',str(seed))
        evaluations('B_repeats',seed,[('direct',[False]),('mixed',[False,True])])
    add('D','rl','--probe')
    for variant in ('std','std32','nk','nk_prefix'):
        add('D','rl','--variant',variant)
        evaluations('D',42,[(f'rl_{variant}',[False,True])])
    add('E','sft','--variant','direct_tokenmatched')
    evaluations('E',42,[('direct_tokenmatched',[False])])
    for variant in ('mixed_recent_history','mixed_balanced'):add('E','sft','--variant',variant)
    for split in ('validation','test'):add('E','history','--split',split)
    add('F','report')
    prior_protocol=CODE/'artifacts/execution_protocol.json'
    protocol=dict(config=config(),jobs=jobs,config_sha256=hashlib.sha256((CODE/'config.json').read_bytes()).hexdigest(),
                  revision=2,parent_sha256=hashlib.sha256(prior_protocol.read_bytes()).hexdigest() if prior_protocol.exists() else None,
                  amendments=['Add standard GRPO with 32 completions','Batch answer continuations and enable CUDA graphs; all model comparisons rerun in isolated results_v2 and predictions_v2 directories'])
    protocol_path=CODE/'artifacts/execution_protocol_v2.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Frozen execution protocol changed; use a new run before changing comparisons')
    write_json(protocol_path,protocol)
    order=scheduled_jobs(jobs)
    active_indices={index for index,_ in order}
    deferred=[index for index in range(len(jobs)) if index not in active_indices]
    state['deferred']=[f"{index:03d}_"+'_'.join(x for x in jobs[index]['args'] if not x.startswith('--')) for index in deferred]
    write_json(CODE/'artifacts/scheduling_revision.json',dict(
        protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        reason='User requested the complete seed-42 chain first on 2026-10-21; seeds 43/44 require a later decision',
        scope='seed_42',active_seeds=[42],deferred_seeds=[43,44],
        stage_order=['B','D','C','E','F'],job_indices=[index for index,_ in order],deferred_job_indices=deferred))
    for index,job in order:
        key=f"{index:03d}_"+'_'.join(x for x in job['args'] if not x.startswith('--'))
        if key in state['done'] or key in state['skipped']:continue
        if job['when'] and not (ROOT/'checkpoints/main'/job['when']/'selected').exists():
            state['skipped'].append(key);status(job['stage'],'skipped missing adapter: '+job['when']);continue
        status(job['stage'],' '.join(job['args']))
        log=CODE/'reports/jobs'/f'{key}.log';log.parent.mkdir(exist_ok=True)
        with log.open('a') as output:
            process=subprocess.Popen([sys.executable,str(CODE/'run.py'),*job['args']],cwd=CODE,stdout=output,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid;write_json(state_path,state);code=process.wait()
        if code:
            state['failure']=dict(job=key,exit_code=code,log=str(log));status('failed','Inspect '+str(log));raise RuntimeError(state['failure'])
        state['done'].append(key);state.pop('failure',None);status(job['stage'],'completed '+key)
        subprocess.run([sys.executable,str(CODE/'run.py'),'report'],cwd=CODE,check=True)
    state.pop('child_pid',None)
    status('complete','Seed-42 chain finished; seeds 43/44 remain deferred pending a later user decision')


if __name__=='__main__':
    try:
        main()
    except BlockingIOError:
        # A second launch must not overwrite the active owner's status.
        raise
    except Exception as exc:
        import datetime,sys
        path=CODE/'reports/pipeline_status.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        if sys.argv[1:2] in (['pipeline'],['supplement']) or (sys.argv[1:2]==['rl'] and state.get('stage')=='RL_preflight'):
            state.update(stage='failed',current=f'{type(exc).__name__}: {exc}',updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            state.setdefault('failure',dict(command=sys.argv[1:],error=type(exc).__name__,message=str(exc)))
            write_json(path,state)
        raise
