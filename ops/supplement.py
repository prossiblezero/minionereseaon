"""Run the two missing RL cost evaluations without changing the main protocol."""
import hashlib
import json
from pathlib import Path

from data import CODE, ROOT, config, write_json


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_protocol():
    parent=CODE/'artifacts/execution_protocol_v2.json'
    if sha(parent)!='8364fd75e1aa407ebf45c41e99ef05788dba7fd7a8ca5ea8b083b6b5f43e937a':
        raise RuntimeError('Main execution protocol changed')
    if sha(CODE/'config.json')!=json.loads(parent.read_text())['config_sha256']:
        raise RuntimeError('Main configuration changed')
    candidates={};sources={}
    for variant in ('rl_std','rl_std32','rl_nk','rl_nk_prefix'):
        checkpoint=ROOT/'checkpoints/main'/f'{variant}_42'
        complete=checkpoint/'complete.json'
        if not complete.exists():continue
        if json.loads(complete.read_text())['steps']!=config()['rl']['steps']:continue
        result=CODE/'reports/results_v2'/f'{variant}_42_validation_recent_in2048_thinkTrue_budgetNone_n64_vllm.json'
        candidates[variant]=json.loads(result.read_text())['recall@32']
        sources[variant]=dict(path=str(result),sha256=sha(result))
    best=max(candidates,key=lambda variant:(candidates[variant],variant))
    adapter=ROOT/'checkpoints/main'/f'{best}_42/selected/adapter_model.safetensors'
    value=dict(seed=42,variant=best,selected_on='validation',selection_metric='recall@32',
               candidates=candidates,selection_sources=sources,adapter=str(adapter),adapter_sha256=sha(adapter),
               parent_protocol_sha256=sha(parent),config_sha256=sha(CODE/'config.json'),
               thinkcost=config()['thinkcost'],final_attempts=32,probe_attempts=8,remainder_attempts=24,
               core_source_sha256=hashlib.sha256(b''.join((CODE/name).read_bytes() for name in ('data.py','history.py','model.py'))).hexdigest(),
               runtime=json.loads((CODE/'artifacts/runtime_identity.json').read_text()),
               reason='Complete the RL-weight budget and routing requirement recorded at the 5090 handoff; preserve all mixed-SFT results.')
    path=CODE/'artifacts/rl_thinkcost_protocol_42.json'
    if path.exists() and json.loads(path.read_text())!=value:
        raise RuntimeError('Frozen supplemental protocol changed')
    write_json(path,value)
    return value


def audit_supplement():
    path=CODE/'artifacts/rl_thinkcost_protocol_42.json';missing=[];invalid=[]
    protocol=json.loads(path.read_text()) if path.exists() else None
    if not protocol:missing.append(str(path))
    else:
        namespace='main_'+protocol['variant']
        selection=CODE/'artifacts'/f'thinking_selection_{namespace}_42.json'
        if not selection.exists():missing.append(str(selection))
        for split in ('validation','test'):
            summary=CODE/'reports'/f'{namespace}_thinking_42_{split}_summary.json'
            required=[summary]
            tags=[f'{namespace}_thinkcost_42_{split}_{budget}' for budget in protocol['thinkcost']['budgets']]
            tags += [f'{namespace}_route_42_{split}_{part}' for part in ('probe','remainder','all_non_thinking')]
            tags += [f'{namespace}_thinkcost_42_{split}_mixed16']
            for signal in ('uncertainty','history_length'):
                for fraction in protocol['thinkcost']['routing_fractions']:
                    tag=f'{namespace}_route_42_{split}_{signal}_{fraction}'
                    tags.extend([tag,tag+'_random'])
                    required.append(CODE/'reports'/f'{tag}_vs_random.json')
            required.extend(CODE/'reports/results_v2'/f'{tag}.json' for tag in tags)
            required.extend(ROOT/'predictions_v2/main'/f'{tag}.jsonl' for tag in tags)
            missing.extend(str(p) for p in required if not p.exists())
            for tag in tags:
                result=CODE/'reports/results_v2'/f'{tag}.json'
                if result.exists():
                    settings=json.loads(result.read_text())['settings']
                    if settings['variant']!=protocol['variant'] or settings['adapter_sha256']!=protocol['adapter_sha256']:
                        invalid.append(dict(path=str(result),reason='wrong RL adapter'))
            if summary.exists() and selection.exists() and json.loads(summary.read_text())['selection']!=json.loads(selection.read_text()):
                invalid.append(dict(path=str(summary),reason='routing selection differs from frozen validation selection'))
    official_path=CODE/'reports/official_status.json'
    official=json.loads(official_path.read_text()) if official_path.exists() else dict(status='pending',reason='Official product data/code compatibility check is in progress.')
    value=dict(rl_thinkcost_complete=not missing and not invalid,missing=missing,invalid=invalid,official=official,
               requirements_resolved=not missing and not invalid and official['status'] in ('complete','blocked'),
               all_experiments_complete=not missing and not invalid and official['status']=='complete')
    write_json(CODE/'reports/supplement_audit.json',value)
    return value


def run():
    import datetime,fcntl,os,subprocess,sys
    lock=(CODE/'reports/pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if (proc/'cwd').resolve()==CODE and any(arg==b'run.py' or arg.endswith(b'/run.py') for arg in (proc/'cmdline').read_bytes().split(b'\0')):
                raise RuntimeError(f'Project worker {proc.name} is still active; inspect it before resuming')
        except (FileNotFoundError,PermissionError):pass
    if not json.loads((CODE/'reports/final_audit.json').read_text())['complete']:
        raise RuntimeError('Finish the main seed-42 audit before supplemental work')
    protocol=frozen_protocol();variant=protocol['variant']
    state_path=CODE/'reports/pipeline_status.json';state=json.loads(state_path.read_text())
    state.pop('child_pid',None);state.pop('failure',None)
    state.update(scope='seed_42_supplement',active_seeds=[42],deferred_seeds=[43,44])
    (CODE/'reports/pipeline.pid').write_text(str(os.getpid()))
    def status(stage,current):
        state.update(stage=stage,current=current,pid=os.getpid(),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(state_path,state)
        (CODE/'reports/active_stage.txt').write_text(stage+': '+current+'\nPID '+str(os.getpid()))
    # Re-entering a finished split uses its existing per-sample cache and audits it.
    for split in ('validation','test'):
        key=f'supplement_thinkcost_{variant}_42_{split}'
        status('RL_cost','thinkcost --variant '+variant+' --seed 42 --split '+split)
        log=CODE/'reports/jobs'/f'{key}.log'
        with log.open('a') as output:
            process=subprocess.Popen([sys.executable,str(CODE/'run.py'),'thinkcost','--variant',variant,'--seed','42','--split',split],cwd=CODE,stdout=output,stderr=subprocess.STDOUT)
            state['child_pid']=process.pid;write_json(state_path,state);code=process.wait()
        state.pop('child_pid',None)
        if code:
            state['failure']=dict(job=key,exit_code=code,log=str(log));status('failed','Inspect '+str(log));raise RuntimeError(state['failure'])
        done=state.setdefault('supplement_done',[])
        if key not in done:done.append(key)
        status('RL_cost','completed '+key)
    audit=audit_supplement()
    if not audit['rl_thinkcost_complete']:raise RuntimeError(audit)
    subprocess.run([sys.executable,str(CODE/'run.py'),'report'],cwd=CODE,check=True)
    status('complete','RL thinking-cost supplement finished; official product evaluation status: '+audit['official']['status']+'; seeds 43/44 remain deferred.')
