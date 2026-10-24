"""Run the pinned official product task after the local seed-42 supplement."""
import json
import os
from pathlib import Path

from data import CODE, ROOT, write_json
from ops.supplement import sha


TASK='challenge_recommendation_product'


def run_checks_baseline():
    """Run only the mixed-SFT baseline in the separate checks namespace."""
    import datetime,subprocess,sys
    from ops.prepare_official import merge_adapter
    checks=CODE/'reports/checks';storage=ROOT/'predictions_checks'
    frozen=json.loads((checks/'protocol.json').read_text())
    runtime_path=checks/'official_runtime_identity.json'
    runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else frozen['runtime']
    preparation=json.loads((CODE/'artifacts/official_preparation.json').read_text())
    original=json.loads((CODE/'artifacts/official_execution_protocol_42.json').read_text())
    repo=Path(preparation['repo'])
    for name,expected in preparation['runtime_files'].items():
        if sha(repo/name)!=expected:raise RuntimeError('Official runtime changed: '+name)
    adapter=ROOT/'checkpoints/main/mixed_42/selected'
    if sha(adapter/'adapter_model.safetensors')!=frozen['adapters']['mixed']:raise RuntimeError('Mixed adapter changed')
    merged=storage/'models/mixed_42_merged_bf16'
    merge=merge_adapter(adapter,merged)
    output=storage/'official/mixed_42/datav3.1_seed42_race';cache=storage/'official/mixed_42/batch_cache'
    command=list(original['command']);command[0]=sys.executable
    for flag,value in [('--model_path',str(merged)),('--output_dir',str(output))]:
        command[command.index(flag)+1]=value
    protocol=dict(command=command,parent_protocol_sha256=sha(CODE/'artifacts/official_execution_protocol_42.json'),
                  checks_protocol_sha256=sha(checks/'protocol.json'),merge=merge,gpu_index=runtime['gpu_index'],
                  batch_cache=str(cache),generation=original['generation'],evaluation=original['evaluation'],race=original['race'])
    path=checks/'official_mixed_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        previous=json.loads(path.read_text())
        if {k:v for k,v in previous.items() if k!='gpu_index'}!={k:v for k,v in protocol.items() if k!='gpu_index'}:
            raise RuntimeError('Frozen mixed official generation/scoring protocol changed')
        protocol['supersedes']=dict(path=str(path),sha256=sha(path),reason='Original GPU reservation became unavailable before any official sample was generated')
        path=checks/'official_mixed_protocol_runtime_recovery.json'
        if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Official recovery protocol changed')
    if not path.exists():write_json(path,protocol)
    generated=output/merged.name/TASK/'test_generated.json';metrics_path=output/'eval_results.json'
    if not (generated.exists() and metrics_path.exists()):
        from ops.checks import gpu_ready
        gpu_ready(runtime)
        ray_root=storage/'ray_official';ray_root.mkdir(parents=True,exist_ok=True)
        short_ray=Path('/tmp/mor_checks_dzh')
        if short_ray.is_symlink():
            if short_ray.resolve()!=ray_root.resolve():raise RuntimeError('Short Ray path belongs to another directory')
        elif short_ray.exists():raise RuntimeError('Short Ray path is already in use')
        else:short_ray.symlink_to(ray_root.resolve(),target_is_directory=True)
        if len(str(short_ray)+'/session_2026-10-24_12-06-33_608852_9999999/sockets/plasma_store')>107:
            raise RuntimeError('Ray socket path still exceeds AF_UNIX limit')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(runtime['gpu_index']),PYTHONPATH=str(repo),
            MINIONEREASON_OFFICIAL_CACHE=str(cache),RAY_TMPDIR=str(short_ray),RAY_ADDRESS='local',
            ONEREC_VLLM_SID_INTERNAL_BEAM='0',TOKENIZERS_PARALLELISM='false')
        Path(env['RAY_TMPDIR']).mkdir(parents=True,exist_ok=True)
        subprocess.run(command,cwd=repo,env=env,check=True)
    samples=json.loads(generated.read_text())['samples']
    if set(samples)!=set(map(str,range(1000))):raise RuntimeError('Official sample IDs differ')
    if any(len(r['generations'])!=64 for r in samples.values()):raise RuntimeError('Incomplete Race candidates')
    scores=json.loads(metrics_path.read_text())[merged.name][TASK]['test']
    write_json(checks/'official_mixed.json',dict(status='complete',scores=scores,examples=1000,seed=42,
        raw_generations=str(generated),raw_sha256=sha(generated),metrics_file=str(metrics_path),
        protocol=str(path),runtime=runtime,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()))


def run(wait_pid=None):
    import datetime,fcntl,subprocess,sys,time
    if wait_pid:
        while True:
            proc=Path('/proc')/str(wait_pid)
            try:
                alive=(proc/'cwd').resolve()==CODE and b'supplement' in (proc/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError):alive=False
            if not alive:break
            time.sleep(10)
    lock=(CODE/'reports/pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=CODE/'reports/pipeline_status.json';state=json.loads(state_path.read_text())
    from ops.supplement import audit_supplement
    if not audit_supplement()['rl_thinkcost_complete']:
        raise RuntimeError('RL cost supplement is unfinished; preserve its status and resume it first')
    state.pop('child_pid',None);state.pop('failure',None)
    state.update(scope='seed_42_official',active_seeds=[42],deferred_seeds=[43,44])
    def status(stage,current):
        state.update(stage=stage,current=current,pid=os.getpid(),updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(state_path,state)
        (CODE/'reports/pipeline.pid').write_text(str(os.getpid()))
        (CODE/'reports/active_stage.txt').write_text(stage+': '+current+'\nPID '+str(os.getpid()))
    status('official','Verify pinned official product protocol and GPU 1 availability')
    try:
        preparation=json.loads((CODE/'artifacts/official_preparation.json').read_text())
        compatibility=json.loads((CODE/'artifacts/official_compatibility.json').read_text())
        if not compatibility['compatible']:raise RuntimeError('Official data/code compatibility check failed')
        repo=Path(preparation['repo']);merged=Path(preparation['merged_model'])
        for name,expected in preparation['runtime_files'].items():
            if sha(repo/name)!=expected:raise RuntimeError('Official runtime changed: '+name)
        for name,expected in preparation['merge']['weights'].items():
            if sha(merged/name)!=expected:raise RuntimeError('Merged RL weights changed: '+name)
        output=ROOT/'official_results/rl_nk_prefix_42/datav3.1_seed42_race'
        cache=ROOT/'official_results/rl_nk_prefix_42/batch_cache'
        command=[sys.executable,str(repo/'scripts/ray-vllm/evaluate.py'),
                 '--model_path',str(merged),'--version','v3.1','--seed','42',
                 '--race_mode','true','--enable_thinking','false','--task_types',TASK,
                 '--data_dir',str(ROOT/'data/raw/official'),'--output_dir',str(output),
                 '--sample_size','full','--ray_address','local','--num_gpus','1',
                 '--tensor_parallel_size','1','--gpu_memory_utilization','0.65',
                 '--max_model_len','8192','--worker_batch_size','4']
        protocol=dict(command=command,seed=42,examples=1000,task=TASK,version='v3.1',data_version=compatibility['data_version'],
                      dataset_revision=compatibility['dataset_revision'],upstream_commit=preparation['upstream_commit'],
                      preparation_sha256=sha(CODE/'artifacts/official_preparation.json'),
                      model_selection='Best fully trained RL model on local validation Recall@32; no external-score selection',
                      adapter_sha256=preparation['adapter_sha256'],gpu_index=1,
                      race=dict(thinking=32,non_thinking=32,thinking_first=True),
                      generation=compatibility['generation'],evaluation=compatibility['evaluation'],
                      batch_cache=str(cache),max_model_len=8192,truncate_inputs=False,
                      note='Official PID and SID scores are independent of the local held-out SID protocol. Resume retains completed batches; timing is diagnostic only.')
        protocol_path=CODE/'artifacts/official_execution_protocol_42.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Frozen official protocol changed')
        write_json(protocol_path,protocol)
        generated=output/merged.name/TASK/'test_generated.json';metrics_path=output/'eval_results.json'
        if not (generated.exists() and metrics_path.exists()):
            gpu=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(', ')
            if gpu[0]!='GPU-316108a9-373b-1984-405a-b9d9f7ee55b3':raise RuntimeError('GPU 1 identity changed')
            if int(gpu[1])<int(gpu[2])*.65+512:raise RuntimeError('GPU 1 free memory is below the official runtime reservation; preserve other jobs')
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='1',PYTHONPATH=str(repo),
                MINIONEREASON_OFFICIAL_CACHE=str(cache),RAY_TMPDIR=str(ROOT/'ray_official'),RAY_ADDRESS='local',
                ONEREC_VLLM_SID_INTERNAL_BEAM='0',TOKENIZERS_PARALLELISM='false')
            Path(env['RAY_TMPDIR']).mkdir(parents=True,exist_ok=True)
            log=CODE/'reports/jobs/official_product_rl_nk_prefix_42.log'
            status('official','Official product v3.1, 1000 rows, Race 32+32, seed 42, GPU 1')
            write_json(CODE/'reports/official_status.json',dict(status='running',task=TASK,examples=1000,log=str(log),protocol=str(protocol_path)))
            with log.open('a') as stream:
                child=subprocess.Popen(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT)
                state['official_child_pid']=child.pid;write_json(state_path,state);code=child.wait()
            state.pop('official_child_pid',None)
            if code:raise RuntimeError(f'Official runner exited {code}; inspect {log}')
        raw=json.loads(generated.read_text());samples=raw['samples']
        if set(samples)!=set(map(str,range(1000))):raise RuntimeError('Official output is missing or has extra sample IDs')
        if any(len(row['generations'])!=64 for row in samples.values()):raise RuntimeError('Official candidate count differs from Race 32+32')
        scores=json.loads(metrics_path.read_text())[merged.name][TASK]['test']
        if not any(key.startswith('pid_') and 'recall' in key for key in scores):raise RuntimeError('Official PID metrics are absent')
        result=dict(status='complete',task=TASK,examples=1000,seed=42,scores=scores,
                    reason='Official v3.1 product task completed with full data, Race 32+32, and official PID/SID scoring.',
                    protocol=str(protocol_path),raw_generations=str(generated),raw_sha256=sha(generated),metrics_file=str(metrics_path),
                    label_mapping_discrepancies=compatibility['label_mapping_discrepancies'],
                    finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(CODE/'reports/official_status.json',result)
        subprocess.run([sys.executable,str(CODE/'run.py'),'report'],cwd=CODE,check=True)
        if not json.loads((CODE/'reports/final_audit.json').read_text())['complete']:raise RuntimeError('Final raw-output audit failed')
        if not audit_supplement()['all_experiments_complete']:raise RuntimeError('Final supplemental audit is incomplete')
        status('complete','Seed-42 main chain, RL cost supplement, and official product evaluation finished; seeds 43/44 remain deferred.')
    except Exception as error:
        state['failure']=dict(job='official_product_rl_nk_prefix_42',error=str(error),log=str(CODE/'reports/jobs/official_product_rl_nk_prefix_42.log'))
        write_json(CODE/'reports/official_status.json',dict(status='failed',reason=str(error),failure=state['failure']))
        status('failed','Official external evaluation failed: '+str(error))
        raise
