"""Prepare the pinned upstream product benchmark and a separate merged RL model."""
import difflib
import json
import shutil
import zipfile

from data import CODE, ROOT, write_json
from ops.supplement import sha


def merge_adapter(adapter, merged):
    """Use the same CPU BF16 merge for every official-task model."""
    from pathlib import Path
    adapter=Path(adapter);merged=Path(merged)
    adapter_sha=sha(adapter/'adapter_model.safetensors')
    marker=merged/'merge_complete.json'
    if marker.exists():
        value=json.loads(marker.read_text())
        if value['adapter_sha256']!=adapter_sha:raise RuntimeError('Merged adapter differs')
        for name,expected in value['weights'].items():
            if sha(merged/name)!=expected:raise RuntimeError('Merged weights changed: '+name)
        return value
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    base=ROOT/'models/onereason-0.8b'
    model=AutoModelForCausalLM.from_pretrained(str(base),torch_dtype=torch.bfloat16,device_map='cpu',low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(model,str(adapter),autocast_adapter_dtype=False).merge_and_unload(safe_merge=True)
    merged.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(merged,safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base)).save_pretrained(merged)
    value=dict(adapter_sha256=adapter_sha,base=str(base),dtype='bfloat16',device='cpu',
               weights={p.name:sha(p) for p in merged.glob('*.safetensors')},
               note='Official runner receives merged weights; BF16 merge may differ numerically from vLLM LoRA inference.')
    write_json(marker,value)
    return value


def prepare():
    compatibility=json.loads((CODE/'artifacts/official_compatibility.json').read_text())
    if not compatibility['compatible']:raise RuntimeError(compatibility)
    protocol=json.loads((CODE/'artifacts/rl_thinkcost_protocol_42.json').read_text())
    adapter=__import__('pathlib').Path(protocol['adapter']).parent
    if sha(adapter/'adapter_model.safetensors')!=protocol['adapter_sha256']:raise RuntimeError('RL adapter changed')
    repo=CODE/'external/OneReason_Eval_Benchmark-main'
    archive=CODE/'external/official-main.zip';diff=[]
    with zipfile.ZipFile(archive) as source:
        if source.comment.decode()!=compatibility['code_commit']:raise RuntimeError('Upstream archive commit differs')
        modifications={
            'scripts/ray-vllm/utils/generator.py':[
                ('        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))',
                 '        # Preserve the physical device assigned by Ray (logical 0 is physical GPU 1 here).\n        os.environ.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, gpu_ids)))'),
                ('    def generate_batch(\n','    from benchmark.minionereason_runtime import resumable_batch\n\n    @resumable_batch\n    def generate_batch(\n')],
            'benchmark/base_generator.py':[
                ('        runtime_env = {"env_vars": runtime_env_vars}',
                 '        if os.environ.get("MINIONEREASON_OFFICIAL_CACHE"):\n            runtime_env_vars["MINIONEREASON_OFFICIAL_CACHE"] = os.environ["MINIONEREASON_OFFICIAL_CACHE"]\n        runtime_env = {"env_vars": runtime_env_vars}'),
                ('\n            ray.init(ignore_reinit_error=True, runtime_env=runtime_env)\n',
                 '\n            ray.init(address="local", ignore_reinit_error=True, runtime_env=runtime_env, num_cpus=4, object_store_memory=256*1024**2, include_dashboard=False, _temp_dir=os.environ.get("RAY_TMPDIR"))\n')]
        }
        for name,replacements in modifications.items():
            original=source.read('OneReason_Eval_Benchmark-main/'+name).decode();patched=original
            for before,after in replacements:
                if patched.count(before)!=1:raise RuntimeError('Unexpected upstream patch context: '+name)
                patched=patched.replace(before,after)
            (repo/name).write_text(patched,encoding='utf-8')
            diff.extend(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='upstream/'+name,tofile='local/'+name))
    shutil.copyfile(CODE/'ops/official_cache.py',repo/'benchmark/minionereason_runtime.py')
    (CODE/'artifacts/official_runtime_patch.diff').write_text(''.join(diff),encoding='utf-8')
    merged=ROOT/'checkpoints/official'/f'{protocol["variant"]}_42_merged_bf16'
    merge=merge_adapter(adapter,merged)
    value=dict(upstream_commit=compatibility['code_commit'],upstream_archive_sha256=sha(archive),repo=str(repo),merged_model=str(merged),
               adapter_sha256=protocol['adapter_sha256'],merge=merge,
               local_patch_sha256=sha(CODE/'artifacts/official_runtime_patch.diff'),
               runtime_files={name:sha(repo/name) for name in [*modifications,'benchmark/minionereason_runtime.py']},
               changes=['Preserve Ray-assigned physical GPU','Use a private local Ray session','Checkpoint completed worker batches; raise on incomplete engine output'],
               unchanged=['Official prompts','Full 1000-row dataset and labels','Race 32 think + 32 non-think beam candidates','Sampling parameters','PID/SID metric code'],
               timing_note='Checkpointed batch timings are operational diagnostics, not a controlled throughput benchmark.')
    write_json(CODE/'artifacts/official_preparation.json',value)
    print(json.dumps(value),flush=True)


if __name__=='__main__':prepare()
