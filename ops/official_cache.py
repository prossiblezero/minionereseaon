"""Checkpoint complete official worker batches; generation/scoring stay upstream."""
import functools
import hashlib
import json
import os
from pathlib import Path


def resumable_batch(function):
    @functools.wraps(function)
    def wrapped(self,prompts,sampling_params,worker_batch_size=8):
        directory=os.environ.get('MINIONEREASON_OFFICIAL_CACHE')
        if not directory or not prompts:
            return function(self,prompts,sampling_params,worker_batch_size)
        directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
        combined=({}, {}, {});items=list(prompts.items())
        for start in range(0,len(items),worker_batch_size):
            batch=dict(items[start:start+worker_batch_size])
            signature=hashlib.sha256(json.dumps([batch,sampling_params],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
            path=directory/(signature+'.json')
            if path.exists():
                result=json.loads(path.read_text())
            else:
                result=function(self,batch,sampling_params,worker_batch_size)
            expected=sampling_params.get('beam_width',1) if sampling_params.get('use_beam_search') else sampling_params.get('n',1)
            # Upstream converts a batch exception to empty outputs and omits MFU
            # entries. Stop here so an engine failure never becomes a model score.
            if set(result[0])!=set(batch) or set(result[2])!=set(batch) or any(len(v)!=expected for v in result[0].values()):
                raise RuntimeError('Official worker batch incomplete; inspect its preceding traceback')
            if not path.exists():
                temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');temporary.replace(path)
            for target,part in zip(combined,result):target.update(part)
            print(f'Official batch checkpoint: {start+len(batch)}/{len(items)} samples, {path.name}',flush=True)
        return combined
    return wrapped
