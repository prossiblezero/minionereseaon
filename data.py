"""Official snapshot adaptation. Labels never enter the input-side contract."""
import collections
import hashlib
import json
import os
import re
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('MINIONEREASON_DATA_ROOT', '/data1/dzh/minionereason'))
SID = re.compile(r'<\|(prod|video|ad|living|sid)_begin\|><s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>')
TASK = '请根据以上历史，推荐一个用户接下来可能点击的商品。最终答案只输出一个完整的商品SID。'
SYSTEM = '你负责根据用户多域行为理解用户兴趣偏好，推荐商品。'


def config():
    return json.loads((CODE / 'config.json').read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(x.get('type') == 'text' for x in value):
        return ''.join(x['text'] for x in value)
    raise ValueError('unsupported_content')


def parse_answer(text, catalog=None):
    """Only the final answer, never a SID mentioned inside thinking."""
    if '<think>' in text or '</think>' in text:
        if text.count('</think>') != 1 or text.count('<think>') > 1:
            return None, 'unclosed_or_multiple_think'
        text = text.split('</think>', 1)[1]
    found = list(SID.finditer(text))
    if len(found) != 1:
        return None, 'missing_or_multiple_sid'
    match = found[0]
    if match[1] != 'prod':
        return None, 'wrong_domain'
    if any(int(x) >= 8192 for x in match.groups()[1:]):
        return None, 'invalid_code'
    remaining = text[:match.start()] + text[match.end():]
    if re.search(r'<\|\w+_begin\|>|<s_[abc]_', remaining):
        return None, 'extra_partial_sid'
    item = match[0]
    if catalog is not None and item not in catalog:
        return None, 'out_of_catalog'
    return item, 'valid'


def codes(sid):
    match = SID.fullmatch(sid)
    if not match:
        raise ValueError('malformed SID')
    return tuple(map(int, match.groups()[1:]))


def split_for(group_id, seed=42):
    bucket = int(digest(group_id + str(seed)), 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'


def input_view(row):
    return {k: row[k] for k in ('sample_id', 'group_id', 'history', 'context_text')}


def normalize(row, source):
    messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('unsupported_turn_structure')
    user = content_text(messages[1]['content'])
    answer = content_text(messages[2]['content'])
    if answer.count('<think>') != 1 or answer.count('</think>') != 1:
        raise ValueError('missing_reasoning_boundary')
    reasoning, final = answer.split('</think>')
    reasoning = reasoning.split('<think>')[1].strip()
    target, reason = parse_answer(final)
    if not target:
        raise ValueError(reason)
    if not ('用户' in user and ('历史' in user or '行为' in user)):
        raise ValueError('not_recommendation_instruction')
    if not ('商品' in final and ('点击' in final or '购买' in final or '推荐' in final)):
        raise ValueError('unclear_product_target')
    user = re.sub(r'\n\s*请[^\n]*$', '', user).strip()
    user = re.sub(r'^(?:以下是一个用户的多域历史行为信息|下面给出用户历史行为线索)[：:]\s*', '', user)
    history, previous, behavior = [], 0, ''
    for i, m in enumerate(SID.finditer(user)):
        label = re.findall(r'[\u4e00-\u9fff][\u4e00-\u9fff /：:()（）]*', user[previous:m.start()])
        if label:
            behavior = label[-1][-48:]
        history.append(dict(domain=m[1], sid=m[0], behavior=behavior, timestamp=None,
                            original_order=i, span=[m.start(), m.end()]))
        previous = m.end()
    if not history:
        raise ValueError('no_history')
    group_id = digest(re.sub(r'\s+', '', user))
    source = dict(source, record_id=row.get('record_id'))
    return dict(sample_id=digest(json.dumps(source, sort_keys=True)), group_id=group_id,
                source=source, history=history, context_text=user, task='product_recommendation',
                targets=[target], train_target=target, reasoning=reasoning,
                metadata={'reasoning_sids': sorted(set(m[0] for m in SID.finditer(reasoning))),
                          'timestamp_order': 'unknown; source presentation order only'})


def load_split(profile, split):
    return read_jsonl(ROOT / 'data/processed' / profile / (split + '.jsonl'))


def load_catalog():
    return set((ROOT / 'data/processed/catalog.txt').read_text(encoding='utf-8').splitlines())


def build_catalog():
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    files = sorted((ROOT / 'data/raw/catalog/data/OneReason_Pid2Sid').glob('*.parquet'))
    expected = json.loads((CODE / 'artifacts/catalog_revision.json').read_text())
    wanted = [s for s in expected['siblings'] if s['rfilename'].startswith('data/OneReason_Pid2Sid/') and s['rfilename'].endswith('.parquet')]
    if len(files) != len(wanted):
        raise RuntimeError(f'Incomplete catalog snapshot: {len(files)}/{len(wanted)}')
    counts, domains, invalid = collections.Counter(), collections.Counter(), 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=65536):
            domains.update(batch.column('domain').to_pylist())
            goods = batch.filter(pc.equal(batch.column('domain'), 'goods'))
            for values in goods.column('sid_three').to_pylist():
                if len(values) != 3 or any(v is None or v != int(v) or not 0 <= v < 8192 for v in values):
                    invalid += 1
                    continue
                counts['<|prod_begin|><s_a_%d><s_b_%d><s_c_%d>' % tuple(values)] += 1
    dest = ROOT / 'data/processed/catalog.txt'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('\n'.join(sorted(counts)) + '\n', encoding='utf-8')
    write_json(CODE / 'reports/catalog_report.json', dict(domains=domains, product_sids=len(counts),
               colliding_sids=sum(v > 1 for v in counts.values()), excess_pid_rows=sum(v-1 for v in counts.values()),
               invalid_rows=invalid, files=len(files), provenance='full independent Pid2Sid; goods domain'))
    return set(counts)


def prepare(profile):
    import numpy as np
    import pyarrow.parquet as pq
    from model import tokenizer
    from history import select_history, prompt_text
    cfg, tok = config(), tokenizer()
    catalog = load_catalog() if (ROOT / 'data/processed/catalog.txt').exists() else build_catalog()
    meta = json.loads((CODE / 'artifacts/sft_revision.json').read_text())
    files = sorted((ROOT / 'data/raw/sft').rglob('*.parquet'))
    expected = [s for s in meta['siblings'] if s['rfilename'].startswith('OneRec_Foundation_UserProfile_CoT_Sft_Demodata_Video/') and s['rfilename'].endswith('.parquet')]
    if len(files) != len(expected):
        raise RuntimeError('Incomplete recommendation SFT snapshot')
    rejection, seen, total = collections.Counter(), set(), 0
    splits = {s: [] for s in ('train', 'validation', 'test')}
    lengths = {s: [] for s in splits}
    quarantine = []
    for f in files:
        offset = 0
        for batch in pq.ParquetFile(f).iter_batches(batch_size=128, columns=['record_id', 'messages']):
            for raw in batch.to_pylist():
                source = dict(dataset=cfg['sft_dataset'], revision=meta['sha'], file=str(f.relative_to(ROOT / 'data/raw/sft')), row=offset)
                offset += 1
                total += 1
                try:
                    row = normalize(raw, source)
                    if row['train_target'] not in catalog:
                        raise ValueError('target_out_of_catalog')
                    split = split_for(row['group_id'], cfg['split_seed'])
                    key = (row['group_id'], row['train_target'])
                    if split == 'train' and key in seen:
                        raise ValueError('duplicate_history_target')
                    seen.add(key)
                    view = input_view(row)
                    full_len = len(tok.encode(prompt_text(view, tok, list(range(len(row['history']))), False), add_special_tokens=False))
                    lengths[split].append((len(row['history']), full_len))
                    selected = select_history(view, tok, cfg['input_budget'], 'recent')
                    kept = {row['history'][i]['sid'] for i in selected}
                    refs = set(row['metadata']['reasoning_sids'])
                    all_history = {e['sid'] for e in row['history']}
                    if refs - all_history - {row['train_target']}:
                        raise ValueError('reasoning_reference_absent_in_original_history')
                    if (refs & all_history) - kept:
                        raise ValueError('reasoning_reference_removed_by_crop')
                    completion = row['reasoning'] + '\n</think>\n\n' + row['train_target'] + tok.eos_token
                    if len(tok.encode(completion, add_special_tokens=False)) > cfg['answer_budget']:
                        raise ValueError('reasoning_exceeds_output_budget')
                    row['metadata'].update(full_prompt_tokens=full_len, full_events=len(row['history']), retained_events=len(selected))
                    splits[split].append(row)
                except ValueError as exc:
                    rejection[str(exc)] += 1
                    if str(exc) != 'wrong_domain':
                        quarantine.append(dict(source=source, reason=str(exc)))
    out = ROOT / 'data/processed' / profile
    actual = {}
    for split, rows in splits.items():
        rows.sort(key=lambda x: digest(x['sample_id'] + 'selection42'))
        rows = rows[:cfg['profiles'][profile][split]]
        write_jsonl(out / (split + '.jsonl'), rows)
        actual[split] = len(rows)
    def stats(values):
        a = np.asarray(values)
        if not len(a):
            return {}
        return dict(mean=float(a.mean()), median=float(np.median(a)), p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)))
    length_report = {}
    for split in splits:
        rows = load_split(profile, split)
        tokens = [r['metadata']['full_prompt_tokens'] for r in rows]
        length_report[split] = dict(events=stats([len(r['history']) for r in rows]), prompt_tokens=stats(tokens),
                                   trimmed_fraction={str(b): sum(v > b for v in tokens) / max(1, len(tokens)) for b in cfg['history']['budgets']})
    write_json(CODE / 'reports' / f'data_report_{profile}.json', dict(raw_rows=total, accepted_before_cap={s:len(v) for s,v in splits.items()}, actual=actual,
               rejected=rejection, length_gate=length_report, scope='paired valid-CoT product subset; train-only deduplication; validation/test repeats retained within group; single target SID; grouped holdout, not temporal/user split',
               raw_product_candidate_lengths={s:dict(events=stats([x[0] for x in v]),prompt_tokens=stats([x[1] for x in v])) for s,v in lengths.items()}))
    write_jsonl(out / 'quarantine.jsonl', quarantine)
    if min(actual.values()) < 8:
        raise RuntimeError(f'Insufficient real product data: {actual}; see data report')
    validate(profile)
    print(json.dumps({'profile':profile,'counts':actual,'rejected':rejection},ensure_ascii=False), flush=True)


def validate(profile):
    from model import tokenizer
    from history import prompt_text, select_history
    tok, catalog, cfg = tokenizer(), load_catalog(), config()
    groups = {}
    for split in ('train','validation','test'):
        rows = load_split(profile,split)
        groups[split] = {r['group_id'] for r in rows}
        for row in rows:
            assert row['train_target'] in catalog
            assert split_for(row['group_id'],cfg['split_seed']) == split
            view = input_view(row)
            chosen = select_history(view,tok,cfg['input_budget'],'recent')
            prompt = prompt_text(view,tok,chosen,False)
            changed = dict(row,targets=['POISON'],train_target='POISON',reasoning='POISON')
            assert prompt == prompt_text(input_view(changed),tok,chosen,False)
            assert len(tok.encode(prompt,add_special_tokens=False)) <= cfg['input_budget']
    assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])
    write_json(CODE/'reports'/f'validation_{profile}.json',dict(status='passed',group_counts={k:len(v) for k,v in groups.items()}))


def manifest():
    import platform
    import subprocess
    records=[]
    for base in (ROOT/'models/onereason-0.8b',ROOT/'data/raw',ROOT/'data/processed'):
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '.cache' in p.parts:
                continue
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):
                    h.update(block)
            records.append(dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest()))
    rev={k:json.loads((CODE/f'artifacts/{k}_revision.json').read_text())['sha'] for k in ('model','sft','catalog')}
    write_json(CODE/'artifacts/manifest.json',dict(host=platform.node(),data_root=str(ROOT),code_root=str(CODE),revisions=rev,files=records,
               gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
               hardware_note='4090 selected by explicit user instruction; shared GPU, latency is not exclusive-GPU latency'))


def audit_catalog_collisions():
    """Count distinct PID/SID pairs, rather than mistaking duplicate rows for collisions."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    pa.set_cpu_count(4);tables=[]
    for path in sorted((ROOT/'data/raw/catalog/data/OneReason_Pid2Sid').glob('*.parquet')):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=65536):
            goods=batch.filter(pc.equal(batch.column('domain'),'goods'))
            if not len(goods):continue
            values=goods.column('sid_three')
            a,b,c=[pc.cast(pc.list_element(values,i),pa.int64()) for i in range(3)]
            packed=pc.add(pc.add(pc.multiply(a,8192**2),pc.multiply(b,8192)),c)
            tables.append(pa.table({'pid':goods.column('pid'),'sid':pc.cast(packed,pa.int64())}))
    table=pa.concat_tables(tables);unique=table.group_by(['pid','sid']).aggregate([])
    counts=unique.group_by('sid').aggregate([('pid','count')]);pid_counts=unique.group_by('pid').aggregate([('sid','count')])
    path=CODE/'reports/catalog_report.json';report=json.loads(path.read_text())
    report.update(duplicate_pid_sid_rows=len(table)-len(unique),product_sids=len(counts),
                  colliding_sids=pc.sum(pc.cast(pc.greater(counts.column('pid_count'),1),pa.int64())).as_py(),
                  excess_pid_rows=len(unique)-len(counts),pids_with_multiple_sids=pc.sum(pc.cast(pc.greater(pid_counts.column('sid_count'),1),pa.int64())).as_py(),
                  collision_definition='distinct goods PID/SID pairs; duplicate raw rows excluded')
    write_json(path,report);print(json.dumps(report),flush=True)


def download():
    import concurrent.futures
    import time
    from huggingface_hub import hf_hub_download
    cfg=config();jobs=[]
    for name,repo,destination in [('model',cfg['model_id'],ROOT/'models/onereason-0.8b'),('sft',cfg['sft_dataset'],ROOT/'data/raw/sft'),('catalog',cfg['catalog_dataset'],ROOT/'data/raw/catalog')]:
        meta=json.loads((CODE/f'artifacts/{name}_revision.json').read_text())
        for item in meta['siblings']:
            filename=item['rfilename']
            keep=(name=='model' and (filename.endswith(('.json','.txt','.safetensors')) or filename=='README.md')) or (name=='sft' and filename.startswith('OneRec_Foundation_UserProfile_CoT_Sft_Demodata_Video/') and filename.endswith('.parquet')) or (name=='catalog' and filename.startswith('data/OneReason_Pid2Sid/') and filename.endswith('.parquet')) or filename=='README.md'
            if keep:jobs.append((name,repo,destination,meta['sha'],filename))
    def fetch(job):
        name,repo,destination,revision,filename=job
        for attempt in range(3):
            try:
                path=hf_hub_download(repo_id=repo,repo_type='model' if name=='model' else 'dataset',filename=filename,revision=revision,
                                     endpoint=os.environ.get('HF_ENDPOINT','https://hf-mirror.com'),local_dir=destination)
                return dict(resource=name,file=filename,bytes=Path(path).stat().st_size)
            except Exception:
                if attempt==2:raise
                time.sleep(2**attempt)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(fetch,jobs):print(json.dumps(result),flush=True)
