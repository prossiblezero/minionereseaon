"""The three mandatory checks in PLAN_checks.md section 0.1; no training."""
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from data import CODE, ROOT, config, load_catalog, load_split, parse_answer, read_jsonl, write_json
from evaluate import evaluate_one, metrics, paired_bootstrap

OUT = CODE / 'reports/checks'
RAW = ROOT / 'predictions_checks'
VARIANTS = ('mixed', 'rl_nk', 'rl_nk_prefix')
HASHES = dict(mixed='af31b185f5e932e93a169db94ebe2a9be45d41a51fd71947067e25f36e699a78',
              rl_nk='937ecc5c0d0f13c6d1a433cd9e9160089b600c844509c6d3ec78c6b2e9af2334',
              rl_nk_prefix='c46b30a4890911b2cf0d67e4b2bdaae824cdec932358b19cbffb4b749686ddd1')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def summary_path(variant, thinking, offset):
    if offset == 0:
        return CODE / 'reports/results_v2' / f'{variant}_42_test_recent_in2048_think{thinking}_budgetNone_n64_vllm.json'
    assert variant in ('mixed', 'rl_nk_prefix') and not thinking and offset in range(1, 5)
    return OUT / 'results' / f'checks_{variant}_42_test_thinkFalse_offset{offset}.json'


def gpu_snapshot():
    fields = 'index,uuid,name,memory.free,memory.total'
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields, '--format=csv,noheader,nounits'], text=True)
    cards = []
    for line in lines.splitlines():
        index, uuid, name, free, total = line.split(', ')
        cards.append(dict(index=int(index), uuid=uuid, name=name, free_mib=int(free), total_mib=int(total)))
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    return dict(recorded_at=now(), cards=cards, compute_processes=processes.splitlines())


def gpu_ready(runtime):
    card = next(x for x in gpu_snapshot()['cards'] if x['index'] == runtime['gpu_index'])
    if card['uuid'] != runtime['gpu_uuid'] or card['free_mib'] < .65 * card['total_mib'] + 1024:
        raise RuntimeError('Frozen GPU reservation unavailable; preserve other jobs and resume after availability returns')


def project_workers():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            cwd = str((proc / 'cwd').resolve())
            if ('minionereason' in cwd or 'minionerec' in cwd) and any(x in args for x in ['run.py', 'ops.checks', 'VLLM::', 'ray::', 'train.py']):
                found.append(dict(pid=int(proc.name), args=args))
        except (OSError, PermissionError):
            pass
    return found


def freeze():
    path = OUT / 'protocol.json'
    if path.exists():
        p = read(path)
        assert p['plan_sha256'] == sha(CODE / 'PLAN_checks.md')
        for v, digest in p['adapters'].items():
            assert sha(ROOT / 'checkpoints/main' / f'{v}_42/selected/adapter_model.safetensors') == digest
        return p
    workers = project_workers()
    if workers:
        raise RuntimeError(f'Existing project workers: {workers}')
    snapshot = gpu_snapshot()
    preferred = read(CODE / 'artifacts/runtime_identity.json')['gpu_index']
    candidates = sorted(snapshot['cards'], key=lambda x: x['index'] != preferred)
    card = next((x for x in candidates if x['free_mib'] >= .65 * x['total_mib'] + 1024), None)
    if card is None:
        raise RuntimeError('No GPU has space for unchanged 0.65 reservation; no experiment started')
    runtime = dict(runtime_id=f"dsj5090:{card['uuid']}", host='dsj5090', gpu_index=card['index'], gpu_uuid=card['uuid'],
                   memory_fraction=.65, data_root=str(ROOT.resolve()), compatible_root=str(ROOT))
    assert ROOT.resolve() == Path('/data3/dzh/minionereason')
    for v, digest in HASHES.items():
        assert sha(ROOT / 'checkpoints/main' / f'{v}_42/selected/adapter_model.safetensors') == digest, v
    sources = []
    for v in VARIANTS:
        for thinking in (False, True):
            s = summary_path(v, thinking, 0); result = read(s); raw = Path(result['raw_predictions'])
            assert result['settings']['adapter_sha256'] == HASHES[v]
            sources.append(dict(summary=str(s), summary_sha256=sha(s), raw=str(raw), raw_sha256=sha(raw),
                                runtime_id=result['settings'].get('runtime_id'), variant=v, thinking=thinking, offset=0))
    protected = {}
    paths = [*CODE.glob('reports/results_v2/*.json'), *CODE.glob('artifacts/*selection*.json'),
             CODE/'config.json', CODE/'reports/RESUME.md', CODE/'reports/official_status.json',
             CODE/'artifacts/execution_protocol_v2.json', CODE/'artifacts/official_preparation.json', CODE/'artifacts/official_execution_protocol_42.json']
    for file in paths:
        protected[str(file)] = sha(file)
    original_stats = {str(x): [x.stat().st_size, x.stat().st_mtime_ns] for x in (ROOT/'predictions_v2').rglob('*') if x.is_file()}
    plan = (CODE / 'PLAN_checks.md').read_text(encoding='utf-8')
    rules = {section: plan.split('### '+section, 1)[1].split('\n## ', 1)[0].split('\n### ', 1)[0] for section in ('2.4 判断规则', '3.4 判断规则', '4.3 判断规则')}
    p = dict(created_at=now(), plan_sha256=sha(CODE/'PLAN_checks.md'), scope='section 0.1 mandatory M1/M2/M3 only',
             seed=42, offsets=[0,1,2,3,4], new_generation_variants=['mixed','rl_nk_prefix'], new_generation_thinking=False,
             official_new_variants=['mixed'], optional_items_run=[], adapters=HASHES, runtime=runtime,
             local=dict(examples=403, attempts=64, history='recent', input_budget=2048, temperature=.8, top_p=.95, backend='vllm'),
             bootstrap=dict(resamples=1000, seed=42, local_resampling_unit='group_id after per-sample offset averaging',
                 validity_resampling_unit='sample_id', official_resampling_unit='official sample_id',
                 validity_cohort='Within each mode, require >=16 valid attempts in both models at every included offset; same cohort for both models',
                 offset0_hardware='Reuse recorded 4090 offset 0; new offsets run on 5090; no deterministic rerun in mandatory scope'),
             decision_rules=rules, original_sources=sources, protected_sha256=protected, original_predictions_stat=original_stats,
             original_project_report=dict(path=str(CODE/'reports/PROJECT_REPORT.md'), bytes=(CODE/'reports/PROJECT_REPORT.md').stat().st_size,
                 sha256=sha(CODE/'reports/PROJECT_REPORT.md')),
             source_sha256={name:sha(CODE/name) for name in ['evaluate.py','data.py','history.py','model.py','ops/checks.py','ops/official.py','ops/prepare_official.py']})
    write_json(path, p)
    write_json(OUT/'gpu_before.json', snapshot)
    write_json(OUT/'runtime_identity.json', runtime)
    return p


def checked_predictions(variant, thinking, offset):
    source = summary_path(variant, thinking, offset); s = read(source)
    raw = read_jsonl(s['raw_predictions']); rows = load_split('main', 'test')
    assert len(raw) == len(rows) == 403
    assert {r['sample_id'] for r in raw} == {r['sample_id'] for r in rows}
    assert all(len(r['outputs']) == 64 for r in raw)
    assert s['settings']['adapter_sha256'] == HASHES[variant]
    assert s['settings'].get('seed_offset', 0) == offset
    score, per = metrics(rows, raw, load_catalog())
    for key in ('recall@8','recall@32','recall@64','invalid_rate'):
        assert abs(s[key]-score[key]) < 1e-12, (source, key)
    return s, raw, per


def average_offsets(records, keys=('recall@8','recall@32','recall@64')):
    maps = [{r['sample_id']:r for r in run} for run in records]
    assert maps and all(set(m) == set(maps[0]) for m in maps)
    return [dict(sample_id=sid, group_id=maps[0][sid]['group_id'],
                 **{key:statistics.mean(m[sid][key] for m in maps) for key in keys}) for sid in maps[0]]


def stability(positive, interval):
    if positive == 5 and interval[0] > 0:
        return '稳定'
    if positive >= 4 and interval[0] <= 0 <= interval[1]:
        return '方向一致、幅度不确定'
    return '未得到支持'


def bootstrap(left, right, key, clustered=True):
    if not clustered:
        left = [dict(r, group_id=r['sample_id']) for r in left]
        right = [dict(r, group_id=r['sample_id']) for r in right]
    return paired_bootstrap(left, right, key, resamples=1000, seed=42, with_se=True)


def seed_statistics():
    results = {}; runs = {}
    keys = ('recall@8','recall@32','recall@64','invalid_rate')
    for v in ('mixed','rl_nk_prefix'):
        entries=[]; runs[v]=[]
        for offset in range(5):
            s, _, per = checked_predictions(v, False, offset); runs[v].append(per)
            entries.append(dict(offset=offset, source=str(summary_path(v,False,offset)), **{k:s[k] for k in keys}))
        zero=[dict(r, **{'recall@32':0.}) for r in runs[v][0]]
        results[v]=dict(runs=entries, mean={k:statistics.mean(x[k] for x in entries) for k in keys},
            sample_std={k:statistics.stdev(x[k] for x in entries) for k in keys},
            original_single_run_bootstrap_se=bootstrap(zero,runs[v][0],'recall@32')['bootstrap_standard_error'])
    diffs=[b['recall@32']-a['recall@32'] for a,b in zip(results['mixed']['runs'],results['rl_nk_prefix']['runs'])]
    merged=bootstrap(average_offsets(runs['mixed']),average_offsets(runs['rl_nk_prefix']),'recall@32')
    comparison=dict(per_offset_deltas=diffs, positive_offsets=sum(d>0 for d in diffs), merged=merged,
        sample_std=statistics.stdev(diffs), original_single_run=bootstrap(runs['mixed'][0],runs['rl_nk_prefix'][0],'recall@32'))
    comparison['judgement']=stability(comparison['positive_offsets'],merged['ci95'])
    existing=[]
    for label,left,right in [
        ('thinking',summary_path('mixed',True,0),summary_path('rl_nk_prefix',True,0)),
        ('non-thinking',summary_path('mixed',False,0),summary_path('rl_nk_prefix',False,0)),
        ('budget 0',CODE/'reports/results_v2/main_thinkcost_42_test_0.json',CODE/'reports/results_v2/main_rl_nk_prefix_thinkcost_42_test_0.json'),
        ('budget 1024',CODE/'reports/results_v2/main_thinkcost_42_test_1024.json',CODE/'reports/results_v2/main_rl_nk_prefix_thinkcost_42_test_1024.json'),
        ('route all non-thinking',CODE/'reports/results_v2/main_route_42_test_all_non_thinking.json',CODE/'reports/results_v2/main_rl_nk_prefix_route_42_test_all_non_thinking.json')]:
        a=read(left);b=read(right)
        existing.append(dict(setting=label,mixed=a['recall@32'],rl_nk_prefix=b['recall@32'],delta=b['recall@32']-a['recall@32'],sources=[str(left),str(right)]))
    write_json(OUT/'eval_seed_summary.json',dict(created_at=now(),mode='non-thinking',models=results,comparison=comparison,
        existing_five_settings=existing,thinking_multi_sampling_evaluated=False,
        hardware_note='Offset 0 uses original recorded 4090 evaluations; offsets 1-4 use 5090. Sample variability is therefore not isolated from cross-device effects.'))


def decompose(initial=False):
    catalog=load_catalog(); targets={r['sample_id']:set(r['targets']) for r in load_split('main','test')}
    runs={}; features={}
    for v in VARIANTS:
        for thinking in (False,True):
            offsets=range(5) if not initial and not thinking and v!='rl_nk' else [0]
            for offset in offsets:
                summary, raw, per=checked_predictions(v,thinking,offset)
                counts=collections.Counter();valid_hits=0;unique=[];per_feature={}
                for row in raw:
                    sid=row['sample_id'];valid=[]
                    for output in row['outputs']:
                        item, reason=parse_answer(output['text'],catalog);counts[reason]+=1
                        if item:
                            valid.append(item);valid_hits+=item in targets[sid]
                    unique.append(len(set(valid)))
                    per_feature[sid]=dict(sample_id=sid,group_id=row['group_id'],valid_count=len(valid),unique_valid_sids=len(set(valid)),
                        **{f'recall@{k}':len(set(valid[:k])&targets[sid])/len(targets[sid]) for k in (8,16)})
                key=f'{v}_think{thinking}_offset{offset}';features[(v,thinking,offset)]=per_feature
                bad=dict(out_of_catalog=counts['out_of_catalog'],unclosed_thinking=counts['unclosed_or_multiple_think'],
                         format_error=sum(counts[x] for x in ['missing_or_multiple_sid','invalid_code','extra_partial_sid']))
                bad['other']=sum(counts.values())-counts['valid']-sum(bad.values())
                runs[key]=dict(variant=v,thinking=thinking,offset=offset,examples=len(raw),attempts=sum(counts.values()),
                    parser_counts=dict(counts),invalid_categories=bad,valid_attempts=counts['valid'],hit_attempts=valid_hits,
                    valid_rate=counts['valid']/sum(counts.values()),hit_rate_given_valid=valid_hits/max(1,counts['valid']),
                    mean_unique_valid_sids=statistics.mean(unique),per_sample=list(per_feature.values()),
                    original_recall32=summary['recall@32'],source=str(summary_path(v,thinking,offset)),raw=summary['raw_predictions'])
    comparisons=[]
    for thinking in (False,True):
        offsets=[0] if initial or thinking else list(range(5))
        eligible_by_offset={o:{sid for sid in targets if all(features[(v,thinking,o)][sid]['valid_count']>=16 for v in ('mixed','rl_nk_prefix'))} for o in offsets}
        eligible=set.intersection(*eligible_by_offset.values())
        per_offset=[]; averaged={}
        for v in ('mixed','rl_nk_prefix'):
            selected=[[features[(v,thinking,o)][sid] for sid in sorted(eligible)] for o in offsets]
            averaged[v]=average_offsets(selected,keys=('recall@8','recall@16')) if eligible else []
        for o in offsets:
            cohort=eligible_by_offset[o]
            scores={v:{f'recall@{k}':statistics.mean(features[(v,thinking,o)][sid][f'recall@{k}'] for sid in cohort) if cohort else None for k in (8,16)} for v in ('mixed','rl_nk_prefix')}
            per_offset.append(dict(offset=o,eligible_examples=len(cohort),models=scores))
        equal={f'recall@{k}':bootstrap(averaged['mixed'],averaged['rl_nk_prefix'],f'recall@{k}',clustered=False) for k in (8,16)} if eligible else {}
        original=[]
        for v in ('mixed','rl_nk_prefix'):
            records=[read_jsonl(Path(read(summary_path(v,thinking,o))['raw_predictions']).with_suffix('.metrics.jsonl')) for o in offsets]
            original.append(average_offsets(records))
        unfiltered=bootstrap(*original,'recall@32',clustered=False)
        label='未能区分有效率与推荐准确率的贡献'
        if equal and equal['recall@16']['ci95'][0]>0:label='收益不只来自有效率，推荐本身更准'
        elif equal and equal['recall@16']['ci95'][0]<=0<=equal['recall@16']['ci95'][1] and unfiltered['ci95'][0]>0:
            label='收益主要来自减少目录外 SID'
        rates={}
        for v in ('mixed','rl_nk_prefix'):
            subset=[runs[f'{v}_think{thinking}_offset{o}'] for o in offsets]
            rates[v]=dict(valid_rate=sum(x['valid_attempts'] for x in subset)/sum(x['attempts'] for x in subset),
                          hit_rate_given_valid=sum(x['hit_attempts'] for x in subset)/max(1,sum(x['valid_attempts'] for x in subset)))
        comparisons.append(dict(thinking=thinking,offsets=offsets,per_offset=per_offset,intersection_examples=len(eligible),
            excluded_examples=len(targets)-len(eligible),equal_valid_budget=equal,original_recall32=unfiltered,rates=rates,
            judgement=label,eligible_sample_ids=sorted(eligible)))
    value=dict(created_at=now(),phase='offset0_only' if initial else 'all_mandatory_offsets',runs=runs,comparisons=comparisons,
               note='Equal-valid-budget analysis conditions on both models having >=16 valid outputs. The prescribed attribution label is a diagnostic, not a causal mediation estimate.')
    write_json(OUT/('validity_decomposition_offset0.json' if initial else 'validity_decomposition.json'),value)


def official_statistics():
    results={};rows={}
    keys=['pass@32','recall@32','pass@64','recall@64','pid_pass@64','pid_recall@64','sid_validity@64']
    for v,path in [('mixed',OUT/'official_mixed.json'),('rl_nk_prefix',CODE/'reports/official_status.json')]:
        status=read(path);raw=Path(status['raw_generations'])
        assert sha(raw)==status['raw_sha256']
        samples=read(raw)['samples'];assert set(samples)==set(map(str,range(1000)))
        assert all(len(r['generations'])==64 for r in samples.values())
        assert all(all('</think>' in x for x in r['generations'][:32]) and all('</think>' not in x for x in r['generations'][32:]) for r in samples.values())
        for k in keys:assert abs(statistics.mean(r[k] for r in samples.values())-status['scores'][k])<1e-12,k
        rows[v]=[dict(sample_id=sid,group_id=sid,**{k:r[k] for k in keys}) for sid,r in samples.items()]
        results[v]=dict(scores={k:status['scores'][k] for k in keys},source=str(path),raw=str(raw),raw_sha256=sha(raw))
    left=read(results['mixed']['raw'])['samples'];right=read(results['rl_nk_prefix']['raw'])['samples']
    assert all(left[sid]['ground_truth']==right[sid]['ground_truth'] and left[sid]['metadata']['answer_pid']==right[sid]['metadata']['answer_pid'] for sid in left)
    comparisons={k:bootstrap(rows['mixed'],rows['rl_nk_prefix'],k,clustered=False) for k in keys}
    for k in ('pid_recall@64','recall@64'):
        c=comparisons[k]
        c['judgement']='官方独立数据上复现了提升' if c['ci95'][0]>0 else ('方向一致，未达显著' if c['delta']>0 else '未在官方数据上复现')
    write_json(OUT/'official_baselines.json',dict(created_at=now(),examples=1000,models=results,comparisons=comparisons,
        reference='mixed SFT only; no pretrained baseline in mandatory scope',bootstrap_resamples=1000,per_sample_means_verified=True))


def verify_protected():
    p=read(OUT/'protocol.json')
    for file,digest in p['protected_sha256'].items():assert sha(file)==digest, file
    current={str(x):[x.stat().st_size,x.stat().st_mtime_ns] for x in (ROOT/'predictions_v2').rglob('*') if x.is_file()}
    assert current==p['original_predictions_stat'], 'Existing predictions_v2 was modified'
    for source in p['original_sources']:assert sha(source['raw'])==source['raw_sha256']
    report=p['original_project_report'];content=Path(report['path']).read_bytes()
    assert hashlib.sha256(content[:report['bytes']]).hexdigest()==report['sha256']
    return dict(checked_at=now(),passed=True,protected_files=len(p['protected_sha256']),existing_prediction_files=len(current),
                original_source_hashes_verified=len(p['original_sources']),project_report_original_bytes_unchanged=True)


def render_report():
    s=read(OUT/'eval_seed_summary.json');d=read(OUT/'validity_decomposition.json');o=read(OUT/'official_baselines.json')
    official_run=read(OUT/'official_mixed.json');official_protocol=Path(official_run['protocol']).name
    def pct(x):return f'{x*100:.3f}%' if x is not None else '无合格样本'
    def pp(c):return f"{c['delta']*100:+.3f} pp [{c['ci95'][0]*100:+.3f}, {c['ci95'][1]*100:+.3f}]"
    c=s['comparison']
    display_time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
    lines=['# MiniOneReason 必做补充评测', '',f'报告更新时间（北京时间）：{display_time}。范围严格按 PLAN_checks.md 第 0.1 节：M1、M2、M3。训练种子仍为 42；采样 offset 为 0–4。', '',
        '## 结论','',f"- Q1 非思考模式：**{c['judgement']}**。5 次差值中 {c['positive_offsets']} 次为正；合并 Recall@32 差值 {pp(c['merged'])}。",
        f"- Q2 官方真实商品 ID：**{o['comparisons']['pid_recall@64']['judgement']}**；SID：**{o['comparisons']['recall@64']['judgement']}**。"]
    for comp in d['comparisons']:lines.append(f"- Q3 {'思考' if comp['thinking'] else '非思考'}模式：**{comp['judgement']}**；共同满足 16 个有效尝试的样本 {comp['intersection_examples']}/403。")
    lines+=['','## M2：非思考模式多采样重评','','来源：[eval_seed_summary.json](eval_seed_summary.json)，逐次 summary 见 results/。所有参数、适配器与判断规则见 [protocol.json](protocol.json)。','',
        '| 模型 | offset | Recall@8 | Recall@32 | Recall@64 | 无效比例 |','|---|---:|---:|---:|---:|---:|']
    for v,model in s['models'].items():
        for r in model['runs']:lines.append(f"| {v} | {r['offset']} | {pct(r['recall@8'])} | {pct(r['recall@32'])} | {pct(r['recall@64'])} | {pct(r['invalid_rate'])} |")
        lines.append(f"| {v} 均值 | 0–4 | {pct(model['mean']['recall@8'])} | {pct(model['mean']['recall@32'])} | {pct(model['mean']['recall@64'])} | {pct(model['mean']['invalid_rate'])} |")
    lines+=['','| 方差来源 | mixed | rl_nk_prefix | 配对差 |','|---|---:|---:|---:|',
        '| 5 次采样的样本标准差（pp） | '+' | '.join(f"{s['models'][v]['sample_std']['recall@32']*100:.3f}" for v in ('mixed','rl_nk_prefix'))+f" | {c['sample_std']*100:.3f} |",
        '| offset 0 样本 bootstrap 标准误（pp） | '+' | '.join(f"{s['models'][v]['original_single_run_bootstrap_se']*100:.3f}" for v in ('mixed','rl_nk_prefix'))+f" | {c['original_single_run']['bootstrap_standard_error']*100:.3f} |",'',
        '逐 offset 配对差（pp）：'+', '.join(f'{x*100:+.3f}' for x in c['per_offset_deltas'])+'。',
        f"采样间配对差标准差{'大于' if c['sample_std']>c['original_single_run']['bootstrap_standard_error'] else '不大于'}原单次样本 bootstrap 标准误。两者刻画不同的不确定性。",'',
        '先对每条样本的 5 次命中取均值，再按输入组做 1,000 次配对 bootstrap。offset 0 来自原 4090，新增 offset 来自 5090；本轮未做可选的确定性重跑，因此这组波动未完全排除设备差异。',
        '本次合并数值只作为非思考模式的依据；原 thinking 模式 +2.233 pp 的幅度没有经过多采样检验，不能用本轮非思考结果替代它。','',
        '| 已有设置 | mixed Recall@32 | rl_nk_prefix Recall@32 |','|---|---:|---:|']
    for r in s['existing_five_settings']:lines.append(f"| {r['setting']} | {pct(r['mixed'])} | {pct(r['rl_nk_prefix'])} |")
    lines+=['','以上五组设置的来源均记录在 eval_seed_summary.json；它们的预算与采样方式不同，不能当作五次相同配置的独立复现。','',
        '## M3：官方商品任务的 mixed 基线','',f'来源：[official_baselines.json](official_baselines.json)、[实际执行协议]({official_protocol})。沿用官方提交、数据、标签、Race 32+32 与评分补丁；1,000 条全量。',
        f"实际设备：{official_run.get('runtime',read(OUT/'runtime_identity.json'))['runtime_id']}；显存比例仍为 0.65。若有环境恢复，详见 [runtime_recovery.json](runtime_recovery.json)。",'',
        '| 指标 | mixed | rl_nk_prefix | RL − mixed，95% CI（pp） |','|---|---:|---:|---|']
    for k,comp in o['comparisons'].items():lines.append(f"| {k} | {pct(o['models']['mixed']['scores'][k])} | {pct(o['models']['rl_nk_prefix']['scores'][k])} | {pp(comp)} |")
    lines+=['','逐样本均值已与官方聚合分数核对；区间按官方样本做 1,000 次配对 bootstrap。本轮没有预训练基线，不作相对预训练的提升结论。官方使用多目标标签、混合候选和 PID 映射，与本地单目标 SID 协议分开解释。','',
        '## M1：有效性分解','','初始 offset 0 结果：[validity_decomposition_offset0.json](validity_decomposition_offset0.json)。并入新采样后的结果：[validity_decomposition.json](validity_decomposition.json)。其中保留逐样本不同有效 SID 数与所有原始路径。','',
        '| 模型 | 思考 | offset | 目录外 | 格式错误 | 思考未闭合 | 其他无效 | 有效率 | 有效尝试命中率 | 平均不同有效 SID |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['runs'].values():
        a=r['invalid_categories'];lines.append(f"| {r['variant']} | {r['thinking']} | {r['offset']} | {a['out_of_catalog']} | {a['format_error']} | {a['unclosed_thinking']} | {a['other']} | {pct(r['valid_rate'])} | {pct(r['hit_rate_given_valid'])} | {r['mean_unique_valid_sids']:.3f} |")
    for comp in d['comparisons']:
        lines+=['',f"### {'思考' if comp['thinking'] else '非思考'}模式，offset {comp['offsets']}",
                f"共同合格样本 {comp['intersection_examples']}，排除 {comp['excluded_examples']}。按样本 bootstrap；跨 offset 采用共同合格样本的交集，两个模型使用同一批样本。",'']
        for r in comp['per_offset']:lines.append(f"- offset {r['offset']}：合格 {r['eligible_examples']} 条；等有效预算 Recall@8 mixed/RL {pct(r['models']['mixed']['recall@8'])}/{pct(r['models']['rl_nk_prefix']['recall@8'])}，Recall@16 {pct(r['models']['mixed']['recall@16'])}/{pct(r['models']['rl_nk_prefix']['recall@16'])}。")
        for k,estimate in comp['equal_valid_budget'].items():lines.append(f'- 合并等有效预算 {k} 差值：{pp(estimate)}。')
        rates=comp['rates'];lines.append(f"- 有效率变化：{pct(rates['mixed']['valid_rate'])} → {pct(rates['rl_nk_prefix']['valid_rate'])}；有效尝试命中率：{pct(rates['mixed']['hit_rate_given_valid'])} → {pct(rates['rl_nk_prefix']['hit_rate_given_valid'])}。")
        lines.append(f"- 判定：{comp['judgement']}。")
    lines+=['','有效预算比较是对合格样本的条件分析；“主要来自减少目录外 SID”是预设规则的诊断标签，不能单凭该检验确定因果贡献。','',
        '## 范围与完整性','','未运行：thinking 多采样、rl_nk 新采样、offset 0 确定性重跑、官方预训练/rl_nk 基线、约束解码、种子 43/44。',
        '旧结果、旧预测、选择文件与 RESUME.md 均保持原样；PROJECT_REPORT.md 只追加本报告链接。最终指纹验收见 [integrity.json](integrity.json)。','']
    (OUT/'CHECKS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def local_eval(variant, offset):
    assert variant in ('mixed','rl_nk_prefix') and offset in range(1,5)
    p=read(OUT/'protocol.json');gpu_ready(p['runtime'])
    assert sha(ROOT/'checkpoints/main'/f'{variant}_42/selected/adapter_model.safetensors')==p['adapters'][variant]
    evaluate_one(variant,42,'test',thinking=False,n=64,seed_offset=offset,
        tag=f'checks_{variant}_42_test_thinkFalse_offset{offset}',prediction_directory=RAW/'local',
        result_directory=OUT/'results',runtime_identity_path=OUT/'runtime_identity.json')


def run():
    import fcntl
    lock=(CODE/'reports/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=freeze();os.environ['CUDA_VISIBLE_DEVICES']=str(p['runtime']['gpu_index'])
    status_path=OUT/'status.json';state=read(status_path) if status_path.exists() else dict(done=[])
    state.pop('failure',None)
    def status(stage,current):
        state.update(stage=stage,current=current,pid=os.getpid(),updated_at=now());write_json(status_path,state)
    jobs=[('M1_offset0',['decompose-initial'])]
    jobs += [(f'M2_{v}_offset{o}',['local',v,str(o)]) for o in range(1,5) for v in ('mixed','rl_nk_prefix')]
    jobs += [('M2_statistics',['seed-statistics']),('M1_merged',['decompose']),('M3_official_mixed',['official']),('M3_statistics',['official-statistics'])]
    try:
        (OUT/'logs').mkdir(parents=True,exist_ok=True)
        for label,args in jobs:
            if label in state['done']:continue
            status('running',label)
            with (OUT/'logs'/f'{label}.log').open('a') as log:
                child=subprocess.Popen([sys.executable,'-m','ops.checks',*args],cwd=CODE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;write_json(status_path,state);code=child.wait()
            state.pop('child_pid',None)
            if code:raise RuntimeError(f'{label} exited {code}; see reports/checks/logs/{label}.log')
            state['done'].append(label);status('running',label+' complete')
        render_report()
        integrity=verify_protected()
        protocol_time=datetime.datetime.fromisoformat(p['created_at']).timestamp()
        for path in [*OUT.glob('results/*.json'),OUT/'validity_decomposition_offset0.json',OUT/'validity_decomposition.json',OUT/'eval_seed_summary.json',OUT/'official_baselines.json']:
            assert path.stat().st_mtime>protocol_time,str(path)
        integrity['new_results_after_protocol']=True
        link='\n补充评测：[必做稳健性检查](checks/CHECKS_REPORT.md)。\n'
        report=CODE/'reports/PROJECT_REPORT.md'
        if link.strip() not in report.read_text(encoding='utf-8'):
            with report.open('a',encoding='utf-8') as handle:handle.write(link)
        write_json(OUT/'integrity.json',integrity)
        write_json(OUT/'gpu_after.json',gpu_snapshot())
        status('complete','Mandatory M1, M2, M3 complete; no optional experiment run')
    except Exception as error:
        state['failure']=str(error);status('failed',str(error));raise


if __name__=='__main__':
    action=sys.argv[1]
    if action=='run':run()
    elif action=='local':local_eval(sys.argv[2],int(sys.argv[3]))
    elif action=='decompose-initial':decompose(initial=True)
    elif action=='decompose':decompose()
    elif action=='seed-statistics':seed_statistics()
    elif action=='official':
        from ops.official import run_checks_baseline
        run_checks_baseline()
    elif action=='official-statistics':official_statistics()
    else:raise ValueError(action)
