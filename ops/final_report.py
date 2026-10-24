"""Build the delivery report only from saved runs; audit every planned job."""
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path

from data import CODE, ROOT, config, load_split, read_jsonl, write_json
from evaluate import paired_bootstrap
from run import scheduled_jobs


def options(arguments):
    values={};i=1
    while i<len(arguments):
        key=arguments[i].removeprefix('--')
        if i+1<len(arguments) and not arguments[i+1].startswith('--'):
            values[key]=arguments[i+1];i+=2
        else:values[key]=True;i+=1
    return values


def audit_jobs():
    protocol=json.loads((CODE/'artifacts/execution_protocol_v2.json').read_text())
    active=scheduled_jobs(protocol['jobs']);active_indices={index for index,_ in active}
    deferred=[dict(index=index,**job) for index,job in enumerate(protocol['jobs']) if index not in active_indices]
    results=CODE/'reports/results_v2';checks=[]
    def require(job,path):
        checks.append(dict(job=job,path=str(path),present=path.exists()))
    for _,job in active:
        args=job['args'];command=args[0];v=options(args)
        profile=v.get('profile','main');seed=int(v.get('seed',42));variant=v.get('variant','mixed');split=v.get('split','validation')
        name=' '.join(args)
        if command=='sft':
            require(name,ROOT/'checkpoints'/profile/f'{variant}_{seed}/complete.json')
            require(name,ROOT/'checkpoints'/profile/f'{variant}_{seed}/selected/adapter_model.safetensors')
        elif command=='baseline':
            require(name,ROOT/'checkpoints'/profile/f'sasrec_{seed}/complete.json')
        elif command=='evaluate':
            skip=CODE/'reports/rl'/f'{profile}_{variant.removeprefix("rl_")}_{seed}_skipped.json'
            if variant.startswith('rl_') and skip.exists():
                require(name,skip);continue
            tag=v.get('tag') or f'{variant}_{seed}_{split}_recent_in2048_think{bool(v.get("thinking"))}_budgetNone_n{v.get("samples",64)}_vllm'
            require(name,results/f'{tag}.json')
            require(name,ROOT/'predictions_v2'/profile/f'{tag}.jsonl')
        elif command=='thinkcost':
            require(name,CODE/'reports'/f'{profile}_thinking_{seed}_{split}_summary.json')
            for budget in protocol['config']['thinkcost']['budgets']:
                require(name,results/f'{profile}_thinkcost_{seed}_{split}_{budget}.json')
            for signal in ('uncertainty','history_length'):
                for fraction in protocol['config']['thinkcost']['routing_fractions']:
                    require(name,CODE/'reports'/f'{profile}_route_{seed}_{split}_{signal}_{fraction}_vs_random.json')
        elif command=='rl':
            if v.get('probe'):
                for item in ('std','std32','nk','nk_prefix','nk_prefix_div'):
                    require(name,CODE/'reports/rl'/f'{profile}_probe_{item}_{seed}.json')
            else:
                skip=CODE/'reports/rl'/f'{profile}_{variant}_{seed}_skipped.json'
                probe=CODE/'reports/rl'/f'{profile}_probe_{variant}_{seed}.json'
                require(name,probe)
                if skip.exists():
                    if probe.exists() and json.loads(probe.read_text())['eligible_for_training']:
                        checks.append(dict(job=name,path=str(skip),present=False,reason='An eligible RL run cannot be counted as sparse-reward skip'))
                else:
                    complete=ROOT/'checkpoints'/profile/f'rl_{variant}_{seed}/complete.json'
                    require(name,complete)
                    require(name,ROOT/'checkpoints'/profile/f'rl_{variant}_{seed}/selected/adapter_model.safetensors')
                    if complete.exists():
                        actual=json.loads(complete.read_text())['steps']
                        checks.append(dict(job=name,path=str(complete),present=actual==protocol['config']['rl']['steps'],actual_steps=actual))
        elif command=='history':
            selection=CODE/'artifacts'/f'history_selection_{profile}_{seed}.json';require(name,selection)
            if selection.exists():
                selected=json.loads(selection.read_text());best=selected['variant'];budget=selected['main_budget']
                for item in [best,'mixed_recent_history','mixed_balanced']:
                    require(name,CODE/'reports'/f'history_comparison_{profile}_{item}_{seed}_{split}.json')
                    for size in protocol['config']['history']['budgets'] if item==best else [budget]:
                        for strategy in ('recent','domain_recent','balanced') if item==best else ('recent','balanced'):
                            require(name,results/f'{profile}_history_{item}_{seed}_{split}_{strategy}_{size}.json')
    bad=[]
    for path in results.glob('*.json'):
        row=json.loads(path.read_text());s=row.get('settings',{})
        if not s:continue
        if s.get('protocol_version')!='2.0-runtime-isolated':bad.append(dict(path=str(path),reason='mixed runtime protocol'))
        if row['examples']!=len(s['sample_ids']):bad.append(dict(path=str(path),reason='incomplete evaluation'))
        for key in ('recall@32','invalid_rate','tokens_per_example'):
            if key in row and not math.isfinite(row[key]):bad.append(dict(path=str(path),reason='nonfinite metric'))
    missing=[c for c in checks if not c['present']]
    if not missing and not bad:
        raw_signature=[]
        for path in results.glob('*.json'):
            row=json.loads(path.read_text())
            if not row.get('settings'):continue
            raw=Path(row['raw_predictions']);stat=raw.stat()
            raw_signature.append((str(path),path.stat().st_mtime_ns,str(raw),stat.st_size,stat.st_mtime_ns))
        signature=hashlib.sha256(json.dumps(sorted(raw_signature)).encode()).hexdigest()
        cache=CODE/'reports/raw_prediction_audit.json'
        prior=json.loads(cache.read_text()) if cache.exists() else {}
        if prior.get('signature')!=signature or not prior.get('passed'):
            checked=0
            for result_path,_,raw,_,_ in raw_signature:
                row=json.loads(Path(result_path).read_text());settings=row['settings'];ids=[]
                with Path(raw).open() as stream:
                    for line in stream:
                        prediction=json.loads(line);ids.append(prediction['sample_id'])
                        if len(prediction['outputs'])!=settings['n']:
                            bad.append(dict(path=raw,reason='raw attempt budget mismatch'));break
                if len(ids)!=len(set(ids)) or set(ids)!=set(settings['sample_ids']):
                    bad.append(dict(path=raw,reason='raw input IDs incomplete or duplicated'))
                checked+=1
            write_json(cache,dict(signature=signature,passed=not bad,checked_files=checked,invalid=bad))
    value=dict(complete=not missing and not bad,scope='seed_42',seeds=[42],active_job_count=len(active),deferred_jobs=deferred,
               checks=checks,missing=missing,invalid=bad,
               protocol_sha256=hashlib.sha256((CODE/'artifacts/execution_protocol_v2.json').read_bytes()).hexdigest())
    write_json(CODE/'reports/final_audit.json',value)
    return value


def read_main_results():
    results={}
    for path in (CODE/'reports/results_v2').glob('*.json'):
        row=json.loads(path.read_text());s=row.get('settings',{})
        if s.get('profile')=='main' and s.get('n')==64 and s.get('strategy')=='recent' and s.get('input_budget')==2048 and s.get('budget') is None and not s.get('monitor') and '_history_' not in path.stem:
            results[(s['variant'],s['seed'],s['split'],s['thinking'])]=row
    return results


def comparison(left,right):
    a=read_jsonl(Path(left['raw_predictions']).with_suffix('.metrics.jsonl'))
    b=read_jsonl(Path(right['raw_predictions']).with_suffix('.metrics.jsonl'))
    return paired_bootstrap(a,b,'recall@32')


def delivery_assets(complete):
    reports=CODE/'reports'
    manifest_path=CODE/'artifacts/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    combined={name:json.loads((reports/file).read_text()) for name,file in (
        ('main','data_report_main.json'),('catalog','catalog_report.json'),('validation','validation_main.json'))}
    combined.update(revisions=manifest['revisions'],fingerprints=[row for row in manifest['files'] if '/data/processed/main/' in row['path']])
    write_json(reports/'data_report.json',combined)
    invalid=[]
    for path in sorted((reports/'results_v2').glob('*.json')):
        value=json.loads(path.read_text());settings=value.get('settings',{})
        if settings.get('profile')!='main' or settings.get('monitor'):continue
        invalid.append(dict(experiment=path.stem,attempts=value['attempts'],invalid_rate=value['invalid_rate'],
                            parser_counts=value['parser_counts'],raw_predictions=value['raw_predictions']))
    write_json(reports/'invalid_output_audit.json',invalid)
    render_delivery_figures()
    if not complete:return
    def fingerprint(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())
    paths=[*CODE.glob('*.py'),CODE/'config.json',CODE/'env.sh',CODE/'requirements.txt',
           *CODE.glob('tests/*.py'),*CODE.glob('ops/*.py'),reports/'environment.lock.txt',
           CODE/'artifacts/execution_protocol_v2.json',*CODE.glob('artifacts/*_revision.json')]
    manifest['completion_code']=[fingerprint(path) for path in sorted(paths)]
    manifest['selected_adapters']=[fingerprint(path) for path in sorted((ROOT/'checkpoints/main').glob('*/selected/adapter_model.safetensors'))]
    manifest['rl_prompt_schedules']=[fingerprint(path) for path in sorted(CODE.glob('artifacts/rl_prompt_schedule_main_*.json'))]
    manifest['prediction_root']=str(ROOT/'predictions_v2')
    manifest['results_root']=str(reports/'results_v2')
    write_json(manifest_path,manifest)


def render_delivery_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports=CODE/'reports';directory=reports/'figures';directory.mkdir(exist_ok=True)
    for split in ('validation','test'):
        summary=reports/f'main_thinking_42_{split}_summary.json'
        if summary.exists():
            value=json.loads(summary.read_text());fig,ax=plt.subplots(figsize=(11,5))
            for row in value['routes']:
                name=row['experiment'].removeprefix(f'main_route_42_{split}_').replace(f'main_thinkcost_42_{split}_','')
                ax.scatter(row['tokens_per_example'],row['recall@32']*100,marker='x' if name.endswith('random') else 'o',label=name)
            ax.set(xlabel='Generated tokens per user (including discarded probes)',ylabel='Recall@32 (%)',title=f'Routing and equal-fraction random controls: {split}')
            ax.margins(.15);ax.legend(bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
            fig.tight_layout();fig.savefig(directory/f'main_routing_42_{split}.png',dpi=180);plt.close(fig)
        selection=CODE/'artifacts/history_selection_main_42.json'
        if selection.exists():
            best=json.loads(selection.read_text())['variant'];curves=collections.defaultdict(list)
            for path in (reports/'results_v2').glob(f'main_history_{best}_42_{split}_*.json'):
                row=json.loads(path.read_text());s=row['settings'];curves[s['strategy']].append((s['input_budget'],row['recall@32']*100))
            if curves:
                fig,ax=plt.subplots(figsize=(7,4))
                for strategy,points in sorted(curves.items()):
                    points.sort();ax.plot([p[0] for p in points],[p[1] for p in points],'o-',label=strategy)
                ax.set(xlabel='Input token budget',ylabel='Recall@32 (%)',title=f'History selection ({best}): {split}',xticks=config()['history']['budgets'])
                ax.legend();fig.tight_layout();fig.savefig(directory/f'main_history_42_{split}.png',dpi=180);plt.close(fig)


def write_final_report():
    audit=audit_jobs();results=read_main_results();seeds=audit['seeds']
    from ops.supplement import audit_supplement
    supplement=audit_supplement()
    write_json(CODE/'reports/completion_status.json',dict(main_chain_complete=audit['complete'],
        rl_thinkcost_complete=supplement['rl_thinkcost_complete'],official_status=supplement['official']['status'],
        all_experiments_complete=audit['complete'] and supplement['all_experiments_complete'],deferred_seeds=[43,44]))
    delivery_assets(audit['complete'])
    lines=['# MiniOneReason 项目报告','',
           '状态：'+('seed 42 主队列 43 项的审计通过。' if audit['complete'] else f'seed 42 实验进行中；当前范围尚缺 {len(audit["missing"])} 项产物，以下只展示已有结果。'),
           'RL 权重成本评测：'+('完成' if supplement['rl_thinkcost_complete'] else '待完成')+'；官方外部评测：'+supplement['official']['status']+'。两项补充要求的证据见 supplement_audit.json。','',
           f'按用户要求，本轮只运行 seed 42；seed 43、44 的 {len(audit["deferred_jobs"])} 个重复实验已移出自动队列，是否补做由用户后续决定。当前结果不包含多种子稳定性验证。','',
           '基于官方 0.8B 预训练模型与公开 SFT 的商品子集，研究推理成本、稀疏奖励和历史选择。此项目不构成官方 8B 全流程复现。','',
           '## 数据与比较边界','',
           '训练 2,647 条、验证 385 条、测试 403 条；按规范化输入分组留出，不能保证用户级或时间级互斥，也不能排除预训练重叠。评价目标是单个商品 SID；官方结果在真实物品 ID 上计分，数值不可直接比较。',
           '模型对比统一来自 results_v2。旧结果保留在 results 和 archive_v1。主要训练与模型对比在共享 RTX 4090 上完成，后续成本与历史实验在 RTX 5090 上完成；runtime_id 标明各次评测的设备。跨设备、共享负载的耗时不能直接作为方法收益，成本结论优先使用生成 token。','',
           '## SFT 对照','',
           '| 设置 | 已完成种子 | Recall@32 均值 | 种子标准差 | Recall@64 均值 | 生成 token/用户 |',
           '| --- | --- | ---: | ---: | ---: | ---: |']
    for variant,thinking,label in [('base',False,'预训练基线'),('direct',False,'直接 SFT'),('mixed',False,'混合 SFT，不思考'),('mixed',True,'混合 SFT，思考')]:
        rows=[(seed,results[(variant,seed,'test',thinking)]) for seed in seeds if (variant,seed,'test',thinking) in results]
        if not rows:continue
        scores=[r['recall@32'] for _,r in rows];std=f'{statistics.stdev(scores)*100:.3f} 个百分点' if len(scores)>1 else '单种子'
        lines.append(f'| {label} | {", ".join(str(seed) for seed,_ in rows)} | {statistics.mean(scores)*100:.3f}% | {std} | {statistics.mean(r["recall@64"] for _,r in rows)*100:.3f}% | {statistics.mean(r["tokens_per_example"] for _,r in rows):.1f} |')
    paired=[]
    for seed in seeds:
        left=results.get(('direct',seed,'test',False))
        if left:
            for mode in (False,True):
                right=results.get(('mixed',seed,'test',mode))
                if right:paired.append(dict(seed=seed,comparison=f'mixed thinking={mode} minus direct',**comparison(left,right)))
    lines+=['','配对差值按相同输入组做 1,000 次 bootstrap；区间覆盖零表示证据不足，不能改写成稳定提升。']
    for row in paired:
        lines.append(f'- seed={row["seed"]}，{row["comparison"]}：{row["delta"]*100:+.3f} 个百分点，95% 区间 [{row["ci95"][0]*100:+.3f}, {row["ci95"][1]*100:+.3f}]。')
    lines+=['','## 推理预算与路由','']
    summary=CODE/'reports/main_thinking_42_test_summary.json'
    if summary.exists():
        value=json.loads(summary.read_text());selected=value['selection']['thinking_budget']
        lines+=['| 推理预算 | Recall@32 | token/用户 | 无效比例 |','| ---: | ---: | ---: | ---: |']
        for budget,row in value['budgets'].items():lines.append(f'| {budget} | {row["recall@32"]*100:.3f}% | {row["tokens_per_example"]:.1f} | {row["invalid_rate"]*100:.2f}% |')
        lines+=['',f'路由使用验证集选定的 {selected} token 思考预算，阈值在测试前冻结；探测被丢弃的答案也计入成本。','',
                '| 路由信号 | 与随机路由的 Recall@32 差值 | 95% 区间（百分点） |','| --- | ---: | --- |']
        for path in sorted((CODE/'reports').glob('main_route_42_test_*_vs_random.json')):
            row=json.loads(path.read_text());lines.append(f'| {path.stem.removeprefix("main_route_42_test_").removesuffix("_vs_random")} | {row["delta"]*100:+.3f} 个百分点 | [{row["ci95"][0]*100:+.3f}, {row["ci95"][1]*100:+.3f}] |')
    else:lines.append('完整测试预算与路由结果尚未生成。')
    lines+=['','### RL 权重的同预算评测','']
    protocol=CODE/'artifacts/rl_thinkcost_protocol_42.json'
    if protocol.exists():
        variant=json.loads(protocol.read_text())['variant'];namespace='main_'+variant
        lines.append(f'只依据验证集 Recall@32 选择 {variant}，沿用 SFT 的 0/128/256/512/1024 token 预算、32 次最终尝试、8 次探测和同等比例随机路由对照。探测被丢弃的输出计入成本。')
        summary=CODE/'reports'/f'{namespace}_thinking_42_test_summary.json'
        if summary.exists():
            value=json.loads(summary.read_text())
            lines+=['','| 推理预算 | Recall@32 | token/用户 | 无效比例 |','| ---: | ---: | ---: | ---: |']
            for budget,row in value['budgets'].items():lines.append(f'| {budget} | {row["recall@32"]*100:.3f}% | {row["tokens_per_example"]:.1f} | {row["invalid_rate"]*100:.2f}% |')
            lines+=['',f'验证集选定的路由思考预算为 {value["selection"]["thinking_budget"]} token；测试使用冻结阈值。','',
                    '| 路由信号 | 相对随机路由的 Recall@32 差值 | 95% 区间（百分点） |','| --- | ---: | --- |']
            for path in sorted((CODE/'reports').glob(f'{namespace}_route_42_test_*_vs_random.json')):
                row=json.loads(path.read_text());lines.append(f'| {path.stem.removeprefix(namespace+"_route_42_test_").removesuffix("_vs_random")} | {row["delta"]*100:+.3f} 个百分点 | [{row["ci95"][0]*100:+.3f}, {row["ci95"][1]*100:+.3f}] |')
            direct_route=next(row for row in value['routes'] if row['experiment'].endswith('_all_non_thinking'))
            lines+=['',f'路由的全非思考对照 Recall@32={direct_route["recall@32"]*100:.3f}%，成本为 {direct_route["tokens_per_example"]:.1f} token/用户。它由实际 8+24 次调用合并，随机序列与上表单次 32 候选的预算 0 不同；路由收益应与这组同源对照比较。各路由的绝对成绩和相对非思考对照区间保存在对应 summary.json 的 routes 中。']
        else:lines.append('测试结果尚未生成。')
    else:lines.append('尚未冻结 RL 权重成本评测协议。')
    lines+=['','## RL 奖励信号与推荐收益','',
            '标准 8 答案、标准 32 答案、N×K=4×8、N×K 加前缀奖励均从同一 mixed SFT 出发。训练门槛只取决于训练集的奖励差异，不能根据测试集决定是否保留某个变体。相同答案数不等于相同计算量；两种实现的损失归一化与采样后端差异已列入 method.md。','']
    probe=CODE/'reports/rl/main_probe_summary_42.json'
    if probe.exists():
        lines+=['| 变体 | 有效奖励组 | 精确命中/尝试 | 无效输出 | 是否达到训练门槛 |','| --- | ---: | ---: | ---: | --- |']
        for row in json.loads(probe.read_text()):lines.append(f'| {row["variant"]} | {row["active_group_fraction"]*100:.2f}% | {row["exact_hit_rate"]*100:.3f}% | {row["invalid_rate"]*100:.2f}% | {row["eligible_for_training"]} |')
    rl_comparisons=[]
    for variant in ('std','std32','nk','nk_prefix'):
        skip=CODE/'reports/rl'/f'main_{variant}_42_skipped.json'
        if skip.exists():lines.append(f'- {variant}：按预定稀疏奖励门槛跳过训练，原因见 {skip.name}。')
        for mode in (False,True):
            left=results.get(('mixed',42,'test',mode));right=results.get((f'rl_{variant}',42,'test',mode))
            if left and right:
                row=dict(variant=variant,thinking=mode,**comparison(left,right));rl_comparisons.append(row)
                lines.append(f'- {variant}，thinking={mode}：Recall@32={right["recall@32"]*100:.3f}%，相对对应 SFT 模式差 {row["delta"]*100:+.3f} 个百分点，95% 区间 [{row["ci95"][0]*100:+.3f}, {row["ci95"][1]*100:+.3f}]。')
    lines+=['','RL 目前按计划只使用 seed 42，属于探索性对照。训练奖励提高、前缀命中增加和精确推荐改善是不同结论，不能相互替代。','',
            '## 官方商品外部评测','',
            '状态：'+supplement['official']['status']+'。本项目验证集选出的 0.8B RL 模型在官方 v3.1 商品任务的完整 1,000 条样本上评测，Race 按思考 32 个、直接 32 个候选合并，保留官方 prompt、beam search 和 SID/PID 评分。它与本地 403 条单目标 SID 测试是两套独立协议，成绩分开解释。']
    official=supplement['official']
    if official.get('scores'):
        lines+=['','| 指标（本项目 RL 模型） | 分数 |','| --- | ---: |']
        for name in ('pass@32','recall@32','pass@64','recall@64','pid_pass@32','pid_recall@32','pid_pass@64','pid_recall@64'):
            if name in official['scores']:lines.append(f'| {name} | {official["scores"][name]*100:.3f}% |')
    if (CODE/'artifacts/official_compatibility.json').exists():
        compatibility=json.loads((CODE/'artifacts/official_compatibility.json').read_text())
        lengths=compatibility['prompt_tokens']
        lines+=['',f'官方输入 token 长度：最小 {lengths["min"]}，中位数 {lengths["median"]}，最大 {lengths["max"]}。上下文上限 8,192，保留全部输入和官方 4,096 token 思考预算。',
                '官方 3 条样本的 answer_pid 集合包含 answer SID 未覆盖的商品；保留官方两类标签分别计分。运行补丁只处理物理 GPU 隔离、失败上报和批次保存；不改变标签、生成预算或指标。BF16 合并后的 RL 模型用于官方运行，合并可能带来数值差异；不能把耗时当成独占 GPU 基准。']
    lines+=['',
            '## 历史选择与 token 匹配','']
    selection=CODE/'artifacts/history_selection_main_42.json'
    if selection.exists():
        budget=json.loads(selection.read_text())['main_budget']
        lines += [f'训练与评测策略的 2×2 对照使用共同的 {budget} token 输入上限：','',
                  '| 训练策略 | recent 评测 Recall@32 | balanced 评测 Recall@32 |','| --- | ---: | ---: |']
        for variant in ('mixed_recent_history','mixed_balanced'):
            cells=[]
            for strategy in ('recent','balanced'):
                path=CODE/'reports/results_v2'/f'main_history_{variant}_42_test_{strategy}_{budget}.json'
                cells.append(f'{json.loads(path.read_text())["recall@32"]*100:.3f}%' if path.exists() else '待完成')
            lines.append(f'| {variant} | {cells[0]} | {cells[1]} |')
    for path in sorted((CODE/'reports').glob('history_comparison_main_*_42_test.json')):
        for row in json.loads(path.read_text()):lines.append(f'- {path.stem}，{row["strategy"]} 相对 recent：{row["delta"]*100:+.3f} 个百分点，95% 区间 [{row["ci95"][0]*100:+.3f}, {row["ci95"][1]*100:+.3f}]。')
    left=results.get(('mixed',42,'test',False));right=results.get(('direct_tokenmatched',42,'test',False))
    if left and right:
        row=comparison(left,right);lines.append(f'- 匹配训练 token 的直接 SFT 相对混合 SFT non-thinking：{row["delta"]*100:+.3f} 个百分点，95% 区间 [{row["ci95"][0]*100:+.3f}, {row["ci95"][1]*100:+.3f}]。Token 数匹配不等于 FLOPs 严格相同。')
        ledger=json.loads((ROOT/'checkpoints/main/direct_tokenmatched_42/complete.json').read_text())
        lines.append(f'- Token 匹配目标 {ledger["token_target"]:,}，实际 {ledger["processed_tokens"]:,}；在完整梯度累积步骤结束时停止，超出 {(ledger["processed_tokens"]/ledger["token_target"]-1)*100:.3f}%。')
    lines+=['','## 输出失败分析','',
            '以下为 seed 42 测试集的尝试级统计。目录外 SID 的格式可以完全正确，但没有对应目录项，不能通过放宽解析规则改成合法推荐。','',
            '| 模式 | 总尝试数 | 目录外 SID | 思考未正确闭合 | 其它无效输出 |','| --- | ---: | ---: | ---: | ---: |']
    for variant,mode,label in [('base',False,'预训练'),('direct',False,'直接 SFT'),('mixed',False,'混合 SFT 不思考'),('mixed',True,'混合 SFT 思考')]:
        row=results.get((variant,42,'test',mode))
        if not row:continue
        counts=row['parser_counts'];out=counts.get('out_of_catalog',0);closure=counts.get('unclosed_or_multiple_think',0)
        other=sum(counts.values())-counts.get('valid',0)-out-closure
        lines.append(f'| {label} | {row["attempts"]} | {out} | {closure} | {other} |')
    lines+=['','## 失败结果与限制','',
            '- SASRec 与热门基线在该子集上全零；测试目标只有 3 条曾作为训练目标，353 条连训练历史也未出现。它们是该稀疏设置下的弱参考，不能支撑广泛的优越性声明。',
            '- 商品 SID 存在碰撞，同一 SID 可对应多个商品；本项目的 SID 命中不保证定位唯一真实商品。',
            '- 官方外部评测状态见上节。项目完成了 SFT 与 RL 的离线实验，RFT、多教师整合和线上 A/B 不在本轮实测范围内。',
            '- 对 6 个验证案例的阅读发现，部分推理补充了输入没有的日期、搜索记录或历史 SID，命中案例也存在依据问题。证据见 case_review.md；该定性样本不代表全量错误率。',
            '- 测试集仅 403 条，重复输入按组处理，单次命中差异可能明显改变百分比。保留所有失败变体，不按测试结果选择性删除。','',
            '## 产物与复现','',
            '- `results.csv`：每行保留配置、模型路径、原始预测路径和源结果文件。',
            '- `final_audit.json`：逐项检查本轮 seed 42 范围的产物；多种子任务列为 deferred_jobs，不计作已完成。',
            '- `completion_revision.md`：相对初版的运行修订与隔离方式。',
            '- `data_report.json`、`invalid_output_audit.json`：数据筛选、目录碰撞、数据指纹与逐项无效输出原因。',
            '- `figures/`：推理预算、随机路由对照、奖励探测及历史预算曲线。',
            '- 大文件保存在 `/data1/dzh/minionereason`；代码与报告保存在 `/home/dzh/minionereason`，小文件镜像位于 `D:\\P_project\\minionereason`。','']
    (CODE/'reports/PROJECT_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    write_json(CODE/'reports/final_paired_comparisons.json',dict(sft=paired,rl=rl_comparisons))
    if audit['complete']:
        measured=[]
        for variant,mode,label in [('direct',False,'直接 SFT'),('mixed',False,'混合 SFT 不思考'),('mixed',True,'混合 SFT 思考')]:
            values=[results[(variant,seed,'test',mode)]['recall@32'] for seed in seeds]
            measured.append(f'{label} {statistics.mean(values)*100:.3f}%')
        resume=['# 简历与面试材料','',
                '**MiniOneReason：生成式推荐的推理成本与后训练实验**','',
                '- 基于 OneReason-0.8B 和公开商品推荐样本，完成 LoRA SFT、seed 42 的四种 RL 对照及历史选择消融；主要训练使用 RTX 4090，后续评测迁移至 RTX 5090。',
                '- 建立固定 32/64 次生成尝试的 SID 评测，记录无效输出、重复率和 token 成本，用配对 bootstrap 检查收益的不确定性，所有结果可追溯至适配器和原始预测。',
                '- 测试集 403 条，seed 42 的 Recall@32：'+'；'.join(measured)+'。']
        for row in rl_comparisons:
            if row['variant']=='nk_prefix' and row['thinking']:
                score=results[('rl_nk_prefix',42,'test',True)]['recall@32']
                resume.append(f'- 实现 N×K=4×8 分支采样和三级 SID 前缀奖励，并保留 GRPO-8、GRPO-32 与无前缀 N×K 对照；前缀奖励模型的本地测试 Recall@32 为 {score*100:.3f}%，相对混合 SFT 思考模式提高 {row["delta"]*100:.3f} 个百分点，配对 bootstrap 95% 区间为 [{row["ci95"][0]*100:.3f}, {row["ci95"][1]*100:.3f}] 个百分点。')
        if official.get('scores'):
            resume.append(f'- 完成官方 v3.1 商品任务的 1,000 条外部评测：本项目 0.8B RL 模型的真实商品 ID Recall@64 为 {official["scores"]["pid_recall@64"]*100:.3f}%，Pass@64 为 {official["scores"]["pid_pass@64"]*100:.3f}%；使用官方 Race 32+32 候选和评分程序。')
        resume+=['','统计范围：单次 seed 42。主线成绩使用分组留出的单目标 SID 指标，官方商品任务单独在商品 ID 上计分。多种子稳定性和线上收益尚未验证。','',
                 '面试重点：混合 SFT 与 RFT 的区别；32 答案 GRPO 对照与 N×K 信用分配；前缀奖励如何增加组内差异；固定尝试次数与 token 成本的区别；SID 碰撞如何造成 SID 和真实商品 ID 成绩差距；随机路由对照、历史消融和 SASRec 全零的解释。','']
        (CODE/'reports/RESUME.md').write_text('\n'.join(resume),encoding='utf-8')
        import tarfile
        archive=CODE/'artifacts/completion_v2_code.tgz'
        files=[*CODE.glob('*.py'),*CODE.glob('tests/*.py'),*CODE.glob('ops/*.py'),CODE/'config.json',CODE/'env.sh',
               CODE/'requirements.txt',CODE/'README.md',CODE/'artifacts/manifest.json',CODE/'artifacts/execution_protocol.json',CODE/'artifacts/execution_protocol_v2.json',
               CODE/'reports/PROJECT_REPORT.md',CODE/'reports/RESUME.md']
        for name in ('method.md','completion_revision.md','case_review.md',
                     'validation_case_review_inputs.json','case_input_origin_audit.json'):
            path=CODE/'reports'/name
            if path.is_file():files.append(path)
        files.extend(CODE.glob('artifacts/*_revision.json'))
        files.extend(CODE.glob('artifacts/rl_thinkcost_protocol_*.json'))
        files.extend(CODE.glob('artifacts/official*.json'))
        files.extend(CODE.glob('reports/supplement_audit.json'))
        files.extend(CODE.glob('reports/official_status.json'))
        files.extend(CODE.glob('reports/completion_status.json'))
        files.extend(CODE.glob('artifacts/official_runtime_patch.diff'))
        files.extend(CODE.glob('external/official-main.zip'))
        files.extend(CODE.glob('artifacts/monitor_samples_*.json'))
        files.extend(CODE.glob('artifacts/rl_prompt_schedule_*.json'))
        with tarfile.open(archive.with_suffix('.tmp'),'w:gz') as stream:
            for path in files:stream.add(path,arcname=str(path.relative_to(CODE)))
        archive.with_suffix('.tmp').replace(archive)
    return audit
