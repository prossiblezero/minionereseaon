import json
import unittest
from data import normalize,parse_answer,input_view,split_for
from evaluate import metrics
from history import select_history,prompt_text,history_gate

A='<|prod_begin|><s_a_1><s_b_2><s_c_3>'
B='<|prod_begin|><s_a_4><s_b_5><s_c_6>'
V='<|video_begin|><s_a_1><s_b_2><s_c_3>'


class ToyTokenizer:
    def encode(self,text,add_special_tokens=False):
        return list(text)
    def apply_chat_template(self,messages,**kwargs):
        return '\n'.join(m['content'] for m in messages)+'ASSISTANT'
    def decode(self,ids,**kwargs):
        return ''.join(ids)


def example():
    text='下面给出用户历史行为线索：\n用户购物行为: 浏览了商品 '+A+', '+B+'。\n\n请基于这些线索总结该用户在各场景中的目标内容。'
    messages=[dict(role='system',content=[dict(type='text',text='系统')]),dict(role='user',content=[dict(type='text',text=text)]),
              dict(role='assistant',content=[dict(type='text',text='<think>偏好'+A+'</think>该用户最近点击了商品: '+B)])]
    return normalize(dict(record_id=1,messages=json.dumps(messages)),dict(revision='test',file='test',row=0))


class PipelineTests(unittest.TestCase):
    def test_supplement_audit_cannot_complete_with_missing_outputs_or_wrong_adapter(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import ops.supplement as supplement
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'artifacts').mkdir();(root/'reports/results_v2').mkdir(parents=True)
            with patch.object(supplement,'CODE',root),patch.object(supplement,'ROOT',root):
                self.assertFalse(supplement.audit_supplement()['requirements_resolved'])
                protocol=dict(variant='rl_nk_prefix',adapter_sha256='expected',thinkcost=dict(budgets=[0,128],routing_fractions=[.5]))
                (root/'artifacts/rl_thinkcost_protocol_42.json').write_text(json.dumps(protocol))
                (root/'reports/official_status.json').write_text(json.dumps(dict(status='complete')))
                result=root/'reports/results_v2/main_rl_nk_prefix_thinkcost_42_validation_0.json'
                result.write_text(json.dumps(dict(settings=dict(variant='mixed',adapter_sha256='wrong'))))
                audit=supplement.audit_supplement()
                self.assertFalse(audit['all_experiments_complete'])
                self.assertTrue(audit['missing'])
                self.assertEqual(audit['invalid'][0]['reason'],'wrong RL adapter')

    def test_reference_preserves_fp32_weights_and_stays_frozen(self):
        try:
            import torch
            from peft import LoraConfig,PeftModel,get_peft_model,get_peft_model_state_dict
        except ModuleNotFoundError:
            self.skipTest('Requires the server training environment with torch and PEFT')
        import tempfile
        from collections import OrderedDict
        from pathlib import Path
        from train import load_reference_adapter
        def base():
            return torch.nn.Sequential(OrderedDict(linear=torch.nn.Linear(8,8,bias=False))).to(torch.bfloat16)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            original=get_peft_model(base(),LoraConfig(r=2,lora_alpha=2,target_modules=['linear'],lora_dropout=0))
            with torch.no_grad():
                for name,parameter in original.named_parameters():
                    if 'lora_' in name:parameter.fill_(0.12345678)
            original.save_pretrained(directory)
            policy=PeftModel.from_pretrained(base(),directory,is_trainable=True)
            load_reference_adapter(policy,directory)
            initial=get_peft_model_state_dict(policy,adapter_name='default')
            reference={k:v.clone() for k,v in get_peft_model_state_dict(policy,adapter_name='reference').items()}
            self.assertTrue(all(torch.equal(initial[k],reference[k]) for k in initial))
            inputs=torch.ones(2,8,dtype=torch.bfloat16)
            with torch.no_grad():
                expected=policy(inputs)
                policy.set_adapter('reference');self.assertTrue(torch.equal(expected,policy(inputs)))
            policy.set_adapter('default')
            self.assertFalse(any(p.requires_grad for name,p in policy.named_parameters() if '.reference.' in name))
            optimizer=torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],lr=.01)
            policy(inputs).float().square().mean().backward();optimizer.step()
            after=get_peft_model_state_dict(policy,adapter_name='reference')
            self.assertTrue(all(torch.equal(reference[k],after[k]) for k in reference))
            updated=get_peft_model_state_dict(policy,adapter_name='default')
            self.assertTrue(any(not torch.equal(updated[k],reference[k]) for k in reference))

    def test_single_seed_audit_keeps_repeats_deferred(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from ops.final_report import audit_jobs
        from run import scheduled_jobs
        jobs=[dict(stage='B',args=['sft','--variant','mixed','--seed','42'],when=None),
              dict(stage='B_repeats',args=['sft','--variant','mixed','--seed','43'],when=None),
              dict(stage='B_repeats',args=['sft','--variant','mixed','--seed','44'],when=None),
              dict(stage='F',args=['report'],when=None)]
        self.assertEqual([index for index,_ in scheduled_jobs(jobs)],[0,3])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root=Path(directory);(root/'artifacts').mkdir()
            (root/'artifacts/execution_protocol_v2.json').write_text(json.dumps(dict(jobs=jobs,config={})))
            with patch('ops.final_report.CODE',root),patch('ops.final_report.ROOT',root/'data'):
                before=audit_jobs()
                self.assertFalse(before['complete'])
                self.assertTrue(all('--seed 42' in row['job'] for row in before['missing']))
                for name in ('complete.json','selected/adapter_model.safetensors'):
                    path=root/'data/checkpoints/main/mixed_42'/name
                    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
                after=audit_jobs()
                self.assertTrue(after['complete'])
                self.assertEqual(after['scope'],'seed_42')
                self.assertEqual([row['index'] for row in after['deferred_jobs']],[1,2])
                self.assertFalse((root/'data/checkpoints/main/mixed_43').exists())

    def test_rl_variants_share_a_frozen_prompt_schedule(self):
        import tempfile
        import numpy as np
        from pathlib import Path
        from unittest.mock import patch
        from train import rl_prompt_schedule
        root=Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(root))
            rows=[dict(sample_id=str(i)) for i in range(19)]
            with patch('train.CODE',Path(directory)),patch('train.config',return_value={'rl':{'steps':7}}):
                indices=rl_prompt_schedule(rows,42,'test')
                rng=np.random.default_rng(42)
                self.assertEqual(indices,[int(rng.integers(len(rows))) for _ in range(7)])
                self.assertEqual(indices,rl_prompt_schedule(rows,42,'test'))
                with self.assertRaises(RuntimeError):rl_prompt_schedule(rows[::-1],42,'test')

    def test_branch_batch_keeps_user_seeds_and_groups_separate(self):
        from train import sample_branch_batch
        class Gen:
            calls=[]
            def raw_many(self,prompts,n,max_tokens,seeds,**kwargs):
                self.calls.append((prompts,n,seeds))
                return [[dict(text=f'prefix-{seed}-{i}',generated_tokens=2,seconds=.1) for i in range(n)] for seed in seeds]
        gen=Gen();groups=sample_branch_batch(gen,[dict(prompt='P',seed=42),dict(prompt='Q',seed=100)],2,3)
        self.assertEqual(gen.calls[0][2],[42,100])
        self.assertEqual(gen.calls[1][2],[1051,1052,1109,1110])
        self.assertEqual([len(g['branches']) for g in groups],[2,2])
        self.assertTrue(all(len(b['answers'])==3 for g in groups for b in g['branches']))
        self.assertTrue(all(b['branch_prompt'].startswith('Qprefix-100') for b in groups[1]['branches']))

    def test_cache_requires_identical_sampling_and_prompt(self):
        import tempfile
        from pathlib import Path
        from evaluate import identical_input_cache
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root=Path(folder).resolve()
            assert root.is_relative_to(Path(__file__).resolve().parent)
            settings=dict(adapter_sha256='model',seed=42,n=2,thinking=False,profile='main',split='test')
            (root/'a.settings.json').write_text(json.dumps(settings))
            record=dict(sample_id='row',prompt_sha256='prompt',outputs=[dict(text=A),dict(text=B)])
            (root/'a.jsonl').write_text(json.dumps(record)+'\n')
            cache=identical_input_cache(root,settings,root/'b.jsonl')
            self.assertIn(('row','prompt'),cache)
            self.assertNotIn(('row','different prompt'),cache)
            for change in (dict(n=1),dict(seed=43),dict(adapter_sha256='other'),dict(thinking=True),dict(runtime_id='dsj5090:gpu0')):
                self.assertFalse(identical_input_cache(root,dict(settings,**change),root/'b.jsonl'))
            self.assertFalse(identical_input_cache(root,settings,root/'a.jsonl'))

    def test_thinking_models_keep_separate_outputs_and_validation_thresholds(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import evaluate
        from data import write_jsonl
        rows=[dict(sample_id=str(i),group_id=str(i),targets=[A],history=[{}]*i) for i in range(4)]
        cfg={'thinkcost':{'budgets':[0,128],'routing_fractions':[.5]}}
        calls=[]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root=Path(folder)
            def fake_eval(**kwargs):
                calls.append(kwargs)
                path=root/'predictions_v2/main'/(kwargs['tag']+'.jsonl')
                predictions=[dict(sample_id=r['sample_id'],group_id=r['group_id'],outputs=[
                    dict(text=A,generated_tokens=4,seconds=.1)]*kwargs['samples']) for r in rows]
                write_jsonl(path,predictions)
                summary,_=evaluate.metrics(rows,predictions,{A,B},ks=(8,32))
                summary.update(raw_predictions=str(path),settings=dict(variant=kwargs['variant'],n=kwargs['samples']))
                return summary
            with patch.multiple(evaluate,CODE=root,ROOT=root),patch.object(evaluate,'config',return_value=cfg),\
                 patch.object(evaluate,'load_split',return_value=rows),patch.object(evaluate,'load_catalog',return_value={A,B}),\
                 patch.object(evaluate,'run_eval_process',side_effect=fake_eval):
                evaluate.thinking_experiments(42,'validation','main')
                sft_path=root/'artifacts/thinking_selection_main_42.json'
                sft_selection=sft_path.read_bytes()
                with self.assertRaisesRegex(RuntimeError,'Freeze validation'):
                    evaluate.thinking_experiments(42,'test','main','rl_nk_prefix')
                evaluate.thinking_experiments(42,'validation','main','rl_nk_prefix')
                rl_path=root/'artifacts/thinking_selection_main_rl_nk_prefix_42.json'
                frozen=rl_path.read_bytes()
                evaluate.thinking_experiments(42,'test','main','rl_nk_prefix')
                self.assertEqual(rl_path.read_bytes(),frozen)
                self.assertEqual(sft_path.read_bytes(),sft_selection)
                self.assertTrue(all(c['variant']=='rl_nk_prefix' for c in calls[4:]))
                self.assertTrue(all('main_rl_nk_prefix_' in c['tag'] for c in calls[4:]))
                route=json.loads((root/'reports/results_v2/main_rl_nk_prefix_route_42_test_history_length_0.5.json').read_text())
                self.assertEqual(route['attempts'],4*32)
                self.assertEqual(route['generated_tokens'],4*32*4+2*8*4)
                self.assertEqual(route['settings']['variant'],'rl_nk_prefix')

    def test_batched_budget_keeps_seeds_and_token_accounting(self):
        from model import Generator
        gen=Generator.__new__(Generator);gen.cfg={'generation':{'direct_tokens':32}}
        gen.raw=lambda *a,**k: [dict(text='reason1',generated_tokens=2,seconds=1,stop_reason='</think>'),
                                dict(text='reason2',generated_tokens=3,seconds=1,stop_reason=None)]
        calls=[]
        def raw_many(prompts,n,max_tokens,seeds,temperature=None):
            calls.append((prompts,n,max_tokens,seeds))
            return [[dict(text=A,generated_tokens=4,seconds=.5)] for _ in prompts]
        gen.raw_many=raw_many
        outputs=gen.generate('PROMPT',2,42,True,128)
        self.assertEqual(calls[0][3],[100045,100046])
        self.assertEqual(calls[0][1:3],(1,32))
        self.assertEqual([o['generated_tokens'] for o in outputs],[6,7])
        self.assertEqual([o['forced_close'] for o in outputs],[False,True])
        self.assertEqual(sum(o['seconds'] for o in outputs),3)
        self.assertTrue(all(parse_answer(o['text'],{A})[0]==A for o in outputs))

    def test_standard_32_probe_counts_one_reward_group(self):
        from train import reward_diagnostics
        outputs=[dict(text=A,generated_tokens=4,seconds=.1)]+[dict(text=B,generated_tokens=4,seconds=.1)]*31
        report=reward_diagnostics([dict(sample_id='a',outputs=outputs)],
                    [dict(sample_id='a',targets=[A],train_target=A)],{A,B},'std32')
        self.assertEqual(report['groups'],1)
        self.assertEqual(report['exact_hit_rate'],1/32)
        self.assertEqual(report['active_group_fraction'],1)
        self.assertEqual(report['generated_tokens'],128)

    def test_label_boundary_and_fixed_attempts(self):
        self.assertEqual(parse_answer('<think>'+A+'</think>'+B,{A,B})[0],B)
        for bad in ('<think>'+A, A+B, V, A+'<s_a_4>', A.replace('s_a_1','s_a_8192')):
            self.assertIsNone(parse_answer(bad,{A,B})[0])
        row=example()
        outputs=[dict(text='bad'),dict(text=A),dict(text=A),dict(text=B)]
        summary,_=metrics([row],[dict(sample_id=row['sample_id'],outputs=outputs)],{A,B},ks=(1,3,4))
        self.assertEqual(summary['recall@3'],0)
        self.assertEqual(summary['recall@4'],1)
        self.assertEqual(summary['invalid_rate'],.25)
        self.assertEqual(summary['duplicate_rate'],.25)

    def test_target_mutation_and_whole_event_crop(self):
        row=example();tok=ToyTokenizer();view=input_view(row)
        original=prompt_text(view,tok,[0,1],False)
        mutated=dict(row,targets=['poison'],train_target='poison',reasoning='poison')
        self.assertEqual(original,prompt_text(input_view(mutated),tok,[0,1],False))
        for strategy in ('recent','domain_recent','balanced'):
            self.assertEqual(select_history(view,tok,9999,strategy),[0,1])
            budget=len(tok.encode(original))-len(A)+2
            chosen=select_history(view,tok,budget,strategy)
            self.assertLessEqual(len(prompt_text(view,tok,chosen,False)),budget)
            self.assertEqual(chosen,sorted(chosen))
        self.assertEqual(split_for(row['group_id']),split_for(mutated['group_id']))

    def test_history_gate(self):
        report={'length_gate':{'test':{'trimmed_fraction':{'512':.7,'1024':.4,'2048':.1}}}}
        self.assertEqual(history_gate(report),1024)
        report['length_gate']['test']['trimmed_fraction']={str(k):0 for k in (512,1024,2048)}
        with self.assertRaises(RuntimeError):history_gate(report)

    def test_two_prompt_branch_credit_and_reward(self):
        from train import branch_advantages,reward,branch_rewards,branch_segments
        shared,individual=branch_advantages([[1,0],[0,0]])
        self.assertAlmostEqual(shared[0],(individual[0][0]+individual[0][1])/2)
        self.assertAlmostEqual(sum(sum(r) for r in individual),0)
        self.assertGreater(shared[0],0);self.assertLess(shared[1],0)
        flat_shared,flat_individual=branch_advantages([[.2,.2],[.2,.2]])
        self.assertEqual(flat_shared,[0,0]);self.assertEqual(flat_individual,[[0,0],[0,0]])
        prefix2='<|prod_begin|><s_a_1><s_b_2><s_c_7>'
        prefix1='<|prod_begin|><s_a_1><s_b_9><s_c_7>'
        catalog={A,B,prefix1,prefix2}
        self.assertEqual([reward(s,[A],catalog,True) for s in (A,prefix2,prefix1,B)],[1,.5,.2,0])
        self.assertEqual(reward('<think>'+A,[A],catalog,True),0)
        self.assertEqual(reward(prefix2,[A],{A},True),0)
        branches=[{'answers':[{'text':A},{'text':prefix2}]}]
        self.assertEqual(branch_rewards(branches,[A],catalog,'nk_prefix_div'),[[0,0]])
        branches=[dict(prefix={'text':'abc'},branch_prompt='PROMPTabc</think>',answers=[dict(token_ids=['x']),dict(token_ids=['y'])]) for _ in range(2)]
        segments=branch_segments('PROMPT',branches,[[1,0],[0,0]],ToyTokenizer())
        self.assertEqual(sum(len(s[1]) for s in segments),10)
        self.assertEqual([s[3] for s in segments].count('reasoning'),2)
        self.assertTrue(all('PROMPT' not in s[1] for s in segments))


if __name__=='__main__':
    unittest.main()
