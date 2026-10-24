import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate
from ops.checks import average_offsets, bootstrap, stability


class ChecksTests(unittest.TestCase):
    def test_separate_output_paths_resume_without_touching_legacy(self):
        sid='<|prod_begin|><s_a_1><s_b_2><s_c_3>'
        rows=[dict(sample_id=str(i),group_id=str(i),targets=[sid],history=[],context_text='') for i in range(2)]
        calls=[]
        class Tokenizer:
            def encode(self,*args,**kwargs):return [1]
        class Generator:
            def __init__(self,*args):pass
            def generate(self,prompt,n,seed,*args):
                calls.append(seed)
                return [dict(text=sid) for _ in range(n)]
        with tempfile.TemporaryDirectory() as folder,contextlib.ExitStack() as stack:
            root=Path(folder);runtime=root/'checks_runtime.json';runtime.write_text(json.dumps(dict(runtime_id='test-device')))
            stack.enter_context(patch.object(evaluate,'CODE',root/'code'))
            stack.enter_context(patch.object(evaluate,'ROOT',root/'data'))
            stack.enter_context(patch.object(evaluate,'config',return_value=dict(input_budget=2048)))
            stack.enter_context(patch.object(evaluate,'load_split',return_value=rows))
            stack.enter_context(patch.object(evaluate,'load_catalog',return_value={sid}))
            stack.enter_context(patch.object(evaluate,'select_history',return_value=[]))
            stack.enter_context(patch.object(evaluate,'prompt_text',return_value='prompt'))
            stack.enter_context(patch('model.tokenizer',return_value=Tokenizer()))
            stack.enter_context(patch('model.Generator',Generator))
            (root/'code').mkdir()
            for name in ('data.py','history.py','model.py'):(root/'code'/name).write_text('source')
            kwargs=dict(variant='base',seed=42,split='test',seed_offset=1,tag='check',n=64,
                prediction_directory=root/'predictions_checks',result_directory=root/'reports_checks',runtime_identity_path=runtime)
            with contextlib.redirect_stdout(io.StringIO()):
                first=evaluate.evaluate_one(**kwargs);again=evaluate.evaluate_one(**kwargs)
            self.assertEqual(len(calls),2)
            self.assertEqual(first,again)
            self.assertEqual(first['recall@32'],1)
            self.assertEqual(first['settings']['runtime_id'],'test-device')
            self.assertTrue((root/'predictions_checks/check.metrics.jsonl').exists())
            self.assertTrue((root/'reports_checks/check.json').exists())
            self.assertFalse((root/'data/predictions_v2').exists())
            self.assertFalse((root/'code/reports/results_v2').exists())

    def test_offset_average_is_per_sample_and_bootstrap_retains_pairing(self):
        a=[dict(sample_id='a',group_id='g',**{'recall@32':0.}),dict(sample_id='b',group_id='h',**{'recall@32':0.})]
        b=[dict(sample_id='b',group_id='h',**{'recall@32':0.}),dict(sample_id='a',group_id='g',**{'recall@32':1.})]
        mean=average_offsets([a,b],keys=['recall@32'])
        self.assertEqual(mean[0]['recall@32'],.5)
        old=evaluate.paired_bootstrap(a,mean,'recall@32')
        extended=bootstrap(a,mean,'recall@32')
        self.assertEqual(old,{k:v for k,v in extended.items() if k!='bootstrap_standard_error'})
        self.assertEqual(extended['delta'],.25)
        self.assertGreater(extended['bootstrap_standard_error'],0)
        with self.assertRaises(AssertionError):average_offsets([a,b[:1]],keys=['recall@32'])

    def test_preregistered_stability_boundaries(self):
        self.assertEqual(stability(5,[.001,.02]),'稳定')
        self.assertEqual(stability(5,[0,.02]),'方向一致、幅度不确定')
        self.assertEqual(stability(4,[-.001,.02]),'方向一致、幅度不确定')
        self.assertEqual(stability(3,[-.001,.02]),'未得到支持')


if __name__=='__main__':unittest.main()
