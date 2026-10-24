import os
import tempfile
import unittest
from unittest.mock import patch

from ops.official_cache import resumable_batch


class OfficialCacheTests(unittest.TestCase):
    def test_resume_retains_completed_batches_and_does_not_cache_engine_failure(self):
        calls=[]
        class Worker:
            fail=True
            @resumable_batch
            def generate(self,prompts,params,batch_size):
                calls.append(list(prompts))
                if 'c' in prompts and self.fail:return ({'c':['']},{},{})
                return ({key:[text] for key,text in prompts.items()},{},{key:dict(output_tokens=[1]) for key in prompts})
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,MINIONEREASON_OFFICIAL_CACHE=directory):
            worker=Worker();prompts=dict(a='A',b='B',c='C')
            with self.assertRaises(RuntimeError):worker.generate(prompts,dict(n=1),2)
            worker.fail=False
            actual=worker.generate(prompts,dict(n=1),2)
            self.assertEqual(actual[0],dict(a=['A'],b=['B'],c=['C']))
            self.assertEqual(calls,[['a','b'],['c'],['c']])
            worker.generate(dict(a='changed'),dict(n=1),2)
            self.assertEqual(len(calls),4)
