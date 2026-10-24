import json
import tarfile
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from ops import final_report


class ReportArchiveTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        (self.root / 'reports').mkdir()
        (self.root / 'artifacts').mkdir()
        self.required = [
            'config.json', 'env.sh', 'requirements.txt', 'README.md',
            'artifacts/manifest.json', 'artifacts/execution_protocol.json',
            'artifacts/execution_protocol_v2.json',
        ]
        self.optional = [
            'reports/method.md', 'reports/completion_revision.md',
            'reports/case_review.md', 'reports/validation_case_review_inputs.json',
            'reports/case_input_origin_audit.json',
        ]
        for name in self.required:
            (self.root / name).write_text('{}', encoding='utf-8')
        row = {'recall@32': .1, 'recall@64': .1, 'tokens_per_example': 5,
               'attempts': 64, 'parser_counts': {'valid': 64}}
        results = {(variant, 42, 'test', mode): row
                   for variant, mode in [('direct', False), ('mixed', False), ('mixed', True)]}
        audit = dict(complete=True, seeds=[42], missing=[], deferred_jobs=[])
        supplement = dict(rl_thinkcost_complete=False, all_experiments_complete=False,
                          official=dict(status='pending'))
        for name, value in [('CODE', self.root), ('ROOT', self.root / 'data')]:
            self.stack.enter_context(patch.object(final_report, name, value))
        for name, value in [('audit_jobs', audit), ('read_main_results', results),
                            ('comparison', dict(delta=0, ci95=[0, 0]))]:
            self.stack.enter_context(patch.object(final_report, name, return_value=value))
        self.stack.enter_context(patch.object(final_report, 'delivery_assets'))
        self.stack.enter_context(patch('ops.supplement.audit_supplement', return_value=supplement))

    def archive_names(self):
        final_report.write_final_report()
        with tarfile.open(self.root / 'artifacts/completion_v2_code.tgz') as archive:
            return {name.replace('\\', '/') for name in archive.getnames()}

    def test_new_run_archives_current_reports_without_historical_attachments(self):
        names = self.archive_names()
        self.assertTrue(set(self.required).issubset(names))
        self.assertTrue({'reports/PROJECT_REPORT.md', 'reports/RESUME.md',
                         'reports/completion_status.json'}.issubset(names))
        self.assertTrue(set(self.optional).isdisjoint(names))
        status = json.loads((self.root / 'reports/completion_status.json').read_text())
        self.assertTrue(status['main_chain_complete'])

    def test_existing_historical_attachments_are_preserved(self):
        for name in self.optional:
            (self.root / name).write_text('historical evidence', encoding='utf-8')
        names = self.archive_names()
        self.assertTrue(set(self.optional).issubset(names))

    def test_missing_required_configuration_still_fails(self):
        (self.root / 'config.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.archive_names()
        self.assertFalse((self.root / 'artifacts/completion_v2_code.tgz').exists())
