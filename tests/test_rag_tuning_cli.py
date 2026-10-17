"""操作入口仅消费已有记录；假本地模型不能绕过持久化门禁。"""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from agent.rag.models import ModelUnavailable
from agent.rag.tuning.policy import fingerprint
from eval.rag_tuning import main


IDENTITY = {'model_revision': 'test-model', 'runtime_revision': 'test-runtime',
            'tokenizer_revision': 'test-tokenizer', 'prompt_revision': 'test-prompt'}


class FakeModels:
    identity = IDENTITY
    choice = 'c1'
    calls = []

    def __init__(self, path=None):
        self.path = path

    def laya_provider(self):
        return self

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {'answers': {'decision': {'choice': self.choice}}}


class RAGTuningCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'tuning.sqlite3'
        FakeModels.calls = []
        FakeModels.choice = 'c1'
        self.metrics = self.save('metrics', {'recall_at_100': 0.8, 'failed_query_ids': ['opt-1']})

    def save(self, name, data):
        path = self.root / f'{name}.json'
        path.write_text(json.dumps(data), encoding='utf-8')
        return path

    def run_cli(self, *arguments, factory=FakeModels):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(['--db', str(self.db), *map(str, arguments)],
                        models_factory=factory, clock=lambda: 100_000)
        return code, json.loads(output.getvalue())

    def propose(self):
        code, proposal = self.run_cli('propose', '--metrics', self.metrics)
        self.assertEqual(code, 0, proposal)
        return proposal

    def spec(self):
        return {
            'queries': {f'q{i}': ['hard_case', 'no_answer', 'permission_boundary']
                        if i == 0 else ['ordinary'] for i in range(100)},
            'optimization_ids': ['opt-1'], 'dataset_sha256': fingerprint('holdout'),
            'qualification_dataset_sha256': fingerprint('human-set'),
            'primary_metric': 'ndcg', 'min_effect': 0.01, 'bootstrap_samples': 200,
            'judge_identity': IDENTITY, 'generation_identity': {'model': 'recorded-work-model'},
        }

    def shadow(self, proposal):
        code, result = self.run_cli('shadow', '--proposal', self.save('proposal', proposal),
                                   '--spec', self.save('spec', self.spec()))
        self.assertEqual(code, 0, result)
        return result['trial']

    def test_status_does_not_initialize_database_or_call_model(self):
        code, state = self.run_cli('status')
        self.assertEqual(code, 0)
        self.assertFalse(state['initialized'])
        self.assertFalse(self.db.exists())
        self.assertEqual(FakeModels.calls, [])

    def test_proposal_only_selects_program_candidate_and_does_not_activate(self):
        proposal = self.propose()
        self.assertEqual(proposal['choice'], 'c1')
        self.assertEqual(len(FakeModels.calls), 1)
        options = FakeModels.calls[0][1]['decision']['criteria']
        self.assertIn('no_change', options)
        self.assertLess(len(options), 16)
        _, status = self.run_cli('status')
        self.assertIsNone(status['state']['canary'])
        self.assertEqual(status['trials'], [])

    def test_missing_local_model_is_explicit_and_does_not_fake_a_proposal(self):
        def absent(_):
            raise ModelUnavailable('No qualified local model is available')

        code, result = self.run_cli('propose', '--metrics', self.metrics, factory=absent)
        self.assertEqual(code, 2)
        self.assertNotIn('candidate_version', result)
        self.assertFalse(result['paid_generation_started'])

    def test_tampered_or_no_change_proposal_cannot_create_shadow_trial(self):
        proposal = self.propose()
        proposal['values']['vector_weight'] = 0.9
        code, _ = self.run_cli('shadow', '--proposal', self.save('tampered', proposal),
                               '--spec', self.save('spec', self.spec()))
        self.assertEqual(code, 2)
        FakeModels.choice = 'no_change'
        proposal = self.propose()
        code, _ = self.run_cli('shadow', '--proposal', self.save('unchanged', proposal),
                               '--spec', self.save('spec', self.spec()))
        self.assertEqual(code, 2)

    def test_missing_metrics_fail_evaluation_and_canary_gate(self):
        trial = self.shadow(self.propose())
        code, report = self.run_cli('evaluate', '--trial', trial, '--rows', self.save('rows', []),
                                    '--qualification', self.save('qualification', {}))
        self.assertEqual(code, 2)
        self.assertFalse(report['passed'])
        code, result = self.run_cli('canary', '--trial', trial)
        self.assertEqual(code, 2)
        self.assertIn('passing shadow', result['error'])

    def test_existing_failed_trial_reports_its_actual_status(self):
        proposal = self.propose()
        trial = self.shadow(proposal)
        self.run_cli('evaluate', '--trial', trial, '--rows', self.save('rows', []),
                     '--qualification', self.save('qualification', {}))
        code, result = self.run_cli('shadow', '--proposal', self.root / 'proposal.json',
                                    '--spec', self.root / 'spec.json')
        self.assertEqual(code, 0)
        self.assertEqual(result['status'], 'shadow_failed')
        self.assertEqual(result['trial'], trial)

    def activate_recorded_candidate(self):
        proposal = self.propose()
        trial = self.shadow(proposal)
        spec = self.spec()
        labels = ('supported', 'contradicted', 'unknown')
        qualification = {
            'human_labeled': True, 'dataset_sha256': spec['qualification_dataset_sha256'],
            'judge_identity': IDENTITY,
            'cases': [{'id': f'h{i}', 'expected': labels[i % 3], 'predicted': labels[i % 3]}
                      for i in range(120)],
        }
        rows = []
        for query_id in spec['queries']:
            row = {'query_id': query_id, 'dataset_sha256': spec['dataset_sha256']}
            for side, version in (('baseline', proposal['base_version']),
                                  ('candidate', proposal['candidate_version'])):
                row[side] = {
                    'version': version, 'generator': spec['generation_identity'],
                    'judge_identity': IDENTITY, 'input_sha256': fingerprint([version, query_id]),
                    'answer_sha256': fingerprint(['answer', query_id]),
                    'record_sha256': fingerprint([side, query_id]), 'usage_known': True,
                    'permission_pass': True, 'safety_pass': True,
                    'ndcg': 0.8 if side == 'baseline' else 0.9, 'faithfulness': 1.0,
                    'citation_precision': 1.0, 'latency_ms': 10, 'memory_bytes': 50_000_000,
                    'cost': 0.01,
                }
            rows.append(row)
        code, report = self.run_cli('evaluate', '--trial', trial,
                                    '--rows', self.save('rows', rows),
                                    '--qualification', self.save('qualification', qualification))
        self.assertEqual(code, 0, report)
        self.assertTrue(report['passed'])
        code, result = self.run_cli('canary', '--trial', trial)
        self.assertEqual(code, 0, result)
        self.assertEqual(result['new_task_percentage'], 5)
        return trial

    def test_recorded_paired_evaluation_canary_and_rollback(self):
        trial = self.activate_recorded_candidate()
        code, result = self.run_cli('rollback', '--trial', trial, '--reason', 'manual_review')
        self.assertEqual((code, result['status']), (0, 'rolled_back'))
        self.assertEqual(len(FakeModels.calls), 1)

    def test_unreadable_canary_evaluation_record_rolls_back(self):
        trial = self.activate_recorded_candidate()
        broken = self.root / 'invalid.json'
        broken.write_text('{', encoding='utf-8')
        code, report = self.run_cli('evaluate', '--phase', 'canary', '--trial', trial,
                                    '--rows', broken,
                                    '--qualification', self.root / 'qualification.json')
        self.assertEqual(code, 2)
        self.assertEqual(report['status'], 'error')
        _, status = self.run_cli('status')
        self.assertIsNone(status['state']['canary'])
        self.assertEqual(status['trials'][0]['status'], 'rolled_back')

    def test_existing_output_is_not_overwritten_and_model_is_not_called(self):
        output = self.save('existing', {'retain': True})
        before = output.read_bytes()
        code, _ = self.run_cli('--output', output, 'propose', '--metrics', self.metrics)
        self.assertEqual(code, 2)
        self.assertEqual(output.read_bytes(), before)
        self.assertEqual(FakeModels.calls, [])


if __name__ == '__main__':
    unittest.main()
