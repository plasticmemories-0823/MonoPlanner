"""调参候选、冻结门禁、任务绑定与 Laya 薄适配的离线回归。"""

from copy import deepcopy
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from agent.model.errors import ModelInterrupted
from agent.rag.tuning.controller import (
    TuningController, evaluate_gate, freeze_spec, qualification_errors,
)
from agent.rag.tuning.laya import LayaAdapter
from agent.rag.tuning.policy import Config, enumerate_candidates, fingerprint, validate_candidate


def initial_config():
    return Config.create({
        'recall_top_k': 100, 'rewrite_threshold': 0.15,
        'chunk_target': 256, 'memory_limit_bytes': 1024 ** 3, 'message_token_limit': 8192,
        'embedding_revision': 'frozen-model', 'acl_revision': 'frozen-acl',
    })


def evaluation_spec():
    return freeze_spec({
        'queries': {f'q{index}': ['hard_case', 'no_answer', 'permission_boundary']
                    if index == 0 else ['ordinary'] for index in range(100)},
        'optimization_ids': ['optimization-1'], 'dataset_sha256': fingerprint('held-out-data'),
        'qualification_dataset_sha256': fingerprint('human-labeled-data'),
        'primary_metric': 'ndcg', 'min_effect': 0.01, 'bootstrap_samples': 200,
        'judge_identity': {'model': 'local-model', 'runtime': 'fixed-runtime',
                           'tokenizer': 'fixed-tokenizer', 'prompt': 'fixed-question'},
        'generation_identity': {'model': 'frozen-work-model', 'prompt': 'evaluation-v1'},
        'max_trials': 2,
    })


def qualification(spec):
    labels = ('supported', 'contradicted', 'unknown')
    return {
        'human_labeled': True, 'dataset_sha256': spec['qualification_dataset_sha256'],
        'judge_identity': spec['judge_identity'],
        'cases': [{'id': f'h{index}', 'expected': labels[index % 3],
                   'predicted': labels[index % 3]} for index in range(120)],
    }


def measurements(base, candidate, spec):
    rows = []
    for query_id in spec['queries']:
        row = {'query_id': query_id, 'dataset_sha256': spec['dataset_sha256']}
        for side, config in (('baseline', base), ('candidate', candidate)):
            row[side] = {
                'version': config.version, 'generator': spec['generation_identity'],
                'judge_identity': spec['judge_identity'],
                'input_sha256': fingerprint([query_id, config.version, 'input']),
                'answer_sha256': fingerprint([query_id, 'same-answer-is-allowed']),
                'record_sha256': fingerprint([query_id, config.version, 'record']),
                'usage_known': True, 'permission_pass': True, 'safety_pass': True,
                'ndcg': 0.8 if side == 'baseline' else 0.9,
                'faithfulness': 1.0, 'citation_precision': 1.0,
                'latency_ms': 100 if side == 'baseline' else 90,
                'memory_bytes': 400_000_000, 'cost': 0.01,
            }
        rows.append(row)
    return rows


class RAGTuningPolicyTests(unittest.TestCase):
    def test_candidates_are_single_steps_and_cannot_change_fixed_policy(self):
        base = initial_config()
        candidates = enumerate_candidates(base)
        self.assertEqual(candidates[0].id, 'no_change')
        for candidate in candidates[1:]:
            self.assertEqual(validate_candidate(base, candidate.config), candidate.knob)
            changed = [key for key in base.values
                       if base.values[key] != candidate.config.values[key]]
            self.assertEqual(changed, [candidate.knob])
        self.assertEqual({item.config.values['recall_top_k'] for item in candidates
                          if item.knob == 'recall_top_k'}, {90, 110})
        self.assertEqual({item.config.values['chunk_target'] for item in candidates
                          if item.knob == 'chunk_target'}, {384})

        for changes in ({'memory_limit_bytes': 2 * 1024 ** 3},
                        {'recall_top_k': 110, 'chunk_target': 384},
                        {'chunk_target': 512}, {'embedding_revision': 'new-model'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_candidate(base, Config.create(base.values | changes, base.version))

    def test_version_owns_its_values_and_rounding_never_exceeds_ten_percent(self):
        base = initial_config()
        exposed = base.values
        exposed['recall_top_k'] = 500
        self.assertEqual(base.values['recall_top_k'], 100)
        other = Config.create(base.values | {'recall_top_k': 19, 'rewrite_threshold': 1.0})
        for item in enumerate_candidates(other):
            if item.knob == 'recall_top_k':
                self.assertLessEqual(abs(item.config.values['recall_top_k'] - 19), 1.9)
            self.assertLessEqual(item.config.values['rewrite_threshold'], 1.0)
        with self.assertRaises(ValueError):
            Config.create(base.values | {'recall_top_k': True})


class RAGTuningGateTests(unittest.TestCase):
    def setUp(self):
        self.base = initial_config()
        self.candidate = enumerate_candidates(self.base)[1].config
        self.spec = evaluation_spec()
        self.human = qualification(self.spec)
        self.rows = measurements(self.base, self.candidate, self.spec)

    def gate(self, rows=None, human=None):
        return evaluate_gate(self.spec, self.rows if rows is None else rows,
                             self.human if human is None else human,
                             self.base.version, self.candidate.version)

    def test_reproducible_paired_gain_with_complete_provenance_passes(self):
        first = self.gate()
        reversed_result = self.gate(list(reversed(self.rows)))
        self.assertTrue(first['passed'], first)
        self.assertEqual(first['intervals'], reversed_result['intervals'])

    def test_missing_data_unknown_usage_and_wrong_config_fail_closed(self):
        variants = []
        for key, value in (('usage_known', False), ('version', self.base.version),
                           ('cost', None), ('permission_pass', False),
                           ('memory_bytes', 1024 ** 3 + 1), ('ndcg', float('inf'))):
            rows = deepcopy(self.rows)
            rows[0]['candidate'][key] = value
            variants.append((key, rows))
        variants.extend([
            ('missing-query', self.rows[:-1]), ('duplicate', self.rows + self.rows[:1]),
        ])
        for label, rows in variants:
            with self.subTest(label=label):
                if label == 'ndcg':
                    with self.assertRaises(ValueError):
                        self.gate(rows)
                else:
                    self.assertFalse(self.gate(rows)['passed'])

    def test_old_generation_record_cannot_be_reused_for_new_configuration(self):
        self.rows[0]['candidate']['record_sha256'] = self.rows[0]['baseline']['record_sha256']
        self.assertIn('q0:candidate_reuses_baseline_generation', self.gate()['reasons'])

    def test_small_gain_or_quality_and_resource_regression_do_not_pass(self):
        for metric, value in (('ndcg', 0.802), ('faithfulness', 0.99),
                              ('citation_precision', 0.99), ('latency_ms', 101),
                              ('memory_bytes', 400_000_001), ('cost', 0.0101)):
            rows = deepcopy(self.rows)
            for row in rows:
                row['candidate'][metric] = value
            with self.subTest(metric=metric):
                self.assertFalse(self.gate(rows)['passed'])

    def test_qualification_is_human_labeled_version_bound_and_covers_all_classes(self):
        for key, value in (('human_labeled', False), ('judge_identity', {'model': 'another'}),
                           ('dataset_sha256', fingerprint('another-set')), ('cases', [])):
            human = deepcopy(self.human)
            human[key] = value
            self.assertTrue(qualification_errors(self.spec, human))
        human = deepcopy(self.human)
        for case in human['cases']:
            if case['expected'] == 'unknown':
                case['predicted'] = 'supported'
        self.assertIn('qualification_class_accuracy_failed', qualification_errors(self.spec, human))

    def test_frozen_holdout_requires_scope_and_optimization_separation(self):
        for changes in ({'queries': {'one': ['ordinary']}},
                        {'optimization_ids': ['q0']}, {'min_effect': 0},
                        {'primary_metric': 'best-after-looking'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                freeze_spec(self.spec | changes)


class RAGTuningPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'tuning.sqlite3'
        self.base = initial_config()
        self.candidate = enumerate_candidates(self.base)[1].config
        self.spec = evaluation_spec()
        self.now = 100_000.0
        self.controller = TuningController(self.path, self.base, lambda: self.now)

    def passing_shadow(self):
        trial = self.controller.start_shadow(self.candidate, self.spec)
        rows = measurements(self.base, self.candidate, self.spec)
        result = self.controller.evaluate_shadow(trial, rows, qualification(self.spec))
        self.assertTrue(result['passed'], result)
        return trial, rows

    def test_shadow_has_no_effect_then_stable_canary_survives_restart_and_rollback(self):
        original = self.controller.bind_task('running-before-canary')
        trial, _ = self.passing_shadow()
        self.assertEqual(self.controller.bind_task('during-shadow').version, self.base.version)
        self.controller.activate_canary(trial, self.spec['judge_identity'])
        bindings = {f'task-{index}': self.controller.bind_task(f'task-{index}').version
                    for index in range(200)}
        canary_tasks = [key for key, value in bindings.items() if value == self.candidate.version]
        self.assertTrue(canary_tasks)
        self.assertLess(len(canary_tasks), 30)
        restarted = TuningController(self.path, self.base, lambda: self.now)
        self.assertEqual(restarted.bind_task('running-before-canary'), original)
        for task_id, version in bindings.items():
            self.assertEqual(restarted.bind_task(task_id).version, version)
        self.assertTrue(restarted.rollback(trial, 'resource_budget_exceeded'))
        self.assertEqual(restarted.bind_task('new-after-rollback').version, self.base.version)
        self.assertEqual(restarted.bind_task(canary_tasks[0]).version, self.candidate.version)
        self.assertFalse(restarted.rollback('stale-trial', 'late_failure'))

    def test_failed_shadow_cannot_activate_and_results_cannot_be_overwritten(self):
        trial = self.controller.start_shadow(self.candidate, self.spec)
        report = self.controller.evaluate_shadow(trial, [], None)
        self.assertFalse(report['passed'])
        with self.assertRaises(ValueError):
            self.controller.activate_canary(trial, self.spec['judge_identity'])
        with self.assertRaises(ValueError):
            self.controller.evaluate_shadow(trial, [], None)

    def test_regression_review_rolls_back_and_cooldown_is_persistent(self):
        trial, rows = self.passing_shadow()
        self.controller.activate_canary(trial, self.spec['judge_identity'])
        rows[0]['candidate']['safety_pass'] = False
        self.assertFalse(self.controller.review_canary(
            trial, rows, qualification(self.spec)
        )['passed'])
        self.assertEqual(self.controller.bind_task('after-failure').version, self.base.version)

        self.candidate = enumerate_candidates(self.base)[2].config
        next_trial, _ = self.passing_shadow()
        restarted = TuningController(self.path, self.base, lambda: self.now)
        with self.assertRaises(ValueError):
            restarted.activate_canary(next_trial, self.spec['judge_identity'])
        self.now += 86_400
        restarted.activate_canary(next_trial, self.spec['judge_identity'])

    def test_changed_judge_and_tampered_configuration_are_rejected(self):
        trial, _ = self.passing_shadow()
        with self.assertRaises(ValueError):
            self.controller.activate_canary(trial, {'model': 'changed'})
        with closing(sqlite3.connect(self.path)) as connection:
            with connection:
                connection.execute('UPDATE configs SET parent=? WHERE version=?',
                                   ('tampered', self.base.version))
        with self.assertRaises(ValueError):
            self.controller.bind_task('not-trusted')

    def test_invalid_canary_metrics_roll_back_before_propagating_error(self):
        trial, rows = self.passing_shadow()
        self.controller.activate_canary(trial, self.spec['judge_identity'])
        rows[0]['candidate']['cost'] = float('nan')
        with self.assertRaises(ValueError):
            self.controller.review_canary(trial, rows, qualification(self.spec))
        self.assertEqual(self.controller.trial_status(trial), 'rolled_back')
        self.assertEqual(self.controller.bind_task('after-invalid-record').version,
                         self.base.version)


class FakeProvider:
    identity = {'model': 'offline-fake', 'runtime': 'test', 'tokenizer': 'test', 'prompt': 'test'}

    def __init__(self, choice='no_change'):
        self.choice = choice
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {'answers': {'decision': {'choice': self.choice}}}


class LayaAdapterTests(unittest.TestCase):
    def test_unavailable_identity_is_read_once_and_does_not_dispatch(self):
        class MissingIdentity(FakeProvider):
            reads = 0

            @property
            def identity(self):
                self.reads += 1
                raise RuntimeError('Optional inference runtime is not installed')

        provider = MissingIdentity()
        decision = LayaAdapter(provider).choose({}, {'no_change': 'Keep current'})
        self.assertEqual(decision.status, 'unavailable')
        self.assertEqual(decision.reason, 'local_model_identity_unavailable')
        self.assertEqual(provider.reads, 1)
        self.assertEqual(provider.calls, [])

    def test_pause_during_identity_or_prediction_is_not_swallowed(self):
        class PausedIdentity(FakeProvider):
            @property
            def identity(self):
                raise ModelInterrupted('pause', sent=False)

        class PausedPrediction(FakeProvider):
            def predict(self, state, questions):
                raise ModelInterrupted('pause', sent=False)

        for provider in (PausedIdentity(), PausedPrediction()):
            with self.subTest(provider=type(provider).__name__):
                with self.assertRaises(ModelInterrupted):
                    LayaAdapter(provider).choose({}, {'no_change': 'Keep current'})

    def test_prediction_failure_reuses_the_verified_identity(self):
        class FailingPrediction(FakeProvider):
            reads = 0

            @property
            def identity(self):
                self.reads += 1
                if self.reads > 1:
                    raise AssertionError('Identity must not be re-read during failure handling')
                return {'model_revision': 'test-model'}

            def predict(self, state, questions):
                raise RuntimeError('Model loading failed')

        provider = FailingPrediction()
        decision = LayaAdapter(provider).choose({}, {'no_change': 'Keep current'})
        self.assertEqual(decision.status, 'unavailable')
        self.assertEqual(decision.identity, {'model_revision': 'test-model'})
        self.assertEqual(provider.reads, 1)

    def test_unavailable_does_not_install_or_call_any_provider(self):
        self.assertEqual(LayaAdapter().choose({}, {'no_change': 'Keep current'}).status,
                         'unavailable')

    def test_only_predefined_choices_are_accepted(self):
        provider = FakeProvider()
        adapter = LayaAdapter(provider)
        self.assertEqual(adapter.choose({'recall': 0.8}, {'no_change': 'Keep current'}).choice,
                         'no_change')
        provider.choice = 'change_acl'
        self.assertEqual(adapter.choose({}, {'no_change': 'Keep current'}).status, 'invalid')
        provider.choice = 'supported'
        self.assertEqual(adapter.classify('source text', 'claim').choice, 'supported')
        self.assertEqual(provider.calls[-1][0], {'evidence': 'source text', 'claim': 'claim'})

    def test_large_input_and_missing_no_change_do_not_dispatch(self):
        provider = FakeProvider()
        adapter = LayaAdapter(provider)
        self.assertEqual(adapter.choose({}, {'c1': 'Change'}).status, 'invalid')
        self.assertEqual(adapter.choose({'large': 'x' * 70_000},
                                        {'no_change': 'Keep'}).status, 'unavailable')
        self.assertEqual(provider.calls, [])


if __name__ == '__main__':
    unittest.main()
