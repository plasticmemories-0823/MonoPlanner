"""迁移旧轮数测试中的原件、崩溃和不重放保障，不保留轮数停机。"""

import json
import unittest
from unittest.mock import patch

from agent.storage.history import LocalHistory
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from support import interrupt_after_persist
import test_task_runner as fixtures


class ReworkRecoveryTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    candidate = fixtures.TaskRunnerTests.candidate
    shell = fixtures.TaskRunnerTests.shell

    def run_plans(self, plans, **options):
        self.model = fixtures.Model(plans)
        self.runner = TaskRunner(self.model, self.env, state_dir=self.root / 'state', **options)
        return self.runner.run([], 'return 4', StepBudget(max_steps=12))

    def change_state(self, change):
        store = LocalHistory(self.root / 'state')
        try:
            with store.lease():
                state = store.load()
                change(state)
                store.record(state, 'offline_checkpoint', {})
                return state
        finally:
            store.close()

    def test_finished_acceptance_commit_never_replays_on_restart(self):
        for value in (2, 4):
            with self.subTest(value=value):
                self.setUp()
                interrupt, error = interrupt_after_persist('final_acceptance_finished')
                with interrupt, self.assertRaises(error):
                    self.run_plans([self.candidate(value)])
                with patch('agent.task.operations.verify_final', side_effect=AssertionError('replay')):
                    result = self.run_plans([], yield_boundary=True)
                self.assertEqual(result['status'], 'verified_success' if value == 4 else 'needs_plan')
                self.assertEqual(result['calls_this_invocation'], 0)
                self.assertEqual(len(result['rounds']), 1)
                self.assertIsNone(self.runner.state['pending'])

    def test_completed_edits_survive_user_pause_before_acceptance(self):
        first = self.run_plans([self.candidate()], pause_after_steps=1)
        self.assertEqual(first['status'], 'paused')
        before = self.change_state(lambda state: None)
        with patch('agent.execution.executor.OperationExecutor.execute',
                   side_effect=AssertionError('completed edit replayed')):
            result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success')
        self.assertEqual(result['calls_this_invocation'], 0)
        self.assertEqual(result['planner_calls'], before['budget']['used_steps'])
        self.assertEqual(len(result['rounds']), 1)

    def test_unknown_operation_does_not_turn_into_ordinary_rework(self):
        self.run_plans([self.candidate()], pause_after_steps=1)
        self.change_state(lambda state: state.update(running_step={'id': 'unknown'}, pending=None))
        result = self.run_plans([])
        self.assertEqual(result['status'], 'execution_unknown')
        self.assertEqual(result['calls_this_invocation'], 0)

    def test_changed_source_stops_before_rework(self):
        self.run_plans([self.candidate()], pause_after_steps=1)
        (self.repo / 'sample.py').write_text('external modification')
        result = self.run_plans([])
        self.assertEqual(result['status'], 'state_conflict')
        self.assertEqual(result['calls_this_invocation'], 0)

    def test_changed_frozen_contract_stops_before_rework(self):
        self.run_plans([self.candidate()], pause_after_steps=1)
        path = self.root / 'state/frozen-acceptance.json'
        saved = json.loads(path.read_text())
        saved['regression'] = 'changed'
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, 'Frozen acceptance'):
            self.run_plans([])

    def test_unknown_request_remains_reserved_across_restarts(self):
        self.run_plans([self.candidate()], pause_after_steps=1)
        def pending(state):
            state['budget']['request_sequence'] += 1
            state['budget']['used_steps'] += 1
            state['budget']['inflight_reserved_tokens'] += 100
            state['request_pending'] = {
                'call': state['budget']['request_sequence'], 'reservation': 100, 'purpose': 'plan',
            }
        self.change_state(pending)
        for _ in range(2):
            result = self.run_plans([])
            self.assertEqual(result['status'], 'model_request_unknown', result)
            self.assertEqual(result['calls_this_invocation'], 0)
            self.assertEqual(self.runner.budget.unknown_reserved_tokens, 100)

    def test_committed_failed_command_is_not_replayed_or_sent_to_acceptance(self):
        candidate = self.candidate()
        candidate['steps'] = [self.shell('import sys; sys.exit(3)')]
        interrupt, error = interrupt_after_persist('step_finished')
        with interrupt, self.assertRaises(error):
            self.run_plans([candidate])
        with patch('agent.execution.executor.OperationExecutor.execute',
                   side_effect=AssertionError('command replayed')), patch(
                       'agent.task.operations.verify_final', side_effect=AssertionError('acceptance')):
            result = self.run_plans([], yield_boundary=True)
        self.assertEqual(result['status'], 'needs_plan')
        self.assertEqual(self.runner.state['last_result']['outcome'], 'execution_failed')
        self.assertEqual(result['calls_this_invocation'], 0)


def test_freeze_feedback_is_specific_without_changing_attribution_guard():
    from agent.acceptance.contracts import tests_ready_for_candidate
    from agent.acceptance.revisions import freeze_feedback

    report = {'status': 'failed', 'test_ids': ['check'], 'test_results': [
        {'id': 'check', 'status': 'failed', 'exceptions': [
            {'phase': 'unknown', 'origin': 'external', 'ordinary_exception': True}]}],
        'test_evidence_valid': True, 'integrity_ok': True, 'tests_run': 1}
    assert not tests_ready_for_candidate(report)
    assert 'phase=unknown' in freeze_feedback(report)
    assert 'exec/eval' in freeze_feedback(report)
    report['test_results'][0]['exceptions'][0]['phase'] = 'test'
    assert tests_ready_for_candidate(report)


def test_default_call_count_has_no_cutoff_but_explicit_cap_is_enforced():
    from agent.model.errors import ModelRequestNotSent
    import pytest

    budget = StepBudget()
    budget.used_steps = 1000
    ticket = budget.reserve_request()
    assert budget.used_steps == 1001
    budget.cancel_request(ticket)
    restricted = StepBudget(max_steps=1, used_steps=1)
    with pytest.raises(ModelRequestNotSent):
        restricted.reserve_request()
