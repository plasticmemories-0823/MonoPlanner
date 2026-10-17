"""候选次数用尽后完整收尾，已完成操作和验收不得因恢复重复执行。"""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from agent.storage.history import LocalHistory
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from support import interrupt_after_persist
import test_task_runner as fixtures


class CandidateLimitTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    candidate = fixtures.TaskRunnerTests.candidate
    shell = fixtures.TaskRunnerTests.shell

    def run_plans(self, plans, **options):
        self.model = fixtures.Model(plans)
        self.runner = TaskRunner(
            self.model, self.env, state_dir=self.root / 'state',
            max_rounds=options.pop('max_rounds', 1), **options,
        )
        return self.runner.run([], 'return 4', StepBudget(max_steps=8))

    def stored(self):
        store = LocalHistory(self.root / 'state')
        try:
            with store.lease():
                return store.load()
        finally:
            store.close()

    def assert_limit_result(self, state):
        self.assertEqual(state['status'], 'stalled')
        self.assertIsNone(state['pending'])
        self.assertEqual(state['reason'], 'Final candidate attempt limit reached')
        result = state['last_result']
        self.assertEqual(result['outcome'], 'candidate_limit_reached')
        self.assertEqual(result['final_acceptance'], {
            'status': 'not_run', 'reason': 'Final candidate attempt limit reached',
        })
        self.assertNotIn('verification', result)
        request, response = state['history'][-2:]
        self.assertEqual(response['type'], 'tool')
        self.assertEqual(
            request['data']['tool_calls'][0]['id'], response['data']['tool_call_id'],
        )
        self.assertEqual(json.loads(response['data']['content']), result)

    def test_limit_stops_new_candidate_before_any_step_and_pairs_its_result(self):
        result = self.run_plans([self.candidate(2), self.candidate(3)])
        self.assertEqual(result['status'], 'stalled', result)
        self.assertEqual((self.repo / 'sample.py').read_text(), 'def value():\n    return 2\n')
        state = self.stored()
        self.assert_limit_result(state)
        self.assertEqual(state['last_result']['steps'], [])
        self.assertEqual(len(state['rounds']), 1)
        self.assertEqual(len(self.model.calls), 3)

    def test_last_failed_round_still_allows_inspection_and_blocked_report(self):
        inspect = {
            'kind': 'inspect', 'note': 'Read the last failed candidate',
            'steps': [{'id': 'read', 'op': 'read', 'path': 'sample.py'}],
        }
        result = self.run_plans([
            self.candidate(2), inspect, {'kind': 'blocked', 'note': 'Cannot repair within limit'},
        ])
        self.assertEqual(result['status'], 'task_blocked', result)
        self.assertEqual(len(result['rounds']), 1)
        self.assertEqual(self.runner.state['last_result']['outcome'], 'evidence_ready')
        self.assertEqual(len(self.model.calls), 4)

    def test_last_allowed_candidate_can_pass(self):
        result = self.run_plans([self.candidate()])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(result['rounds']), 1)
        self.assertEqual(len(self.model.calls), 2)
        self.assertIsNone(self.stored()['pending'])

    def test_completed_last_acceptance_is_paired_on_resume_without_repeating_it(self):
        interrupt, Interrupted = interrupt_after_persist('final_acceptance_finished')
        with interrupt, self.assertRaises(Interrupted):
            self.run_plans([self.candidate()])
        with patch('agent.task.operations.verify_final', side_effect=AssertionError('repeated')):
            result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(result['calls_this_invocation'], 0)
        self.assertIsNone(self.stored()['pending'])
        self.assertEqual(len(result['rounds']), 1)

    def test_previously_executed_candidate_preserves_rows_without_replay(self):
        # 模拟旧版在步骤完成后才发现上限的现场；此次恢复只降低测试夹具中的上限。
        result = self.run_plans(
            [self.candidate(2), self.candidate(3)], max_rounds=2, pause_after_steps=2,
        )
        self.assertEqual(result['status'], 'paused', result)
        before = self.stored()
        rows = deepcopy(before['pending']['rows'])
        self.assertEqual(before['pending']['cursor'], len(before['pending']['plan']['steps']))
        with patch('agent.task.operations.verify_final', side_effect=AssertionError('repeated')):
            result = self.run_plans([])
        state = self.stored()
        self.assert_limit_result(state)
        self.assertEqual(state['last_result']['steps'], [
            {key: value for key, value in row.items() if key != 'cache_key'} for row in rows
        ])
        self.assertEqual(state['history'][:-2], before['history'])
        self.assertEqual(state['source'], before['source'])
        self.assertEqual((self.repo / 'sample.py').read_text(), 'def value():\n    return 3\n')
        self.assertEqual(result['calls_this_invocation'], 0)
        self.assertEqual(len(state['rounds']), 1)

    def test_saved_step_failure_keeps_its_outcome_when_limit_is_already_reached(self):
        class Interrupted(BaseException):
            pass

        original = TaskRunner.persist

        def persist(runner, event, payload=None):
            original(runner, event, payload)
            if event == 'step_finished' and payload['status'] == 'failed':
                raise Interrupted()

        failed = self.candidate(3)
        failed['steps'] = [self.shell('import sys; sys.exit(3)')]
        with patch.object(TaskRunner, 'persist', persist), self.assertRaises(Interrupted):
            self.run_plans([self.candidate(2), failed], max_rounds=2)
        with patch('agent.task.operations.verify_final', side_effect=AssertionError('repeated')):
            result = self.run_plans([{'kind': 'blocked', 'note': 'Observed command failure'}])
        self.assertEqual(result['status'], 'task_blocked', result)
        state = self.stored()
        self.assertEqual(state['last_result']['outcome'], 'execution_failed')
        self.assertNotIn('final_acceptance', state['last_result'])
        self.assertEqual(state['last_result']['steps'][0]['data']['exit_code'], 3)
        self.assertEqual(len(state['rounds']), 1)
        self.assertIsNone(state['pending'])

    def test_final_acceptance_defensive_limit_also_completes_pairing(self):
        result = self.run_plans(
            [self.candidate(2), self.candidate(3)], max_rounds=2, pause_after_steps=2,
        )
        self.assertEqual(result['status'], 'paused', result)
        self.runner.max_rounds = 1
        store = LocalHistory(self.root / 'state')
        self.runner.history_store = store
        try:
            with store.lease(), patch(
                'agent.task.operations.verify_final', side_effect=AssertionError('repeated'),
            ):
                self.runner.state = store.load()
                self.runner.final_acceptance()
        finally:
            store.close()
        self.assert_limit_result(self.stored())

    def test_terminal_status_and_pairing_are_one_commit_and_resume_is_idempotent(self):
        class Interrupted(BaseException):
            pass

        original = TaskRunner.persist
        captured = []

        def persist(runner, event, payload=None):
            original(runner, event, payload)
            if event == 'plan_finished' and payload.get('outcome') == 'candidate_limit_reached':
                captured.append(deepcopy(runner.history_store.load()))
                raise Interrupted()

        with patch.object(TaskRunner, 'persist', persist), self.assertRaises(Interrupted):
            self.run_plans([self.candidate(2), self.candidate(3)])
        self.assertEqual(len(captured), 1)
        self.assert_limit_result(captured[0])
        with patch('agent.task.operations.verify_final', side_effect=AssertionError('repeated')):
            result = self.run_plans([])
        self.assertEqual(result['calls_this_invocation'], 0)
        state = self.stored()
        self.assert_limit_result(state)
        self.assertEqual(state['history'], captured[0]['history'])
        events = [json.loads(line) for line in (
            self.root / 'state/events.jsonl'
        ).read_text(encoding='utf-8').splitlines()]
        self.assertEqual(sum(
            row['kind'] == 'plan_finished'
            and row['payload'].get('outcome') == 'candidate_limit_reached'
            for row in events
        ), 1)


if __name__ == '__main__':
    unittest.main()
