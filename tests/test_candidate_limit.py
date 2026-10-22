"""验收次数耗尽后等待人工判断；批准后按原断点继续且不重放已完成操作。"""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import message_to_dict

from agent.storage.history import LocalHistory
from agent.task import acceptance_limit
from agent.task.budget import StepBudget
from agent.task.contracts import WorkRequest
from agent.task.runner import TaskRunner
from support import interrupt_after_persist
import test_task_runner as fixtures


class CandidateLimitTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    candidate = fixtures.TaskRunnerTests.candidate
    shell = fixtures.TaskRunnerTests.shell

    def run_plans(self, plans, controls=(), budget=None, **options):
        self.model = fixtures.Model(plans)
        self.runner = TaskRunner(
            self.model, self.env, state_dir=self.root / 'state',
            max_rounds=options.pop('max_rounds', 1), control_source=lambda: list(controls),
            **options,
        )
        return self.runner.run([], 'return 4', budget or StepBudget(max_steps=12))

    def stored(self):
        store = LocalHistory(self.root / 'state')
        try:
            with store.lease():
                return store.load()
        finally:
            store.close()

    def change_state(self, change):
        store = LocalHistory(self.root / 'state')
        try:
            with store.lease():
                state = store.load()
                change(state)
                store.record(state, 'test_checkpoint', {})
                return state
        finally:
            store.close()

    def resume(self, identity='continue-1', pause_id=None):
        value = self.stored()['acceptance_limit']
        return {'id': identity, 'kind': 'resume',
                'acceptance_pause_id': pause_id or value['pause_id']}

    def assert_paused(self, state, rounds=1):
        self.assertEqual(state['status'], acceptance_limit.STATUS)
        self.assertEqual(len(state['rounds']), rounds)
        self.assertTrue(state['acceptance_limit']['active'])
        self.assertEqual(state['acceptance_limit']['completed_rounds'], rounds)
        self.assertIn('是否增加', state['reason'])

    def test_last_failed_round_pauses_before_another_model_request(self):
        result = self.run_plans([self.candidate(2), self.candidate(3)])
        self.assertEqual(result['status'], acceptance_limit.STATUS, result)
        self.assertEqual((self.repo / 'sample.py').read_text(), 'def value():\n    return 2\n')
        state = self.stored()
        self.assert_paused(state)
        self.assertIsNone(state['pending'])
        self.assertEqual(state['last_result']['outcome'], 'verification_failed')
        self.assertEqual(len(self.model.calls), 3)
        request, response = state['history'][-2:]
        self.assertEqual(request['data']['tool_calls'][0]['id'], response['data']['tool_call_id'])

    def test_refresh_and_restart_do_not_grant_more_rounds(self):
        self.run_plans([self.candidate(2)])
        before = self.stored()
        for _ in range(2):
            result = self.run_plans([])
            self.assertEqual(result['calls_this_invocation'], 0)
            self.assert_paused(self.stored())
        self.assertEqual(self.stored()['history'], before['history'])
        self.assertEqual(self.stored()['acceptance_budget'], before['acceptance_budget'])

    def test_human_continue_keeps_task_history_and_cumulative_consumption(self):
        self.run_plans([self.candidate(2)])
        before = self.stored()
        result = self.run_plans([self.candidate()], controls=[self.resume()])
        self.assertEqual(result['status'], 'verified_success', result)
        state = self.stored()
        self.assertEqual(len(state['rounds']), 2)
        self.assertEqual(state['rounds'][:1], before['rounds'])
        self.assertEqual(state['history'][:len(before['history'])], before['history'])
        self.assertEqual(state['identity'], before['identity'])
        self.assertEqual(state['budget']['used_steps'], before['budget']['used_steps'] + 1)
        self.assertEqual(state['budget']['reported_tokens'], before['budget']['reported_tokens'] + 20)
        self.assertEqual(state['budget']['max_steps'], before['budget']['max_steps'])
        self.assertEqual(state['acceptance_budget'], {'round_increment': 1, 'round_limit': 2})
        self.assertEqual(len(state['acceptance_limit_history']), 1)

    def test_each_exhausted_batch_needs_a_new_explicit_approval(self):
        self.run_plans([self.candidate(2)])
        first = self.resume()
        result = self.run_plans([self.candidate(3)], controls=[first])
        self.assertEqual(result['status'], acceptance_limit.STATUS, result)
        self.assert_paused(self.stored(), 2)
        result = self.run_plans([], controls=[first])
        self.assertEqual(result['calls_this_invocation'], 0)
        stale = {**first, 'id': 'delayed-second-click'}
        result = self.run_plans([], controls=[stale])
        self.assertEqual(result['calls_this_invocation'], 0)
        self.assertEqual(self.stored()['acceptance_budget']['round_limit'], 2)
        result = self.run_plans([self.candidate()], controls=[self.resume('continue-2')])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(result['rounds']), 3)

    def test_duplicate_clicks_only_grant_one_batch(self):
        self.run_plans([self.candidate(2)])
        first = self.resume()
        second = {**first, 'id': 'second-click'}
        result = self.run_plans([self.candidate(3)], controls=[first, second])
        self.assertEqual(result['status'], acceptance_limit.STATUS, result)
        state = self.stored()
        self.assertEqual(state['acceptance_budget']['round_limit'], 2)
        self.assertIn('second-click', state['applied_inputs'])
        result = self.run_plans([], controls=[first, second])
        self.assertEqual(result['calls_this_invocation'], 0)

    def test_supplement_and_pause_never_grant_rounds(self):
        self.run_plans([self.candidate(2)])
        result = self.run_plans([], controls=[
            {'id': 'ask', 'kind': 'message', 'text': 'What happened?'},
            {'id': 'pause', 'kind': 'pause'},
        ])
        self.assertEqual(result['calls_this_invocation'], 0)
        self.assert_paused(self.stored())
        self.assertEqual(self.stored()['acceptance_budget']['round_limit'], 1)

    def test_unbound_resume_cannot_approve_a_later_pause(self):
        self.run_plans([self.candidate(2)])
        result = self.run_plans([], controls=[{'id': 'old-resume', 'kind': 'resume'}])
        self.assertEqual(result['calls_this_invocation'], 0)
        self.assert_paused(self.stored())
        self.assertEqual(self.stored()['acceptance_budget']['round_limit'], 1)

    def test_cancelled_task_cannot_be_resumed_or_granted_more_rounds(self):
        self.run_plans([self.candidate(2)])
        resume = self.resume()
        budget = StepBudget(max_steps=12)
        budget.request_cancel('cancelled by user')
        result = self.run_plans([], controls=[resume], budget=budget)
        self.assertEqual(result['status'], 'task_cancelled', result)
        self.assertEqual(self.stored()['acceptance_budget']['round_limit'], 1)

    def test_saved_round_policy_does_not_change_with_constructor(self):
        self.run_plans([self.candidate(2)])
        result = self.run_plans([], max_rounds=99)
        self.assertEqual(result['status'], acceptance_limit.STATUS, result)
        self.assertEqual(self.stored()['acceptance_budget']['round_limit'], 1)

    def test_last_allowed_candidate_can_pass(self):
        result = self.run_plans([self.candidate()])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(result['rounds']), 1)
        self.assertEqual(len(self.model.calls), 3)
        self.assertIsNone(self.stored()['pending'])

    def test_completed_last_acceptance_is_paired_without_repeating_it(self):
        for value, status in [(4, 'verified_success'), (2, acceptance_limit.STATUS)]:
            with self.subTest(value=value):
                self.setUp()
                interrupt, Interrupted = interrupt_after_persist('final_acceptance_finished')
                with interrupt, self.assertRaises(Interrupted):
                    self.run_plans([self.candidate(value)])
                with patch('agent.task.operations.verify_final', side_effect=AssertionError('repeated')):
                    result = self.run_plans([])
                self.assertEqual(result['status'], status, result)
                self.assertEqual(result['calls_this_invocation'], 0)
                self.assertIsNone(self.stored()['pending'])
                self.assertEqual(len(result['rounds']), 1)

    def test_defensive_cap_preserves_completed_rows_and_resumes_at_acceptance(self):
        result = self.run_plans(
            [self.candidate(2), self.candidate()], max_rounds=2, pause_after_steps=2,
        )
        self.assertEqual(result['status'], 'paused', result)
        self.change_state(lambda state: state['acceptance_budget'].update(round_limit=1))
        before = self.stored()
        rows = deepcopy(before['pending']['rows'])
        with patch('agent.task.operations.verify_final', side_effect=AssertionError('repeated')):
            result = self.run_plans([])
        self.assertEqual(result['status'], acceptance_limit.STATUS, result)
        state = self.stored()
        self.assertEqual(state['pending']['rows'], rows)
        self.assertEqual(state['pending']['cursor'], len(state['pending']['plan']['steps']))
        with patch('agent.execution.executor.OperationExecutor.execute',
                   side_effect=AssertionError('completed step replayed')):
            result = self.run_plans([], controls=[self.resume()])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(result['calls_this_invocation'], 0)
        self.assertEqual(len(result['rounds']), 2)

    def test_unknown_operation_source_and_frozen_contract_block_resume(self):
        for problem in ['running_step', 'source', 'frozen']:
            with self.subTest(problem=problem):
                self.setUp()
                self.run_plans([self.candidate(2)])
                resume = self.resume()
                if problem == 'running_step':
                    self.change_state(lambda state: state.update(running_step={'id': 'unknown'}))
                elif problem == 'source':
                    (self.repo / 'sample.py').write_text('changed externally')
                else:
                    path = self.root / 'state/frozen-acceptance.json'
                    contract = json.loads(path.read_text())
                    contract['regression'] = 'changed'
                    path.write_text(json.dumps(contract))
                if problem == 'frozen':
                    with self.assertRaisesRegex(ValueError, 'Frozen acceptance'):
                        self.run_plans([], controls=[resume])
                else:
                    result = self.run_plans([], controls=[resume])
                    self.assertEqual(result['status'],
                                     'execution_unknown' if problem == 'running_step' else 'state_conflict')
                self.assertEqual(self.stored()['acceptance_budget']['round_limit'], 1)

    def test_unknown_request_stays_blocked_after_marker_is_settled(self):
        self.run_plans([self.candidate(2)])
        continuation = self.resume()
        def pending_request(state):
            state['budget']['request_sequence'] += 1
            state['budget']['used_steps'] += 1
            state['budget']['inflight_reserved_tokens'] += 100
            state['request_pending'] = {
                'call': state['budget']['request_sequence'], 'reservation': 100, 'purpose': 'plan',
            }
        self.change_state(pending_request)
        for _ in range(2):
            result = self.run_plans([], controls=[continuation])
            self.assertEqual(result['status'], 'model_request_unknown', result)
            self.assertEqual(result['calls_this_invocation'], 0)
            self.assertEqual(self.stored()['acceptance_budget']['round_limit'], 1)

    def test_saved_failed_step_keeps_failure_precedence_at_limit(self):
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
        self.change_state(lambda state: state['acceptance_budget'].update(round_limit=1))
        with patch('agent.task.operations.verify_final', side_effect=AssertionError('repeated')):
            result = self.run_plans([{'kind': 'blocked', 'note': 'Observed command failure'}])
        self.assertEqual(result['status'], 'task_blocked', result)
        state = self.stored()
        self.assertEqual(state['last_result']['outcome'], 'execution_failed')
        self.assertEqual(state['last_result']['steps'][0]['data']['exit_code'], 3)
        self.assertEqual(len(state['rounds']), 1)
        self.assertIsNone(state['pending'])





    def test_pause_commit_survives_interruption_and_releases_history_lease(self):
        interrupt, Interrupted = interrupt_after_persist('plan_finished')
        with interrupt, self.assertRaises(Interrupted):
            self.run_plans([self.candidate(2)])
        self.assert_paused(self.stored())
        result = self.run_plans([self.candidate()], controls=[self.resume()])
        self.assertEqual(result['status'], 'verified_success', result)


if __name__ == '__main__':
    unittest.main()
