"""普通返工不产生停止；权限、完整性和显式预算仍是真实边界。"""

import json
import unittest

from agent.task.budget import StepBudget
import test_task_runner as fixtures


class ReworkPolicyTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    shell = fixtures.TaskRunnerTests.shell
    candidate = fixtures.TaskRunnerTests.candidate
    loop = fixtures.TaskRunnerTests.loop

    def test_model_blocked_is_paired_feedback_then_real_work(self):
        result = self.loop([
            {'kind': 'blocked', 'note': 'I cannot solve this yet'}, self.candidate(),
        ]).run([], 'return 4', StepBudget(max_steps=6))
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertIn('Model-reported obstacle', str(self.model.calls[1]))
        self.assertEqual(self.agent.state['repair_count'], 0)
        self.assertNotIn('blocked_report', self.agent.state)

    def test_duplicate_plan_is_not_replayed_and_does_not_stop(self):
        read = {'kind': 'inspect', 'note': 'Read once',
                'steps': [{'id': 'read', 'op': 'read', 'path': 'sample.py'}]}
        result = self.loop([read, read, self.candidate()]).run(
            [], 'return 4', StepBudget(max_steps=8))
        self.assertEqual(result['status'], 'verified_success', result)
        events = [json.loads(line) for line in
                  (self.agent.root / 'events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(row['kind'] == 'plan_rejected' for row in events), 1)
        self.assertEqual(self.agent.state['plan_count'], 2)

    def test_four_failed_rounds_rework_without_round_grants(self):
        result = self.loop([self.candidate(v) for v in (1, 2, 3, 5, 4)]).run(
            [], 'return 4', StepBudget(max_steps=9))
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(result['rounds']), 5)
        self.assertNotIn('acceptance_budget', self.agent.state)
        self.assertNotIn('acceptance_limit', self.agent.state)
        self.assertNotIn('budget_grants', self.agent.state)

    def test_rework_preserves_token_limit(self):
        result = self.loop([{'kind': 'blocked', 'note': 'Investigate again'}]).run(
            [], 'return 4', StepBudget(max_steps=5, max_tokens=1))
        self.assertEqual(result['status'], 'token_budget_exhausted', result)
        self.assertEqual(len(self.model.calls), 0)

    def test_rework_preserves_explicit_user_pause(self):
        read = {'kind': 'inspect', 'note': 'Read once',
                'steps': [{'id': 'read', 'op': 'read', 'path': 'sample.py'}]}
        result = self.loop([read], pause_after_steps=1).run(
            [], 'return 4', StepBudget(max_steps=5))
        self.assertEqual(result['status'], 'paused', result)
        self.assertEqual(len(self.model.calls), 1)

    def test_response_arriving_after_deadline_is_saved_without_starting_trial(self):
        from agent.task.runner import TaskRunner

        budget = StepBudget(max_steps=6, timeout_seconds=300)
        class LateResponse(fixtures.Model):
            def invoke(inner, messages):
                response = super().invoke(messages)
                if response.tool_calls[0]['name'] == 'submit_plan':
                    budget.started_at -= 301
                return response

        model = LateResponse([self.candidate()])
        runner = TaskRunner(model, self.env, state_dir=self.root / 'state')
        result = runner.run([], 'return 4', budget)
        self.assertEqual(result['status'], 'task_timeout', result)
        self.assertIsNotNone(runner.state.get('response_pending'))
        self.assertIsNone(runner.state.get('request_pending'))
        self.assertEqual(runner.budget.unknown_usage_steps, 0)
        self.assertFalse(list(runner.root.glob('acceptance/*/proposal.json')))
        original = (runner.root / 'requests/call-0002/response.json').read_bytes()
        continuing = fixtures.Model([])
        resumed = TaskRunner(continuing, self.env, state_dir=runner.root,
                             recovery_id='continue-after-time-limit').run(
                                 [], 'return 4', StepBudget(max_steps=6, timeout_seconds=300))
        self.assertEqual(resumed['status'], 'verified_success', resumed)
        self.assertEqual(len(continuing.calls), 1)  # 只请求后续冻结确认，不重发保存的候选。
        self.assertEqual((runner.root / 'requests/call-0002/response.json').read_bytes(), original)
