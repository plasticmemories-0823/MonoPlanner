"""验证任务运行器的少量回移修复：模拟模型、真实本地执行、无网络。"""

import json
from pathlib import Path
import unittest
from unittest.mock import patch
from langchain_core.messages import SystemMessage, HumanMessage
from agent.task.budget import StepBudget
from agent.model.payload import wire_input
from agent.model.errors import ModelRequestNotSent
from agent.model.context import admit
from agent.task.contracts import WorkRequest
from agent.model.tools import submit_plan
from agent.execution.probe import run_probe
import test_task_runner as f


class TaskSafetyTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp
    candidate = f.TaskRunnerTests.candidate
    shell = f.TaskRunnerTests.shell
    loop = f.TaskRunnerTests.loop

    def test_all_requests_append_even_after_rejection_and_rework(self):
        bad = {'kind': 'inspect', 'note': 'invalid', 'steps': []}
        result = self.loop([bad, self.candidate(2), self.candidate()]).run(
            [], 'return 4', StepBudget(max_steps=6)
        )
        self.assertEqual(result['status'], 'verified_success', result)
        payloads = [wire_input(m, [submit_plan]) for m in self.model.calls]
        for a, b in zip(payloads, payloads[1:]):
            self.assertEqual(a['messages'], b['messages'][: len(a['messages'])])
        self.assertIn('Plan rejected', str(self.model.calls[-1]))

    def test_paging_keeps_requested_tail_and_search_meaning(self):
        (self.repo / 'long.txt').write_text(''.join('L%d\n' % i for i in range(230)))
        plan = {
            'kind': 'inspect',
            'note': 'locate data',
            'steps': [
                {'id': 's', 'op': 'search', 'path': '.', 'query': 'L200', 'limit': 2},
                {'id': 'r', 'op': 'read', 'path': 'long.txt', 'offset': 0, 'limit': 202},
            ],
        }
        result = self.loop([plan, self.candidate()]).run([], 'return 4', StepBudget(max_steps=4))
        self.assertEqual(result['status'], 'verified_success', result)
        text = str(self.model.calls[1])
        self.assertIn('L201', text)
        self.assertIn('requested_lines', text)
        self.assertIn('query', text)
        self.assertIn('scope', text)

    def test_more_than_two_corrections_can_succeed(self):
        bad = {'kind': 'inspect', 'note': 'invalid', 'steps': []}
        result = self.loop([bad] * 5 + [self.candidate()]).run(
            [], 'return 4', StepBudget(max_steps=8)
        )
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(self.model.calls), 8)

    def test_corrections_still_obey_total_task_budget(self):
        bad = {'kind': 'inspect', 'note': 'invalid', 'steps': []}
        result = self.loop([bad] * 6).run([], 'return 4', StepBudget(max_steps=6))
        self.assertEqual(result['status'], 'budget_exhausted', result)
        self.assertEqual(len(self.model.calls), 6)
        self.assertEqual(result['rounds'], [])
        self.assertIn('return 0', (self.repo / 'sample.py').read_text())

    def test_guard_refuses_changes(self):
        state = {}
        original = [SystemMessage(content='stable'), HumanMessage(content='task')]
        admit(state, original, [submit_plan], {})
        with self.assertRaises(ModelRequestNotSent):
            admit(state, [SystemMessage(content='changed')], [submit_plan], {})

    def test_probe_never_falls_back_to_host(self):
        marker = self.repo / 'unsafe'
        command = self.shell("from pathlib import Path; Path('unsafe').write_text('bad')")[
            'command'
        ]
        result = run_probe(self.env, command, self.root / 'probe', 5)
        self.assertEqual(result['error_kind'], 'probe_backend_unavailable')
        self.assertFalse(marker.exists())

    def test_inspect_probe_does_not_freeze_acceptance(self):
        plan = {'kind': 'inspect', 'note': 'observe', 'steps': [self.shell('print(1)')]}
        observed = {'exit_code': 0, 'stdout': '1', 'stderr': '', 'mode': 'isolated_probe'}
        with patch('agent.execution.probe.run_probe', return_value=observed) as probe:
            result = self.loop([plan, self.candidate()]).run(
                [], 'return 4', StepBudget(max_steps=4)
            )
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(len(result['rounds']), 1)

    def test_oversized_and_unsafe_paths_are_not_reinterpreted(self):
        with self.assertRaises(ValueError):
            WorkRequest.model_validate(
                {
                    'kind': 'inspect',
                    'note': 'huge',
                    'steps': [{'id': 'r', 'op': 'read', 'limit': 2001}],
                }
            )
        self.loop([self.candidate()]).run([], 'return 4', StepBudget(max_steps=3))
        with self.assertRaises(ValueError):
            self.agent.executor.path('../private.txt')


if __name__ == '__main__':
    unittest.main()
