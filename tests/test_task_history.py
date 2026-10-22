"""P0339 扩展原始归档，不增加规划器或上下文投影。"""

import copy
import json
import re
from pathlib import Path
import shutil
import sqlite3
import unittest
from unittest.mock import patch
from agent.task.budget import StepBudget
from agent.model.payload import wire_input
from agent.task.runner import TaskRunner
from agent.storage.history import LocalHistory
from agent.model.tools import submit_plan
from support import interrupt_after_persist
import test_task_runner as f


class TaskHistoryTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp
    candidate = f.TaskRunnerTests.candidate
    shell = f.TaskRunnerTests.shell

    def run_plans(self, plans, **options):
        self.model = f.Model(plans)
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / 'state', **options)
        return self.engine.run([], 'return 4', StepBudget(max_steps=8))

    def store(self):
        return LocalHistory(self.root / 'state')

    def test_confirmed_step_resume_does_not_repeat_write_or_model(self):
        plan = self.candidate()
        counter = self.shell(
            "from pathlib import Path; p=Path('counter'); p.write_text(str(int(p.read_text())+1 if p.exists() else 1))",
            'counter',
        )
        plan['steps'].insert(0, counter)
        result = self.run_plans([plan], pause_after_steps=1)
        self.assertEqual(result['status'], 'paused')
        result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(result['calls_this_invocation'], 0)
        self.assertEqual((self.repo / 'counter').read_text(), '1')
        self.assertFalse(self.model.calls)

    def test_completed_return_has_no_model_or_verification(self):
        self.run_plans([self.candidate()])
        with patch(
            'agent.task.operations.verify_final', side_effect=AssertionError('must not rerun')
        ):
            result = self.run_plans([])
        self.assertTrue(result['result_reused'])
        self.assertFalse(result['fresh_verification'])
        self.assertEqual(result['calls_this_invocation'], 0)

    def test_external_edit_conflicts_without_overwrite(self):
        self.run_plans([self.candidate()])
        (self.repo / 'sample.py').write_text('def value():\n    return 99\n')
        result = self.run_plans([])
        self.assertEqual(result['status'], 'state_conflict')
        self.assertFalse(self.model.calls)
        self.assertIn('99', (self.repo / 'sample.py').read_text())

    def test_received_response_is_consumed_without_resending(self):
        interrupt, Interrupted = interrupt_after_persist(
            'model_response_received', tool_name='submit_plan',
        )
        with interrupt, self.assertRaises(Interrupted):
            self.run_plans([self.candidate()])
        result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(result['calls_this_invocation'], 1)
        self.assertEqual(result['planner_calls'], 3)

    def test_unfinished_operation_is_not_repeated(self):
        interrupt, Interrupted = interrupt_after_persist('step_started')
        with interrupt, self.assertRaises(Interrupted):
            self.run_plans([self.candidate()])
        result = self.run_plans([])
        self.assertEqual(result['status'], 'execution_unknown')
        self.assertIn('return 0', (self.repo / 'sample.py').read_text())

    def test_read_cache_and_existing_history_search_entry(self):
        first = {
            'kind': 'inspect',
            'note': 'first read',
            'steps': [
                {'id': 'a', 'op': 'read', 'path': 'sample.py'},
                {'id': 'b', 'op': 'search', 'path': '.', 'query': 'def value'},
            ],
        }
        second = {
            'kind': 'inspect',
            'note': 'repeat exact read and locate record',
            'steps': [
                {'id': 'a', 'op': 'read', 'path': 'sample.py'},
                {'id': 'b', 'op': 'search', 'path': '$record', 'query': 'def value'},
                {'id': 'c', 'op': 'read', 'from_step': 'b'},
            ],
        }
        result = self.run_plans([first, second, self.candidate()])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(result['local_state_stats']['read_cache_hits'], 0)
        repeated = self.model.calls[2][-1].content
        self.assertIn('already_visible', repeated)
        self.assertIn('$record/plans/', str(self.model.calls[-1]))

    def test_database_contains_refs_not_chat_bodies(self):
        (self.repo / 'sample.py').write_text('# UNIQUE_LOCAL_BODY\ndef value():\n    return 0\n')
        read = {
            'kind': 'inspect',
            'note': 'read',
            'steps': [{'id': 'r', 'op': 'read', 'path': 'sample.py'}],
        }
        self.run_plans([read, self.candidate()])
        raw = (self.root / 'state/history.sqlite3').read_bytes()
        self.assertNotIn(b'UNIQUE_LOCAL_BODY', raw)
        marker = json.loads((self.root / 'state/state.json').read_text())
        self.assertEqual(marker['format'], 'task-history-reference')
        self.assertTrue(list((self.root / 'state/history/messages').glob('*.json')))

    def test_corrupt_original_is_rejected(self):
        self.run_plans([self.candidate()])
        (self.root / 'state/plans/p0001/plan.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Original record changed'):
            self.run_plans([])

    def test_sqlite_does_not_change_cold_planner_input(self):
        scan = {
            'kind': 'inspect',
            'note': 'read needed file',
            'steps': [
                {'id': 's', 'op': 'search', 'query': 'def value'},
                {'id': 'r', 'op': 'read', 'from_step': 's'},
            ],
        }
        self.assertEqual(
            self.run_plans([scan, self.candidate()], persistence=False)["status"],
            "verified_success",
        )
        a = [wire_input(x, [submit_plan]) for x in self.model.calls]
        shutil.rmtree(self.root / 'state')
        shutil.rmtree(self.env.artifact_root)
        self.env.artifact_root.mkdir()
        (self.repo / 'sample.py').write_text('def value():\n    return 0\n')
        self.assertEqual(self.run_plans([scan, self.candidate()])["status"], "verified_success")
        b = [wire_input(x, [submit_plan]) for x in self.model.calls]

        def normalize(value):
            if isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items() if k != 'seconds'}
            if isinstance(value, list):
                return [normalize(v) for v in value]
            if isinstance(value, str) and value.startswith('{'):
                try:
                    return normalize(json.loads(value))
                except ValueError:
                    pass
            if isinstance(value, str):
                value = re.sub(r"acceptance-[a-z0-9_]{8}(?=[/\\]|$)", "acceptance-TEMP", value)
                return re.sub(r"(Ran \d+ tests? in )\d+\.\d+s", r"\1TIME", value)
            return value

        self.assertEqual(normalize(a), normalize(b))

    def test_no_concurrent_task_owner(self):
        first, second = self.store(), self.store()
        try:
            with first.lease(), self.assertRaises(RuntimeError):
                with second.lease():
                    pass
        finally:
            first.close()
            second.close()

    def test_interrupted_request_keeps_unknown_usage_without_resend(self):
        interrupt, Interrupted = interrupt_after_persist('model_request_started')
        with interrupt, self.assertRaises(Interrupted):
            self.run_plans([self.candidate()])
        result = self.run_plans([])
        self.assertEqual(result['status'], 'model_request_unknown')
        self.assertFalse(self.model.calls)
        self.assertEqual(self.engine.budget.unknown_usage_steps, 1)
        self.assertGreater(self.engine.budget.unknown_reserved_tokens, 0)
        result = self.run_plans([])
        self.assertEqual(self.engine.budget.unknown_usage_steps, 1)

    def test_budget_and_contract_changes_are_refused(self):
        self.run_plans([self.candidate()])
        changed = TaskRunner(f.Model([]), self.env, state_dir=self.root / 'state')
        with self.assertRaisesRegex(ValueError, 'Budget policy'):
            changed.run([], 'return 4', StepBudget(max_steps=9))
        path = self.root / 'state/frozen-acceptance.json'
        data = json.loads(path.read_text())
        data['regression'] = 'echo passed'
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Frozen acceptance'):
            self.run_plans([])
