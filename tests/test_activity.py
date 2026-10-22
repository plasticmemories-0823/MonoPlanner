"""行内展示读取原始内容，不参与模型输入或接管执行。"""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import uuid

from agent.storage.activity import projection, read_activity
from tests.fixtures.models import ScriptedModel, text_reply, investigation
from agent.execution.local import LocalEnvironment
from agent.task.runner import TaskRunner
from agent.task.budget import StepBudget


class ActivityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'first.py').write_text('VALUE = 1\n', encoding='utf-8')
        self.runs, self.task = self.root / 'runs', uuid.uuid4().hex
        self.environment = LocalEnvironment(self.repo, artifact_root=self.root / 'execution')

    def run_model(self, model):
        runner = TaskRunner(model, self.environment,
                            state_dir=self.runs / 'tasks' / self.task, interactive=True)
        result = runner.run([], 'Read the source without editing.', StepBudget(max_steps=5))
        return {**result, 'task_id': self.task}

    def read(self, cursors=None, page_size=100):
        return read_activity(self.runs, [{'id': self.task, 'input_id': 'first'}],
                             cursors or {}, page_size)


    def test_display_reads_exact_thinking_and_operation_without_changing_history(self):
        model = ScriptedModel([investigation(['first.py']), text_reply()])
        result = self.run_model(model)
        root = self.runs / 'tasks' / result['task_id']
        before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
        display = self.read()
        self.assertIn('PRIVATE_REASONING', repr(display))
        self.assertNotIn('PRIVATE_NOTE', repr(display))
        self.assertNotIn('PRIVATE_REASONING', repr(result))
        reads = [r for r in display['items'] if r['kind'] == 'operation']
        self.assertEqual(reads[-1]['status'], 'succeeded')
        self.assertIn('first.py', reads[-1]['detail'])
        self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()})
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.read(display['cursors'])['items'], [])

    def test_waiting_is_visible_while_model_holds_execution_lease(self):
        waiting = []
        model = ScriptedModel([text_reply()], hook=lambda _: waiting.extend(self.read()['items']))
        self.run_model(model)
        self.assertEqual(waiting[-1]['status'], 'running')
        self.assertEqual(waiting[-1]['detail'], '')
        self.assertEqual(waiting[-1]['after_id'], 'first')

    def test_boundary_lookup_is_visible_without_triggering_an_extra_model_call(self):
        from langchain_core.messages import AIMessage

        lookup = AIMessage(
            content='', tool_calls=[{
                'name': 'get_boundary_hints', 'id': 'boundary-call',
                'args': {'keys': ['general', 'state']},
            }],
        )
        model = ScriptedModel([lookup, text_reply()])
        self.run_model(model)
        display = self.read()
        rows = [row for row in display['items'] if row['kind'] == 'operation']
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['detail'], 'general, state')
        self.assertEqual(rows[0]['status'], 'completed')
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.read(display['cursors'])['items'], [])

    def test_side_conversation_web_operations_follow_their_user_message(self):
        record = {
            'state': {
                'public_events': [{'id': 'old-work-reply'}],
                'conversation_turn': {'id': 'new-web-question'},
            },
            'event': 'web_finished',
            'payload': {
                'id': 'web-call',
                'name': 'web_search',
                'args': {'query': '公开资料'},
                'status': 'succeeded',
            },
        }
        row = projection(self.root, {'id': 'task', 'input_id': 'original'}, 1, record, None)
        self.assertEqual(row['after_id'], 'new-web-question')
        self.assertEqual(row['detail'], '公开资料')


    def test_incremental_pages_converge_to_same_activity_without_replay(self):
        model = ScriptedModel([investigation(['first.py']), text_reply()])
        self.run_model(model)
        cursors, merged = {}, {}
        for _ in range(50):
            page = self.read(cursors, page_size=1)
            cursors = page['cursors']
            merged.update({r['id']: r for r in page['items']})
            if not page['more']:
                break
        self.assertEqual(merged, {r['id']: r for r in self.read()['items']})
        self.assertEqual(len(model.calls), 2)

    def test_corrupted_local_record_fails_display_without_retrying_model(self):
        model = ScriptedModel([text_reply()])
        result = self.run_model(model)
        root = self.runs / 'tasks' / result['task_id']
        with closing(sqlite3.connect(root / 'history.sqlite3')) as db:
            path = db.execute('SELECT path FROM commits ORDER BY seq LIMIT 1').fetchone()[0]
        (root / path).write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.read()
        self.assertEqual(len(model.calls), 1)

    def test_invalid_cursors_and_task_paths_are_rejected(self):
        self.run_model(ScriptedModel([text_reply()]))
        task = self.task
        with self.assertRaisesRegex(ValueError, 'cursor'):
            self.read({task: -1})
        with self.assertRaisesRegex(ValueError, 'task ID'):
            read_activity(self.runs, [{'id': '../outside', 'input_id': 'first'}], {})
        self.assertEqual(read_activity(self.runs, [], {})['items'], [])

    def test_command_string_is_displayed_verbatim_without_execution(self):
        command = "echo '中文 \\\\ 引号'\npython -c \"print(1)\""
        state = {
            'public_events': [],
            'pending': {
                'id': 'plan',
                'plan': {'steps': [{'id': 'run', 'op': 'shell', 'command': command}]},
            },
        }
        row = projection(
            self.root,
            {'id': '0' * 32, 'input_id': 'first'},
            1,
            {'state': state, 'event': 'step_started', 'payload': {'id': 'run'}},
            None,
        )
        self.assertEqual(row['detail'], command)
        self.assertEqual(row['status'], 'running')

    def trial_activity(self, report):
        raw = json.dumps(report, ensure_ascii=False).encode()
        path = self.root / 'trial.json'
        path.write_bytes(raw)
        ref = {'record': 'trial.json', 'sha256': hashlib.sha256(raw).hexdigest()}
        state = {'acceptance_trial': {'baseline_result': ref}}
        record = {
            'state': state,
            'event': 'acceptance_trial_finished',
            'payload': {'trial_id': 'trial', 'status': report['status']},
        }
        return projection(self.root, {'id': 'task', 'input_id': 'input'}, 1, record, None), record

    def test_baseline_behavior_problem_is_progress_not_final_task_failure(self):
        report = {
            'status': 'failed',
            'tests_run': 1,
            'test_ids': ['test_behavior'],
            'test_results': [{'id': 'test_behavior', 'status': 'failed', 'exceptions': []}],
            'test_evidence_valid': True,
            'integrity_ok': True,
            'stderr': '实际断言未通过',
        }
        row, _ = self.trial_activity(report)
        self.assertIn('接下来审查验收与修复方案', row['summary'])
        self.assertEqual(row['status'], 'failed')
        self.assertEqual(row['stage'], 'trial')
        self.assertEqual(json.loads(row['detail']), report)

    def test_unusable_baseline_test_is_not_reported_as_reproduced_defect(self):
        for invalid in (
            {'status': 'error', 'tests_run': 0, 'error_kind': 'load_error'},
            {'status': 'passed', 'tests_run': 0, 'test_ids': [], 'test_results': []},
        ):
            with self.subTest(report=invalid):
                row, record = self.trial_activity(invalid)
                self.assertIn('未形成可用验收', row['summary'])
                self.assertNotIn('发现未通过项', row['summary'])
                (self.root / 'trial.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    projection(self.root, {'id': 'task', 'input_id': 'input'}, 1, record, None)

    def test_actual_candidate_results_remain_distinct_and_preserve_diagnostics(self):
        for status, phrase in [
            ('passed', '实际验收通过'),
            ('failed', '仍有未通过项'),
            ('error', '未完整完成'),
        ]:
            with self.subTest(status=status):
                report = {'status': status, 'reason': '真实诊断'}
                row = projection(
                    self.root,
                    {'id': 'task', 'input_id': 'input'},
                    1,
                    {
                        'state': {'verification': report},
                        'event': 'final_acceptance_finished',
                        'payload': {'plan': 'candidate', 'status': status},
                    },
                    None,
                )
                self.assertIn(phrase, row['summary'])
                self.assertEqual(row['status'], status)
                self.assertEqual(json.loads(row['detail']), report)
