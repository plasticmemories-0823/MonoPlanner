"""Restored material handoff and single-owner continuous work; no provider calls."""
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from agent.model.tools import CHAT_TOOLS, CODING_TOOLS, WORK_TOOLS
from agent.model.payload import wire_input
from agent.server.interrupts import Signal
from agent.storage.history import LocalHistory
from agent.settings import PROJECT_ROOT
import test_web_graph as fixtures


class CodingHandoffTests(unittest.TestCase):
    setUp = fixtures.ConversationTests.setUp
    invoke = fixtures.ConversationTests.invoke

    def test_original_four_schema_and_other_task_rules_are_unchanged(self):
        for name in ('plan.schema.json', 'repair.schema.json', 'acceptance-review.schema.json'):
            original = subprocess.check_output(['git', 'show', '98b3851:resources/' + name], cwd=PROJECT_ROOT)
            expected = json.loads(original)
            if name == 'plan.schema.json':
                expected['properties']['steps']['items']['properties']['limit']['maximum'] = 2000
                expected['properties']['steps']['items']['properties']['limit']['description'] = (
                    'read: default 2000, range 1..2000. Other operations: default 80, maximum 1000.')
            self.assertEqual(json.loads((PROJECT_ROOT/'resources'/name).read_bytes()), expected)
        original = subprocess.check_output(['git', 'show', '98b3851:resources/prompts/task.txt'], cwd=PROJECT_ROOT).decode()
        rules = (PROJECT_ROOT/'resources/prompts/task.txt').read_text(encoding='utf-8')
        expected = original.replace('submit_work', 'submit_plan')
        expected = expected.replace("limit=80, expected_exit=[0].", "limit=80, expected_exit=[0]. A read defaults to limit=2000; choose a smaller window when appropriate. Other operation defaults and limits are unchanged.")
        expected = expected.replace('a total line limit (1..1000), automatically paged at 200 lines; search',
            'a total line limit (1..2000), automatically paged at 200 lines. All pages and from_step locations share 50 KiB of UTF-8 text per read step. Truncation returns only an original prefix and continuation.next_offset; a partial long line is not a complete read. search')
        self.assertEqual(rules, expected)
        self.assertEqual(wire_input([], CODING_TOOLS)['tools'][:4], wire_input([], WORK_TOOLS)['tools'])
        self.assertEqual([t.name for t in CODING_TOOLS], [t.name for t in WORK_TOOLS] + ['web_search', 'web_fetch'])
        self.assertNotIn('submit_plan', [t.name for t in CHAT_TOOLS])

    def test_first_coding_request_contains_initial_material_and_no_chat_instructions(self):
        (self.repo/'README.md').write_text('Public repository introduction.', encoding='utf-8')
        (self.repo/'pytest.ini').write_text('[pytest]\n', encoding='utf-8')
        model = fixtures.ScriptedModel([fixtures.start_reply(), fixtures.investigation(['first.py']), fixtures.text_reply()])
        facts = {'python': '3.9.20 (container)', 'python_executable': '/container/python',
                 'checkers': {'pytest': False, 'flake8': True, 'ruff': None}}
        class ProbedDocker(fixtures.FixtureDocker):
            def start(self):
                self.runtime_facts = facts
                return self
        with patch('agent.execution.task_docker.TaskDocker', ProbedDocker):
            result = self.invoke(model, 'Explicitly inspect first.py without editing.')
        self.assertEqual(result['status'], 'awaiting_user')
        first = model.calls[1]
        initial = json.loads(first[1].content)
        self.assertEqual(initial['request'], 'Explicitly inspect first.py without editing.')
        self.assertIn('Public repository introduction.', repr(initial['initial_evidence']['readme']))
        self.assertIn('pytest.ini', initial['initial_evidence']['test_config_paths'])
        self.assertEqual(initial['initial_evidence']['runtime_facts'], facts)
        self.assertNotIn('work_materials', initial['initial_evidence'])
        self.assertTrue(first[0].content.startswith((PROJECT_ROOT/'resources/prompts/task.txt').read_text(encoding='utf-8')))
        self.assertNotIn('general assistant', first[0].content)
        self.assertNotIn('application_id', first[0].content)
        self.assertEqual(len(list((Path(result['result']['artifact_dir'])/'requests').glob('call-*'))), 3)

    def test_invalid_start_arguments_and_direct_coding_call_are_rejected_in_chat(self):
        invalid = AIMessage(content='', tool_calls=[{'name': 'start_work', 'id': 'bad', 'args': {'request': 'replacement'}}])
        model = fixtures.ScriptedModel([invalid, fixtures.investigation(['first.py']), fixtures.text_reply('Discuss only')])
        with patch('agent.server.runtime.ConversationRunner.activate', side_effect=AssertionError('must not execute')):
            result = self.invoke(model)
        self.assertEqual(result['status'], 'chat_complete')
        self.assertIn('start_work takes no arguments', model.calls[1][-1].content)
        self.assertFalse(result['workspace_bound'])

    def test_six_inspection_steps_reuse_one_copy_and_container(self):
        from test_probe_batch import FakeDocker
        class ProbeFixture(fixtures.FixtureDocker, FakeDocker):
            def clone_options(self):
                return {}
        FakeDocker.instances, FakeDocker.outcomes = [], []
        plan = AIMessage(content='Inspect six independent observations.', tool_calls=[{'name': 'submit_plan', 'id': 'six',
            'args': {'kind': 'inspect', 'note': 'Readonly observations', 'steps': [
                {'id': 's'+str(i), 'op': 'shell', 'command': 'echo observation'+str(i)} for i in range(6)]}}])
        model = fixtures.ScriptedModel([fixtures.start_reply(), plan, fixtures.text_reply()])
        with patch('agent.execution.probe_batch.DockerEnvironment', FakeDocker), \
                patch('agent.execution.task_docker.TaskDocker', ProbeFixture):
            result = self.invoke(model, 'Directly investigate six observations.')
        self.assertEqual(result['status'], 'awaiting_user')
        self.assertEqual(len(FakeDocker.instances), 1)
        self.assertEqual((FakeDocker.instances[0].starts, FakeDocker.instances[0].stops), (1, 1))
        root = Path(result['result']['artifact_dir'])
        sessions = list(root.glob('plans/*/probe-sessions/*/session.json'))
        self.assertEqual(len(sessions), 1)
        self.assertEqual(json.loads(sessions[0].read_text())['commands'], 6)
        self.assertEqual(len(list(root.glob('plans/*/p*-s*/stdout.log'))), 6)
        events = [json.loads(line)['kind'] for line in (root/'events.jsonl').read_text().splitlines()]
        self.assertEqual(events.count('task_resumed'), 0)

    def test_coding_and_chat_admissions_survive_a_status_question(self):
        model = fixtures.ScriptedModel([fixtures.start_reply(), fixtures.investigation(['first.py']),
                                       fixtures.text_reply('Work retained'), fixtures.text_reply('Status only')])
        work = self.invoke(model, 'Directly inspect first.py')
        chatted = self.invoke(model, 'How is it going?', 'status-question')
        self.assertEqual(chatted['task_id'], work['task_id'])
        self.assertEqual(chatted['status'], 'awaiting_user')
        root = Path(work['result']['artifact_dir'])
        archive = LocalHistory(root)
        try:
            with archive.lease():
                state = archive.load()
            self.assertNotEqual(state['last_admission']['envelope'], state['conversation_admission']['last_admission']['envelope'])
            self.assertNotIn('Status only', repr(state['history']))
            self.assertIn('Status only', repr(state['chat_history']))
        finally:
            archive.close()


class SignalSequenceTests(unittest.TestCase):
    def test_consuming_old_message_cannot_clear_new_message_or_pause(self):
        signal = Signal()
        signal.interrupt('message', 'one')
        signal.interrupt('message', 'two')
        signal.consume_input('one')
        self.assertEqual(signal.kind, 'message')
        signal.interrupt('pause', 'pause')
        signal.consume_input('two')
        self.assertEqual(signal.kind, 'pause')
        signal.consume_input('pause')
        self.assertIsNone(signal.kind)
