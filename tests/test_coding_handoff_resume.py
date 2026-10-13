"""A saved work handoff survives missing environment and ordinary chat."""
import json
import unittest
from unittest.mock import patch

from agent.storage.history import LocalHistory
import test_web_graph as fixtures


class WorkPreparationResumeTests(unittest.TestCase):
    setUp = fixtures.ConversationTests.setUp
    invoke = fixtures.ConversationTests.invoke

    def remove_environment(self):
        path = self.root/'runs/project-environments.json'
        self.configuration = path.read_bytes()
        path.unlink()

    def configure_environment(self):
        (self.root/'runs/project-environments.json').write_bytes(self.configuration)

    def checkpoint(self, result):
        archive = LocalHistory(self.root/'runs/tasks'/result['task_id'])
        try:
            with archive.lease():
                return archive.load()
        finally:
            archive.close()

    def resume(self, model, identity='resume'):
        self.store.enqueue(self.thread, identity, 'resume')
        with patch('agent.server.graph.model_factory', return_value=model):
            return self.compiled.invoke({'messages': []}, self.config)

    def test_environment_resume_prepares_work_without_repeating_chat_decision(self):
        self.remove_environment()
        model = fixtures.ScriptedModel([fixtures.start_reply(), fixtures.investigation(['first.py']),
                                       fixtures.text_reply('The inspected source is saved.')])
        missing = self.invoke(model, 'Directly inspect first.py.')
        self.assertEqual(missing['status'], 'awaiting_user')
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(self.checkpoint(missing).get('coding_preparation_pending'))
        self.assertFalse(missing['workspace_bound'])
        self.configure_environment()
        resumed = self.resume(model)
        self.assertEqual(resumed['task_id'], missing['task_id'])
        self.assertEqual(resumed['status'], 'awaiting_user')
        self.assertEqual(len(model.calls), 3)
        self.assertIn('initial_evidence', json.loads(model.calls[1][1].content))
        self.assertNotIn('general assistant', model.calls[1][0].content)
        state = self.checkpoint(resumed)
        self.assertFalse(state.get('coding_preparation_pending'))
        self.assertEqual(state['execution_binding']['image'], 'sha256:'+'1'*64)
        self.assertEqual(len(list((self.root/'runs/tasks'/resumed['task_id']).glob('plans/*/*/result.json'))), 1)
        history = state['chat_history']
        self.assertEqual(sum(m['type'] == 'tool' and m['data']['name'] == 'start_coding' for m in history), 1)

    def test_resume_without_configuration_never_calls_model_or_starts_docker(self):
        self.remove_environment()
        model = fixtures.ScriptedModel([fixtures.start_reply()])
        missing = self.invoke(model, 'Directly inspect first.py.')
        with patch('agent.execution.task_docker.TaskDocker.start', side_effect=AssertionError('No environment')):
            again = self.resume(model)
            repeated = self.resume(model, 'resume-again')
        self.assertEqual(again['status'], 'awaiting_user')
        self.assertEqual(repeated['task_id'], missing['task_id'])
        self.assertEqual(repeated['status'], 'awaiting_user')
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(self.checkpoint(repeated).get('coding_preparation_pending'))

    def test_ordinary_chat_retains_handoff_and_does_not_start_work(self):
        self.remove_environment()
        model = fixtures.ScriptedModel([fixtures.start_reply(), fixtures.text_reply('Environment setup is pending.'),
                                       fixtures.investigation(['first.py']), fixtures.text_reply('Read complete.')])
        missing = self.invoke(model, 'Directly inspect first.py.')
        self.configure_environment()
        with patch('agent.server.runtime.ConversationRunner.activate', side_effect=AssertionError('Chat must stay chat')):
            chatted = self.invoke(model, 'What is the current status?', 'question')
        self.assertEqual(chatted['task_id'], missing['task_id'])
        self.assertEqual(chatted['status'], 'awaiting_user')
        self.assertEqual(chatted['turn_status'], 'chat_complete')
        self.assertTrue(self.checkpoint(chatted).get('coding_preparation_pending'))
        resumed = self.resume(model)
        self.assertEqual(resumed['task_id'], missing['task_id'])
        self.assertEqual(len(model.calls), 4)
        self.assertTrue(resumed['workspace_bound'])

    def test_pause_before_preparation_preserves_handoff_until_explicit_resume(self):
        self.remove_environment()
        model = fixtures.ScriptedModel([fixtures.start_reply(), fixtures.investigation(['first.py']),
                                       fixtures.text_reply('Read complete.')])
        missing = self.invoke(model, 'Directly inspect first.py.')
        self.configure_environment()
        self.store.enqueue(self.thread, 'pause', 'pause')
        with patch('agent.server.graph.model_factory', return_value=model), \
                patch('agent.server.runtime.ConversationRunner.activate', side_effect=AssertionError('Paused')):
            paused = self.compiled.invoke({'messages': []}, self.config)
        self.assertEqual(paused['status'], 'paused')
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(self.checkpoint(paused).get('coding_preparation_pending'))
        resumed = self.resume(model)
        self.assertTrue(resumed['workspace_bound'])
        self.assertEqual(resumed['task_id'], missing['task_id'])
        self.assertEqual(len(model.calls), 3)
