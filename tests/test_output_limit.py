"""当前 Chatbot 与工作内核的输出触顶、原件保存及明确续接；全部离线。"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import Mock

from langchain_core.messages import AIMessage

from agent.application.scheduler import Scheduler
from agent.application.store import ApplicationStore
from agent.execution.local import LocalEnvironment
from agent.model.client import ModelPolicy, ConfiguredModel
from agent.storage.history import LocalHistory
from agent.task.budget import StepBudget
from agent.task.output_limit import STATUS
from agent.task.runner import TaskRunner


def truncated(message=None):
    message = message or AIMessage(content='尚未完成的说明')
    message.response_metadata['finish_reason'] = 'length'
    message.additional_kwargs['reasoning_content'] = 'RETAINED_PARTIAL_REASONING'
    message.usage_metadata = {
        'input_tokens': 10, 'output_tokens': 256000, 'total_tokens': 256010,
        'output_token_details': {'reasoning': 255990},
    }
    return message


def investigation(name):
    return AIMessage(content='', tool_calls=[{
        'name': 'submit_plan', 'id': 'read-' + name,
        'args': {'kind': 'inspect', 'note': 'Inspect this source once',
                 'steps': [{'id': 'source', 'op': 'read', 'path': name}]},
    }])


def finished():
    return AIMessage(content='', tool_calls=[{
        'name': 'submit_plan', 'id': 'finished',
        'args': {'kind': 'blocked', 'note': 'The offline continuation is complete'},
    }])


class Replies:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return next(self.replies)


class OutputLimitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        for name in ('first.py', 'second.py'):
            (self.repo / name).write_text('VALUE = 1\n', encoding='utf-8')
        self.data = self.root / 'runs'
        self.task = uuid.uuid4().hex
        self.artifacts = self.data / 'tasks' / self.task
        self.environment = LocalEnvironment(self.repo, artifact_root=self.root / 'runtime')
        self.commands = []

    def invoke(self, model):
        runner = TaskRunner(
            model, self.environment, state_dir=self.artifacts,
            control_source=lambda: self.commands,
        )
        return runner.run([], 'Inspect the specified files', StepBudget(max_steps=20))

    def resume(self, model, identity='resume-one'):
        if not any(row['id'] == identity for row in self.commands):
            self.commands.append({'id': identity, 'kind': 'resume'})
        return self.invoke(model)

    def saved_state(self, edit):
        history = LocalHistory(self.artifacts)
        try:
            with history.lease():
                state = history.load()
                edit(state)
                history.record(state, 'offline_fixture', {})
        finally:
            history.close()

    def chat(self, replies, *, active_work=False):
        store = ApplicationStore(self.root / 'application')
        conversation = store.create(repo=str(self.repo) if active_work else '')['thread_id']
        if active_work:
            original = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Original work'}
            store.submit(conversation, original)
            task = store.create_work(conversation, original['id'])['task_id']
            store.work_state(task, STATUS)
        model = Replies(replies)
        work = Mock(side_effect=AssertionError('Chat must not execute or resume work'))
        scheduler = Scheduler(store, model_factory=lambda _settings: model, work_factory=work)
        self.addCleanup(scheduler.close)
        return store, conversation, model, scheduler, work

    def test_defaults_and_auxiliary_caps(self):
        from agent.application.titles import POLICY
        from agent.model.compaction import CONFIG

        self.assertEqual(ModelPolicy().max_output_tokens, 256000)
        self.assertEqual(ModelPolicy().reasoning_effort, 'high')
        configured = ConfiguredModel(object())
        self.assertEqual(configured.request_options['max_tokens'], 256000)
        self.assertEqual(configured.for_purpose('compaction').max_output_tokens,
                         CONFIG['summary_max_tokens'])
        self.assertEqual(POLICY.max_output_tokens, 128)

    def test_complete_looking_tool_in_truncated_reply_never_executes(self):
        model = Replies([truncated(investigation('second.py'))])
        result = self.invoke(model)
        self.assertEqual(result['status'], STATUS, result)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(list(self.artifacts.glob('plans/*/*/result.json')), [])
        self.assertEqual(result['output_limit']['output_tokens'], 256000)
        self.assertIn('达到输出上限', repr(result['messages']))
        self.assertNotIn('RETAINED_PARTIAL_REASONING', repr(result))
        from agent.storage.activity import read_activity

        display = read_activity(self.data, [{'id': self.task, 'input_id': 'initial'}], {})
        stopped = [row for row in display['items'] if row['label'] == '回复达到输出上限']
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0]['status'], 'interrupted')

    def test_chat_followup_does_not_activate_work_and_resume_does_not_replay(self):
        model = Replies([investigation('first.py'), truncated(investigation('second.py')),
                         investigation('second.py'), finished()])
        self.assertEqual(self.invoke(model)['status'], STATUS)
        raw_path = self.artifacts / 'requests/call-0002/response.json'
        original = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        self.assertEqual(len(list(self.artifacts.glob('plans/*/*/result.json'))), 1)
        store, conversation, chat_model, scheduler, work = self.chat(
            [AIMessage(content='回复达到输出上限，尚未执行下一步。')], active_work=True,
        )
        command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '为什么未完成'}
        store.submit(conversation, command)
        scheduler._chat(conversation, command)
        self.assertEqual(store.snapshot(conversation)['work']['status'], STATUS)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(chat_model.calls), 1)
        work.assert_not_called()
        resumed = self.resume(model)
        self.assertEqual(resumed['status'], 'task_blocked', resumed)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(len(list(self.artifacts.glob('plans/*/*/result.json'))), 2)
        self.assertIn('RETAINED_PARTIAL_REASONING', repr(model.calls[2]))
        self.assertNotIn('second.py', repr([
            message.tool_calls for message in model.calls[2] if isinstance(message, AIMessage)
        ]))
        self.assertEqual(hashlib.sha256(raw_path.read_bytes()).hexdigest(), original)
        self.resume(model)
        self.assertEqual(len(model.calls), 4)

    def test_plain_chat_can_hit_limit_and_continue_without_workspace(self):
        store, conversation, model, scheduler, work = self.chat([
            truncated(), AIMessage(content='说明完整'),
        ])
        first = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '解释这个概念'}
        store.submit(conversation, first)
        scheduler._chat(conversation, first)
        self.assertEqual(store.snapshot(conversation)['chat_status'], STATUS)
        self.assertEqual(len(model.calls), 1)
        next_message = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '请继续说明'}
        store.submit(conversation, next_message)
        scheduler._chat(conversation, next_message)
        self.assertEqual(store.snapshot(conversation)['chat_status'], 'chat_complete')
        self.assertEqual(len(model.calls), 2)
        self.assertFalse(store.snapshot(conversation)['work'])
        work.assert_not_called()

    def test_pause_queued_before_explicit_resume_does_not_cancel_new_continuation(self):
        model = Replies([truncated(), finished()])
        self.invoke(model)
        self.commands.append({'id': 'older-pause', 'kind': 'pause'})
        result = self.resume(model)
        self.assertEqual(result['status'], 'task_blocked', result)
        self.assertEqual(len(model.calls), 2)
        history = LocalHistory(self.artifacts)
        try:
            with history.lease():
                self.assertTrue({'older-pause', 'resume-one'}.issubset(history.load()['applied_inputs']))
        finally:
            history.close()

    def test_changed_source_blocks_continuation(self):
        model = Replies([truncated()])
        self.invoke(model)
        (self.repo / 'first.py').write_text('VALUE = 99', encoding='utf-8')
        result = self.resume(model)
        self.assertEqual(result['status'], 'state_conflict', result)
        self.assertEqual(len(model.calls), 1)

    def test_saved_failure_response_is_identified_as_limit_without_tool_execution(self):
        response = truncated(AIMessage(content=''))
        response.usage_metadata.update(
            output_tokens=65536, total_tokens=65546, output_token_details={'reasoning': 65536},
        )
        result = self.invoke(Replies([response]))
        self.assertEqual(result['status'], STATUS)
        self.assertEqual(result['output_limit']['reasoning_tokens'], 65536)

    def test_model_cannot_resume_work_when_user_only_asks_a_question(self):
        start = AIMessage(content='', tool_calls=[{
            'name': 'prepare_work', 'id': 'unexpected-resume',
            'args': {'project_path': '.', 'create': False},
        }])
        store, conversation, model, scheduler, work = self.chat(
            [start, AIMessage(content='需要你点击继续任务。')], active_work=True,
        )
        command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '为什么停了'}
        store.submit(conversation, command)
        scheduler._chat(conversation, command)
        self.assertEqual(store.snapshot(conversation)['work']['status'], STATUS)
        self.assertEqual(store.snapshot(conversation)['chat_status'], 'chat_complete')
        self.assertIn('actions_executed', repr(model.calls[-1]))
        self.assertIn('false', repr(model.calls[-1]))
        work.assert_not_called()

    def test_resume_checks_frozen_acceptance_before_request(self):
        model = Replies([truncated()])
        self.invoke(model)
        def edit(state):
            from test_task_runner import EXECUTION_CHECK
            state['verification_spec'] = {'test_code': 'original', 'execution_checks': [dict(EXECUTION_CHECK)]}
            state['regression_command'] = 'original regression'
        self.saved_state(edit)
        (self.artifacts / 'frozen-acceptance.json').write_text(
            json.dumps({'spec': {'test_code': 'changed'}, 'regression': 'original regression'}),
            encoding='utf-8',
        )
        with self.assertRaisesRegex(ValueError, 'Frozen acceptance was changed'):
            self.resume(model)
        self.assertEqual(len(model.calls), 1)

    def test_unknown_operation_blocks_resume(self):
        model = Replies([truncated()])
        self.invoke(model)
        self.saved_state(lambda state: state.update(running_step={'id': 'unknown'}))
        result = self.resume(model)
        self.assertEqual(result['status'], 'execution_unknown', result)
        self.assertEqual(len(model.calls), 1)


class ContainerContinuationTests(unittest.TestCase):
    def setUp(self):
        from agent.execution.task_docker import TaskDocker
        from unittest.mock import Mock
        from copy import deepcopy

        self.container = TaskDocker.__new__(TaskDocker)
        self.container._container_id = 'a' * 64
        self.container.image = 'sha256:' + 'b' * 64
        self.actual = {
            'Id': self.container._container_id,
            'Image': self.container.image,
            'Name': '/original',
            'Config': {},
            'HostConfig': {'NetworkMode': 'none'},
            'Mounts': [],
            'State': {
                'Running': False,
                'StartedAt': 'start',
                'FinishedAt': 'finish',
                'ExitCode': 0,
            },
        }
        self.container._continuation_container = lambda: deepcopy(self.actual)
        self.container._prepare_project_masks = Mock()
        self.container.lifetime = Mock()
        self.container.start = Mock()
        self.container._docker = Mock(return_value=type('Result', (), {'returncode': 0})())
        self.receipt = self.container.output_limit_stop_receipt()

    def test_only_same_container_is_started_and_start_intent_can_resume(self):
        self.container.restart_after_output_limit(self.receipt)
        self.container._prepare_project_masks.assert_called_once_with(
            allow_create=False, full_check=True,
        )
        self.assertEqual(self.container._docker.call_args.args[0], ['start', 'a' * 64])
        self.actual['State']['Running'] = True
        self.container.restart_after_output_limit(self.receipt, allow_running=True)
        self.assertEqual(self.container._docker.call_count, 1)

    def test_changed_container_configuration_or_stop_state_cannot_resume(self):
        for mutate in (
            lambda: self.actual['HostConfig'].update(NetworkMode='bridge'),
            lambda: self.actual['State'].update(FinishedAt='changed'),
            lambda: self.actual['State'].update(Running=True),
        ):
            with self.subTest(mutate=mutate):
                self.setUp()
                mutate()
                with self.assertRaises(ValueError):
                    self.container.restart_after_output_limit(self.receipt)
                self.container._docker.assert_not_called()
