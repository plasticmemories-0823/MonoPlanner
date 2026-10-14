"""触顶的完整响应归档、聊天及用户主动续接；不调用真实模型。"""

import hashlib
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import test_web_graph as fixtures
from agent.model.client import ModelPolicy, ConfiguredModel
from agent.task.output_limit import STATUS


def truncated(message=None):
    message = message or AIMessage(content='尚未完成的说明')
    message.response_metadata['finish_reason'] = 'length'
    message.additional_kwargs['reasoning_content'] = 'RETAINED_PARTIAL_REASONING'
    message.usage_metadata = {
        'input_tokens': 10,
        'output_tokens': 256000,
        'total_tokens': 256010,
        'output_token_details': {'reasoning': 255990},
    }
    return message


class OutputLimitTests(unittest.TestCase):
    setUp = fixtures.ConversationTests.setUp
    invoke = fixtures.ConversationTests.invoke

    def resume(self, model, identity='resume-one'):
        self.store.enqueue(self.thread, identity, 'resume')
        with patch('tests.legacy.graph.model_factory', return_value=model):
            return self.compiled.invoke({'messages': []}, self.config)

    def test_defaults_and_auxiliary_caps(self):
        from agent.application.titles import POLICY
        from agent.model.compaction import CONFIG

        self.assertEqual(ModelPolicy().max_output_tokens, 256000)
        self.assertEqual(ModelPolicy().reasoning_effort, 'high')
        configured = ConfiguredModel(object())
        self.assertEqual(configured.request_options['max_tokens'], 256000)
        self.assertEqual(
            configured.for_purpose('compaction').max_output_tokens, CONFIG['summary_max_tokens']
        )
        self.assertEqual(POLICY.max_output_tokens, 128)

    def test_complete_looking_tool_in_truncated_reply_never_executes(self):
        model = fixtures.ScriptedModel(
            [fixtures.start_reply(), truncated(fixtures.investigation(['second.py']))]
        )
        result = self.invoke(model, '开始编码')
        self.assertEqual(result['status'], STATUS, result)
        self.assertEqual(len(model.calls), 2)
        root = self.root / 'runs' / 'tasks' / result['task_id']
        self.assertEqual(list(root.glob('plans/*/*/result.json')), [])
        self.assertEqual(result['result']['output_limit']['output_tokens'], 256000)
        self.assertIn('达到输出上限', repr(result['messages']))
        self.assertNotIn('RETAINED_PARTIAL_REASONING', repr(result))
        from agent.storage.activity import read_activity

        display = read_activity(
            self.root / 'runs', self.store.session(self.thread)['activity_tasks'], {}
        )
        stopped = [row for row in display['items'] if row['label'] == '回复达到输出上限']
        self.assertEqual(len(stopped), 1)
        self.assertEqual(stopped[0]['status'], 'interrupted')

    def test_chat_followup_does_not_activate_container_and_resume_does_not_replay(self):
        model = fixtures.ScriptedModel(
            [
                fixtures.start_reply(),
                fixtures.investigation(['first.py']),
                truncated(fixtures.investigation(['second.py'])),
                fixtures.text_reply('回复达到输出上限，尚未执行下一步。'),
                fixtures.investigation(['second.py']),
                fixtures.text_reply('本轮读取完成'),
            ]
        )
        stopped = self.invoke(model, '开始编码')
        root = self.root / 'runs' / 'tasks' / stopped['task_id']
        raw_path = root / 'requests/call-0003/response.json'
        original = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        self.assertEqual(len(list(root.glob('plans/*/*/result.json'))), 1)
        with patch(
            'tests.legacy.runtime.ConversationRunner.activate',
            side_effect=AssertionError('chat must not activate'),
        ):
            chatted = self.invoke(model, '为什么未完成', 'why')
        self.assertEqual(chatted['status'], STATUS, chatted)
        self.assertEqual(chatted['turn_status'], 'chat_complete')
        self.assertIn('回复达到输出上限', repr(chatted['messages']))
        self.assertIn('output_limit', repr(model.calls[3]))
        resumed = self.resume(model)
        self.assertEqual(resumed['status'], 'awaiting_user', resumed)
        self.assertEqual(len(model.calls), 6)
        self.assertEqual(len(list(root.glob('plans/*/*/result.json'))), 2)
        self.assertIn('RETAINED_PARTIAL_REASONING', repr(model.calls[4]))
        self.assertNotIn(
            'second.py', repr([m.tool_calls for m in model.calls[4] if isinstance(m, AIMessage)])
        )
        self.assertEqual(hashlib.sha256(raw_path.read_bytes()).hexdigest(), original)
        self.resume(model)
        self.assertEqual(len(model.calls), 6)

    def test_plain_chat_can_hit_limit_and_continue_without_workspace(self):
        model = fixtures.ScriptedModel([truncated(), fixtures.text_reply('说明完整')])
        result = self.invoke(model)
        self.assertEqual(result['status'], STATUS)
        with patch(
            'tests.legacy.runtime.ConversationRunner.activate',
            side_effect=AssertionError('no execution'),
        ):
            resumed = self.resume(model)
        self.assertEqual(resumed['status'], 'chat_complete', resumed)
        self.assertEqual(len(model.calls), 2)

    def test_pause_queued_before_explicit_resume_does_not_cancel_new_continuation(self):
        model = fixtures.ScriptedModel([truncated(), fixtures.text_reply('继续后的完整回复')])
        self.invoke(model)
        self.store.enqueue(self.thread, 'older-pause', 'pause')
        result = self.resume(model)
        self.assertEqual(result['status'], 'chat_complete', result)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(self.store.pending(self.thread), [])

    def test_changed_source_blocks_continuation(self):
        model = fixtures.ScriptedModel([fixtures.start_reply(), truncated()])
        self.invoke(model, '开始编码')
        (self.repo / 'first.py').write_text('VALUE = 99', encoding='utf-8')
        result = self.resume(model)
        self.assertEqual(result['status'], 'state_conflict', result)
        self.assertEqual(len(model.calls), 2)

    def test_saved_failure_response_is_identified_as_limit_without_tool_execution(self):
        # 固定结构复现现场：65,536 思考 Token、零正文、零工具。
        response = truncated(AIMessage(content=''))
        response.usage_metadata.update(
            output_tokens=65536, total_tokens=65546, output_token_details={'reasoning': 65536}
        )
        result = self.invoke(fixtures.ScriptedModel([response]))
        self.assertEqual(result['status'], STATUS)
        self.assertEqual(result['result']['output_limit']['reasoning_tokens'], 65536)

    def test_model_cannot_resume_work_when_user_only_asks_a_question(self):
        model = fixtures.ScriptedModel(
            [
                fixtures.start_reply(),
                truncated(),
                fixtures.start_reply(),
                fixtures.text_reply('需要你点击继续任务。'),
            ]
        )
        self.invoke(model, '开始编码')
        with patch(
            'tests.legacy.runtime.ConversationRunner.activate',
            side_effect=AssertionError('no activation'),
        ):
            result = self.invoke(model, '为什么停了', 'why')
        self.assertEqual(result['status'], STATUS, result)
        self.assertEqual(result['turn_status'], 'chat_complete')
        self.assertIn('No operation was executed', repr(model.calls[-1]))

    def test_resume_checks_frozen_acceptance_before_request(self):
        from agent.storage.history import LocalHistory

        model = fixtures.ScriptedModel([fixtures.start_reply(), truncated()])
        result = self.invoke(model, '开始编码')
        root = self.root / 'runs/tasks' / result['task_id']
        history = LocalHistory(root)
        with history.lease():
            state = history.load()
            state['verification_spec'] = {'test_code': 'original'}
            state['regression_command'] = 'original regression'
            history.record(state, 'test_frozen_fixture', {})
        history.close()
        (root / 'frozen-acceptance.json').write_text(
            json.dumps({'spec': {'test_code': 'changed'}, 'regression': 'original regression'}),
            encoding='utf-8',
        )
        resumed = self.resume(model)
        self.assertEqual(resumed['status'], 'state_conflict', resumed)
        self.assertEqual(len(model.calls), 2)

    def test_unknown_operation_blocks_resume(self):
        from agent.storage.history import LocalHistory

        model = fixtures.ScriptedModel([fixtures.start_reply(), truncated()])
        result = self.invoke(model, '开始编码')
        root = self.root / 'runs/tasks' / result['task_id']
        history = LocalHistory(root)
        with history.lease():
            state = history.load()
            state['running_step'] = {'id': 'unknown'}
            history.record(state, 'test_unknown_fixture', {})
        history.close()
        resumed = self.resume(model)
        self.assertEqual(resumed['status'], 'execution_unknown', resumed)
        self.assertEqual(len(model.calls), 2)


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
