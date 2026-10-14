"""验证图的首次回复路由；不调用意图分类器或付费模型服务。"""

import os
from pathlib import Path
from unittest.mock import patch
import unittest

from langchain_core.messages import AIMessage, HumanMessage

import test_web_graph as fixtures
from agent.storage.preferences import Preferences


class ConversationRoutingTests(unittest.TestCase):
    setUp = fixtures.ConversationTests.setUp
    invoke = fixtures.ConversationTests.invoke

    def web_reply(self, name='web_fetch', **args):
        return AIMessage(
            content='读取公开网页。',
            tool_calls=[{'name': name, 'id': 'web-1', 'args': args}],
            additional_kwargs={'reasoning_content': 'PRIVATE_WEB_THINKING'},
        )

    def test_chat_without_repository_never_initializes_source_or_executor(self):
        model = fixtures.ScriptedModel([fixtures.text_reply('你好')])
        with patch('agent.server.graph.model_factory', return_value=model), patch(
            'agent.server.runtime.ConversationRunner.activate',
            side_effect=AssertionError('chat must not initialize source'),
        ):
            result = self.compiled.invoke(
                {'messages': [HumanMessage(content='你好', id='one')]}, self.config
            )
        self.assertEqual(result['status'], 'chat_complete')
        self.assertFalse(result['workspace_bound'])
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(list((self.root / 'runs' / 'tasks').glob('*/baseline/*')))
        rules = model.calls[0][0].content
        self.assertIn('Linux Docker, Bash shell', rules)
        self.assertNotIn('Windows; shell commands use cmd.exe', rules)
        self.assertNotIn('native LPAC isolation', rules)

    def test_chat_work_chat_routes_share_public_context_without_replaying_reads(self):
        model = fixtures.ScriptedModel(
            [
                fixtures.text_reply('你好'),
                fixtures.start_reply(),
                fixtures.investigation(['first.py']),
                fixtures.text_reply('读完了'),
                fixtures.text_reply('我是一名编码助手'),
            ]
        )
        self.invoke(model, '你好')
        work = self.invoke(model, '读取文件', 'two')
        chat = self.invoke(model, '介绍自己', 'three')
        self.assertEqual(len(model.calls), 5)
        self.assertEqual(chat['task_id'], work['task_id'])
        self.assertEqual(chat['status'], 'awaiting_user')
        self.assertEqual(chat['turn_status'], 'chat_complete')
        self.assertIn('我是一名编码助手', repr(chat['messages']))
        self.assertNotIn('PRIVATE_REASONING', repr(chat['messages']))
        results = list(
            (self.root / 'runs' / 'tasks' / work['task_id'] / 'plans').glob('*/*/result.json')
        )
        self.assertEqual(len(results), 1)

    def test_paused_task_can_chat_without_resuming_or_losing_original_response(self):
        model = fixtures.ScriptedModel(
            [
                fixtures.start_reply(),
                fixtures.investigation(['first.py']),
                fixtures.text_reply('你好'),
                fixtures.text_reply('读取已完成'),
            ],
            hook=lambda n: self.store.enqueue(self.thread, 'pause', 'pause') if n == 2 else None,
        )
        paused = self.invoke(model, '读取文件')
        self.assertEqual(paused['status'], 'paused')
        chatted = self.invoke(model, '你好', 'two')
        self.assertEqual(chatted['status'], 'paused')
        self.assertEqual(chatted['turn_status'], 'chat_complete')
        self.assertFalse(
            list(
                (self.root / 'runs' / 'tasks' / paused['task_id'] / 'plans').glob('*/*/result.json')
            )
        )
        self.store.enqueue(self.thread, 'resume', 'resume')
        with patch('agent.server.graph.model_factory', return_value=model):
            result = self.compiled.invoke({'messages': []}, self.config)
        self.assertNotEqual(result['status'], 'runtime_error')
        self.assertEqual(len(model.calls), 4)
        self.assertEqual(
            len(
                list(
                    (self.root / 'runs' / 'tasks' / paused['task_id'] / 'plans').glob(
                        '*/*/result.json'
                    )
                )
            ),
            1,
        )

    def test_new_defaults_do_not_change_active_task_or_restored_tool_envelope(self):
        factory_settings = []
        model = fixtures.ScriptedModel(
            [fixtures.start_reply(), fixtures.investigation(['first.py']), fixtures.text_reply()],
            hook=lambda n: self.store.enqueue(self.thread, 'pause', 'pause') if n == 2 else None,
        )
        self.invoke(model, '读取文件')
        Preferences().update({'reasoning_effort': 'low', 'online': True})
        self.store.enqueue(self.thread, 'resume', 'resume')

        def factory(settings):
            factory_settings.append(settings)
            return model

        with patch('agent.server.graph.model_factory', side_effect=factory):
            self.compiled.invoke({'messages': []}, self.config)
        self.assertTrue(
            all(v['reasoning_effort'] == 'high' and not v['online'] for v in factory_settings)
        )
        model = fixtures.ScriptedModel([fixtures.text_reply()])
        with patch(
            'agent.server.graph.model_factory',
            side_effect=lambda s: factory_settings.append(s) or model,
        ):
            # 已结束的聊天会使用变更后的默认设置开始新任务。
            from agent.server.graph import builder
            import uuid

            result = builder.compile().invoke(
                {'messages': [HumanMessage(content='你好', id='new')]},
                {'configurable': {'thread_id': str(uuid.uuid4())}},
            )
        self.assertEqual(result['runtime_settings']['reasoning_effort'], 'low')
        self.assertTrue(factory_settings[-1]['online'])

    def test_disabled_web_tool_returns_error_without_network_and_keeps_pair(self):
        model = fixtures.ScriptedModel(
            [self.web_reply(url='https://example.com'), fixtures.text_reply('联网关闭')]
        )
        with patch(
            'agent.execution.web.WebProvider.fetch', side_effect=AssertionError('no network')
        ):
            result = self.invoke(model, '查看网页')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 2)
        self.assertIn('Networking is disabled', repr(model.calls[1]))
        self.assertEqual(model.calls[1][-1].tool_call_id, 'web-1')
        self.assertFalse(result['workspace_bound'])

    def test_web_then_text_routes_without_workspace_and_no_code_execution(self):
        Preferences().update({'online': True})
        model = fixtures.ScriptedModel(
            [self.web_reply(url='https://example.com'), fixtures.text_reply('网页摘要')]
        )
        with patch('agent.server.graph.model_factory', return_value=model), patch(
            'agent.execution.web.WebProvider.fetch',
            return_value={'url': 'https://example.com', 'text': '原文'},
        ) as fetch:
            result = self.compiled.invoke(
                {'messages': [HumanMessage(content='联网看网页', id='net')]}, self.config
            )
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(fetch.call_count, 1)
        self.assertIn('原文', repr(model.calls[-1]))
        self.assertIn('PRIVATE_WEB_THINKING', repr(model.calls[-1]))
        self.assertNotIn('PRIVATE_WEB_THINKING', repr(result))
        from agent.server.app import activity_value

        self.assertIn('正在读取网页', repr(activity_value(self.thread, {}).body.decode()))

    def test_unknown_web_operation_is_not_replayed_on_resume(self):
        from agent.execution.web import WebUnknown

        Preferences().update({'online': True})
        model = fixtures.ScriptedModel([self.web_reply(url='https://example.com')])
        with patch(
            'agent.execution.web.WebProvider.fetch', side_effect=WebUnknown('offline')
        ) as fetch:
            result = self.invoke(model, '联网')
            self.assertEqual(result['status'], 'web_request_unknown')
            again = self.invoke(model, '继续', 'two')
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(len(model.calls), 1)
        self.assertNotEqual(again['status'], 'verified_success')

    def test_truncated_text_is_not_complete_delivery(self):
        message = fixtures.text_reply('半句话')
        message.response_metadata['finish_reason'] = 'length'
        result = self.invoke(fixtures.ScriptedModel([message]))
        self.assertEqual(result['status'], 'output_limit_reached')
        self.assertNotIn('半句话', repr(result['messages']))

    def test_missing_repository_pairs_handoff_without_initializing_source(self):
        model = fixtures.ScriptedModel([fixtures.start_reply()])
        with patch('agent.server.graph.model_factory', return_value=model), patch(
            'agent.server.runtime.ConversationRunner.activate',
            side_effect=AssertionError('no repository'),
        ):
            result = self.compiled.invoke(
                {'messages': [HumanMessage(content='读取文件', id='one')]}, self.config
            )
        self.assertEqual(result['status'], 'awaiting_user')
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(result['workspace_bound'])

    def test_missing_key_can_be_configured_before_a_new_turn(self):
        with patch(
            'agent.server.graph.model_factory', side_effect=ValueError('请配置 DEEPSEEK_API_KEY')
        ):
            result = self.compiled.invoke(
                {'messages': [HumanMessage(content='你好', id='one')]}, self.config
            )
        self.assertEqual(result['status'], 'settings_required')
        model = fixtures.ScriptedModel([fixtures.text_reply('你好')])
        again = self.invoke(model, '你好', 'two')
        self.assertEqual(again['status'], 'chat_complete')
        self.assertNotEqual(result['task_id'], again['task_id'])

    def test_multiple_web_reads_resume_cursor_without_replaying_completed_call(self):
        Preferences().update({'online': True})
        response = self.web_reply(url='https://example.com/one')
        response.tool_calls.append(
            {
                'name': 'web_fetch',
                'id': 'web-2',
                'args': {'url': 'https://example.com/two'},
                'type': 'tool_call',
            }
        )
        model = fixtures.ScriptedModel([response, fixtures.text_reply('读完两个网页')])
        with patch(
            'agent.execution.web.WebProvider.fetch', return_value={'text': '真实只读样本'}
        ) as fetch:
            result = self.invoke(model, '读取两个网页')
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual([m.tool_call_id for m in model.calls[-1][-2:]], ['web-1', 'web-2'])
        self.assertEqual(result['status'], 'chat_complete')

    def test_two_consecutive_web_turns_do_not_reuse_completed_cursor(self):
        Preferences().update({'online': True})
        search = self.web_reply(name='web_search', query='公开查询')
        fetch = self.web_reply(url='https://example.com')
        fetch.tool_calls[0]['id'] = 'web-2'
        model = fixtures.ScriptedModel([search, fetch, fixtures.text_reply('读完了')])
        with patch(
            'agent.execution.web.WebProvider.search',
            return_value={'sources': [{'url': 'https://example.com'}]},
        ) as first, patch(
            'agent.execution.web.WebProvider.fetch', return_value={'text': '网页正文'}
        ) as second:
            result = self.invoke(model, '搜索后读取')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(first.call_count, 1)
        self.assertEqual(second.call_count, 1)
        self.assertEqual(model.calls[1][-1].tool_call_id, 'web-1')
        self.assertEqual(model.calls[2][-1].tool_call_id, 'web-2')
