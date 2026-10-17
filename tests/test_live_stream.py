"""流式展示不能改变原始工具交付、用量或执行门控。"""

import json
from pathlib import Path
import tempfile
import unittest
import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from agent.model.client import ConfiguredModel
from agent.model.deepseek import DeepSeekClient
from agent.model.payload import RAW_TOOL_CALLS, wire_input
from agent.storage.live_reasoning import LiveReasoning
from agent.storage.activity import read_activity
from agent.storage.history import LocalHistory
from support import read_record


class StreamTests(unittest.TestCase):
    def test_real_sdk_stream_preserves_thought_raw_arguments_and_usage(self):
        requests, fragments = [], []
        argument = '{\n "name" : "中文\\\\path\\nnext"\n}'

        def http(request):
            requests.append(json.loads(request.content))
            deltas = [
                {'role': 'assistant', 'reasoning_content': '先核对'},
                {'reasoning_content': '原始记录。'},
                {'content': '我会读取记录。'},
                {
                    'tool_calls': [
                        {
                            'index': 0,
                            'id': 'raw-call',
                            'type': 'function',
                            'function': {'name': 'read_record', 'arguments': argument[:15]},
                        }
                    ]
                },
                {'tool_calls': [{'index': 0, 'function': {'arguments': argument[15:]}}]},
            ]
            events = [
                {
                    'id': 'fixture',
                    'object': 'chat.completion.chunk',
                    'created': 1,
                    'model': 'deepseek-flash',
                    'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}],
                }
                for delta in deltas
            ]
            events.append(
                {**events[0], 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]}
            )
            events.append(
                {
                    **events[0],
                    'choices': [],
                    'usage': {
                        'prompt_tokens': 100,
                        'completion_tokens': 20,
                        'total_tokens': 120,
                        'prompt_cache_hit_tokens': 80,
                        'completion_tokens_details': {'reasoning_tokens': 10},
                    },
                }
            )
            return httpx.Response(
                200,
                headers={'content-type': 'text/event-stream'},
                text=''.join('data: ' + json.dumps(event) + '\n\n' for event in events)
                + 'data: [DONE]\n\n',
            )

        with httpx.Client(transport=httpx.MockTransport(http)) as transport:
            model = ConfiguredModel(
                DeepSeekClient(
                    model='deepseek-flash',
                    api_key='offline-placeholder',
                    http_client=transport,
                    max_retries=0,
                )
            ).bind_tools([read_record])
            response = model.invoke_with_progress(
                [HumanMessage(content='检查记录')], fragments.append
            )
        self.assertEqual(fragments, ['先核对', '原始记录。'])
        self.assertEqual(response.additional_kwargs['reasoning_content'], ''.join(fragments))
        self.assertEqual(
            response.additional_kwargs[RAW_TOOL_CALLS][0]['function']['arguments'], argument
        )
        replay = wire_input(
            [response, ToolMessage(content='结果', tool_call_id='raw-call')], [read_record]
        )
        self.assertEqual(replay['messages'][0]['tool_calls'][0]['function']['arguments'], argument)
        self.assertEqual(response.usage_metadata['total_tokens'], 120)
        self.assertEqual(response.usage_metadata['input_token_details']['cache_read'], 80)
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0]['stream'])
        self.assertTrue(requests[0]['stream_options']['include_usage'])

    def test_incomplete_stream_does_not_deliver_partial_tool(self):
        class Partial:
            def stream(self, *args, **kwargs):
                from langchain_core.messages import AIMessageChunk

                yield AIMessageChunk(
                    content='',
                    additional_kwargs={'reasoning_content': '部分思考'},
                    tool_call_chunks=[
                        {'name': 'submit_plan', 'args': '{', 'id': 'call', 'index': 0}
                    ],
                )

        with self.assertRaisesRegex(RuntimeError, 'without a complete'):
            ConfiguredModel(Partial()).invoke_with_progress([], lambda text: None)

    def test_live_partial_is_local_and_polling_never_changes_execution_history(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            identity = 'a' * 32
            task = root / 'tasks' / identity
            history = LocalHistory(task)
            with history.lease():
                history.record(
                    {'history': [], 'budget': {'used_steps': 1}, 'request_pending': {'call': 1}},
                    'model_request_started',
                    {'call': 1},
                )
            history.close()
            before = (task / 'history.sqlite3').read_bytes()
            live = LiveReasoning(task, 1, 'user-message')
            live.append('部分思考')
            result = read_activity(root, [{'id': identity, 'input_id': 'user-message'}], {})
            self.assertEqual(result['live'][0]['detail'], '部分思考')
            self.assertEqual(result['live'][0]['status'], 'running')
            self.assertEqual(
                read_activity(
                    root, [{'id': identity, 'input_id': 'user-message'}], result['cursors']
                )['live'],
                [],
            )
            self.assertEqual((task / 'history.sqlite3').read_bytes(), before)
            self.assertFalse((task / 'requests/call-0001/response.json').exists())
            live.finish('unknown')
            self.assertEqual(
                read_activity(root, [{'id': identity, 'input_id': 'user-message'}], {})['live'][0][
                    'status'
                ],
                'unknown',
            )
            self.assertEqual((task / 'history.sqlite3').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()


class CurrentChatStreamTests(unittest.TestCase):
    def test_chat_emits_reasoning_before_response_and_retains_only_canonical_reply(self):
        import time
        import uuid
        from unittest.mock import patch
        from agent.application.chat import Chatbot
        from agent.application.store import ApplicationStore
        from tests.fixtures.models import ScriptedModel, text_reply

        with tempfile.TemporaryDirectory() as folder:
            store = ApplicationStore(folder)
            conversation = store.create()['thread_id']
            command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Explain the project.'}
            store.submit(conversation, command)
            test = self
            clock = time.monotonic

            class Streaming(ScriptedModel):
                def invoke_with_progress(self, messages, receive):
                    self.calls.append(messages)
                    for number, fragment in enumerate(('first fragment', 'second fragment'), 1):
                        with patch('agent.application.chat.time.monotonic', return_value=clock() + number):
                            receive(fragment)
                        activities = [event['payload'] for event in store.events(conversation)['events']
                                      if event['kind'] == 'activity']
                        test.assertEqual(activities[-1]['detail'], fragment)
                        test.assertEqual(activities[-1]['status'], 'running')
                        test.assertTrue(activities[-1]['append'])
                        test.assertFalse(any(raw['type'] == 'ai' for _, raw in store.model_history(conversation)))
                    return next(self.replies)

            model = Streaming([text_reply('Canonical reply')])
            bot = Chatbot(store, lambda *args: self.fail('No work handoff expected'),
                          factory=lambda settings: model)
            result = bot.run(conversation, command)
            self.assertTrue(result['done'])
            self.assertEqual(len(model.calls), 1)
            snapshot = store.snapshot(conversation)
            self.assertEqual([m['content'] for m in snapshot['messages']],
                             [command['text'], 'Canonical reply'])
            self.assertEqual(store.usage(conversation)['chat']['calls'], 1)

    def test_chat_stream_failure_is_recorded_unknown_without_resend(self):
        import uuid
        from agent.application.chat import Chatbot
        from agent.application.store import ApplicationStore
        from tests.fixtures.models import ScriptedModel

        with tempfile.TemporaryDirectory() as folder:
            store = ApplicationStore(folder)
            conversation = store.create()['thread_id']
            command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Explain.'}
            store.submit(conversation, command)

            class Interrupted(ScriptedModel):
                def invoke_with_progress(self, messages, receive):
                    self.calls.append(messages)
                    receive('partial fragment')
                    raise RuntimeError('stream disconnected')

            model = Interrupted([])
            bot = Chatbot(store, lambda *args: self.fail('No work handoff expected'),
                          factory=lambda settings: model)
            with self.assertRaisesRegex(RuntimeError, 'stream disconnected'):
                bot.run(conversation, command)
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(store.usage(conversation)['chat']['unknown_calls'], 1)
            self.assertFalse(any(raw['type'] == 'ai' for _, raw in store.model_history(conversation)))
