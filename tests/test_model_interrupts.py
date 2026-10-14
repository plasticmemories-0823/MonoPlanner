"""用户控制可取消网络等待、保留记录，且绝不执行回复片段。"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
import time
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from agent.model.client import ConfiguredModel
from agent.model.deepseek import DeepSeekClient
from agent.model.errors import ModelInterrupted
from agent.model.payload import RAW_TOOL_CALLS, wire_input
from agent.task import control as interrupts
from tests.legacy.app_adapter import control_value
from agent.storage.history import LocalHistory
from support import read_record
import test_web_graph as fixtures


class WaitingClient:
    def __init__(self, before_headers=False):
        self.started, self.closed = Event(), Event()
        self.calls, self.before_headers = [], before_headers

    def bind_tools(self, tools, strict):
        self.tools, self.strict = tools, strict
        return self

    async def astream(self, messages, **options):
        self.calls.append((messages, options))
        if len(self.calls) == 1:
            try:
                self.started.set()
                if not self.before_headers:
                    yield AIMessageChunk(
                        content='',
                        additional_kwargs={'reasoning_content': '部分思考'},
                        tool_call_chunks=[
                            {'name': 'submit_plan', 'args': '{', 'id': 'partial', 'index': 0}
                        ],
                    )
                await asyncio.Event().wait()
            finally:
                self.closed.set()
        else:
            yield AIMessageChunk(
                content='按最新要求回答。',
                response_metadata={'finish_reason': 'stop'},
                usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5},
            )


class CancellationTests(unittest.TestCase):
    def test_sdk_http_wait_and_sse_body_are_cancellable(self):
        for before_headers in (True, False):
            started, closed = Event(), Event()

            class Body(httpx.AsyncByteStream):
                async def __aiter__(self):
                    delta = {
                        'id': 'partial',
                        'object': 'chat.completion.chunk',
                        'created': 1,
                        'model': 'deepseek-flash',
                        'choices': [
                            {
                                'index': 0,
                                'delta': {'role': 'assistant', 'reasoning_content': '正在核对'},
                                'finish_reason': None,
                            }
                        ],
                    }
                    yield ('data: ' + json.dumps(delta) + '\n\n').encode()
                    started.set()
                    await asyncio.Event().wait()

                async def aclose(self):
                    closed.set()

            async def http(request):
                if before_headers:
                    try:
                        started.set()
                        await asyncio.Event().wait()
                    finally:
                        closed.set()
                return httpx.Response(
                    200, headers={'content-type': 'text/event-stream'}, stream=Body()
                )

            transport = httpx.AsyncClient(transport=httpx.MockTransport(http))
            model = ConfiguredModel(
                DeepSeekClient(
                    model='deepseek-flash',
                    api_key='offline-placeholder',
                    http_async_client=transport,
                    max_retries=0,
                )
            )
            with self.subTest(before_headers=before_headers), ThreadPoolExecutor() as pool:

                def invoke():
                    with interrupts.active('sdk-cancel', 'thread'):
                        return model.invoke_with_progress(
                            [HumanMessage(content='只读')], lambda fragment: None
                        )

                future = pool.submit(invoke)
                self.assertTrue(started.wait(3))
                interrupts.interrupt('sdk-cancel', 'thread', 'pause')
                with self.assertRaises(ModelInterrupted):
                    future.result(timeout=3)
                self.assertTrue(closed.is_set())
            with interrupts.active('close', 'thread') as signal:
                from agent.model.streaming import run

                run(transport.aclose(), signal)

    def test_cancel_before_headers_or_between_chunks_closes_stream(self):
        for before_headers in (True, False):
            with self.subTest(before_headers=before_headers), ThreadPoolExecutor() as pool:
                client = WaitingClient(before_headers)
                fragments = []

                def invoke():
                    with interrupts.active('test-root', 'test-thread'):
                        return ConfiguredModel(client).invoke_with_progress([], fragments.append)

                future = pool.submit(invoke)
                self.assertTrue(client.started.wait(3))
                started = time.monotonic()
                interrupts.interrupt('test-root', 'test-thread', 'message')
                with self.assertRaises(ModelInterrupted):
                    future.result(timeout=3)
                self.assertLess(time.monotonic() - started, 1)
                self.assertTrue(client.closed.is_set())
                self.assertFalse(interrupts._active)
                self.assertEqual(fragments, [] if before_headers else ['部分思考'])

    def test_pause_wins_over_later_message_and_scopes_are_isolated(self):
        with interrupts.active('root', 'first') as first:
            with interrupts.active('root', 'second') as second:
                interrupts.interrupt('root', 'first', 'pause')
                interrupts.interrupt('root', 'first', 'message')
                self.assertEqual(first.kind, 'pause')
                self.assertIsNone(second.kind)
                self.assertIs(interrupts.current(), second)
            self.assertIs(interrupts.current(), first)
        self.assertIsNone(interrupts.current())
        self.assertFalse(interrupts._active)

    def test_cancelled_before_send_never_calls_client(self):
        client = WaitingClient()
        with interrupts.active('root', 'thread') as signal:
            signal.interrupt('pause')
            with self.assertRaises(ModelInterrupted) as caught:
                ConfiguredModel(client).invoke_with_progress([], lambda fragment: None)
        self.assertFalse(caught.exception.sent)
        self.assertFalse(client.calls)

    def test_provider_cancellation_is_not_a_user_pause(self):
        class ProviderCancelled:
            async def astream(self, messages, **options):
                raise asyncio.CancelledError()
                yield  # 与真实 SDK 一样，保持此函数为异步生成器。

        with interrupts.active('provider', 'thread'):
            with self.assertRaisesRegex(RuntimeError, 'without a user interruption'):
                ConfiguredModel(ProviderCancelled()).invoke_with_progress([], lambda fragment: None)

    def test_real_async_sdk_preserves_tools_reasoning_arguments_and_continuation(self):
        requests, fragments = [], []
        argument = '{\n "name" : "中文\\\\path\\nnext"\n}'

        def http(request):
            requests.append(json.loads(request.content))
            deltas = [
                {'role': 'assistant', 'reasoning_content': '核对原件'},
                {'content': '开始读取。'},
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

        async_transport = httpx.AsyncClient(transport=httpx.MockTransport(http))
        with httpx.Client(transport=httpx.MockTransport(http)) as transport:
            model = ConfiguredModel(
                DeepSeekClient(
                    model='deepseek-flash',
                    api_key='offline-placeholder',
                    http_client=transport,
                    http_async_client=async_transport,
                    max_retries=0,
                )
            ).bind_tools([read_record])
            with interrupts.active('sdk', 'thread') as signal:
                response = model.invoke_with_progress(
                    [HumanMessage(content='检查原件')], fragments.append
                )
                continuation = [response, ToolMessage(content='结果', tool_call_id='raw-call')]
                model.invoke_with_progress(continuation, fragments.append)
                from agent.model.streaming import run

                run(async_transport.aclose(), signal)
        self.assertEqual(
            response.additional_kwargs[RAW_TOOL_CALLS][0]['function']['arguments'], argument
        )
        self.assertEqual(response.additional_kwargs['reasoning_content'], '核对原件')
        self.assertEqual(response.usage_metadata['total_tokens'], 120)
        self.assertEqual(response.usage_metadata['input_token_details']['cache_read'], 80)
        self.assertEqual(
            requests[1]['messages'], wire_input(continuation, [read_record])['messages']
        )
        self.assertTrue(requests[0]['tools'][0]['function']['strict'])
        self.assertEqual(requests[0]['tools'], requests[1]['tools'])
        self.assertEqual(requests[0]['reasoning_effort'], 'high')
        self.assertTrue(requests[0]['stream_options']['include_usage'])


class GraphInterruptionTests(unittest.TestCase):
    setUp = fixtures.ConversationTests.setUp
    invoke = fixtures.ConversationTests.invoke

    def interrupted(self, kind, text=''):
        client = WaitingClient()
        model = ConfiguredModel(client)
        with ThreadPoolExecutor() as pool:
            future = pool.submit(self.invoke, model, '原始要求')
            self.assertTrue(client.started.wait(3))
            started = time.monotonic()
            value = control_value(self.thread, {'id': 'control-one', 'kind': kind, 'text': text})
            self.assertTrue(json.loads(value.body)['received'])
            result = future.result(timeout=5)
            self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(client.closed.is_set())
        task = self.root / 'runs' / 'tasks' / result['task_id']
        archive = LocalHistory(task)
        with archive.lease():
            state = archive.load()
        archive.close()
        self.assertIsNone(state['request_pending'])
        self.assertEqual(state['interrupted_requests'][0]['call'], 1)
        self.assertTrue(state['interrupted_requests'][0]['request_sent'])
        self.assertFalse((task / 'requests/call-0001/response.json').exists())
        self.assertFalse(list(task.glob('plans/*/plan.json')))
        self.assertFalse(
            any(
                c.get('id') == 'partial'
                for m in state['history']
                for c in m['data'].get('tool_calls', [])
            )
        )
        self.assertEqual(state['budget']['unknown_usage_steps'], 1)
        from agent.storage.activity import read_activity

        view = read_activity(
            self.root / 'runs', [{'id': result['task_id'], 'input_id': 'user-one'}], {}
        )
        interrupted = [r for r in view['items'] if r['status'] == 'interrupted']
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0]['detail'], '部分思考')
        return result, model, client, task

    def test_pause_returns_without_reply_and_explicit_resume_uses_new_call(self):
        result, model, client, task = self.interrupted('pause')
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.store.session(self.thread)['status'], 'paused')
        control_value(self.thread, {'id': 'resume', 'kind': 'resume'})
        with patch('tests.legacy.graph.model_factory', return_value=model):
            resumed = self.compiled.invoke({'messages': []}, self.config)
        self.assertEqual(resumed['status'], 'chat_complete')
        self.assertEqual(len(client.calls), 2)
        self.assertTrue((task / 'requests/call-0002/response.json').exists())
        self.assertTrue((task / 'requests/call-0001/interruption.json').exists())

    def test_supplement_interrupts_and_is_in_next_request_without_duplicate_input(self):
        result, _, client, _ = self.interrupted('message', '新要求：只解释，不读取')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(client.calls), 2)
        self.assertIn('新要求：只解释，不读取', repr(client.calls[1][0]))
        public = result['messages']
        self.assertEqual(sum(m.type == 'human' and m.id == 'control-one' for m in public), 1)
        self.assertTrue(any(m.type == 'human' and '原始要求' in m.content for m in public))
        self.assertFalse(self.store.pending(self.thread))

    def test_interrupting_a_supplement_retains_both_user_messages(self):
        paused = fixtures.ScriptedModel(
            [fixtures.start_reply(), fixtures.investigation(['first.py'])],
            hook=lambda n: self.store.enqueue(self.thread, 'initial-pause', 'pause'),
        )
        self.assertEqual(self.invoke(paused, '最初要求')['status'], 'paused')
        client = WaitingClient()
        with ThreadPoolExecutor() as pool:
            future = pool.submit(self.invoke, ConfiguredModel(client), '第一条补充', 'middle-input')
            self.assertTrue(client.started.wait(3))
            control_value(
                self.thread, {'id': 'latest-input', 'kind': 'message', 'text': '第二条补充：只讨论'}
            )
            result = future.result(timeout=5)
        self.assertEqual(len(client.calls), 2)
        self.assertIn('第一条补充', repr(client.calls[-1][0]))
        self.assertIn('第二条补充：只讨论', repr(client.calls[-1][0]))
        for identity in ('middle-input', 'latest-input'):
            self.assertEqual(
                sum(m.id == identity and m.type == 'human' for m in result['messages']), 1
            )
        self.assertFalse(
            list((self.root / 'runs' / 'tasks' / result['task_id']).glob('plans/*/plan.json'))
        )


if __name__ == '__main__':
    unittest.main()
