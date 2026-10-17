"""用户控制可取消网络等待、保留记录，且绝不执行回复片段。"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time
import unittest

import httpx
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from agent.model.client import ConfiguredModel
from agent.model.deepseek import DeepSeekClient
from agent.model.errors import ModelInterrupted
from agent.model.payload import RAW_TOOL_CALLS, wire_input
from agent.task import control as interrupts
from support import read_record


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


if __name__ == '__main__':
    unittest.main()
