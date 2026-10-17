"""使用模拟 HTTP 验证 Search / Fetch 的真实边界，不调用付费服务。"""

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agent.execution.web import (
    WebProvider, WebUnknown, public_url, search_sources, validate_web_args,
)
from agent.model.client import ModelPolicy
from agent.model.errors import ModelInterrupted
from agent.task import control
from agent.task.budget import StepBudget


class WebRetrievalTests(unittest.TestCase):
    def loop(self, **settings):
        return SimpleNamespace(
            runtime_settings={'model_name': 'deepseek-flash', 'reasoning_effort': 'high',
                              'online': True, **settings},
            budget=StepBudget(), state={}, persist=lambda *args: None,
        )

    @contextmanager
    def mocked_provider(self, response):
        self.requests = []

        def respond(request):
            self.requests.append(request)
            return response(request) if callable(response) else httpx.Response(200, json=response)

        with tempfile.TemporaryDirectory() as name, httpx.Client(
            transport=httpx.MockTransport(respond),
        ) as client, patch(
            'agent.execution.web.load_settings', return_value={'api_key': 'offline-secret'},
        ), patch(
            'agent.execution.web.model_http_clients', return_value={'http_client': client},
        ) as factory:
            yield WebProvider(), self.loop(), Path(name), factory

    def entry(self, url='https://example.com/a', title='标题'):
        return {'url': url, 'title': title, 'snippet': '短摘要', 'summary': '完整资料说明',
                'content': '', 'rerank_score': 2.5, 'authority': 3, 'site_name': None,
                'site_icon': None, 'date': None}

    def test_search_posts_original_query_without_a_chat_model_request(self):
        query = ' 中文 foo.bar OR exact_word site:example.com '
        raw = {'results': [self.entry()]}
        with self.mocked_provider(raw) as (provider, loop, folder, factory):
            result = provider.search(query, loop, folder)
            self.assertEqual(result['sources'], raw['results'])
            self.assertFalse(result['truncated'])
            self.assertEqual(loop.budget.used_steps, 0)
            self.assertEqual(loop.budget.reported_tokens, 0)
            self.assertEqual(loop.budget.service_calls, 1)
            row = loop.state['web_usage'][0]
            self.assertEqual(row['kind'], 'service')
            self.assertEqual(row['cost_micros'], 5000)
            self.assertEqual(row['reserved_cost_micros'], 0)
            self.assertNotIn('usage', row)
            self.assertIsNone(loop.state['web_request_pending'])
            self.assertEqual(json.loads((folder / 'provider-response.json').read_bytes()), raw)
            self.assertNotIn('offline-secret', (folder / 'provider-request.json').read_text())
        request = self.requests[0]
        self.assertEqual(json.loads(request.content),
                         {'query': query, 'count': 10, 'need_content': False})
        self.assertEqual(request.headers['authorization'], 'Bearer offline-secret')
        self.assertEqual(request.headers['content-type'], 'application/json')
        self.assertNotIn('x-api-key', request.headers)
        self.assertEqual(str(request.url), 'https://api.deepseek.com/v1/tools/search')
        self.assertEqual(factory.call_args.args, (90,))

    def test_search_same_url_keeps_latest_full_entry_at_latest_position(self):
        old = {**self.entry(title='old'), 'obsolete_extra': 'not preserved'}
        latest = self.entry(title='new')
        latest['summary'] = 'updated'
        raw = {'results': [old, self.entry('https://example.com/b'), latest]}
        original = deepcopy(raw)
        result = search_sources(raw)
        self.assertEqual(result['sources'], raw['results'][1:])
        self.assertNotIn('obsolete_extra', result['sources'][-1])
        self.assertEqual(raw, original)

    def test_provider_does_not_silently_cut_returned_sources_at_ten(self):
        entries = [self.entry('https://example.com/' + str(index)) for index in range(11)]
        result = search_sources({'results': entries})
        self.assertEqual(result['sources'], entries)
        self.assertFalse(result['truncated'])

    def test_provider_archive_preserves_all_occurrences_and_response_bytes(self):
        raw = {'results': [self.entry(title='old'), self.entry(title='new')]}
        body = json.dumps(raw, ensure_ascii=False, indent=4).encode('utf-8')
        with self.mocked_provider(lambda request: httpx.Response(200, content=body)) as values:
            provider, loop, folder, _ = values
            result = provider.search('query', loop, folder)
            self.assertEqual(result['sources'], raw['results'][1:])
            self.assertEqual((folder / 'provider-response.json').read_bytes(), body)

    def test_empty_search_is_known_and_free(self):
        with self.mocked_provider({'results': []}) as (provider, loop, folder, _):
            self.assertEqual(provider.search('missing', loop, folder)['sources'], [])
            self.assertEqual(loop.state['web_usage'][0]['status'], 'empty')
            self.assertEqual(loop.state['web_usage'][0]['cost_micros'], 0)
            self.assertEqual(loop.budget.unknown_usage_steps, 0)

    def test_search_query_limits_are_unicode_not_bytes_and_never_truncated(self):
        self.assertEqual(validate_web_args('web_search', {'query': '文' * 128}), '文' * 128)
        for query in ('', '  \t', 'x' * 129, None, 5):
            with self.subTest(query=query), self.assertRaises(ValueError):
                validate_web_args('web_search', {'query': query})
        for args in ({}, {'query': 'x', 'count': 5}, {'url': 'https://example.com'}, []):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_web_args('web_search', args)

    def test_fetch_posts_to_provider_and_preserves_full_markdown(self):
        markdown = '# 中文\n\n' + 'x' * 25000 + '\n```python\nprint("x")\n```\n'
        raw = {'title': '标题', 'content': markdown}
        url = 'https://example.com/page?foo=a%20b'
        with self.mocked_provider(raw) as (provider, loop, folder, _), patch(
            'socket.getaddrinfo', side_effect=AssertionError('No local target DNS lookup'),
        ):
            result = provider.fetch(url, loop, folder)
            self.assertEqual(result['text'], markdown)
            self.assertEqual(result['title'], raw['title'])
            self.assertEqual(result['format'], 'markdown')
            self.assertFalse(result['truncated'])
            self.assertFalse((folder / 'page.body').exists())
            self.assertEqual(loop.budget.used_steps, 0)
            self.assertEqual(loop.state['web_usage'][0]['cost_micros'], 5000)
        request = self.requests[0]
        self.assertEqual(str(request.url), 'https://api.deepseek.com/v1/tools/fetch')
        self.assertEqual(json.loads(request.content), {'url': url})
        self.assertEqual(request.headers['authorization'], 'Bearer offline-secret')

    def test_fetch_empty_body_is_known_and_free(self):
        with self.mocked_provider({'title': '', 'content': ''}) as values:
            provider, loop, folder, _ = values
            self.assertEqual(provider.fetch('https://example.com', loop, folder)['text'], '')
            self.assertEqual(loop.state['web_usage'][0]['status'], 'empty')
            self.assertEqual(loop.state['web_usage'][0]['cost_micros'], 0)

    def test_private_credentials_and_invalid_urls_are_rejected_before_request(self):
        urls = ('http://localhost/secret', 'http://127.0.0.1/', 'http://10.0.0.1/',
                'http://[::1]/', 'http://[fe80::1]/', 'http://machine.local/',
                'http://2130706433/', 'http://user:pass@example.com/', 'file:///local',
                'https://example.com:70000/', 'https://example.com/\nsecret', '', None,
                'https://example.com/' + 'x' * 8192)
        with self.mocked_provider({'title': '', 'content': ''}) as values:
            provider, loop, folder, _ = values
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    provider.fetch(url, loop, folder)
            self.assertEqual(self.requests, [])
            self.assertEqual(loop.budget.service_calls, 0)
        url = 'https://example.com/' + 'x' * (8192 - len('https://example.com/'))
        self.assertEqual(public_url(url), url)

    def test_transport_unknown_retains_fee_and_cannot_replay(self):
        def timeout(request):
            raise httpx.ReadTimeout('offline timeout')

        with self.mocked_provider(timeout) as (provider, loop, folder, _):
            with self.assertRaises(WebUnknown):
                provider.search('query', loop, folder)
            pending = loop.state['web_request_pending']
            self.assertEqual(pending['reserved_cost_micros'], 5000)
            self.assertEqual(loop.state['web_usage'][0]['status'], 'unknown')
            self.assertIsNone(loop.state['web_usage'][0]['cost_micros'])
            self.assertEqual(loop.budget.unknown_usage_steps, 0)
            with self.assertRaises(WebUnknown):
                provider.search('query', loop, folder)
            self.assertEqual(len(self.requests), 1)

    def test_http_failures_are_free_archived_and_never_retried(self):
        for code in (400, 401, 402, 403, 404, 408, 429, 500, 504, 302):
            with self.subTest(code=code), self.mocked_provider(
                lambda request: httpx.Response(code, text='error',
                                               headers={'location': 'https://other.example/'}),
            ) as values:
                provider, loop, folder, _ = values
                with self.assertRaisesRegex(ValueError, 'HTTP ' + str(code)):
                    provider.search('query', loop, folder)
                self.assertIsNone(loop.state['web_request_pending'])
                self.assertEqual(loop.state['web_usage'][0]['status'], 'failed')
                self.assertEqual(loop.state['web_usage'][0]['cost_micros'], 0)
                self.assertEqual((folder / 'provider-response.json').read_bytes(), b'error')
                self.assertEqual(len(self.requests), 1)

    def test_malformed_success_preserves_original_and_unknown_reservation(self):
        responses = ({'content': [{'type': 'text', 'text': 'old chat API format'}]},
                     {'results': [{'url': 'http://localhost'}]}, {'results': [42]},
                     {'results': [{'url': 'https://example.com', 'summary': 42}]})
        for raw in responses:
            with self.subTest(raw=raw), self.mocked_provider(raw) as values:
                provider, loop, folder, _ = values
                with self.assertRaises(WebUnknown):
                    provider.search('query', loop, folder)
                self.assertIsNotNone(loop.state['web_request_pending'])
                self.assertEqual(loop.state['web_usage'][0]['status'], 'unknown')
                self.assertEqual(json.loads((folder / 'provider-response.json').read_bytes()), raw)
        with self.mocked_provider(
            lambda request: httpx.Response(200, content=b'invalid'),
        ) as values:
            provider, loop, folder, _ = values
            with self.assertRaises(WebUnknown):
                provider.fetch('https://example.com', loop, folder)
            self.assertEqual((folder / 'provider-response.json').read_bytes(), b'invalid')

    def test_known_provider_error_clears_pending_and_cost(self):
        with self.mocked_provider({'error': {'message': 'failure'}}) as values:
            provider, loop, folder, _ = values
            with self.assertRaisesRegex(ValueError, 'error response'):
                provider.fetch('https://example.com', loop, folder)
            self.assertIsNone(loop.state['web_request_pending'])
            self.assertEqual(loop.state['web_usage'][0]['cost_micros'], 0)

    def test_network_disabled_never_loads_key_or_creates_request(self):
        with self.mocked_provider({'results': []}) as (provider, loop, folder, factory):
            loop.runtime_settings['online'] = False
            with self.assertRaisesRegex(ValueError, 'disabled'):
                provider.search('query', loop, folder)
            self.assertEqual(factory.call_count, 0)
            self.assertEqual(loop.budget.service_calls, 0)

    def test_explicit_timeout_and_remaining_task_limit(self):
        with self.mocked_provider({'results': []}) as (provider, loop, folder, factory):
            loop.model = SimpleNamespace(policy=ModelPolicy(api_timeout=42), timeout=None)
            loop.budget.remaining_time = lambda *args: 12
            provider.search('query', loop, folder)
            self.assertEqual(factory.call_args.args, (12,))

    def test_cancel_before_dispatch_never_reserves_or_sends(self):
        signal = control.Signal()
        signal.interrupt('pause')
        with self.mocked_provider({'results': []}) as values, control.using(signal):
            provider, loop, folder, factory = values
            with self.assertRaises(ModelInterrupted) as caught:
                provider.search('query', loop, folder)
            self.assertFalse(caught.exception.sent)
            self.assertEqual(factory.call_count, 0)
            self.assertEqual(loop.budget.service_calls, 0)

    def test_inflight_pause_cancels_async_request_and_retains_unknown_reservation(self):
        import asyncio
        import threading

        signal = control.Signal()
        started, closed = threading.Event(), threading.Event()
        requests = []

        async def waiting(request):
            requests.append(request)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        def pause():
            if started.wait(2):
                signal.interrupt('pause')

        worker = threading.Thread(target=pause)
        worker.start()
        self.addCleanup(worker.join, 3)
        client = httpx.Client(transport=httpx.MockTransport(lambda request: None))
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(waiting))
        with tempfile.TemporaryDirectory() as name, patch(
            'agent.execution.web.load_settings', return_value={'api_key': 'offline'},
        ), patch('agent.execution.web.model_http_clients', return_value={
            'http_client': client, 'http_async_client': async_client,
        }), control.using(signal):
            loop = self.loop()
            with self.assertRaises(WebUnknown):
                WebProvider().fetch('https://example.com', loop, Path(name))
            self.assertTrue(closed.wait(1))
            self.assertTrue(async_client.is_closed)
            self.assertEqual(len(requests), 1)
            self.assertIsNotNone(loop.state['web_request_pending'])
            self.assertEqual(loop.state['web_usage'][0]['status'], 'unknown')

    def test_pre_dispatch_transport_setup_failure_releases_fee_with_not_sent_event(self):
        events = []
        with self.mocked_provider({'results': []}) as (provider, loop, folder, factory):
            loop.persist = lambda event, payload: events.append((event, payload))
            factory.side_effect = OSError('cannot create client')
            with self.assertRaisesRegex(OSError, 'cannot create client'):
                provider.search('query', loop, folder)
            self.assertEqual(loop.budget.service_calls, 0)
            self.assertIsNone(loop.state['web_request_pending'])
            self.assertEqual(loop.state['web_usage'][0]['status'], 'not_sent')
            self.assertEqual(events[0][1]['reserved_cost_micros'], 5000)
            self.assertEqual(events[-1][0], 'web_request_not_sent')
            self.assertEqual(self.requests, [])

    def test_received_response_archive_failure_retains_unknown_fee_without_replay(self):
        with self.mocked_provider({'results': []}) as (provider, loop, folder, _):
            with patch('pathlib.Path.write_bytes', side_effect=OSError('disk full')):
                with self.assertRaises(WebUnknown):
                    provider.search('query', loop, folder)
            self.assertIsNotNone(loop.state['web_request_pending'])
            self.assertEqual(loop.state['web_usage'][0]['status'], 'unknown')
            self.assertEqual(len(self.requests), 1)
