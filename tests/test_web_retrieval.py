"""Real HTTP adapter replay, using mock transport only, never paid search."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import httpx
from agent.execution.web import PageText, WebProvider, WebUnknown, public_url, search_sources
from agent.task.budget import StepBudget


class WebRetrievalTests(unittest.TestCase):
    def loop(self):
        return SimpleNamespace(runtime_settings={'model_name': 'deepseek-flash', 'reasoning_effort': 'high', 'online': True},
            budget=StepBudget(), state={}, persist=lambda *a: None)

    def test_search_uses_dsh_shape_real_blocks_and_records_usage_without_key(self):
        requests = []
        raw = {'content': [{'type': 'web_search_tool_result', 'content': [
            {'type': 'web_search_result', 'url': 'https://example.com', 'title': '标题'}]},
            {'type': 'text', 'text': 'This generated summary is not evidence', 'citations': [
                {'url': 'https://example.com', 'cited_text': '真实原文'}]}],
               'usage': {'input_tokens': 10, 'output_tokens': 20, 'cache_read_input_tokens': 5}}
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=raw)
        with tempfile.TemporaryDirectory() as name, httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                patch('agent.execution.web.load_settings', return_value={'api_key': 'offline-secret'}), \
                patch('agent.execution.web.model_http_clients', return_value={'http_client': client}):
            loop = self.loop()
            result = WebProvider().search('中文 query', loop, Path(name))
            self.assertEqual(result['sources'][0]['snippet'], '真实原文')
            self.assertNotIn('generated summary', repr(result))
            self.assertEqual(loop.budget.used_steps, 1)
            self.assertEqual(loop.budget.reported_tokens, 35)
            self.assertEqual(loop.state['web_usage'][0]['usage']['input_token_details']['cache_read'], 5)
            self.assertNotIn('offline-secret', (Path(name)/'provider-request.json').read_text())
        body = json.loads(requests[0].content)
        self.assertEqual(body['max_tokens'], 4096)
        self.assertEqual(body['tools'][0]['max_uses'], 5)
        self.assertEqual(requests[0].headers['x-api-key'], 'offline-secret')
        self.assertEqual(str(requests[0].url), 'https://api.deepseek.com/anthropic/v1/messages')

    def test_missing_search_blocks_never_become_fabricated_results(self):
        with self.assertRaisesRegex(ValueError, 'not a search result'):
            search_sources({'content': [{'type': 'text', 'text': 'made up answer'}]})
        self.assertEqual(search_sources({'content': [{'type': 'web_search_tool_result', 'content': []}]})['sources'], [])

    def test_page_text_drops_scripts_and_preserves_unicode_and_code(self):
        parser = PageText()
        parser.feed('<h1>中文</h1><script>evil()</script><style>hide</style><pre>print("x")\nback\\slash</pre>')
        self.assertEqual(parser.text(), '中文\nprint("x")\nback\\slash')

    def test_local_url_and_redirect_are_rejected_without_local_http(self):
        with patch('agent.execution.public_http.socket.getaddrinfo', return_value=[(0, 0, 0, '', ('127.0.0.1', 80))]):
            with self.assertRaises(ValueError):
                public_url('http://localhost/secret')
        for url in ('file:///local', 'http://user:pass@example.com', ''):
            with self.assertRaises(ValueError):
                public_url(url)

    def test_fetch_records_body_no_auth_header_and_never_executes_scripts(self):
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, headers={'content-type': 'text/html; charset=utf-8'}, content='<p>公开内容</p><script>evil()</script>'.encode())
        with tempfile.TemporaryDirectory() as name, httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                patch('agent.execution.web.public_client', return_value=client), \
                patch('agent.execution.web.public_url', side_effect=lambda value: value):
            loop = self.loop()
            result = WebProvider().fetch('https://example.com', loop, Path(name))
            self.assertEqual(result['text'], '公开内容')
            self.assertTrue((Path(name)/'page.body').exists())
            self.assertEqual(loop.budget.used_steps, 0)
        self.assertNotIn('authorization', requests[0].headers)
        self.assertNotIn('x-api-key', requests[0].headers)

    def test_dispatched_search_timeout_is_unknown_and_not_retried(self):
        calls = []
        def respond(request):
            calls.append(request)
            raise httpx.ReadTimeout('offline timeout')
        with tempfile.TemporaryDirectory() as name, httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                patch('agent.execution.web.load_settings', return_value={'api_key': 'offline'}), \
                patch('agent.execution.web.model_http_clients', return_value={'http_client': client}):
            loop = self.loop()
            with self.assertRaises(WebUnknown):
                WebProvider().search('query', loop, Path(name))
            self.assertEqual(loop.budget.unknown_usage_steps, 1)
            self.assertIsNotNone(loop.state['web_request_pending'])
        self.assertEqual(len(calls), 1)

    def test_known_http_failure_clears_inflight_without_fabricated_search_result(self):
        with tempfile.TemporaryDirectory() as name, httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(503, text='Service unavailable'))) as client, \
                patch('agent.execution.web.load_settings', return_value={'api_key': 'offline'}), \
                patch('agent.execution.web.model_http_clients', return_value={'http_client': client}):
            loop = self.loop()
            with self.assertRaisesRegex(ValueError, 'HTTP 503'):
                WebProvider().search('query', loop, Path(name))
            self.assertIsNone(loop.state['web_request_pending'])
            self.assertEqual(loop.budget.unknown_usage_steps, 1)
            self.assertEqual(len(loop.state['web_usage']), 1)

    def test_redirect_to_private_address_stops_before_second_http_request(self):
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(302, headers={'location': 'http://127.0.0.1/private'})
        def validate(url):
            if '127.0.0.1' in url:
                raise ValueError('Private pages are not permitted')
            return url
        with tempfile.TemporaryDirectory() as name, httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                patch('agent.execution.web.public_client', return_value=client), \
                patch('agent.execution.web.public_url', side_effect=validate):
            with self.assertRaisesRegex(ValueError, 'Private'):
                WebProvider().fetch('https://example.com', self.loop(), Path(name))
        self.assertEqual(len(requests), 1)
