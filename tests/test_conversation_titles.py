"""Auxiliary titles cannot delay, replay or rewrite work conversations."""
import os
import json
from pathlib import Path
import tempfile
from threading import Event
import unittest
import uuid
from unittest.mock import Mock, patch
import httpx

from langchain_core.messages import AIMessage, HumanMessage
from tests.http_client import TestClient

from agent.model.client import ConfiguredModel
from agent.server import titles
from agent.server.titles import model_factory as real_title_model_factory
from agent.server.app import app
from agent.server.graph import intake
from agent.storage.conversations import ConversationStore
from agent.storage.purge import begin_purge, finish_purge
from agent.storage.titles import Titles


def response(content='修复会话记录与标题更新', **metadata):
    return AIMessage(content=content, response_metadata={'finish_reason': 'stop', **metadata},
                     usage_metadata={'input_tokens': 40, 'output_tokens': 9, 'total_tokens': 49,
                                     'input_token_details': {'cache_read': 30}})


class ConversationTitleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root, self.thread = Path(temp.name), str(uuid.uuid4())
        self.store = ConversationStore(self.root)
        self.records = Titles(self.root)
        self.session = {'status': 'chat_complete', 'task_id': 'a' * 32, 'last_input': 'u1',
                        'repo': '', 'request': '公开要求', 'public_messages': []}

    def dispatcher(self, model):
        dispatcher = titles.TitleDispatcher(self.root, lambda: model)
        self.addCleanup(dispatcher.close)
        return dispatcher

    def reserve(self, text='请修复会话列表', thread=None):
        thread = thread or self.thread
        return self.store.save_session(thread, self.session, title_request=titles.reservation('u1', text))

    def test_once_only_and_new_turn_does_not_replace_title_or_change_session(self):
        model = Mock(); model.invoke.return_value = response()
        dispatcher = self.dispatcher(model)
        self.assertTrue(self.reserve())
        dispatcher.submit(self.thread).result(timeout=2)
        first = self.records.get(self.thread)
        self.assertEqual(first['status'], 'complete')
        self.assertFalse(self.reserve('第二轮另一个问题'))
        dispatcher.submit(self.thread).result(timeout=2)
        model.invoke.assert_called_once()
        self.assertEqual(self.records.get(self.thread), first)
        self.assertEqual(self.store.session(self.thread), self.session)

    def test_only_first_public_message_is_passed_and_input_is_bounded(self):
        model = Mock(); model.invoke.return_value = response()
        dispatcher = self.dispatcher(model)
        text = '公开需求' * 1200
        self.reserve(text)
        self.store.save_session(self.thread, {**self.session, 'history': ['PRIVATE_REASONING'],
                                             'secret_test': 'PRIVATE_TEST'})
        dispatcher.submit(self.thread).result(timeout=2)
        messages = model.invoke.call_args.args[0]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1].content, text[:4000])
        self.assertNotIn('PRIVATE_', str(messages))
        self.assertNotIn('secret_test', str(self.records.get(self.thread)))

    def test_exact_flash_disabled_request_and_no_tools_or_retry(self):
        with patch('agent.server.titles.load_settings', return_value={'api_key': 'SECRET'}) as load, \
                patch('agent.server.titles.create_model') as create:
            real_title_model_factory()
        self.assertEqual(load.call_args.args[0], {'model_name': 'deepseek-flash', 'reasoning_effort': 'none', 'online': False})
        self.assertEqual(create.call_args.args[1], titles.POLICY)
        client = Mock(); client.invoke.return_value = response()
        model = ConfiguredModel(client, titles.POLICY)
        model.invoke([HumanMessage(content='公开要求')])
        self.assertEqual(client.invoke.call_args.kwargs, {'max_tokens': 128, 'timeout': 15,
                         'extra_body': {'thinking': {'type': 'disabled'}}})
        client.bind_tools.assert_not_called()
        self.assertEqual(titles.reservation('u1', 'test')['request']['max_retries'], 0)

    def test_real_sdk_wire_is_text_only_disabled_flash_and_usage_persisted(self):
        bodies = []
        def respond(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'offline-title-sdk', 'object': 'chat.completion',
                'created': 1, 'model': 'deepseek-flash', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '修复会话列表和自动标题'}}],
                'usage': {'prompt_tokens': 40, 'completion_tokens': 9, 'total_tokens': 49,
                          'prompt_cache_hit_tokens': 30, 'prompt_cache_miss_tokens': 10}})
        with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                patch('agent.model.network.model_http_clients', return_value={'http_client': client}), \
                patch('agent.server.titles.load_settings', return_value={'model_name': 'deepseek-flash', 'api_key': 'offline-secret'}):
            dispatcher = titles.TitleDispatcher(self.root, real_title_model_factory)
            try:
                self.reserve()
                dispatcher.submit(self.thread).result(timeout=3)
            finally:
                dispatcher.close()
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        self.assertEqual(body['model'], 'deepseek-flash')
        self.assertEqual(body['max_tokens'], 128)
        self.assertEqual(body['thinking'], {'type': 'disabled'})
        for key in ('tools', 'tool_choice', 'response_format', 'reasoning_effort'):
            self.assertNotIn(key, body)
        saved = self.records.get(self.thread)
        self.assertEqual(saved['status'], 'complete')
        self.assertEqual(saved['response']['data']['response_metadata']['model_name'], 'deepseek-flash')
        self.assertEqual(saved['usage']['total_tokens'], 49)
        self.assertNotIn('offline-secret', json.dumps(saved))

    def test_background_worker_is_nonblocking_and_concurrent_duplicate_calls_once(self):
        entered, release = Event(), Event()
        model = Mock()
        def invoke(messages):
            entered.set()
            if not release.wait(3): raise TimeoutError('test release missing')
            return response()
        model.invoke.side_effect = invoke
        dispatcher = self.dispatcher(model)
        self.addCleanup(release.set)
        self.reserve()
        future = dispatcher.submit(self.thread)
        self.assertTrue(entered.wait(2))
        self.assertFalse(future.done())
        duplicate = dispatcher.submit(self.thread)
        duplicate.result(timeout=2)
        self.assertEqual(self.store.session(self.thread), self.session)
        self.assertEqual(self.records.public(self.thread)['status'], 'running')
        release.set(); future.result(timeout=2)
        model.invoke.assert_called_once()

    def test_failure_keeps_fallback_and_unknown_is_never_replayed_on_restart(self):
        model = Mock(); model.invoke.side_effect = TimeoutError('may have been billed')
        dispatcher = self.dispatcher(model)
        self.reserve('请修复会话列表')
        dispatcher.submit(self.thread).result(timeout=2)
        record = self.records.get(self.thread)
        self.assertEqual(record['status'], 'unknown')
        self.assertEqual(record['title'], '请修复会话列表')
        dispatcher.close()
        next_model = Mock()
        restart = self.dispatcher(next_model)
        restart.submit(self.thread).result(timeout=2)
        next_model.invoke.assert_not_called()
        usage = self.records.activity(self.thread)['usage']['title:' + self.thread]
        self.assertEqual((usage['calls'], usage['unknown_calls']), (1, 1))
        self.assertIsNone(usage['context'])

    def test_missing_key_is_not_a_paid_call_and_invalid_response_is_recorded(self):
        dispatcher = titles.TitleDispatcher(self.root, lambda: (_ for _ in ()).throw(ValueError('missing key')))
        self.addCleanup(dispatcher.close)
        self.reserve(); dispatcher.submit(self.thread).result(timeout=2)
        record = self.records.get(self.thread)
        self.assertEqual(record['status'], 'failed')
        self.assertFalse(record['request_sent'])
        self.assertEqual(self.records.activity(self.thread)['usage']['title:' + self.thread]['calls'], 0)
        other = str(uuid.uuid4())
        self.reserve(thread=other)
        dispatcher.factory = lambda: Mock(invoke=Mock(return_value=response('', finish_reason='length')))
        dispatcher.submit(other).result(timeout=2)
        record = self.records.get(other)
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['response']['data']['response_metadata']['finish_reason'], 'length')
        self.assertEqual(record['usage']['total_tokens'], 49)

    def test_startup_settles_incomplete_title_without_backfilling_old_threads(self):
        self.reserve()
        self.records.update(self.thread, {'pending'}, status='running', request_sent=True)
        old = str(uuid.uuid4())
        self.store.save_session(old, self.session)
        model = Mock()
        dispatcher = self.dispatcher(model)
        self.assertEqual(self.records.get(self.thread)['status'], 'unknown')
        self.assertIsNone(self.records.get(old))
        dispatcher.submit(self.thread).result(timeout=2)
        model.invoke.assert_not_called()

    def test_archive_keeps_title_and_late_response_cannot_recreate_purged_record(self):
        entered, release = Event(), Event()
        model = Mock()
        def invoke(messages):
            entered.set(); release.wait(3)
            return response()
        model.invoke.side_effect = invoke
        dispatcher = self.dispatcher(model); self.addCleanup(release.set)
        self.reserve(); self.store.set_visibility(self.thread, 'archived')
        self.assertIsNotNone(self.records.public(self.thread))
        future = dispatcher.submit(self.thread)
        self.assertTrue(entered.wait(2))
        self.store.set_visibility(self.thread, 'deleted')
        identities = begin_purge(self.store, self.thread)
        finish_purge(self.store, self.thread, identities)
        release.set(); future.result(timeout=2)
        self.assertIsNone(self.records.get(self.thread))
        self.assertIsNone(self.records.public(self.thread))
        self.assertEqual(self.store.visibility()[self.thread], 'purged')

    def test_service_lifespan_intake_schedules_and_read_endpoints_are_passive(self):
        model = Mock(); model.invoke.return_value = response()
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), \
                patch('agent.server.titles.model_factory', return_value=model), TestClient(app) as client:
            output = intake({'messages': [HumanMessage(content='请修复会话列表', id='u1')]},
                            {'configurable': {'thread_id': self.thread}})
            self.assertEqual(output['status'], 'needs_plan')
            titles._dispatcher.pool.submit(lambda: None).result(timeout=2)
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), TestClient(app) as client:
            listing = client.get('/conversations').json()
            activity = client.get(f'/threads/{self.thread}/activity').json()
            self.assertEqual(listing['titles'][self.thread]['title'], response().content)
            self.assertEqual(activity['title'], listing['titles'][self.thread])
            self.assertFalse(activity['title_pending'])
            usage = activity['usage']['title:' + self.thread]
            self.assertEqual((usage['total_tokens'], usage['cached_input_tokens']), (49, 30))
            self.assertEqual(usage['purpose'], 'title')
            self.assertIsNone(usage['context'])
            self.assertIsInstance(activity['title']['revision'], int)
            client.get('/conversations'); client.get(f'/threads/{self.thread}/activity')
        model.invoke.assert_called_once()

    def test_graph_without_service_lifespan_does_not_create_model(self):
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), \
                patch('agent.server.titles.model_factory') as factory:
            intake({'messages': [HumanMessage(content='你好', id='u1')]},
                   {'configurable': {'thread_id': self.thread}})
        factory.assert_not_called()
        self.assertEqual(self.records.get(self.thread)['status'], 'failed')

    def test_title_prompt_failure_still_saves_and_starts_main_session(self):
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), \
                patch('agent.server.titles.PROJECT_ROOT', self.root / 'missing-prompt'):
            output = intake({'messages': [HumanMessage(content='你好', id='u1')]},
                            {'configurable': {'thread_id': self.thread}})
        self.assertEqual(output['status'], 'needs_plan')
        self.assertIsNotNone(self.store.session(self.thread))
        self.assertEqual(self.records.get(self.thread)['status'], 'failed')

    def test_failed_dispatch_submission_stops_polling_without_api_or_task_failure(self):
        self.reserve()
        dispatcher = Mock(root=self.root)
        dispatcher.submit.side_effect = RuntimeError('executor is closed')
        with patch('agent.server.titles._dispatcher', dispatcher):
            titles.schedule(self.root, self.thread)
        self.assertEqual(self.records.get(self.thread)['status'], 'failed')
        self.assertFalse(self.records.activity(self.thread)['title_pending'])
        self.assertFalse(self.records.get(self.thread)['request_sent'])

    def test_full_queue_has_bounded_workers_and_rejects_excess_without_waiting(self):
        release = Event()
        model = Mock()
        model.invoke.side_effect = lambda messages: release.wait(3) and response()
        dispatcher = self.dispatcher(model); self.addCleanup(release.set)
        futures = []
        for _ in range(16):
            thread = str(uuid.uuid4()); self.reserve(thread=thread)
            futures.append(dispatcher.submit(thread))
        self.reserve()
        self.assertIsNone(dispatcher.submit(self.thread))
        self.assertEqual(self.records.get(self.thread)['error_type'], 'TitleQueueUnavailable')
        self.assertFalse(self.records.get(self.thread)['request_sent'])
        self.assertLessEqual(model.invoke.call_count, 2)
        release.set()
        for future in futures:
            future.result(timeout=2)

