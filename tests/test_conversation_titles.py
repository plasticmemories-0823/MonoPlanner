"""当前应用标题使用独立的一次性请求，读取接口不触发新生成。"""

import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid
from unittest.mock import Mock, patch
from langchain_core.messages import AIMessage, HumanMessage
from tests.http_client import TestClient
from agent.model.client import ConfiguredModel
from agent.application import titles
from agent.application.titles import model_factory as real_title_model_factory
from agent.server.app import app


def response(content='修复会话记录与标题更新', **metadata):
    return AIMessage(
        content=content,
        response_metadata={'finish_reason': 'stop', **metadata},
        usage_metadata={
            'input_tokens': 40,
            'output_tokens': 9,
            'total_tokens': 49,
            'input_token_details': {'cache_read': 30},
        },
    )


class ConversationTitleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root, self.thread = Path(temporary.name), str(uuid.uuid4())

    def test_exact_flash_disabled_request_and_no_tools_or_retry(self):
        with patch(
            'agent.application.titles.load_settings', return_value={'api_key': 'SECRET'}
        ) as load, patch('agent.application.titles.create_model') as create:
            real_title_model_factory()
        self.assertEqual(
            load.call_args.args[0],
            {'model_name': 'deepseek-flash', 'reasoning_effort': 'none', 'online': False},
        )
        self.assertEqual(create.call_args.args[1], titles.POLICY)
        client = Mock()
        client.invoke.return_value = response()
        model = ConfiguredModel(client, titles.POLICY)
        model.invoke([HumanMessage(content='公开要求')])
        self.assertEqual(
            client.invoke.call_args.kwargs,
            {'max_tokens': 128, 'timeout': 15, 'extra_body': {'thinking': {'type': 'disabled'}}},
        )
        client.bind_tools.assert_not_called()


    def test_service_lifespan_intake_schedules_and_read_endpoints_are_passive(self):
        from agent.application.api import get_service

        class ChatModel:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                return AIMessage(content='离线普通回复', response_metadata={'finish_reason': 'stop'})

        model = Mock()
        model.invoke.return_value = response()
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), patch(
            'agent.application.titles.model_factory', return_value=model
        ), patch('agent.application.chat.model_factory', return_value=ChatModel()
        ), TestClient(app) as client:
            created = client.post('/conversations', json={'id': self.thread})
            self.assertEqual(created.status_code, 201, created.text)
            output = client.post(f'/conversations/{self.thread}/commands', json={
                'id': str(uuid.uuid4()), 'kind': 'message', 'text': '请修复会话列表'})
            self.assertEqual(output.status_code, 202, output.text)
            get_service().title_pool.submit(lambda: None).result(timeout=2)
            deadline = time.monotonic() + 2
            while get_service().store.get(self.thread)['chat_status'] in {'queued', 'running'}:
                if time.monotonic() >= deadline:
                    self.fail('Fake conversation did not finish')
                time.sleep(0.01)
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), TestClient(
            app
        ) as client:
            listing = client.get('/conversations').json()
            activity = client.get(f'/conversations/{self.thread}/snapshot').json()
            title = next(row['title'] for row in listing['conversations'] if row['thread_id'] == self.thread)
            self.assertEqual(title['title'], response().content)
            self.assertEqual(activity['title'], title)
            self.assertEqual(title['status'], 'complete')
            usage = activity['usage']['title']
            self.assertEqual((usage['total_tokens'], usage['cached_input_tokens']), (49, 30))
            self.assertEqual(usage['purpose'], 'title')
            self.assertIsNone(usage['context'])
            self.assertIsInstance(activity['title']['revision'], int)
            client.get('/conversations')
            client.get(f'/conversations/{self.thread}/snapshot')
        model.invoke.assert_called_once()
