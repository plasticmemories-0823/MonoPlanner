import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch
from starlette.testclient import TestClient
from agent.storage.conversations import ConversationStore
from agent.server.app import app


class ConversationManagementTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.thread = str(uuid.uuid4())
        self.store = ConversationStore(self.root)
        self.session = {'status': 'chat_complete', 'task_id': 'a'*32, 'request': '原始要求',
                        'public_messages': [{'type': 'human', 'data': {'id': 'u', 'content': '中文消息'}}]}
        self.store.save_session(self.thread, self.session)

    def test_archive_delete_and_restore_preserve_session_and_task_files(self):
        task = self.root/'tasks'/('a'*32); task.mkdir(parents=True)
        original = task/'checkpoint.json'; original.write_text('original', encoding='utf-8')
        for visibility in ('archived', 'deleted', 'active'):
            self.store.set_visibility(self.thread, visibility)
            self.assertEqual(ConversationStore(self.root).visibility()[self.thread], visibility)
            self.assertEqual(self.store.session(self.thread), self.session)
            self.assertEqual(original.read_text(), 'original')

    def test_running_or_queued_input_cannot_be_hidden(self):
        self.store.save_session(self.thread, {**self.session, 'status': 'executing'})
        with self.assertRaisesRegex(ValueError, '暂停'): self.store.set_visibility(self.thread, 'deleted')
        self.store.save_session(self.thread, self.session)
        self.store.enqueue(self.thread, 'queued', 'message', '继续')
        with self.assertRaisesRegex(ValueError, '暂停'): self.store.set_visibility(self.thread, 'archived')
        self.assertEqual(self.store.visibility(), {})

    def test_endpoint_validates_origin_and_allows_only_presentation_state(self):
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), TestClient(app) as client:
            url = f'/threads/{self.thread}/visibility'
            self.assertEqual(client.post(url, json={'visibility': 'deleted'}, headers={'origin': 'https://evil.example'}).status_code, 403)
            self.assertEqual(client.post(url, json={'visibility': 'permanent'}).status_code, 400)
            self.assertEqual(client.post(url, json={'visibility': 'deleted', 'task_id': 'other'}).status_code, 400)
            self.assertEqual(client.post(url, json={'visibility': 'archived'}).status_code, 200)
            self.assertEqual(client.get('/conversations').json()['visibility'][self.thread], 'archived')
        self.assertEqual(self.store.session(self.thread), self.session)
