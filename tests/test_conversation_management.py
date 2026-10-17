import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch
from tests.http_client import TestClient
from tests.legacy.conversations import ConversationStore
from agent.server.app import app
from agent.application.store import ApplicationStore


class ConversationManagementTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.thread = str(uuid.uuid4())
        self.store = ConversationStore(self.root)
        self.session = {
            'status': 'chat_complete',
            'task_id': 'a' * 32,
            'request': '原始要求',
            'public_messages': [{'type': 'human', 'data': {'id': 'u', 'content': '中文消息'}}],
        }
        self.store.save_session(self.thread, self.session)

    def application_thread(self):
        self.thread = str(uuid.uuid4())
        application = ApplicationStore(self.root)
        application.create(self.thread)
        return application


    def test_endpoint_validates_origin_and_allows_only_presentation_state(self):
        application = self.application_thread()
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), TestClient(
            app
        ) as client:
            url = f'/threads/{self.thread}/visibility'
            self.assertEqual(
                client.post(
                    url, json={'visibility': 'deleted'}, headers={'origin': 'https://evil.example'}
                ).status_code,
                403,
            )
            self.assertEqual(client.post(url, json={'visibility': 'permanent'}).status_code, 400)
            self.assertEqual(
                client.post(url, json={'visibility': 'deleted', 'task_id': 'other'}).status_code,
                400,
            )
            self.assertEqual(client.post(url, json={'visibility': 'archived'}).status_code, 200)
            self.assertEqual(
                next(row for row in client.get('/conversations').json()['conversations']
                     if row['thread_id'] == self.thread)['visibility'], 'archived'
            )
        self.assertEqual(application.get(self.thread)['status'], 'idle')


    def test_purge_endpoint_confirms_target_and_cleans_owned_records(self):
        application = self.application_thread()
        application.visibility(self.thread, 'deleted')
        owned = self.root / 'application-records' / self.thread
        owned.mkdir(parents=True)
        (owned / 'request.json').write_text('private record')
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), TestClient(app) as client:
            url = f'/threads/{self.thread}/purge'
            body = {'confirm_thread_id': self.thread}
            self.assertEqual(
                client.post(url, json=body, headers={'origin': 'https://evil.example'}).status_code,
                403,
            )
            self.assertEqual(
                client.post(url, json={'confirm_thread_id': str(uuid.uuid4())}).status_code, 400
            )
            self.assertTrue(owned.exists())
            response = client.post(url, json=body)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertFalse(owned.exists())
            self.assertEqual(application.get(self.thread)['visibility'], 'purged')
            self.assertIsNone(self.store.session(self.thread))

    def test_resource_failure_preserves_retry_context_without_automatic_retry(self):
        application = self.application_thread()
        application.visibility(self.thread, 'deleted')
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), TestClient(app) as client:
            url = f'/threads/{self.thread}/purge'
            body = {'confirm_thread_id': self.thread}
            with patch('agent.application.purge.cleanup', side_effect=OSError('unknown')) as cleanup:
                self.assertEqual(client.post(url, json=body).status_code, 503)
                cleanup.assert_called_once()
            self.assertIsNone(self.store.session(self.thread))
            self.assertEqual(application.get(self.thread)['visibility'], 'purging')
            response = client.post(url, json=body)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsNone(self.store.session(self.thread))

    def test_busy_application_cannot_be_deleted_without_any_legacy_mirror(self):
        application = self.application_thread()
        application.visibility(self.thread, 'deleted')
        self.store.set_visibility(self.thread, 'deleted')
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), TestClient(app) as client:
            application.chat_state(self.thread, 'running')
            response = client.post(
                f'/threads/{self.thread}/purge', json={'confirm_thread_id': self.thread}
            )
            self.assertEqual(response.status_code, 409)
            self.assertEqual(self.store.visibility()[self.thread], 'deleted')
            self.assertIsNone(self.store.session(self.thread))
            application.chat_state(self.thread, 'paused')


    def test_partial_cleanup_remains_visible_and_can_be_retried(self):
        application = self.application_thread()
        application.visibility(self.thread, 'deleted')
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), TestClient(app) as client:
            url = f'/threads/{self.thread}/purge'
            body = {'confirm_thread_id': self.thread}
            with patch('agent.application.purge.cleanup', side_effect=OSError('cleanup interrupted')):
                self.assertEqual(client.post(url, json=body).status_code, 503)
            pending = client.get('/conversations').json()['conversations']
            self.assertTrue(any(row['thread_id'] == self.thread and row['visibility'] == 'purging'
                                for row in pending))
            self.assertIsNone(self.store.session(self.thread))
            self.assertEqual(client.post(url, json=body).status_code, 200)
            self.assertFalse(any(row['thread_id'] == self.thread for row in
                                 client.get('/conversations').json()['conversations']))
