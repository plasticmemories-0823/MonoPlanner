import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import AsyncMock, patch
import httpx
from tests.http_client import TestClient
from tests.legacy.conversations import ConversationStore
from agent.server.app import app
from tests.legacy.purge import begin_purge, finish_purge
from agent.application.store import ApplicationStore
from agent.application.api import get_service


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

    def test_archive_delete_and_restore_preserve_session_and_task_files(self):
        task = self.root / 'tasks' / ('a' * 32)
        task.mkdir(parents=True)
        original = task / 'checkpoint.json'
        original.write_text('original', encoding='utf-8')
        for visibility in ('archived', 'deleted', 'active'):
            self.store.set_visibility(self.thread, visibility)
            self.assertEqual(ConversationStore(self.root).visibility()[self.thread], visibility)
            self.assertEqual(self.store.session(self.thread), self.session)
            self.assertEqual(original.read_text(), 'original')

    def test_running_or_queued_input_cannot_be_hidden(self):
        self.store.save_session(self.thread, {**self.session, 'status': 'executing'})
        with self.assertRaisesRegex(ValueError, '暂停'):
            self.store.set_visibility(self.thread, 'deleted')
        self.store.save_session(self.thread, self.session)
        self.store.enqueue(self.thread, 'queued', 'message', '继续')
        with self.assertRaisesRegex(ValueError, '暂停'):
            self.store.set_visibility(self.thread, 'archived')
        self.assertEqual(self.store.visibility(), {})

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

    def test_permanent_delete_removes_only_owned_messages_and_tasks_and_blocks_stale_tabs(self):
        self.store.set_visibility(self.thread, 'deleted')
        self.store.enqueue(self.thread, 'old', 'message', '已处理消息')
        self.store.acknowledge(self.thread, ['old'])
        task = self.root / 'tasks' / ('a' * 32)
        task.mkdir(parents=True)
        (task / 'reasoning.txt').write_text('private', encoding='utf-8')
        execution = self.root / 'execution' / ('a' * 32)
        execution.mkdir(parents=True)
        (execution / 'private-tests.py').write_text('private generated tests')
        other = str(uuid.uuid4())
        self.store.save_session(other, {'status': 'chat_complete', 'task_id': 'b' * 32})
        another = self.root / 'tasks' / ('b' * 32)
        another.mkdir()
        (another / 'record.txt').write_text('keep')
        repo = self.root / 'workspace'
        repo.mkdir()
        (repo / 'main.py').write_text('keep')
        evaluation = self.root / 'validation'
        evaluation.mkdir()
        (evaluation / 'score.json').write_text('keep')
        identities = begin_purge(self.store, self.thread)
        with self.assertRaisesRegex(ValueError, '删除'):
            self.store.enqueue(self.thread, 'stale', 'message', '旧页面')
        with self.assertRaisesRegex(ValueError, '删除'):
            self.store.set_visibility(self.thread, 'active')
        self.assertTrue(task.exists())
        result = finish_purge(self.store, self.thread, identities)
        self.assertTrue(result['deleted'])
        self.assertFalse(task.exists())
        self.assertFalse(execution.exists())
        self.assertIsNone(self.store.session(self.thread))
        self.assertEqual(self.store.pending(self.thread), [])
        self.assertEqual(self.store.visibility()[self.thread], 'purged')
        with self.assertRaisesRegex(ValueError, '删除'):
            self.store.save_session(self.thread, self.session)
        self.assertEqual((repo / 'main.py').read_text(), 'keep')
        self.assertEqual((evaluation / 'score.json').read_text(), 'keep')
        self.assertEqual((another / 'record.txt').read_text(), 'keep')
        self.assertTrue(
            finish_purge(self.store, self.thread, begin_purge(self.store, self.thread))['deleted']
        )

    def test_purge_requires_trash_and_blocks_running_pending_shared_and_invalid_tasks(self):
        with self.assertRaisesRegex(ValueError, '回收站'):
            begin_purge(self.store, self.thread)
        self.store.set_visibility(self.thread, 'deleted')
        for status in ('needs_plan', 'executing'):
            self.store.save_session(self.thread, {**self.session, 'status': status})
            with self.assertRaisesRegex(ValueError, '暂停'):
                begin_purge(self.store, self.thread)
        self.store.save_session(self.thread, self.session)
        self.store.enqueue(self.thread, 'pending', 'message', '排队')
        with self.assertRaisesRegex(ValueError, '暂停'):
            begin_purge(self.store, self.thread)
        self.store.acknowledge(self.thread, ['pending'])
        self.store.save_session(str(uuid.uuid4()), self.session)
        with self.assertRaisesRegex(ValueError, '其他对话'):
            begin_purge(self.store, self.thread)
        self.store.save_session(self.thread, {**self.session, 'task_id': '../workspace'})
        with self.assertRaisesRegex(ValueError, '编号无效'):
            begin_purge(self.store, self.thread)
        self.assertEqual(self.store.visibility()[self.thread], 'deleted')

    def test_linked_task_directory_cannot_escape_storage(self):
        import subprocess

        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'keep.txt').write_text('keep')
        tasks = self.root / 'tasks'
        tasks.mkdir()
        link = tasks / ('a' * 32)
        if os.name == 'nt':
            result = subprocess.run(
                ['cmd', '/c', 'mklink', '/J', str(link), str(outside)], capture_output=True
            )
            self.assertEqual(result.returncode, 0)
            self.addCleanup(lambda: os.rmdir(link))
        else:
            link.symlink_to(outside, target_is_directory=True)
        self.store.set_visibility(self.thread, 'deleted')
        with self.assertRaisesRegex(ValueError, '路径'):
            begin_purge(self.store, self.thread)
        self.assertEqual((outside / 'keep.txt').read_text(), 'keep')

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

    def test_unknown_operation_in_archive_prevents_purge(self):
        from agent.storage.history import LocalHistory

        history = LocalHistory(self.root / 'tasks' / ('a' * 32))
        self.addCleanup(history.close)
        with history.lease():
            history.record({'history': [], 'request_pending': {'call': 1}}, 'test_pending', {})
        self.store.set_visibility(self.thread, 'deleted')
        with self.assertRaisesRegex(ValueError, '结果不明'):
            begin_purge(self.store, self.thread)
        self.assertEqual(self.store.visibility()[self.thread], 'deleted')
        self.assertTrue(history.root.exists())

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

    def test_readonly_original_in_owned_archive_can_be_removed(self):
        import stat

        task = self.root / 'tasks' / ('a' * 32)
        task.mkdir(parents=True)
        original = task / 'original.py'
        original.write_text('source')
        original.chmod(stat.S_IREAD)
        self.store.set_visibility(self.thread, 'deleted')
        finish_purge(self.store, self.thread, begin_purge(self.store, self.thread))
        self.assertFalse(task.exists())

    def test_workspace_inside_managed_archive_is_not_deleted(self):
        task = self.root / 'execution' / ('a' * 32)
        task.mkdir(parents=True)
        (task / 'main.py').write_text('user file')
        self.store.save_session(self.thread, {**self.session, 'repo': str(task)})
        self.store.set_visibility(self.thread, 'deleted')
        with self.assertRaisesRegex(ValueError, '工作仓库重叠'):
            begin_purge(self.store, self.thread)
        self.assertEqual((task / 'main.py').read_text(), 'user file')
        self.assertEqual(self.store.visibility()[self.thread], 'deleted')

    def test_retry_can_finish_an_archive_partly_removed_after_purge_started(self):
        from agent.storage.history import LocalHistory

        history = LocalHistory(self.root / 'tasks' / ('a' * 32))
        self.addCleanup(history.close)
        with history.lease():
            history.record({'history': []}, 'test_complete', {})
            checkpoint = history.connection.execute('SELECT path FROM commits').fetchone()[0]
        history.close()
        self.store.set_visibility(self.thread, 'deleted')
        begin_purge(self.store, self.thread)
        (history.root / checkpoint).unlink()
        result = finish_purge(self.store, self.thread, begin_purge(self.store, self.thread))
        self.assertTrue(result['deleted'])
        self.assertFalse(history.root.exists())
