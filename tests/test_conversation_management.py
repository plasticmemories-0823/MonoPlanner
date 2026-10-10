import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import AsyncMock, patch
import httpx
from starlette.testclient import TestClient
from agent.storage.conversations import ConversationStore
from agent.server.app import app
from agent.storage.purge import begin_purge, finish_purge


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

    def test_permanent_delete_removes_only_owned_messages_and_tasks_and_blocks_stale_tabs(self):
        self.store.set_visibility(self.thread, 'deleted')
        self.store.enqueue(self.thread, 'old', 'message', '已处理消息')
        self.store.acknowledge(self.thread, ['old'])
        task = self.root/'tasks'/('a'*32); task.mkdir(parents=True)
        (task/'reasoning.txt').write_text('private', encoding='utf-8')
        execution = self.root/'execution'/('a'*32); execution.mkdir(parents=True)
        (execution/'private-tests.py').write_text('private generated tests')
        other = str(uuid.uuid4())
        self.store.save_session(other, {'status': 'chat_complete', 'task_id': 'b'*32})
        another = self.root/'tasks'/('b'*32); another.mkdir()
        (another/'record.txt').write_text('keep')
        repo = self.root/'workspace'; repo.mkdir(); (repo/'main.py').write_text('keep')
        evaluation = self.root/'validation'; evaluation.mkdir(); (evaluation/'score.json').write_text('keep')
        identities = begin_purge(self.store, self.thread)
        with self.assertRaisesRegex(ValueError, '删除'): self.store.enqueue(self.thread, 'stale', 'message', '旧页面')
        with self.assertRaisesRegex(ValueError, '删除'): self.store.set_visibility(self.thread, 'active')
        self.assertTrue(task.exists())
        result = finish_purge(self.store, self.thread, identities)
        self.assertTrue(result['deleted'])
        self.assertFalse(task.exists())
        self.assertFalse(execution.exists())
        self.assertIsNone(self.store.session(self.thread))
        self.assertEqual(self.store.pending(self.thread), [])
        self.assertEqual(self.store.visibility()[self.thread], 'purged')
        with self.assertRaisesRegex(ValueError, '删除'): self.store.save_session(self.thread, self.session)
        self.assertEqual((repo/'main.py').read_text(), 'keep')
        self.assertEqual((evaluation/'score.json').read_text(), 'keep')
        self.assertEqual((another/'record.txt').read_text(), 'keep')
        self.assertTrue(finish_purge(self.store, self.thread, begin_purge(self.store, self.thread))['deleted'])

    def test_purge_requires_trash_and_blocks_running_pending_shared_and_invalid_tasks(self):
        with self.assertRaisesRegex(ValueError, '回收站'): begin_purge(self.store, self.thread)
        self.store.set_visibility(self.thread, 'deleted')
        for status in ('needs_plan', 'executing'):
            self.store.save_session(self.thread, {**self.session, 'status': status})
            with self.assertRaisesRegex(ValueError, '暂停'): begin_purge(self.store, self.thread)
        self.store.save_session(self.thread, self.session)
        self.store.enqueue(self.thread, 'pending', 'message', '排队')
        with self.assertRaisesRegex(ValueError, '暂停'): begin_purge(self.store, self.thread)
        self.store.acknowledge(self.thread, ['pending'])
        self.store.save_session(str(uuid.uuid4()), self.session)
        with self.assertRaisesRegex(ValueError, '其他对话'): begin_purge(self.store, self.thread)
        self.store.save_session(self.thread, {**self.session, 'task_id': '../workspace'})
        with self.assertRaisesRegex(ValueError, '编号无效'): begin_purge(self.store, self.thread)
        self.assertEqual(self.store.visibility()[self.thread], 'deleted')

    def test_linked_task_directory_cannot_escape_storage(self):
        import subprocess
        outside = self.root/'outside'; outside.mkdir(); (outside/'keep.txt').write_text('keep')
        tasks = self.root/'tasks'; tasks.mkdir()
        link = tasks/('a'*32)
        if os.name == 'nt':
            result = subprocess.run(['cmd', '/c', 'mklink', '/J', str(link), str(outside)], capture_output=True)
            self.assertEqual(result.returncode, 0)
            self.addCleanup(lambda: os.rmdir(link))
        else:
            link.symlink_to(outside, target_is_directory=True)
        self.store.set_visibility(self.thread, 'deleted')
        with self.assertRaisesRegex(ValueError, '路径'): begin_purge(self.store, self.thread)
        self.assertEqual((outside/'keep.txt').read_text(), 'keep')

    def test_purge_endpoint_confirms_target_and_cleans_sdk_before_local_records(self):
        self.store.set_visibility(self.thread, 'deleted')
        sdk = AsyncMock(); sdk.threads.get.return_value = {'status': 'idle'}
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), \
                patch('agent.server.app.get_client', return_value=sdk), TestClient(app) as client:
            url = f'/threads/{self.thread}/purge'
            body = {'confirm_thread_id': self.thread}
            self.assertEqual(client.post(url, json=body, headers={'origin': 'https://evil.example'}).status_code, 403)
            self.assertEqual(client.post(url, json={'confirm_thread_id': str(uuid.uuid4())}).status_code, 400)
            sdk.threads.delete.assert_not_called()
            response = client.post(url, json=body)
            self.assertEqual(response.status_code, 200, response.text)
            sdk.threads.delete.assert_awaited_once_with(self.thread)
            sdk.aclose.assert_awaited_once()
            self.assertIsNone(self.store.session(self.thread))

    def test_sdk_failure_preserves_retry_context_without_automatic_retry(self):
        self.store.set_visibility(self.thread, 'deleted')
        sdk = AsyncMock(); sdk.threads.get.return_value = {'status': 'idle'}
        sdk.threads.delete.side_effect = httpx.ReadTimeout('unknown')
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), \
                patch('agent.server.app.get_client', return_value=sdk), TestClient(app) as client:
            url = f'/threads/{self.thread}/purge'; body = {'confirm_thread_id': self.thread}
            self.assertEqual(client.post(url, json=body).status_code, 503)
            self.assertEqual(self.store.session(self.thread), self.session)
            self.assertEqual(self.store.visibility()[self.thread], 'purging')
            sdk.threads.delete.assert_awaited_once()
            missing = httpx.Response(404, request=httpx.Request('GET', 'http://api/threads/x'))
            sdk.threads.get.side_effect = httpx.HTTPStatusError('missing', request=missing.request, response=missing)
            response = client.post(url, json=body)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIsNone(self.store.session(self.thread))
            sdk.threads.delete.assert_awaited_once()

    def test_busy_sdk_thread_cannot_be_deleted_even_if_local_state_is_stale(self):
        self.store.set_visibility(self.thread, 'deleted')
        sdk = AsyncMock(); sdk.threads.get.return_value = {'status': 'busy'}
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), \
                patch('agent.server.app.get_client', return_value=sdk), TestClient(app) as client:
            response = client.post(f'/threads/{self.thread}/purge', json={'confirm_thread_id': self.thread})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(self.store.visibility()[self.thread], 'deleted')
            sdk.threads.delete.assert_not_called()
            self.assertEqual(self.store.session(self.thread), self.session)

    def test_unknown_operation_in_archive_prevents_purge(self):
        from agent.storage.history import LocalHistory
        history = LocalHistory(self.root/'tasks'/('a'*32))
        self.addCleanup(history.close)
        with history.lease():
            history.record({'history': [], 'request_pending': {'call': 1}}, 'test_pending', {})
        self.store.set_visibility(self.thread, 'deleted')
        with self.assertRaisesRegex(ValueError, '结果不明'): begin_purge(self.store, self.thread)
        self.assertEqual(self.store.visibility()[self.thread], 'deleted')
        self.assertTrue(history.root.exists())

    def test_partial_cleanup_remains_visible_after_sdk_delete_and_can_be_retried(self):
        self.store.set_visibility(self.thread, 'deleted')
        sdk = AsyncMock(); sdk.threads.get.return_value = {'status': 'idle'}
        with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root)}), \
                patch('agent.server.app.get_client', return_value=sdk), TestClient(app) as client:
            url = f'/threads/{self.thread}/purge'; body = {'confirm_thread_id': self.thread}
            with patch('agent.server.app.finish_purge', side_effect=OSError('cleanup interrupted')):
                self.assertEqual(client.post(url, json=body).status_code, 503)
            pending = client.get('/conversations').json()['pending_deletions']
            self.assertEqual(pending, [{'thread_id': self.thread, 'title': '中文消息'}])
            self.assertEqual(self.store.session(self.thread), self.session)
            missing = httpx.Response(404, request=httpx.Request('GET', 'http://api/threads/x'))
            sdk.threads.get.side_effect = httpx.HTTPStatusError('missing', request=missing.request, response=missing)
            self.assertEqual(client.post(url, json=body).status_code, 200)
            self.assertEqual(client.get('/conversations').json()['pending_deletions'], [])
            sdk.threads.delete.assert_awaited_once()

    def test_readonly_original_in_owned_archive_can_be_removed(self):
        import stat
        task = self.root/'tasks'/('a'*32); task.mkdir(parents=True)
        original = task/'original.py'; original.write_text('source')
        original.chmod(stat.S_IREAD)
        self.store.set_visibility(self.thread, 'deleted')
        finish_purge(self.store, self.thread, begin_purge(self.store, self.thread))
        self.assertFalse(task.exists())

    def test_workspace_inside_managed_archive_is_not_deleted(self):
        task = self.root/'execution'/('a'*32); task.mkdir(parents=True)
        (task/'main.py').write_text('user file')
        self.store.save_session(self.thread, {**self.session, 'repo': str(task)})
        self.store.set_visibility(self.thread, 'deleted')
        with self.assertRaisesRegex(ValueError, '工作仓库重叠'): begin_purge(self.store, self.thread)
        self.assertEqual((task/'main.py').read_text(), 'user file')
        self.assertEqual(self.store.visibility()[self.thread], 'deleted')

    def test_retry_can_finish_an_archive_partly_removed_after_purge_started(self):
        from agent.storage.history import LocalHistory
        history = LocalHistory(self.root/'tasks'/('a'*32))
        self.addCleanup(history.close)
        with history.lease():
            history.record({'history': []}, 'test_complete', {})
            checkpoint = history.connection.execute('SELECT path FROM commits').fetchone()[0]
        history.close()
        self.store.set_visibility(self.thread, 'deleted')
        begin_purge(self.store, self.thread)
        (history.root/checkpoint).unlink()
        result = finish_purge(self.store, self.thread, begin_purge(self.store, self.thread))
        self.assertTrue(result['deleted'])
        self.assertFalse(history.root.exists())
