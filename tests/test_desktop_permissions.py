import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from tests.http_client import TestClient
from agent.execution.desktop import DesktopStore, applications
from agent.server.desktop import routes
from agent.storage.permissions import Permissions
from starlette.applications import Starlette


class DesktopPermissionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root/'repo'; self.repo.mkdir()
        self.file = self.repo/'document.html'
        self.file.write_text('<h1>预览</h1><script>fetch("/settings")</script>', encoding='utf-8')
        self.thread = str(uuid.uuid4())
        self.store = DesktopStore(self.root/'state')
        env = patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root/'state')})
        env.start(); self.addCleanup(env.stop)

    def test_permissions_persist_per_conversation_and_wait_for_boundary(self):
        permissions = Permissions(self.root/'state')
        self.assertEqual(permissions.get(self.thread)['effective'], 'sandbox')
        self.assertEqual(permissions.request(self.thread, 'full')['effective'], 'sandbox')
        self.assertEqual(permissions.boundary(self.thread)['effective'], 'full')
        self.assertEqual(Permissions(self.root/'state').get(self.thread)['requested'], 'full')
        self.assertEqual(permissions.get(str(uuid.uuid4()))['requested'], 'sandbox')
        permissions.request(self.thread, 'sandbox')
        self.assertEqual(permissions.boundary(self.thread)['effective'], 'sandbox')

    def test_settings_before_first_message_do_not_initialize_a_task(self):
        with TestClient(Starlette(routes=routes)) as client:
            response = client.post(f'/threads/{self.thread}/permission', json={'mode': 'full'},
                                   headers={'Origin': 'http://127.0.0.1:3000'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'requested': 'full', 'effective': 'sandbox'})
        self.assertIsNone(self.store.store.session(self.thread))
        self.assertFalse((self.root/'state/tasks').exists())
        self.assertEqual(self.store.list(self.thread), [])

    def test_file_selection_before_first_message_authorizes_only_selected_file(self):
        from types import SimpleNamespace
        selected = self.root/'selected.txt'; selected.write_text('only this file')
        other = self.root/'other.txt'; other.write_text('not authorized')
        with patch('agent.server.desktop.subprocess.run', return_value=SimpleNamespace(returncode=0, stdout=str(selected))):
            with TestClient(Starlette(routes=routes)) as client:
                result = client.post(f'/threads/{self.thread}/choose-file', json={},
                                    headers={'Origin': 'http://127.0.0.1:3000'})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()['path'], str(selected.resolve()))
        self.assertIsNone(self.store.store.session(self.thread))
        self.assertEqual(Permissions(self.root/'state').get(self.thread)['effective'], 'sandbox')
        with self.assertRaises(ValueError):
            self.store.preview(self.thread, str(other), '', [])

    def test_removed_registration_endpoint_does_not_start_application(self):
        with TestClient(Starlette(routes=routes)) as client, patch('agent.execution.desktop.subprocess.Popen') as start:
            self.assertEqual(client.post(f'/threads/{self.thread}/choose-application', json={},
                headers={'Origin': 'http://127.0.0.1:3000'}).status_code, 404)
            start.assert_not_called()

    def test_preview_is_saved_original_and_contains_no_implicit_model_input(self):
        result = self.store.preview(self.thread, str(self.file), str(self.repo), [])
        self.file.write_text('changed')
        self.assertIn('预览', Path(result['record']).read_text(encoding='utf-8'))
        self.assertNotIn('content', result)
        self.assertEqual(len(self.store.list(self.thread)), 1)

    def test_preview_denies_outside_secrets_and_links(self):
        (self.repo/'.env').write_text('TEST_SECRET=private')
        outside = self.root/'outside.txt'; outside.write_text('outside')
        for path in (str(outside), str(self.repo/'.env')):
            with self.assertRaises(ValueError):
                self.store.preview(self.thread, path, str(self.repo), [])

    def test_single_file_authorization_does_not_authorize_its_directory(self):
        a, b = self.root/'one.txt', self.root/'two.txt'
        a.write_text('one'); b.write_text('two')
        self.store.authorize_file(self.thread, a)
        self.store.preview(self.thread, str(a), '', [])
        with self.assertRaises(ValueError):
            self.store.preview(self.thread, str(b), '', [])

    @unittest.skipUnless(os.name == 'nt', 'Registered Windows applications')
    def test_single_open_is_consumed_and_does_not_change_permissions(self):
        permission = Permissions(self.root/'state')
        permission.request(self.thread, 'ask'); permission.boundary(self.thread)
        app = next(a for a in applications() if a['name'] == '记事本')
        record = self.store.request_open(self.thread, app['id'], self.file)
        with patch('agent.execution.applications.start') as start:
            start.return_value = 123
            self.store.approve(self.thread, record['id'], record['target'])
            DesktopStore(self.root/'state').approve(self.thread, record['id'], record['target'])
            self.assertEqual(start.call_count, 1)
            self.assertEqual(start.call_args.args[0]['id'], app['id'])
            self.assertEqual(start.call_args.args[1], str(self.file))
        self.assertEqual(Permissions(self.root/'state').get(self.thread)['effective'], 'ask')

    @unittest.skipUnless(os.name == 'nt', 'Registered Windows applications')
    def test_changed_target_and_scripts_cannot_be_opened(self):
        permission = Permissions(self.root/'state')
        permission.request(self.thread, 'ask'); permission.boundary(self.thread)
        app = next(a for a in applications() if a['name'] == '记事本')
        record = self.store.request_open(self.thread, app['id'], self.file)
        self.file.write_text('changed')
        with patch('agent.execution.applications.start') as start:
            with self.assertRaisesRegex(ValueError, '文件已变化'):
                self.store.approve(self.thread, record['id'], record['target'])
            start.assert_not_called()
        with self.assertRaises(ValueError):
            self.store.request_open(self.thread, app['id'], self.repo/'payload.cmd')

    def test_preview_cannot_access_controls_and_mutations_require_ui_origin(self):
        record = self.store.preview(self.thread, str(self.file), str(self.repo), [])
        with TestClient(Starlette(routes=routes)) as client:
            result = client.get(f'/threads/{self.thread}/preview/{record["id"]}')
            self.assertEqual(result.status_code, 200)
            policy = result.headers['Content-Security-Policy']
            self.assertIn('sandbox;', policy)
            self.assertIn("connect-src 'none'", policy)
            for origin in ('null', 'https://example.com', ''):
                response = client.post(f'/threads/{self.thread}/permission', json={'mode': 'full'}, headers={'Origin': origin})
                self.assertEqual(response.status_code, 400)
            response = client.post(f'/threads/{self.thread}/permission', json={'mode': 'full'}, headers={'Origin': 'http://127.0.0.1:3000'})
            self.assertEqual(response.status_code, 200)

    @unittest.skipUnless(os.name == 'nt', 'Registered Windows applications')
    def test_unfinished_launch_is_not_replayed(self):
        permission = Permissions(self.root/'state')
        permission.request(self.thread, 'ask'); permission.boundary(self.thread)
        app = next(a for a in applications() if a['name'] == '记事本')
        record = self.store.request_open(self.thread, app['id'])
        with self.store.store.connection() as db:
            db.execute('UPDATE desktop_requests SET status=? WHERE id=?', ('launching', record['id']))
        with patch('agent.execution.desktop.subprocess.Popen') as start:
            self.assertEqual(self.store.approve(self.thread, record['id'], record['target'])['status'], 'launching')
            start.assert_not_called()

    def test_installed_applications_are_discovered_without_launching(self):
        with patch('agent.execution.desktop.subprocess.Popen') as start:
            found = self.store.applications()
            self.assertTrue(any(a['name'] == '记事本' for a in found))
            start.assert_not_called()

    def test_purge_removes_permissions_and_preview_but_not_original_file(self):
        from agent.storage.purge import begin_purge, finish_purge
        record = self.store.preview(self.thread, str(self.file), str(self.repo), [])
        Permissions(self.root/'state').request(self.thread, 'full')
        self.store.store.set_visibility(self.thread, 'deleted')
        identities = begin_purge(self.store.store, self.thread)
        finish_purge(self.store.store, self.thread, identities)
        self.assertFalse(Path(record['record']).exists())
        self.assertTrue(self.file.exists())
        self.assertEqual(self.store.list(self.thread), [])
        self.assertEqual(Permissions(self.root/'state').get(self.thread)['effective'], 'sandbox')

    def test_desktop_routes_remain_nonblocking_under_server_guard(self):
        from blockbuster import blockbuster_ctx
        record = self.store.preview(self.thread, str(self.file), str(self.repo), [])
        with blockbuster_ctx(), TestClient(Starlette(routes=routes)) as client:
            for suffix in ('opens', 'permission', 'preview/'+record['id']):
                self.assertEqual(client.get(f'/threads/{self.thread}/{suffix}').status_code, 200)


if __name__ == '__main__':
    unittest.main()
