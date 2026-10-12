"""Automatic discovery must not grant permission or execute shortcut arguments."""
from pathlib import Path
import os
import tempfile
import unittest
import uuid
from unittest.mock import patch

from agent.execution import applications
from agent.execution.desktop import DesktopStore
from agent.storage.permissions import Permissions


class ApplicationDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = DesktopStore(self.root/'state')
        self.thread = str(uuid.uuid4())
        self.entry = self.root/'Example.exe'
        self.entry.write_bytes(b'original executable')
        self.app = {'name': 'Example', 'path': str(self.entry), 'kind': 'exe', 'id': 'example'}

    def test_missing_cached_entry_refreshes_once_without_launching(self):
        with patch.object(self.store, 'applications', side_effect=[[], [self.app]]) as discover, \
                patch.object(applications, 'start') as start:
            self.assertEqual(self.store.find_applications('Example'), [self.app])
        self.assertEqual(discover.call_args_list[0].kwargs, {})
        self.assertEqual(discover.call_args_list[1].kwargs, {'refresh': True})
        start.assert_not_called()
        self.assertEqual(Permissions(self.root/'state').get(self.thread)['effective'], 'sandbox')

    def test_exact_match_avoids_helper_process_candidates_and_refresh(self):
        other = {**self.app, 'name': 'ExampleExternal', 'id': 'helper'}
        with patch.object(self.store, 'applications', return_value=[other, self.app]) as discover:
            self.assertEqual(self.store.find_applications('example'), [self.app])
        discover.assert_called_once_with()

    def test_multiple_real_candidates_remain_for_user_choice(self):
        other = {**self.app, 'id': 'second', 'path': str(self.root/'Other.exe')}
        with patch.object(self.store, 'applications', return_value=[self.app, other]), \
                patch.object(applications, 'start') as start:
            self.assertEqual(len(self.store.find_applications('Example')), 2)
        start.assert_not_called()

    def test_no_entry_returns_empty_after_one_refresh(self):
        with patch.object(self.store, 'applications', return_value=[]) as discover:
            self.assertEqual(self.store.find_applications('missing'), [])
        self.assertEqual(discover.call_count, 2)

    def test_explicit_uncatalogued_entry_is_located_without_start_or_permission(self):
        with patch.object(applications, 'discover', return_value=[]), patch.object(applications, 'start') as start:
            found = self.store.find_applications(str(self.entry))
            self.assertEqual(len(found), 1)
            self.assertEqual(self.store.applications()[0]['id'], found[0]['id'])
            self.assertEqual(DesktopStore(self.root/'state').applications()[0]['id'], found[0]['id'])
            with self.assertRaisesRegex(ValueError, '沙盒权限'):
                self.store.request_open(self.thread, found[0]['id'])
        start.assert_not_called()

    def test_explicit_path_cannot_be_a_shell_command_or_script(self):
        for value in (str(self.root/'payload.cmd'), str(self.entry)+' --run-script'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.store.find_applications(value)
        self.assertEqual(self.store.list(self.thread), [])

    def test_located_entry_reuses_existing_target_approval_and_restart_protection(self):
        permission = Permissions(self.root/'state')
        permission.request(self.thread, 'ask'); permission.boundary(self.thread)
        with patch.object(applications, 'discover', return_value=[]):
            app = self.store.find_applications(str(self.entry))[0]
            record = self.store.request_open(self.thread, app['id'])
            with patch.object(applications, 'start', return_value=123) as start:
                self.store.approve(self.thread, record['id'], record['target'])
                DesktopStore(self.root/'state').approve(self.thread, record['id'], record['target'])
            start.assert_called_once()

    def test_invalid_and_missing_executable_paths_are_rejected(self):
        for path in ('relative.exe', str(self.root/'missing.exe'), str(self.root/'payload.cmd')):
            with self.subTest(path=path), self.assertRaises(ValueError):
                applications.executable_path(path)
        self.assertEqual(applications.executable_path(str(self.entry)), self.entry.resolve())

    def test_links_are_not_application_entries(self):
        from types import SimpleNamespace
        with patch.object(Path, 'lstat', return_value=SimpleNamespace(st_mode=0, st_file_attributes=0x400)):
            with self.assertRaisesRegex(ValueError, '重解析点'):
                applications.executable_path(str(self.entry))

    def test_shortcut_arguments_do_not_reach_process_start(self):
        app = {**self.app, 'arguments': '--execute arbitrary-code'}
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'credential-canary', 'HTTP_PROXY': 'proxy-canary'}), \
                patch.object(applications.subprocess, 'Popen') as start:
            applications.start(app)
        self.assertEqual(start.call_args.args[0], [str(self.entry.resolve())])
        self.assertNotIn('DEEPSEEK_API_KEY', start.call_args.kwargs['env'])
        self.assertNotIn('HTTP_PROXY', start.call_args.kwargs['env'])

    def test_approval_detects_changed_executable_and_cannot_start(self):
        permission = Permissions(self.root/'state')
        permission.request(self.thread, 'ask'); permission.boundary(self.thread)
        with patch.object(self.store, 'applications', return_value=[self.app]):
            record = self.store.request_open(self.thread, self.app['id'])
            self.entry.write_bytes(b'changed executable')
            with patch.object(applications, 'start') as start, self.assertRaisesRegex(ValueError, '软件已变化'):
                self.store.approve(self.thread, record['id'], record['target'])
        start.assert_not_called()
        self.assertEqual(self.store.get(self.thread, record['id'])['status'], 'awaiting_approval')

    def test_newly_discovered_application_still_needs_approval(self):
        with patch.object(self.store, 'applications', return_value=[self.app]), patch.object(applications, 'start') as start:
            with self.assertRaisesRegex(ValueError, '沙盒权限'):
                self.store.request_open(self.thread, self.app['id'])
            permission = Permissions(self.root/'state')
            permission.request(self.thread, 'ask'); permission.boundary(self.thread)
            record = self.store.request_open(self.thread, self.app['id'])
            self.assertEqual(record['status'], 'awaiting_approval')
            start.assert_not_called()
            start.return_value = 123
            self.store.approve(self.thread, record['id'], record['target'])
            self.store.approve(self.thread, record['id'], record['target'])
            start.assert_called_once()


if __name__ == '__main__':
    unittest.main()
