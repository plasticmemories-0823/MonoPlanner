"""Directory selection reads names, never source contents or task state."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from starlette.testclient import TestClient

from agent.server.app import app
from agent.server.workspaces import directory_page


class WorkspacePickerTests(unittest.TestCase):
    def test_only_immediate_directories_are_listed_in_order(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            for name in ('中文目录', 'Beta', 'alpha'):
                (folder / name).mkdir()
            (folder / 'alpha' / 'nested').mkdir()
            (folder / 'private.key').write_text('not a directory')
            page = directory_page(str(folder))
            self.assertEqual(page['path'], str(folder.resolve()))
            self.assertEqual([p['name'] for p in page['directories']], ['alpha', 'Beta', '中文目录'])
            self.assertEqual(page['parent'], str(folder.parent.resolve()))
            self.assertNotIn('nested', repr(page))
            self.assertNotIn('private.key', repr(page))

    def test_invalid_and_relative_paths_cannot_be_selected(self):
        with self.assertRaises(ValueError):
            directory_page('relative/directory')
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / 'file.txt'
            file.write_text('content')
            with self.assertRaises(ValueError):
                directory_page(str(file))
            with TestClient(app) as client:
                self.assertEqual(client.get('/workspace/directories', params={'path': str(file)}).status_code, 400)
                self.assertEqual(client.get('/workspace/directories', params={'path': str(file / 'missing')}).status_code, 400)

    def test_http_browse_and_select_do_not_change_user_files(self):
        with tempfile.TemporaryDirectory(prefix='picker 中文 ') as root, TestClient(app) as client:
            folder = Path(root)
            (folder / 'child').mkdir()
            response = client.get('/workspace/directories', params={'path': root})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers['cache-control'], 'no-store')
            selected = client.post('/workspace', json={'repo': root})
            self.assertEqual(selected.json()['repo'], str(folder.resolve()))
            self.assertEqual(list(folder.iterdir()), [folder / 'child'])

    def test_no_path_lists_local_places_and_scan_failure_is_reported(self):
        page = directory_page()
        self.assertEqual(page['path'], '')
        self.assertIsNone(page['parent'])
        self.assertTrue(page['directories'])
        with patch('agent.server.workspaces.os.scandir', side_effect=PermissionError('access denied')), TestClient(app) as client:
            denied = client.get('/workspace/directories', params={'path': str(Path.home())})
            self.assertEqual(denied.status_code, 400)
            self.assertIn('access denied', denied.json()['error'])
