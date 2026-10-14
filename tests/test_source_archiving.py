"""源码归档在访问 Windows 链接前剪除依赖目录树。"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eval import swebench_runner


class SourceArchiveTests(unittest.TestCase):
    def test_archive_keeps_web_sources_without_traversing_excluded_trees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = (
                'langgraph.json',
                'README.md',
                'web/LICENSE',
                'web/.env.example',
                'web/package.json',
                'web/src/page.tsx',
                'web/src/nested/style.css',
                'web/node_modules/private/package.json',
                'web/.next/server/page.js',
                'web/src/node_modules/nested/package.json',
                'web/.git/config.json',
            )
            for name in files:
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text('{}', encoding='utf-8')
            output = root / 'archive'
            output.mkdir()
            walk = os.walk
            visited = []

            def guarded_walk(*args, **kwargs):
                for folder, directories, names in walk(*args, **kwargs):
                    relative = Path(folder).relative_to(root / 'web')
                    self.assertFalse({'node_modules', '.next', '.git'}.intersection(relative.parts))
                    visited.append(relative.as_posix())
                    yield folder, directories, names

            with patch.object(swebench_runner, 'PROJECT_ROOT', root), patch.object(
                swebench_runner.os, 'walk', guarded_walk
            ):
                swebench_runner.archive_source(output)
            hashes = json.loads((output / 'source-sha256.json').read_text())
            self.assertIn('web/src/page.tsx', hashes)
            self.assertIn('web/src/nested/style.css', hashes)
            self.assertIn('web/package.json', hashes)
            self.assertEqual(set(visited), {'.', 'src', 'src/nested'})
            self.assertFalse(
                any(
                    {'node_modules', '.next', '.git'}.intersection(Path(name).parts)
                    for name in hashes
                )
            )
