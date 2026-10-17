"""通用工具使用真实临时文件和进程，并采用模拟模型回复。"""

import json
from pathlib import Path
import tempfile
import unittest


from agent.execution.assistant_files import AssistantFiles
from agent.execution.existing import command_for
from agent.model.tools import CHAT_TOOLS, ASSISTANT_SCHEMA
from agent.server.assistant import validate


class GeneralAssistantTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'first.py').write_text('VALUE = 1\n', encoding='utf-8')


    def test_fixed_tools_and_aggregate_parameter_feedback(self):
        self.assertEqual(tuple(tool.name for tool in CHAT_TOOLS), (
            'start_work', 'web_search', 'web_fetch', 'list_files', 'read_file',
            'run_existing', 'list_applications', 'open_file', 'launch_application',
            'request_file_operation', 'retrieve_evidence',
        ))
        for schema in ASSISTANT_SCHEMA.values():
            self.assertFalse(schema['additionalProperties'])
            self.assertEqual(set(schema['required']), set(schema['properties']))
        with self.assertRaises(ValueError) as errors:
            validate('read_file', {'path': 4, 'offset': True, 'unexpected': 3})
        for value in ('path', 'offset', 'limit', 'reread_reason', 'unexpected'):
            self.assertIn(value, str(errors.exception))


    def test_source_continuation_only_reuses_visible_unchanged_lines(self):
        path = self.repo / 'long.txt'
        path.write_text('\n'.join(str(i) for i in range(100)), encoding='utf-8')
        files = AssistantFiles(str(self.repo))
        first = files.read('long.txt', 0, 60, [])
        history = [{'type': 'tool', 'data': {'content': json.dumps(first)}}]
        second = files.read('long.txt', 0, 80, history)
        self.assertEqual(second['visible_ranges'], [[60, 80]])
        self.assertEqual(files.read('long.txt', 0, 80, [])['visible_ranges'], [[0, 80]])
        path.write_text('Changed\n' + path.read_text(), encoding='utf-8')
        self.assertEqual(files.read('long.txt', 0, 80, history)['visible_ranges'], [[0, 80]])

    def test_path_secret_inline_code_and_application_rejection(self):
        files = AssistantFiles(str(self.repo))
        (self.repo / '.env').write_text('SENSITIVE_KEY=secret')
        outside = self.root / 'outside.txt'
        outside.write_text('outside')
        for path in (str(outside), '../outside.txt', '.env'):
            with self.assertRaises(ValueError):
                files.path(path)
        for entry in ('python -c print(1)', 'module:os', 'QQ.exe', 'task.py; echo something'):
            with self.assertRaises((ValueError, OSError)):
                command_for(str(self.repo), entry, [])


    def test_declared_package_script_and_local_module_keep_argv(self):
        package = self.repo / 'package.json'
        package.write_text(json.dumps({'scripts': {'test': 'node existing.js'}}))
        command, _, _ = command_for(
            str(self.repo), 'package.json#test', ['quoted value', '中文\\path']
        )
        self.assertEqual(
            command,
            [
                '/usr/bin/env',
                'npm',
                '--ignore-scripts',
                '--prefix',
                '/testbed',
                'run',
                'test',
                '--',
                'quoted value',
                '中文\\path',
            ],
        )
        (self.repo / 'frontend').mkdir()
        (self.repo / 'frontend/package.json').write_text('{"scripts":{"test":"node test.js"}}')
        nested, _, _ = command_for(str(self.repo), 'frontend/package.json#test', [])
        self.assertEqual(
            nested,
            [
                '/usr/bin/env',
                'npm',
                '--ignore-scripts',
                '--prefix',
                '/testbed/frontend',
                'run',
                'test',
                '--',
            ],
        )
        command, _, _ = command_for(str(self.repo), 'module:first', ['--one'])
        self.assertEqual(command[-3:], ['-m', 'first', '--one'])
        with self.assertRaises(ValueError):
            command_for(str(self.repo), 'package.json#missing', [])
