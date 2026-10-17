import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from agent.execution.desktop import DesktopStore
from agent.execution.assistant_files import AssistantFiles
from agent.storage.permissions import Permissions
from agent.storage.read_sources import ReadSources
from tests.fixtures.models import ScriptedModel, text_reply, reply


class FileActionsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / 'workspace'
        self.repo.mkdir()
        self.thread = str(uuid.uuid4())
        self.store = DesktopStore(self.root / 'state')
        env = patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root / 'state')})
        env.start()
        self.addCleanup(env.stop)
        self.permissions = Permissions(self.root / 'state')

    def mode(self, mode):
        self.permissions.request(self.thread, mode)
        self.permissions.boundary(self.thread)

    def request(self, operation, path, destination='', content=''):
        return self.store.request_file_operation(
            self.thread,
            {
                'operation': operation,
                'path': str(path),
                'destination': str(destination),
                'content': content,
            },
            str(self.repo),
        )

    def approve(self, record):
        return self.store.approve(self.thread, record['id'], record['target'])

    def test_all_five_operations_and_backup_with_real_windows_handles(self):
        self.mode('ask')
        one, two, three = (self.root / name for name in ('one.txt', 'two.txt', 'three.txt'))
        self.assertEqual(
            self.approve(self.request('create', one, content='原文'))['status'], 'completed'
        )
        self.assertEqual(one.read_text(encoding='utf-8'), '原文')
        replaced = self.approve(self.request('replace', one, content='新版'))
        self.assertEqual(replaced['status'], 'completed', replaced)
        self.assertEqual(Path(replaced['backup']).read_text(encoding='utf-8'), '原文')
        self.assertEqual(self.approve(self.request('copy', one, two))['status'], 'completed')
        self.assertTrue(one.exists())
        moved = self.approve(self.request('move', two, three))
        self.assertEqual(moved['status'], 'completed', moved)
        self.assertFalse(two.exists())
        self.assertEqual(three.read_text(encoding='utf-8'), '新版')
        deleted = self.approve(self.request('delete', three))
        self.assertEqual(deleted['status'], 'completed', deleted)
        self.assertFalse(three.exists())
        self.assertTrue(Path(deleted['backup']).is_file())

    def test_sandbox_denial_rejection_and_permission_change(self):
        path = self.root / 'one.txt'
        with self.assertRaises(ValueError):
            self.request('create', path, content='text')
        self.mode('ask')
        request = self.request('create', path, content='text')
        self.assertFalse(path.exists())
        self.store.reject(self.thread, request['id'], request['target'])
        self.assertEqual(self.approve(request)['status'], 'rejected')
        self.assertFalse(path.exists())
        request = self.request('create', path, content='text')
        self.mode('sandbox')
        with self.assertRaises(ValueError):
            self.approve(request)
        self.assertFalse(path.exists())

    def test_target_changes_conflicts_private_paths_and_workspace_edits(self):
        self.mode('ask')
        path = self.root / 'one.txt'
        path.write_text('old')
        request = self.request('replace', path, content='new')
        path.write_text('changed')
        self.assertEqual(self.approve(request)['status'], 'failed')
        self.assertEqual(path.read_text(), 'changed')
        for op, target, dest in [
            ('create', path, ''),
            ('copy', path, path),
            ('create', self.repo / 'code.py', ''),
            ('create', self.root / 'state/private.txt', ''),
        ]:
            with self.subTest(op=op, path=target), self.assertRaises(ValueError):
                self.request(op, target, dest)
        with self.assertRaises(ValueError):
            self.request('create', self.root / '.env', content='secret')
        with self.assertRaises(ValueError):
            self.request('delete', self.root)

    def test_repeated_approval_and_unknown_operation_do_not_replay(self):
        self.mode('ask')
        path = self.root / 'one.txt'
        request = self.request('create', path, content='one')
        self.approve(request)
        path.write_text('afterwards')
        self.assertEqual(self.approve(request)['status'], 'completed')
        self.assertEqual(path.read_text(), 'afterwards')
        request = self.request('replace', path, content='two')
        with self.store.store.connection() as db:
            db.execute(
                'UPDATE desktop_requests SET status=? WHERE id=?', ('executing', request['id'])
            )
        self.assertEqual(self.approve(request)['status'], 'executing')
        self.assertEqual(path.read_text(), 'afterwards')

    def test_read_sources_legacy_grants_and_explicit_revocation(self):
        source = self.root / 'material.txt'
        source.write_text('permitted')
        with self.store.store.connection() as db:
            db.execute('INSERT INTO desktop_files VALUES (?,?)', (self.thread, str(source)))
        records = ReadSources(self.root / 'state')
        self.assertEqual(records.get(self.thread), [{'path': str(source), 'kind': 'file'}])
        records.set(self.thread, [])
        self.assertEqual(records.get(self.thread, [str(self.root)]), [])
        private = self.root / 'state/conversations.sqlite3'
        with self.assertRaises(ValueError):
            records.add(self.thread, private, 'file')

    def test_hardlinks_and_directory_links_are_refused_before_approval(self):
        self.mode('ask')
        path = self.root / 'one.txt'
        path.write_text('original')
        second = self.root / 'linked.txt'
        os.link(path, second)
        with self.assertRaises(ValueError):
            self.request('replace', second, content='no')
        self.assertEqual(path.read_text(), 'original')

    def test_document_open_cannot_run_code_through_interpreters(self):
        self.mode('ask')
        document = self.root / 'program.txt'
        document.write_text('print("not executed")')
        engine = {'id': 'engine', 'name': 'Python', 'path': 'C:/python/python.exe', 'kind': 'exe'}
        with patch.object(self.store, 'applications', return_value=[engine]), patch(
            'agent.execution.desktop.installed.start'
        ) as start:
            with self.assertRaises(ValueError):
                self.store.request_open(self.thread, engine['id'], document)
            start.assert_not_called()

    def test_single_file_query_and_preview_do_not_grant_parent(self):
        source = self.root / 'material.txt'
        source.write_text('permitted')
        sibling = self.root / 'sibling.txt'
        sibling.write_text('private')
        sources = ReadSources(self.root / 'state')
        sources.add(self.thread, source, 'file')
        files = AssistantFiles(str(self.repo), read_sources=sources.get(self.thread))
        self.assertIn('permitted', files.read(str(source), 0, 20, [])['text'])
        self.store.preview(self.thread, str(source), str(self.repo), [])
        with self.assertRaises(ValueError):
            files.read(str(sibling), 0, 20, [])
        with self.assertRaises(ValueError):
            files.list(str(source.parent), 0, 20)


class ApprovalGraphTests(unittest.TestCase):
    def setUp(self):
        from agent.application.store import ApplicationStore

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.application = ApplicationStore(self.root / 'runs')
        self.thread = self.application.create(repo=str(self.repo))['thread_id']

    def invoke(self, model, text, identity='first-message'):
        from agent.application.chat import Chatbot

        command = {'id': str(uuid.uuid5(uuid.UUID(self.thread), identity)),
                   'kind': 'message', 'text': text}
        self.application.submit(self.thread, command)
        bot = Chatbot(self.application, lambda *args: self.fail('No coding handoff expected'),
                      factory=lambda settings: model)
        result = bot.run(self.thread, command)
        self.application.command_state(self.thread, command['id'], 'applied')
        return result

    def test_approval_waits_without_model_polling_and_resumes_with_actual_result(self):
        permissions = Permissions(self.root / 'runs')
        permissions.request(self.thread, 'ask')
        target = self.root / 'created.txt'
        model = ScriptedModel(
            [
                reply(
                    'request_file_operation',
                    operation='create',
                    path=str(target),
                    destination='',
                    content='approved text',
                )
            ]
        )
        result = self.invoke(model, '创建外部文件')
        self.assertTrue(result['waiting_approval'])
        self.assertEqual(len(model.calls), 1)
        self.assertFalse(target.exists())
        desktop = DesktopStore(self.root / 'runs')
        record = desktop.list(self.thread)[0]
        self.assertEqual(
            desktop.approve(self.thread, record['id'], record['target'])['status'], 'completed'
        )
        self.assertEqual(target.read_text(), 'approved text')
        resumed = ScriptedModel([text_reply('文件已经创建。')])
        result = self.invoke(resumed, '继续', 'user-two')
        self.assertTrue(result['done'])
        self.assertEqual(len(resumed.calls), 1)
        self.assertTrue(
            any(record['id'] in m.text and 'completed' in m.text for m in resumed.calls[0])
        )

    def test_full_mode_automatically_executes_and_default_new_chat_cannot(self):
        Permissions(self.root / 'runs').request(self.thread, 'full')
        target = self.root / 'created.txt'
        model = ScriptedModel(
            [
                reply(
                    'request_file_operation',
                    operation='create',
                    path=str(target),
                    destination='',
                    content='full text',
                ),
                text_reply(),
            ]
        )
        result = self.invoke(model, '直接创建外部文件')
        self.assertTrue(result['done'])
        self.assertEqual(target.read_text(), 'full text')
        self.assertEqual(
            DesktopStore(self.root / 'runs').list(self.thread)[0]['status'], 'completed'
        )
        self.assertEqual(
            Permissions(self.root / 'runs').get(str(uuid.uuid4()))['effective'], 'sandbox'
        )
