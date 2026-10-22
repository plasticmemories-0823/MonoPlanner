"""真实来源策略与持久化使用内存容器，验证范围、版本、分页及原件增长。"""

import base64
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.rag.container_sources import (
    ContainerSources, ContainerSourceError, container_original, read_container,
    expire_container_sources,
)
from agent.rag.integration import retrieve_work, validate_arguments, audit_work_citations
from agent.rag.types import canonical
from agent.source.paging import ReadBudget
from agent.task import control
from agent.model.errors import ModelInterrupted


class MemoryContainer:
    def __init__(self):
        self.files = {'/usr/include/a.h': b'first needle\r\nsecond needle\n'}
        self.identity = {'container_id': 'a' * 64, 'image_id': 'sha256:' + 'b' * 64,
                         'repo_digests': [], 'mounts': [],
                         'toolchain_versions': {'status': 'unavailable'}}
        self.calls = []
        self.realpaths = {}
        self.clock = 1

    def binding(self):
        return copy.deepcopy(self.identity)

    def stat(self, path):
        real = self.realpaths.get(path, path)
        directory = any(p.startswith(path.rstrip('/') + '/') for p in self.files)
        if path not in self.files and not directory:
            raise ValueError('missing fake container path: ' + path)
        return {'path': path, 'realpath': real, 'type': 'directory' if directory else 'file',
                'size': len(self.files.get(path, b'')), 'mtime_ns': self.clock}

    def call(self, operation, payload):
        self.calls.append((operation, payload))
        path = payload['path']
        if operation == 'stat':
            return self.stat(path)
        if operation == 'read':
            raw = self.files[path]
            return {**self.stat(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                    'data_base64': base64.b64encode(raw).decode()}
        names = sorted({p[len(path.rstrip('/') + '/'):].split('/')[0] for p in self.files
                        if p.startswith(path.rstrip('/') + '/')})
        names = [name for name in names if name > payload.get('after', '')]
        selected = names[:payload.get('limit', 100)]
        return {**self.stat(path),
                'entries': [self.stat(path.rstrip('/') + '/' + name) | {'name': name}
                            for name in selected],
                'complete': len(names) <= len(selected),
                'next_after': selected[-1] if len(names) > len(selected) else ''}


class ContainerSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = MemoryContainer()
        self.owner = SimpleNamespace(root=self.root / 'task',
                                     environment=SimpleNamespace(repo_root=self.root / 'project'),
                                     state={'status': 'needs_plan'}, runtime_settings={},
                                     history_store=None, visible_reads={})
        self.owner.root.mkdir()
        self.backend_patch = patch('agent.rag.container_sources.ContainerBackend',
                                   return_value=self.backend)
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)

    def arguments(self, path='$container/system_include/usr/include', query='needle', **extra):
        return validate_arguments({'path': path, 'query': query, **extra}, containers=True)

    def finish_scan(self, arguments):
        value = dict(arguments)
        for _ in range(100):
            entries, meta = ContainerSources(self.owner).scan(value)
            if entries is not None:
                return entries, meta
            self.assertFalse(meta['scan_complete'])
            value['cursor'] = meta['next_cursor']
        self.fail('source scan cursor did not progress')

    def test_default_header_grant_is_read_only_versioned_and_lazy(self):
        manager = ContainerSources(self.owner)
        self.assertEqual(self.backend.calls, [])
        grant, _ = manager.authorize('$container/system_include/usr/include/a.h')
        self.assertFalse(grant['temporary'])
        self.assertEqual(grant['binding_facts']['image_id'], self.backend.identity['image_id'])
        self.assertEqual([op for op, _ in self.backend.calls], ['stat'])

    def test_build_source_allowed_but_cache_artifact_and_broad_parent_denied(self):
        self.backend.files['/build/gcc/libstdc++/src/locale.cc'] = b'locale needle'
        manager = ContainerSources(self.owner)
        grant, _ = manager.authorize('$container/toolchain_src/build/gcc/libstdc++/src')
        self.assertTrue(grant['temporary'])
        self.assertTrue(grant['from_build_tree'])
        for path in ['/build', '/build/gcc', '/build/gcc/cache/src/x.h',
                     '/build/gcc/src/x.o', '/build/gcc/src/.env']:
            with self.subTest(path=path), self.assertRaises(ContainerSourceError):
                manager.authorize('$container/toolchain_src' + path)

    def test_secret_roots_mounts_and_symlink_root_escape_cannot_authorize(self):
        manager = ContainerSources(self.owner)
        for path in ['/usr/include/.env', '/usr/include/certs/a.crt', '/usr/include/.git/a',
                     '/usr/include/credentials.json', '/usr/include/private/a.h']:
            with self.subTest(path=path), self.assertRaises(ContainerSourceError):
                manager.authorize('$container/system_include' + path)
        self.backend.identity['mounts'] = [{'Destination': '/usr/include', 'RW': True}]
        with self.assertRaisesRegex(ContainerSourceError, 'mount_authorization_required'):
            ContainerSources(self.owner).authorize('$container/system_include/usr/include/a.h')
        self.backend.identity['mounts'] = []
        self.backend.realpaths['/usr/include/a.h'] = '/etc/secret.h'
        with self.assertRaises(ContainerSourceError):
            ContainerSources(self.owner).authorize('$container/system_include/usr/include/a.h')

    def test_unregistered_high_risk_scope_returns_authorization_without_read(self):
        result = retrieve_work(self.owner, self.arguments('$container/toolchain_src/etc/src', 'x'))
        self.assertTrue(result['requires_authorization'])
        self.assertFalse(result['complete'])
        self.assertFalse(any(op == 'read' for op, _ in self.backend.calls))

    def test_trusted_configuration_pre_registers_exact_root(self):
        self.backend.files['/opt/compiler/doc/manual.md'] = b'compiler'
        self.owner.runtime_settings = {'container_read_sources': [
            {'namespace': 'toolchain_src', 'path': '/opt/compiler/doc'}]}
        grant, _ = ContainerSources(self.owner).authorize('$container/toolchain_src/opt/compiler/doc')
        self.assertFalse(grant['temporary'])

    def test_normal_settings_and_conversation_preserve_trusted_configuration(self):
        from agent.storage.preferences import Preferences, DEFAULTS
        from agent.application.store import ApplicationStore

        roots = [{'namespace': 'toolchain_src', 'path': '/build/gcc/libstdc++/src'}]
        settings = Preferences.validate({**DEFAULTS, 'container_read_sources': roots})
        store = ApplicationStore(self.root / 'product')
        conversation = store.create(runtime_settings=settings)
        self.assertEqual(conversation['runtime_settings']['container_read_sources'], roots)
        preferences = Preferences(self.root / 'settings')
        with patch.object(Preferences, 'key', return_value=''):
            preferences.update({'container_read_sources': roots})
            preferences.update({'online': True})
        self.assertEqual(preferences.defaults()['container_read_sources'], roots)

    def test_terminal_statuses_match_product_but_pause_preserves_authorization(self):
        from agent.application.store import TERMINAL as PRODUCT_TERMINAL
        from agent.rag.container_sources import TERMINAL

        self.assertTrue(PRODUCT_TERMINAL <= TERMINAL)
        path = '$container/system_include/usr/include/a.h'
        container_original(self.owner, path)
        self.owner.state['status'] = 'paused'
        expire_container_sources(self.owner)
        self.assertNotIn('terminated', json.loads(
            (self.owner.root / 'rag/container/grants.json').read_text('utf-8')))
        self.owner.state['status'] = 'task_blocked'
        expire_container_sources(self.owner)
        self.assertTrue(json.loads((self.owner.root / 'rag/container/grants.json')
                                  .read_text('utf-8'))['terminated'])

    def test_originals_are_deduplicated_and_task_end_revokes_access(self):
        path = '$container/system_include/usr/include/a.h'
        first, meta = container_original(self.owner, path)
        container_original(self.owner, path)
        directory = self.owner.root / 'rag/container/originals'
        self.assertEqual(len(list(directory.iterdir())), 1)
        self.assertEqual(next(directory.iterdir()).read_bytes(), first)
        self.owner.state['status'] = 'verified_success'
        expire_container_sources(self.owner)
        self.owner.state['status'] = 'needs_plan'
        with self.assertRaisesRegex(ContainerSourceError, 'task_ended'):
            container_original(self.owner, path)
        self.assertTrue((directory / meta['sha256']).exists())

    def test_same_image_changed_content_and_rebuilt_container_reject_old_version(self):
        path = '$container/system_include/usr/include/a.h'
        _, first = container_original(self.owner, path)
        self.backend.files['/usr/include/a.h'] = b'changed without directory mtime'
        with self.assertRaisesRegex(ContainerSourceError, 'version_changed'):
            container_original(self.owner, path, expected_version=first['version'])
        self.backend.files['/usr/include/a.h'] = b'first needle\r\nsecond needle\n'
        self.backend.identity['container_id'] = 'c' * 64
        with self.assertRaisesRegex(ContainerSourceError, 'version_changed'):
            container_original(self.owner, path, expected_version=first['version'])

    def test_scan_cursor_progress_scope_binding_and_expiry(self):
        self.backend.files.update({f'/usr/include/f{i}.h': b'needle' for i in range(5)})
        args = self.arguments()
        with patch('agent.rag.container_sources.SCAN_BATCH', 2):
            entries, first = ContainerSources(self.owner).scan(args)
            self.assertIsNone(entries)
            with self.assertRaisesRegex(ContainerSourceError, 'stale'):
                ContainerSources(self.owner).scan({**args, 'query': 'different', 'cursor': first['next_cursor']})
            entries, meta = self.finish_scan({**args, 'cursor': first['next_cursor']})
        self.assertEqual(len(entries), 6)
        with patch('agent.rag.container_sources.SCAN_BATCH', 1):
            _, pending = ContainerSources(self.owner).scan(args)
        with patch('agent.rag.container_sources.time.time', return_value=10**12):
            with self.assertRaisesRegex(ContainerSourceError, 'stale'):
                ContainerSources(self.owner).scan({**args, 'cursor': pending['next_cursor']})

    def test_partial_scan_changed_file_is_not_an_old_cached_negative(self):
        args = self.arguments()
        with patch('agent.rag.container_sources.SCAN_BATCH', 2):
            entries, pending = ContainerSources(self.owner).scan(args)
            self.assertIsNone(entries)
            self.backend.files['/usr/include/a.h'] = b'new match in same directory'
            with self.assertRaisesRegex(ContainerSourceError, 'version_changed'):
                ContainerSources(self.owner).scan({**args, 'cursor': pending['next_cursor']})

    def test_exact_hit_pagination_returns_each_hit_once(self):
        self.backend.files['/usr/include/a.h'] = b'needle needle needle'
        args = self.arguments(limit=1)
        starts = []
        for _ in range(4):
            result = retrieve_work(self.owner, args)
            self.assertEqual(result.get('reasons'), [])
            starts.extend(hit['match_start'] for hit in result['hits'])
            if not result['next_cursor']:
                break
            args['cursor'] = result['next_cursor']
        self.assertEqual(starts, [0, 7, 14])

    def test_search_archives_only_delivered_sources_and_preserves_literal_query(self):
        self.backend.files = {'/usr/include/a.h': '中文.foo AND bar\r\n'.encode(),
                              '/usr/include/b.h': b'unrelated'}
        result = retrieve_work(self.owner, self.arguments(query='中文.foo AND bar'))
        self.assertEqual(result['query'], '中文.foo AND bar')
        self.assertEqual(len(result['hits']), 1)
        self.assertEqual(result['hits'][0]['origin'], 'container/toolchain')
        self.assertEqual(len(list((self.owner.root / 'rag/container/originals').iterdir())), 1)
        self.assertNotIn('rewrite', result)

    def test_long_line_byte_cursor_preserves_version_and_partial_read_status(self):
        self.backend.files['/usr/include/a.h'] = ('中' * 200 + '\nend').encode()
        path = '$container/system_include/usr/include/a.h'
        first = read_container(self.owner, path, 0, 2, budget=ReadBudget(100))
        self.assertEqual(first['visible_ranges'], [])
        self.assertTrue(first['next_cursor'].startswith('container-read-v1:'))
        second = read_container(self.owner, path, 0, 2, budget=ReadBudget(100),
                                cursor=first['next_cursor'])
        self.assertGreater(second['byte_range'][0], 0)
        self.backend.identity['container_id'] = 'd' * 64
        with self.assertRaises(ContainerSourceError):
            read_container(self.owner, path, 0, 2, cursor=second['next_cursor'])

    def test_non_utf8_and_secret_content_are_explicit_incomplete_results(self):
        for raw in (b'/* copyright \xe9 */', b'-----BEGIN PRIVATE KEY-----\nprivate'):
            self.backend.files['/usr/include/a.h'] = raw
            result = retrieve_work(self.owner, self.arguments())
            self.assertFalse(result['complete'])
            self.assertIn('container_source_', result['source_error']['reason'])
            self.assertFalse((self.owner.root / 'rag/container/originals').exists())

    def test_repeated_search_does_not_accumulate_scan_manifests_or_source_copies(self):
        args = self.arguments()
        for _ in range(8):
            result = retrieve_work(self.owner, args)
            self.assertTrue(result['scan_complete'])
        base = self.owner.root / 'rag/container'
        self.assertEqual(list((base / 'scans').iterdir()), [])
        self.assertEqual(len(list((base / 'originals').iterdir())), 1)
        self.assertEqual(len(list((base / 'references').iterdir())), 1)
        self.assertEqual(list((base / 'cache').iterdir()), [])

    def test_changed_source_invalidates_hit_pagination(self):
        args = self.arguments(limit=1)
        first = retrieve_work(self.owner, args)
        self.backend.files['/usr/include/a.h'] += b'needle new version'
        with self.assertRaisesRegex(ValueError, '[Cc]ursor'):
            retrieve_work(self.owner, {**args, 'cursor': first['next_cursor']})

    def test_source_priority_only_breaks_ties_and_index_pruning_keeps_neighbor_scope(self):
        from agent.rag.index import Index
        from agent.rag.retrieval import Retriever
        from agent.rag.types import Document, Query, RagConfig, Scope

        first = 'container:$container/toolchain_src/build/gcc/src/a.h'
        other = 'container:$container/toolchain_src/build/gcc/src-extra/b.h'
        documents = {key: Document(key, 'v', 'same needle', path=key) for key in (first, other)}
        with Index(self.root / 'rank.sqlite', Scope('p', 't'), RagConfig(), lambda text: len(text)) as index:
            for document in documents.values():
                index.upsert(document)
            retriever = Retriever(index, len, read_source=lambda source, version: documents[source].text)
            query = Query('needle', {key: 'v' for key in documents}, source_priorities={first: 1})
            result = json.loads(retriever.retrieve(query).message)
            self.assertEqual([hit['source_id'] for hit in result['hits']], [other, first])
            index.prune_sources({}, source_prefix=first.rsplit('/', 1)[0])
            self.assertEqual(index.stats()['documents'], 1)
            self.assertEqual(index.sources(query)[0].source_id, other)

    def test_pause_stops_before_backend_dispatch(self):
        signal = control.Signal()
        with control.using(signal):
            signal.interrupt('pause')
            with self.assertRaises(ModelInterrupted):
                ContainerSources(self.owner)
        self.assertEqual(self.backend.calls, [])


if __name__ == '__main__':
    unittest.main()
