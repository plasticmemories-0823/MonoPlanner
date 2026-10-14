"""引用复用始终受限，实际使用前检查原始内容。"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.storage.history import LocalHistory


class IncrementalHistoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.store = LocalHistory(Path(temp.name))
        self.owner = self.store.lease()
        self.owner.__enter__()
        self.addCleanup(self.store.close)
        self.addCleanup(self.owner.__exit__, None, None, None)

    def test_old_messages_are_not_serialized_again_and_load_is_identical(self):
        state = {'history': [{'type': 'human', 'data': {'content': '中文\\n "path"'}}]}
        self.store.record(state, 'progress', {})
        state['history'].append({'type': 'ai', 'data': {'content': 'complete'}})
        for i in range(20):
            self.store.record(state, 'progress', {'i': i})
        self.assertEqual(self.store.stats['message_serializations'], 2)
        self.assertEqual(self.store.load(), state)
        self.assertEqual(len(list((self.store.root / 'history/messages').glob('*.json'))), 2)

    def test_nested_mutation_is_saved_as_new_original(self):
        message = {'data': {'tool_calls': [{'args': {'code': 'a'}}]}}
        state = {'history': [message]}
        self.store.record(state, 'progress', {})
        message['data']['tool_calls'][0]['args']['code'] = 'b'
        self.store.record(state, 'progress', {})
        self.assertEqual(self.store.load(), state)
        self.assertEqual(self.store.stats['message_serializations'], 2)

    def test_tamper_with_same_size_and_mtime_is_rejected_at_boundary(self):
        state = {'history': [{'content': 'original'}]}
        self.store.record(state, 'progress', {})
        path = next((self.store.root / 'history/messages').glob('*.json'))
        original = path.stat()
        path.write_bytes(path.read_bytes().replace(b'original', b'modified'))
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.store.record(state, 'acceptance_frozen', {})
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            self.store.load()
        self.assertEqual(self.store.last_commit, 1)

    def test_bounded_cache_and_close_release(self):
        self.store.MESSAGE_CACHE_BYTES = 150
        self.store.MESSAGE_CACHE_ITEMS = 2
        state = {'history': [{'content': str(i) * 25} for i in range(20)]}
        self.store.record(state, 'progress', {})
        self.assertLessEqual(self.store.message_cache_bytes, 150)
        self.assertLessEqual(len(self.store.message_cache), 2)
        self.assertEqual(self.store.load(), state)
        self.store.close()
        self.assertEqual(self.store.message_cache, {})
        self.assertEqual(self.store.message_cache_bytes, 0)

    def test_failed_new_original_does_not_publish_checkpoint(self):
        state = {'history': [{'content': 'one'}]}
        self.store.record(state, 'progress', {})
        state['history'].append({'content': 'two'})
        with patch('agent.storage.history.os.fsync', side_effect=OSError('disk failed')):
            with self.assertRaises(OSError):
                self.store.record(state, 'progress', {})
        self.assertEqual(self.store.last_commit, 1)
        self.assertEqual(self.store.load()['history'], [{'content': 'one'}])

    def test_recovery_reuses_no_in_memory_trust(self):
        state = {'history': [{'content': 'original'}]}
        self.store.record(state, 'progress', {})
        other = LocalHistory(self.store.root)
        other._open()
        try:
            self.assertEqual(other.message_cache, {})
            self.assertEqual(other.load(), state)
        finally:
            other.close()
