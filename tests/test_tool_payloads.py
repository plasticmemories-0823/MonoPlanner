"""验证工具载荷卸载、可见历史去重及归档分页，不调用外部服务。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage, message_to_dict

from agent.execution.executor import OperationExecutor
from agent.model import tool_payloads as payloads
from agent.model.compaction import guard_archive
from agent.model.errors import ModelInterrupted, ModelRequestNotSent
from agent.source.paging import OUTPUT_BYTES, ReadBudget
from agent.storage.history import LocalHistory
from agent.storage.records import save
from agent.task.operations import finish_plan
from agent.task.contracts import Operation


class ToolPayloadTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.loop = SimpleNamespace(root=self.root, state={'history': [], 'source': 'source-v1'})
        self.reader = OperationExecutor(SimpleNamespace(repo_root=self.root), None, self.root)

    def remember(self, result, call='original'):
        self.loop.state['history'].append(message_to_dict(ToolMessage(
            content=json.dumps(result, ensure_ascii=False), tool_call_id=call,
        )))

    def open_store(self):
        store = LocalHistory(self.root)
        lease = store.lease()
        lease.__enter__()
        self.addCleanup(store.close)
        self.addCleanup(lease.__exit__, None, None, None)
        self.loop.history_store = store
        self.reader.archive = store
        return store

    def test_large_plain_body_is_stored_once_and_original_result_is_not_mutated(self):
        text = (' \t中文 line \r\n' * 1000) + 'no final newline'
        value = {'stdout': text, 'status': 'failed', 'exit_code': 3, 'error': 'Command failed.'}
        before = deepcopy(value)
        first = payloads.project(self.loop, value)
        second = payloads.project(self.loop, value)
        self.assertEqual(value, before)
        self.assertEqual(first['stdout']['sha256'], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(first['stdout']['path'], second['stdout']['path'])
        self.assertEqual((self.root / first['stdout']['path'][8:]).read_bytes(), text.encode())
        self.assertEqual(len(list((self.root / 'tool-payloads').iterdir())), 1)
        for key in ('status', 'exit_code', 'error'):
            self.assertEqual(first[key], value[key])

    def test_large_error_summary_and_actual_exit_status_remain_visible(self):
        error = 'Git history is unavailable.\n' + 'detail\n' * 2000
        result = payloads.project(self.loop, {'error': error, 'exit_code': 128, 'timed_out': False})
        self.assertEqual(result['error']['summary'], 'Git history is unavailable.')
        self.assertFalse(result['error']['preview_complete'])
        self.assertEqual(result['exit_code'], 128)
        self.assertFalse(result['timed_out'])

    def test_short_first_body_is_inline_then_current_view_duplicate_is_referenced(self):
        value = {'stderr': 'diagnostic\n' * 200}
        first = payloads.project(self.loop, value)
        self.assertEqual(first, value)
        self.remember(first)
        second = payloads.project(self.loop, value)
        self.assertEqual(second['stderr']['duplicate_of']['tool_call_id'], 'original')
        self.assertTrue(second['stderr']['duplicate_of']['body_was_inline'])
        self.assertEqual(second['stderr']['preview'], [])

    def test_uncommitted_projection_is_not_added_to_visible_history(self):
        value = {'stderr': 'diagnostic\n' * 200}
        self.assertEqual(payloads.project(self.loop, value), value)
        self.assertEqual(payloads.project(self.loop, value), value)
        self.assertFalse((self.root / 'tool-payloads').exists())

    def test_large_previous_reference_does_not_claim_full_body_was_read(self):
        value = {'text': 'page\n' * 3000}
        first = payloads.project(self.loop, value)
        self.remember(first)
        again = payloads.project(self.loop, value)
        self.assertFalse(again['text']['duplicate_of']['body_was_inline'])
        self.assertEqual(again['text']['preview'], [])

    def test_identical_large_fields_share_one_body_preview_within_result(self):
        text = 'same diagnostic\n' * 2000
        result = payloads.project(self.loop, {'stdout': text, 'stderr': text})
        self.assertTrue(result['stdout']['preview'])
        self.assertEqual(result['stderr']['preview'], [])
        self.assertEqual(result['stderr']['duplicate_of']['location'], 'earlier_in_this_result')
        self.assertFalse(result['stderr']['duplicate_of']['body_was_inline'])
        self.assertEqual(result['stdout']['path'], result['stderr']['path'])

    def test_compaction_generation_rebuild_drops_removed_seen_body(self):
        value = {'stdout': 'old text\n' * 200}
        self.remember(value)
        self.assertIn('duplicate_of', payloads.project(self.loop, value)['stdout'])
        self.loop.state['context_view'] = {'generation': 1, 'record': {'sha256': 'new'}}
        with patch('agent.model.compaction.history', return_value=[]):
            self.assertEqual(payloads.project(self.loop, value), value)

    def test_incremental_index_does_not_parse_entire_history_again(self):
        for index in range(50):
            self.remember({'stdout': str(index) * 2000}, str(index))
        payloads.project(self.loop, {'stdout': 'small'})
        self.remember({'stdout': 'new ' * 400}, 'new')
        with patch.object(payloads, '_fields', wraps=payloads._fields) as collect:
            payloads.project(self.loop, {'stdout': 'small'})
            self.assertEqual(collect.call_count, 1)
        with patch.object(payloads, '_fields', wraps=payloads._fields) as collect:
            payloads.project(self.loop, {'stdout': 'small'})
            self.assertEqual(collect.call_count, 0)

    def test_existing_prefix_integrity_still_rejects_old_in_place_mutation(self):
        self.remember({'stdout': 'before ' * 300})
        self.remember({'stdout': 'tail'})
        payloads.project(self.loop, {'stdout': 'small'})
        guard_archive(self.loop.state)
        self.loop.state['history'][0]['data']['content'] = json.dumps({'stdout': 'changed'})
        payloads.project(self.loop, {'stdout': 'before ' * 300})
        with self.assertRaisesRegex(ModelRequestNotSent, 'Original transcript changed'):
            guard_archive(self.loop.state)

    def test_new_history_object_cannot_inherit_seen_content(self):
        value = {'stdout': 'search-only ' * 200}
        self.remember(value)
        payloads.project(self.loop, value)
        self.loop.state['history'] = []
        self.assertEqual(payloads.project(self.loop, value), value)

    def test_explicit_web_history_uses_its_own_visible_messages(self):
        value = {'text': 'web page ' * 200}
        self.remember(value)
        explicit = []
        self.assertEqual(payloads.project(self.loop, value, history=explicit), value)

    def test_read_pages_and_readable_ranges_remain_unchanged(self):
        page = {'path': 'code.py', 'sha256': 'f' * 64, 'text': 'x' * 50000,
                'visible_ranges': [[0, 500]], 'truncated': False}
        body = {'steps': [{'op': 'read', 'status': 'succeeded', 'data': page}]}
        self.assertEqual(payloads.project(self.loop, body), body)
        self.assertFalse((self.root / 'tool-payloads').exists())

    def test_large_search_metadata_offloads_without_mutating_raw_hits(self):
        hits = [{'path': str(i), 'text': 'code' * 200} for i in range(30)]
        result = payloads.project(self.loop, {'hits': hits, 'truncated': False})
        raw = (self.root / result['hits']['path'][8:]).read_text(encoding='utf-8')
        self.assertEqual(json.loads(raw), hits)
        self.assertFalse(result['truncated'])

    def test_registered_log_larger_than_eight_mib_reuses_original_without_copy_or_full_read(self):
        store = self.open_store()
        log = self.root / 'plans/p1/s1/stdout.log'
        log.parent.mkdir(parents=True)
        raw = b'line\n' * 1800000
        log.write_bytes(raw)
        store.register_derived(log, 'operation_log', '')
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded read')):
            result = payloads.project(self.loop, {
                'stdout': 'line\n[excerpt]', 'stdout_record': 'plans/p1/s1/stdout.log',
            })
            page = self.reader.read(result['stdout']['path'], 1799998, 2)
        self.assertEqual(result['stdout']['path'], '$record/plans/p1/s1/stdout.log')
        self.assertEqual(result['stdout']['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertEqual(page['text'], '1799999: line\n1800000: line')
        self.assertFalse((self.root / 'tool-payloads').exists())

    def test_registered_log_tampering_stops_without_overwriting_original(self):
        store = self.open_store()
        log = self.root / 'plans/p1/stdout.log'
        log.parent.mkdir(parents=True)
        log.write_bytes(b'original\n' * 2000)
        store.register_derived(log, 'operation_log', '')
        log.write_bytes(b'tampered\n' * 2000)
        with self.assertRaisesRegex(ModelRequestNotSent, 'SHA-256 mismatch'):
            payloads.project(self.loop, {'stdout': 'excerpt', 'stdout_record': 'plans/p1/stdout.log'})
        self.assertTrue(log.read_bytes().startswith(b'tampered'))

    def test_cas_tampering_is_detected_on_repeated_projection_and_read(self):
        body = {'stdout': 'original\n' * 2000}
        ref = payloads.project(self.loop, body)['stdout']
        target = self.root / ref['path'][8:]
        target.write_bytes(b'tampered')
        with self.assertRaisesRegex(ModelRequestNotSent, 'SHA-256 mismatch'):
            payloads.project(self.loop, body)
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            self.reader.read(ref['path'], 0, 1)

    def test_payload_sha_and_utf8_page_survive_chunk_split(self):
        text = 'x' * 65535 + '\r\n中文\rthird\nlast'
        ref = payloads.project(self.loop, {'stdout': text})['stdout']
        page = self.reader.read(ref['path'], 1, 3)
        self.assertEqual(page['text'], '2: 中文\n3: third\n4: last')
        self.assertEqual(page['total_lines'], 4)
        self.assertEqual(page['visible_ranges'], [[1, 4]])

    def test_long_single_line_partial_is_not_a_complete_read(self):
        text = '中文' * 20000
        ref = payloads.project(self.loop, {'stdout': text})['stdout']
        page = self.reader.read(ref['path'], 0, 1)
        self.assertEqual(page['visible_ranges'], [])
        self.assertEqual(page['next_offset'], 0)
        self.assertTrue(page['truncated'])
        self.assertLessEqual(len(page['text'].encode()), OUTPUT_BYTES)
        self.assertNotIn('\ufffd', page['text'])

    def test_payload_read_retains_2000_line_and_50kib_shared_budget(self):
        text = ('x' * 100 + '\n') * 2000
        ref = payloads.project(self.loop, {'stdout': text})['stdout']
        budget = ReadBudget()
        page = self.reader.read(ref['path'], 0, 2000, budget=budget)
        self.assertTrue(page['truncated'])
        self.assertLessEqual(len(page['text'].encode()), OUTPUT_BYTES)
        with self.assertRaises(ValueError):
            self.reader.read(ref['path'], 0, 2001)
        next_page = self.reader.read(ref['path'], page['next_offset'], 10)
        self.assertEqual(next_page['visible_ranges'], [[page['next_offset'], page['next_offset'] + 10]])

    def test_preview_contains_head_tail_and_is_bounded(self):
        text = '\n'.join(str(i) + '中' * 300 for i in range(100))
        ref = payloads.project(self.loop, {'stderr': text})['stderr']
        self.assertEqual(ref['preview'][0]['line'], 0)
        self.assertEqual(ref['preview'][-1]['line'], 99)
        self.assertLessEqual(sum(len(p['text'].encode()) for p in ref['preview']), 2000)
        self.assertFalse(ref['preview_complete'])

    def test_cas_registered_once_and_recovery_verifies_original_payload(self):
        store = self.open_store()
        value = {'text': 'web body\n' * 2000}
        with patch.object(store, 'register_derived', wraps=store.register_derived) as register:
            result = payloads.project(self.loop, value)
            payloads.project(self.loop, value)
            self.assertEqual(register.call_count, 1)
        self.remember(result)
        store.record(self.loop.state, 'progress', {})
        self.assertEqual(store.load(), self.loop.state)
        (self.root / result['text']['path'][8:]).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Original record changed'):
            store.load()

    def test_storage_failure_does_not_publish_a_truncated_success(self):
        with patch.object(payloads, '_store', side_effect=OSError('Disk full')):
            with self.assertRaisesRegex(ModelRequestNotSent, 'no incomplete result'):
                payloads.project(self.loop, {'stdout': 'x' * 10000})

    def test_new_cas_reserves_space_but_reused_object_does_not_reserve_again(self):
        value = {'stdout': 'x' * 10000}
        with patch('agent.storage.resources.preflight_space') as reserve:
            payloads.project(self.loop, value)
            payloads.project(self.loop, value)
            self.assertEqual(reserve.call_count, 1)
            self.assertEqual(reserve.call_args.args[1], 10000)

    def test_space_guard_stops_before_creating_a_new_payload(self):
        with patch('agent.storage.resources.preflight_space', side_effect=OSError('Insufficient space')):
            with self.assertRaisesRegex(ModelRequestNotSent, 'Insufficient space'):
                payloads.project(self.loop, {'stdout': 'x' * 10000})
        self.assertFalse((self.root / 'tool-payloads').exists())

    def test_utf8_character_split_across_chunk_boundary_does_not_corrupt_next_line(self):
        text = 'a' * 65535 + '中\n尾'
        reference = payloads.project(self.loop, {'stdout': text})['stdout']
        self.assertTrue(reference['preview'][0]['partial'])
        page = self.reader.read(reference['path'], 1, 1)
        self.assertEqual(page['text'], '2: 尾')
        self.assertEqual(page['total_lines'], 2)

    def test_blank_lines_and_all_splitline_separators_match_existing_read_semantics(self):
        text = '\n\r\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029尾'
        path, unused = payloads._store(self.loop, text)
        page = self.reader.read(path, 0, 2000)
        lines = text.splitlines()
        self.assertEqual(page['total_lines'], len(lines))
        self.assertEqual(page['text'], '\n'.join(f'{i + 1}: {s}' for i, s in enumerate(lines)))

    def test_empty_payload_read_does_not_invent_a_visible_line(self):
        path, unused = payloads._store(self.loop, '')
        page = self.reader.read(path, 0, 1)
        self.assertEqual(page['total_lines'], 0)
        self.assertEqual(page['text'], '')
        self.assertEqual(page['visible_ranges'], [])

    def test_preview_utf8_cut_cannot_splice_later_chunk_into_prefix(self):
        text = 'a' * 199 + '中' + 'b' * (65536 - 202) + 'Z' * 10000
        result = payloads.project(self.loop, {'stdout': text})['stdout']
        self.assertEqual(result['preview'][0]['text'], 'a' * 199)
        self.assertTrue(result['preview'][0]['partial'])

    def test_byte_fallback_reconstructs_entire_long_utf8_crlf_payload(self):
        original = ('a' * 51199 + '中😀' * 10000 + '\r\n\r\nlast').encode()
        result = payloads.project(self.loop, {'stdout': original.decode()})['stdout']
        first = self.reader.read(result['path'], 0, 2000)
        self.assertEqual(first['next_query'], 'payload_bytes:0')
        query, end, pieces = first['next_query'], 0, []
        while query:
            step = Operation(id='bytes', op='read', path=result['path'], query=query)
            page = self.reader.execute(step, {}, self.root, 10)['data']
            start, next_end = page['byte_range']
            self.assertEqual(start, end)
            self.assertGreater(next_end, start)
            self.assertEqual(page['text'], original[start:next_end].decode())
            self.assertEqual(page['visible_ranges'], [])
            self.assertLessEqual(page['shown_utf8_bytes'], OUTPUT_BYTES)
            self.assertEqual(page['output_budget']['used_bytes'], page['shown_utf8_bytes'])
            pieces.append(page['text'])
            query, end = page['next_query'], next_end
        self.assertEqual(''.join(pieces).encode(), original)
        self.assertEqual(end, len(original))

    def test_byte_fallback_starts_at_original_long_line_after_crlf(self):
        text = 'short\r\n' + '中文' * 20000
        ref = payloads.project(self.loop, {'stdout': text})['stdout']
        first = self.reader.read(ref['path'], 1, 1)
        self.assertEqual(first['next_query'], 'payload_bytes:7')
        page = payloads.read_bytes(self.root, ref['path'], 7)
        self.assertTrue(text.encode()[slice(*page['byte_range'])].decode().startswith('中文'))

    def test_byte_query_cannot_read_source_or_change_parameter_units_silently(self):
        ref = payloads.project(self.loop, {'stdout': 'x' * 90000})['stdout']
        for arguments in ({'path': 'code.py'}, {'path': ref['path'], 'limit': 100},
                          {'path': ref['path'], 'offset': 1}, {'path': ref['path'], 'from_step': 's'}):
            with self.subTest(arguments=arguments):
                step = Operation(id='b', op='read', query='payload_bytes:0', **arguments)
                with self.assertRaises(ValueError):
                    self.reader.execute(step, {}, self.root, 10)
        with self.assertRaisesRegex(ValueError, 'splits a UTF-8'):
            ref = payloads.project(self.loop, {'stdout': '中' * 30000})['stdout']
            payloads.read_bytes(self.root, ref['path'], 1)

    def test_registered_non_utf8_log_keeps_status_and_raw_byte_cursor(self):
        store = self.open_store()
        log = self.root / 'plans/p1/stdout.log'
        log.parent.mkdir(parents=True)
        original = b'normal\r\n' * 2000 + b'\xff\xfe' * 35000 + b'\xe4\xb8'
        log.write_bytes(original)
        store.register_derived(log, 'operation_log', '')
        result = payloads.project(self.loop, {'stdout': 'excerpt', 'exit_code': 0,
                                            'stdout_record': 'plans/p1/stdout.log'})
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['stdout']['decoding_errors'], 'replace')
        self.assertEqual(result['stdout']['sha256'], hashlib.sha256(original).hexdigest())
        offset = 0
        while offset < len(original):
            budget = ReadBudget()
            page = payloads.read_bytes(self.root, result['stdout']['path'], offset,
                                      archive=store, budget=budget)
            start, end = page['byte_range']
            self.assertEqual(start, offset)
            self.assertGreater(end, start)
            self.assertEqual(page['text'], original[start:end].decode(errors='replace'))
            self.assertLessEqual(len(page['text'].encode()), OUTPUT_BYTES)
            self.assertEqual(budget.remaining, OUTPUT_BYTES - len(page['text'].encode()))
            self.assertEqual(page['visible_ranges'], [])
            offset = end
        self.assertEqual(log.read_bytes(), original)

    def test_confirmed_pause_is_not_reclassified_as_storage_conflict(self):
        with patch('agent.storage.resources.owner_cancelled', return_value=True), patch(
            'agent.task.control.check_interrupted', side_effect=ModelInterrupted('pause', sent=False),
        ):
            with self.assertRaises(ModelInterrupted) as stopped:
                payloads.project(self.loop, {'stdout': 'x' * 90000})
        self.assertEqual(stopped.exception.kind, 'pause')
        self.assertFalse(stopped.exception.sent)

    def test_finish_plan_archives_raw_result_and_keeps_original_pairing(self):
        response = message_to_dict(AIMessage(content='', tool_calls=[{
            'id': 'plan-call', 'name': 'submit_plan', 'args': {},
        }]))
        self.loop.state['pending'] = {'id': 'p1', 'response': response, 'rows': [{
            'id': 's1', 'op': 'shell', 'status': 'failed', 'data': {'stderr': 'failure\n' * 2000},
        }]}
        self.loop.persist = lambda *args: None
        finish_plan(self.loop, 'execution_failed')
        archived = json.loads((self.root / 'plans/p1/result.json').read_bytes())
        self.assertEqual(archived['steps'][0]['data']['stderr'], 'failure\n' * 2000)
        message = self.loop.state['history'][-1]['data']
        self.assertEqual(message['tool_call_id'], 'plan-call')
        self.assertEqual(message['name'], 'submit_plan')
        delivered = json.loads(message['content'])
        self.assertEqual(delivered['steps'][0]['status'], 'failed')
        self.assertEqual(self.loop.state['last_result'], archived)
