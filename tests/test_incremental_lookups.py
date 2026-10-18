"""验证记录中的实际可见内容、源码变更和渐进式源码导航集成。"""

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_to_dict
from agent.execution.executor import OperationExecutor
from agent.source.paging import visible_reads, read_pages
from agent.source.snapshots import SourceSnapshot
from agent.source.versions import SourceVersions
from agent.source.exploration import ExplorationMap
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
import test_task_runner as f


def map_fixture(files):
    hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
    executor = SimpleNamespace(
        inventory=lambda: hashes, evidence=SimpleNamespace(current='version-1'),
        prepared=None, source_bytes=lambda name: files[name],
    )
    mapping = ExplorationMap(executor, {})
    pages = [{'path': name, 'sha256': hashes[name], 'text': raw.decode(),
              'visible_ranges': [[0, len(raw.splitlines())]]} for name, raw in files.items()]
    return mapping, {'reads': pages}


def map_nodes(page, field='new_symbols'):
    return [node for update in page['updates'] for node in update.get(field, [])]


def test_focused_navigation_does_not_drain_other_lookups():
    mapping, data = map_fixture({
        'many.py': ''.join(f'def f{i}(): return {i}\n' for i in range(12)).encode(),
        'other.py': b'def other(): pass\n',
    })
    first = mapping.observe(data['reads'][0], record='$record/maps/many.json')
    mapping.acknowledge(first)
    assert mapping.observe({}) is None
    second = mapping.observe(data['reads'][1], record='$record/maps/other.json')
    assert [item['path'] for item in second['updates']] == ['other.py']
    assert len(mapping.state['many.py']['symbols']) == 12
    assert len(mapping.state['many.py']['delivered']['ids']) == 5


def test_current_navigation_selects_only_current_range_and_parents():
    source = ('class Scope:\n' + ''.join(
        f'    def f{i}(self):\n        return {i}\n' for i in range(12))).encode()
    mapping, data = map_fixture({'a.py': source})
    first = mapping.observe(data, record='$record/maps/full.json')
    mapping.acknowledge(first)
    hit = {'hits': [{'path': 'a.py', 'line': 25,
                     'source_sha256': data['reads'][0]['sha256']}]}
    current = mapping.observe(hit, record='$record/maps/current.json')
    assert [node['name'] for node in map_nodes(current)] == ['Scope.f11']
    fresh, _ = map_fixture({'a.py': source})
    current = fresh.observe(hit, record='$record/maps/fresh.json')
    assert [node['name'] for node in map_nodes(current)] == ['Scope', 'Scope.f11']


def test_compact_navigation_keeps_locations_without_repeating_source_fields():
    mapping, data = map_fixture({'a.py': b'def value(argument=123): return argument\n'})
    page = mapping.observe(data, record='$record/maps/a.json')
    update = page['updates'][0]
    assert 'signature' not in map_nodes(page)[0]
    assert not {'read_ranges', 'located_ranges', 'new_dependencies', 'parser'} & update.keys()
    assert mapping.state['a.py']['read_ranges'] == [[0, 1]]
    assert update['sha256'] == hashlib.sha256(data['reads'][0]['text'].encode()).hexdigest()


def test_coverage_changes_omit_unchanged_counters_in_history_and_same_batch():
    for commit in (False, True):
        mapping, data = map_fixture({'a.py': b'def a(): pass\n', 'b.py': b'def b(): pass\n'})
        first = mapping.observe(data['reads'][0], record='$record/maps/a.json')
        assert first['coverage_changes']['total_files'] == 2
        if commit:
            mapping.acknowledge(first)
        second = mapping.observe(data['reads'][1], queued=[] if commit else [first],
                                 record='$record/maps/b.json')
        assert 'total_files' not in second['coverage_changes']
        assert second['coverage_changes']['explored_files'] == 2


def test_compact_navigation_overflow_points_to_local_record():
    mapping, data = map_fixture({'a.py': ''.join(
        f'def f{i}(): pass\n' for i in range(12)).encode()})
    record = '$record/maps/a.json'
    page = mapping.observe(data, record=record)
    assert page['continuation']['path'] == record
    assert page['remaining'] == {'a.py': {'new_symbols': 7}}
    mapping.acknowledge(page)
    assert mapping.observe({}) is None


def test_explicit_lookup_continues_selected_symbols_and_empty_updates_disappear():
    mapping, data = map_fixture({'many.py': ''.join(
        f'def f{i}(): return {i}\n' for i in range(12)).encode()})
    record = '$record/maps/many.json'
    first = mapping.observe(data, record=record)
    assert len(map_nodes(first)) == 5
    assert first['remaining'] == {'many.py': {'new_symbols': 7}}
    assert mapping.observe(data, record=record) == first  # 未追加历史，不能提前确认。
    mapping.acknowledge(first)
    assert mapping.observe({}) is None
    second = mapping.observe(data, record=record)
    assert len(map_nodes(second)) == 5
    assert second['remaining'] == {'many.py': {'new_symbols': 2}}
    assert second['coverage_changes']['delivered_symbols'] == 5
    assert second['continuation']['path'] == record
    mapping.acknowledge(second)
    last = mapping.observe(data, record=record)
    assert len(map_nodes(last)) == 2
    assert not last['truncated']
    mapping.acknowledge(last)
    assert mapping.observe(data) is None
    assert len({node['id'] for page in (first, second, last) for node in map_nodes(page)}) == 12


def test_queued_delivery_is_distinct_from_acknowledged_history_and_survives_restart():
    mapping, data = map_fixture({'a.py': b'def a(): pass\ndef b(): pass\n'})
    first = mapping.observe(data)
    assert mapping.observe(data, queued=[first]) is None
    restored = ExplorationMap(mapping.executor, json.loads(json.dumps(mapping.state)))
    assert restored.observe(data) == first
    restored.acknowledge(first)
    restored.acknowledge(first)
    assert restored.observe(data) is None


def test_file_and_byte_caps_do_not_consume_omitted_entries(monkeypatch):
    import agent.source.exploration as module
    for budget in (4000, 1900):
        monkeypatch.setattr(module, 'MAX_MAP_BYTES', budget)
        mapping, data = map_fixture({f'f{i}.py': f'def value{i}(): pass\n'.encode()
                                     for i in range(14)})
        names = []
        for _ in range(40):
            page = mapping.observe(data)
            if page is None:
                break
            assert len(page['updates']) <= 6
            assert len(json.dumps(page, ensure_ascii=False, indent=2).encode()) <= budget
            names.extend(node['name'] for node in map_nodes(page))
            mapping.acknowledge(page)
        assert len(names) == len(set(names)) == 14
        assert mapping.observe(data) is None


def test_map_references_original_imports_and_counts_partial_scope():
    raw = b'from dep import (\r\n    alpha as a,\r\n    beta,\r\n)\r\n'
    mapping, data = map_fixture({'consumer.py': raw, 'dep.py': b'alpha = beta = 0\n'})
    first = mapping.observe(data['reads'][0])
    assert first['coverage_changes']['total_files'] == 2
    assert first['coverage_changes']['explored_files'] == 1
    assert first['coverage_changes']['status'] == 'partial'
    imports = map_nodes(first, 'new_dependencies')
    assert [item['import_target'] for item in imports] == ['dep.alpha', 'dep.beta']
    for item in imports:
        assert 'evidence' not in item
        assert (item['line'], item['end_line']) == (1, 4)
        update = first['updates'][0]
        assert update['sha256'] == hashlib.sha256(raw).hexdigest()
        assert b''.join(raw.splitlines(keepends=True)[item['line'] - 1:item['end_line']]) == raw


def test_parser_limit_is_explicit_and_hit_is_not_a_read(monkeypatch):
    import agent.source.exploration as module
    monkeypatch.setattr(module, 'MAX_FILE_BYTES', 10)
    mapping, data = map_fixture({'a.py': b'def a(): return 1\n'})
    page = mapping.observe({'hits': [{'path': 'a.py', 'line': 1,
                                     'source_sha256': data['reads'][0]['sha256']}]})
    update = page['updates'][0]
    assert mapping.state['a.py']['read_ranges'] == []
    assert mapping.state['a.py']['located_ranges'] == [[0, 1]]
    assert update['parser']['status'] == 'size_limit'
    assert not update['parser']['complete']
    assert update['parser']['continuation'] == {'path': 'a.py', 'offset': 0, 'limit': 200}
    assert page['coverage_changes']['fully_read_files'] == 0


def test_changed_hash_retires_delivery_and_stale_hits_do_not_grow_map():
    files = {'a.py': b'def a(): return 1\n'}
    mapping, data = map_fixture(files)
    first = mapping.observe(data)
    mapping.acknowledge(first)
    files['a.py'] = b'def replacement(): return 2\n'
    mapping.executor.inventory()['a.py'] = hashlib.sha256(files['a.py']).hexdigest()
    mapping.executor.evidence.current = 'version-2'
    assert mapping.observe(data) is None
    assert mapping.observe({'hits': [{'path': 'a.py', 'line': 1,
                                     'source_sha256': data['reads'][0]['sha256']}]}) is None
    page = mapping.observe({'path': 'a.py', 'sha256': mapping.executor.inventory()['a.py'],
                            'text': files['a.py'].decode(), 'visible_ranges': [[0, 1]]})
    assert [node['name'] for node in map_nodes(page)] == ['replacement']
    assert mapping.state['a.py']['delivered']['ids'] == []


def test_all_imports_and_noncontiguous_ranges_eventually_arrive():
    files = {'a.py': ''.join(f'from dep import value{i}\n' for i in range(30)).encode(),
             'dep.py': b'value = 0\n'}
    mapping, _ = map_fixture(files)
    data = {'hits': [{'path': 'a.py', 'line': i + 1,
                      'source_sha256': mapping.executor.inventory()['a.py']}
                     for i in range(0, 30, 2)]}
    ids = []
    for _ in range(20):
        page = mapping.observe(data)
        if page is None:
            break
        ids.extend(node['id'] for node in map_nodes(page, 'new_dependencies'))
        mapping.acknowledge(page)
    assert len(ids) == len(set(ids)) == 15
    assert mapping.state['a.py']['located_ranges'] == [[i, i + 1] for i in range(0, 30, 2)]
    assert mapping.state['a.py']['read_ranges'] == []
    assert mapping.observe(data) is None


def test_navigation_labels_are_whole_or_remain_in_local_record():
    name = '函数' * 80
    raw = ('def ' + name + '():\n    pass\n').encode()
    mapping, data = map_fixture({'a.py': raw})
    page = mapping.observe(data, record='$record/maps/a.json')
    node = map_nodes(page)[0]
    assert node['name'] == name
    assert not page['truncated']
    original = mapping.state['a.py']['symbols'][0]
    assert original['name'] == name
    assert node['id'] == original['id']
    assert page['updates'][0]['sha256'] == hashlib.sha256(raw).hexdigest()
    large = '函数' * 1000
    mapping, data = map_fixture({'a.py': ('def ' + large + '(): pass\n').encode()})
    page = mapping.observe(data, record='$record/maps/large.json')
    assert page['truncated'] and 'updates' not in page
    assert page['continuation']['path'] == page['record']
    mapping.acknowledge(page)
    assert mapping.state['a.py']['symbols'][0]['name'] == large
    assert mapping.state['a.py']['delivered']['ids'] == []


def test_offloaded_map_reference_is_not_acknowledged():
    mapping, data = map_fixture({'a.py': b'def a(): pass\n'})
    page = mapping.observe(data)
    mapping.acknowledge({'format': 'monoplanner-tool-payload-v1', 'path': '$record/preview'})
    assert mapping.observe(data) == page


def test_oversized_identity_falls_back_to_record_without_false_delivery():
    name = 'nested/' * 1000 + 'a.py'
    mapping, data = map_fixture({name: b'def a(): pass\n'})
    page = mapping.observe(data, record='$record/plans/p1/r/repository-map.json')
    assert page['truncated']
    assert page['continuation']['path'] == page['record']
    assert 'updates' not in page
    mapping.acknowledge(page)
    assert mapping.state[name]['delivered']['ids'] == []


def test_parse_failure_and_unsupported_language_remain_explicit_after_reload():
    mapping, data = map_fixture({'broken.py': b'def broken(\n', 'plain.c': b'int value;\n'})
    page = mapping.observe(data)
    assert [item['parser']['status'] for item in page['updates']] == [
        'SyntaxError', 'unsupported_language']
    assert page['coverage_changes']['status'] == 'complete'
    assert page['coverage_changes']['parsed_files'] == 0
    mapping.acknowledge(page)
    restored = ExplorationMap(mapping.executor, json.loads(json.dumps(mapping.state)))
    assert restored.observe(data) is None


class IncrementalLookupTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp
    candidate = f.TaskRunnerTests.candidate
    shell = f.TaskRunnerTests.shell

    def reader(self):
        (self.repo / 'long.py').write_text(''.join('# line %d\n' % i for i in range(1, 181)))
        baseline = SourceSnapshot.capture(self.repo, self.root / 'baseline')
        evidence = SourceVersions(self.env, self.root / 'evidence', baseline=baseline)
        return OperationExecutor(self.env, evidence, self.root / 'records', versioned_reads=True)

    def inspect(self, name, offset, limit, query=''):
        return {
            'kind': 'inspect',
            'note': name,
            'steps': [
                {
                    'id': name,
                    'op': 'read',
                    'path': 'long.py',
                    'offset': offset,
                    'limit': limit,
                    'query': query,
                }
            ],
        }

    def test_expanding_read_only_returns_new_suffix(self):
        reader = self.reader()
        first = reader.read_continuation('long.py', 0, 60)
        second = reader.read_continuation('long.py', 0, 80)
        self.assertEqual(first['visible_ranges'], [[0, 60]])
        self.assertEqual(second['offset'], 60)
        self.assertEqual(second['visible_ranges'], [[60, 80]])
        self.assertTrue(second['text'].startswith('61: '))
        self.assertNotIn('1: # line 1\n', second['text'])
        self.assertEqual(second['continuation']['next_offset'], 80)

    def test_noncontiguous_missing_ranges_are_not_lost(self):
        reader = self.reader()
        reader.read_continuation('long.py', 0, 10)
        reader.read_continuation('long.py', 20, 10)
        result = reader.read_continuation('long.py', 0, 40)
        self.assertEqual(
            [p['visible_ranges'] for p in read_pages(result)], [[[10, 20]], [[30, 40]]]
        )

    def test_absent_transcript_and_explicit_recheck_return_original(self):
        reader = self.reader()
        reader.read_continuation('long.py', 0, 60)
        self.assertTrue(reader.read_continuation('long.py', 0, 60)['already_visible'])
        repeated = reader.read_continuation('long.py', 0, 60, repeat=True)
        self.assertEqual(repeated['visible_ranges'], [[0, 60]])
        self.assertEqual(repeated['continuation']['reason'], 'explicit_reread')
        reader.visible_reads = {}  # 本地归档或导航存在，并不表示模型已看到其内容。
        self.assertEqual(reader.read_continuation('long.py', 0, 80)['visible_ranges'], [[0, 80]])

    def test_changed_bytes_same_size_and_mtime_invalidate_old_range(self):
        reader = self.reader()
        first = reader.read_continuation('long.py', 0, 60)
        path = self.repo / 'long.py'
        stat = path.stat()
        path.write_text(path.read_text().replace('line', 'LINE'))
        import os

        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        reader.evidence.record('authorized_change')
        second = reader.read_continuation('long.py', 0, 80)
        self.assertEqual(second['offset'], 0)
        self.assertNotEqual(first['sha256'], second['sha256'])
        self.assertIn('LINE', second['text'])

    def test_budget_truncated_suffix_is_still_missing(self):
        reader = self.reader()
        (self.repo / 'long.py').write_text(
            ''.join('# %03d ' % i + 'x' * 400 + '\n' for i in range(180))
        )
        reader.evidence.record('long_excerpt')
        first = reader.read_continuation('long.py', 0, 180)
        self.assertEqual(len(first['visible_ranges']), 1)
        second = reader.read_continuation('long.py', 0, 180)
        missing_start = first['visible_ranges'][0][1]
        self.assertEqual(second['offset'], missing_start)
        self.assertTrue(second['text'].startswith(str(missing_start + 1) + ': '))

    def test_assistant_claims_and_local_paths_do_not_count_as_visible_source(self):
        reader = self.reader()
        page = reader.read('long.py', 0, 60)
        body = json.dumps({'steps': [{'status': 'succeeded', 'op': 'read', 'data': page}]})
        self.assertEqual(
            visible_reads(
                messages_to_dict([HumanMessage(content='request'), AIMessage(content=body)]), []
            ),
            {},
        )
        messages = messages_to_dict(
            [HumanMessage(content='request'), ToolMessage(content=body, tool_call_id='read')]
        )
        self.assertEqual(visible_reads(messages, [])[(page['path'], page['sha256'])], [[0, 60]])

    def test_initial_readme_and_same_batch_reads_share_visible_ranges(self):
        reader = self.reader()
        page = reader.read('long.py', 0, 60)
        initial = messages_to_dict(
            [HumanMessage(content=json.dumps({'initial_evidence': {'readme': [page]}}))]
        )
        self.assertEqual(visible_reads(initial, [])[(page['path'], page['sha256'])], [[0, 60]])
        self.assertEqual(
            visible_reads([], [{'status': 'succeeded', 'op': 'read', 'data': page}])[
                (page['path'], page['sha256'])
            ],
            [[0, 60]],
        )

    def test_map_grows_with_reads_and_search_and_survives_pause_without_replay(self):
        self.reader()
        (self.repo / 'long.py').write_text(
            'from sample import value\n\ndef first():\n    return value()\n'
            + '# filler\n' * 68
            + 'def second():\n    return first()\n'
        )
        plan = {
            'kind': 'inspect',
            'note': 'progressive map',
            'steps': [
                {'id': 'first', 'op': 'read', 'path': 'long.py', 'limit': 60},
                {'id': 'second', 'op': 'read', 'path': 'long.py', 'limit': 80},
            ],
        }
        first = TaskRunner(
            f.Model([plan]), self.env, state_dir=self.root / 'state', pause_after_steps=1
        )
        result = first.run([], 'return 4', StepBudget(max_steps=6))
        self.assertEqual(result['status'], 'paused', result)
        second = TaskRunner(f.Model([self.candidate()]), self.env, state_dir=self.root / 'state')
        result = second.run([], 'return 4', StepBudget(max_steps=6))
        self.assertEqual(result['status'], 'verified_success', result)
        rows = [
            json.loads(p.read_text())
            for p in sorted((self.root / 'state/plans/p0001').glob('p0001-*/result.json'))
        ]
        self.assertEqual(rows[1]['data']['offset'], 60)
        for row in rows:
            record = row['data']['repository_map']['record'][8:]
            mapped = json.loads((self.root / 'state' / record).read_text())['files']['long.py']
            if row['id'] == 'first':
                self.assertEqual([s['name'] for s in mapped['symbols']], ['first'])
                self.assertEqual(mapped['dependencies'][0]['path'], 'sample.py')
            else:
                self.assertEqual([s['name'] for s in mapped['symbols']], ['first', 'second'])
        self.assertEqual(result['planner_calls'], 3)

    def test_map_delivery_commits_with_actual_tool_message_across_pause_and_cache(self):
        self.reader()
        (self.repo / 'long.py').write_text(''.join(
            f'def f{i}(): return {i}\n' for i in range(12)), encoding='utf-8')
        plan = {'kind': 'inspect', 'note': 'map delivery', 'steps': [
            {'id': f'read{i}', 'op': 'read', 'path': 'long.py', 'limit': 12,
             'query': 'reread: verify exact source'} for i in range(4)]}
        first = TaskRunner(f.Model([plan]), self.env, state_dir=self.root / 'state',
                           pause_after_steps=1)
        result = first.run([], 'return 4', StepBudget(max_steps=6))
        self.assertEqual(result['status'], 'paused', result)
        self.assertEqual(first.state['source_map']['long.py']['delivered']['ids'], [])
        model = f.Model([self.candidate()])
        second = TaskRunner(model, self.env, state_dir=self.root / 'state')
        result = second.run([], 'return 4', StepBudget(max_steps=6))
        self.assertEqual(result['status'], 'verified_success', result)
        delivered = next(json.loads(message.content) for message in model.calls[0]
                         if isinstance(message, ToolMessage)
                         and json.loads(message.content).get('plan') == 'p0001')
        pages = [row['data'].get('repository_map') for row in delivered['steps']]
        self.assertEqual([len(map_nodes(page)) if page else 0 for page in pages], [5, 5, 2, 0])
        self.assertEqual(len(second.state['source_map']['long.py']['delivered']['ids']), 12)
        self.assertIsNone(pages[-1])
        self.assertFalse((self.root / 'state/plans/p0001/p0001-read3/repository-map.json').exists())

    def test_omitted_navigation_is_readable_from_registered_record_on_request(self):
        self.reader()
        (self.repo / 'long.py').write_text(''.join(
            f'def f{i}(): return {i}\n' for i in range(12)), encoding='utf-8')
        first = self.inspect('first', 0, 12)
        second = {'kind': 'inspect', 'note': 'read omitted navigation', 'steps': [{
            'id': 'record', 'op': 'read',
            'path': '$record/plans/p0001/p0001-first/repository-map.json', 'limit': 2000,
        }]}
        engine = TaskRunner(f.Model([first, second, self.candidate()]), self.env,
                            state_dir=self.root / 'state')
        result = engine.run([], 'return 4', StepBudget(max_steps=8))
        self.assertEqual(result['status'], 'verified_success', result)
        saved = json.loads((self.root / 'state/plans/p0002/p0002-record/result.json').read_bytes())
        data = saved['data']
        self.assertNotIn('repository_map', data)
        text = '\n'.join(line.split(': ', 1)[1] for line in data['text'].splitlines())
        record = self.root / 'state/plans/p0001/p0001-first/repository-map.json'
        self.assertEqual(text, record.read_text().rstrip('\n'))
        self.assertEqual(data['sha256'], hashlib.sha256(record.read_bytes()).hexdigest())
        original = json.loads(text)
        self.assertEqual(original['format'], 'repository-map-v3')
        self.assertEqual([node['name'] for node in original['files']['long.py']['symbols']],
                         [f'f{i}' for i in range(12)])
        self.assertEqual(len(engine.state['source_map']['long.py']['delivered']['ids']), 5)

    def test_authorized_source_edit_retires_stale_map_details(self):
        self.reader()
        (self.repo / 'long.py').write_text('def before():\n    return 0\n')
        scan = self.inspect('old', 0, 60)
        plan = self.candidate()
        plan['steps'].extend(
            [
                self.shell(
                    "from pathlib import Path; Path('long.py').write_text('def after():\\n    return 1\\n')",
                    'change',
                ),
                {'id': 'new', 'op': 'read', 'path': 'long.py', 'limit': 60},
            ]
        )
        engine = TaskRunner(f.Model([scan, plan]), self.env, state_dir=self.root / 'state')
        result = engine.run([], 'return 4', StepBudget(max_steps=6))
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(
            [s['name'] for s in engine.state['source_map']['long.py']['symbols']], ['after']
        )
        page = json.loads((self.root / 'state/plans/p0002/p0002-new/result.json').read_text())[
            'data'
        ]
        self.assertEqual(page['offset'], 0)
        self.assertIn('def after', page['text'])

    def test_explicit_repeat_can_reuse_complete_archive_cache(self):
        self.reader()
        first = self.inspect('first', 0, 60, 'reread: exact verification')
        second = self.inspect('second', 0, 60, 'reread: exact verification')
        result = TaskRunner(
            f.Model([first, second, self.candidate()]), self.env, state_dir=self.root / 'state'
        ).run([], 'return 4', StepBudget(max_steps=6))
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(result['local_state_stats']['read_cache_hits'], 1)

    def test_visible_range_does_not_bypass_path_or_parameter_checks(self):
        reader = self.reader()
        for path, offset, limit in (
            ('../private', 0, 10),
            ('long.py', -1, 10),
            ('long.py', 0, 2001),
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                reader.read_continuation(path, offset, limit)

    def test_search_grows_map_without_global_rescan_or_executing_modules(self):
        reader = self.reader()
        marker = self.root / 'must_not_execute'
        (self.repo / 'long.py').write_text(
            "from pathlib import Path\nPath(%r).touch()\n\ndef needle():\n    return 1\n"
            % str(marker)
        )
        reader.evidence.record('map_source')
        state = {}
        mapped = ExplorationMap(reader, state)
        from agent.task.contracts import Operation
        from unittest.mock import patch

        with patch('os.walk', side_effect=AssertionError('no global scan')):
            result = mapped.observe(
                reader.search(Operation(id='locate', op='search', query='needle'))
            )
            self.assertEqual(result['coverage_changes']['explored_files'], 1)
            self.assertEqual(result['coverage_changes']['discovered_symbols'], 1)
            mapped.acknowledge(result)
            self.assertEqual(state['long.py']['located_ranges'], [[3, 4]])
            self.assertEqual(state['long.py']['read_ranges'], [])
            mapped.acknowledge(mapped.observe(
                reader.search(Operation(id='other', op='search', query='def value'))))
            self.assertEqual(len(state), 2)
            with patch(
                'agent.source.exploration._parse',
                side_effect=AssertionError('reuse unchanged parse'),
            ):
                again = mapped.observe(
                    reader.search(Operation(id='repeat', op='search', query='needle'))
                )
            self.assertIsNone(again)
        self.assertFalse(marker.exists())
