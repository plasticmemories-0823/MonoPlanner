"""验证记录中的实际可见内容、源码变更和渐进式源码导航集成。"""

import json
from pathlib import Path
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
        self.assertEqual(result['planner_calls'], 4)

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

        import os
        walk = os.walk

        def only_owned_temp_cleanup(path, *args, **kwargs):
            # 新流式扫描清理自己的临时 SQLite/模式文件，不等于重新扫描项目。
            if not Path(path).name.startswith(('monoplanner-search-', 'monoplanner-rg-')):
                raise AssertionError('no project or global scan')
            return walk(path, *args, **kwargs)

        with patch('os.walk', side_effect=only_owned_temp_cleanup):
            result = mapped.observe(
                reader.search(Operation(id='locate', op='search', query='needle'))
            )
            self.assertEqual(result['known_files'], 1)
            self.assertEqual(result['known_symbols'], 1)
            self.assertEqual(state['long.py']['located_ranges'], [[3, 4]])
            self.assertEqual(state['long.py']['read_ranges'], [])
            mapped.observe(reader.search(Operation(id='other', op='search', query='def value')))
            self.assertEqual(len(state), 2)
            with patch(
                'agent.source.exploration._parse',
                side_effect=AssertionError('reuse unchanged parse'),
            ):
                again = mapped.observe(
                    reader.search(Operation(id='repeat', op='search', query='needle'))
                )
            self.assertEqual(again['updates'][0]['new_symbols'], [])
        self.assertFalse(marker.exists())
