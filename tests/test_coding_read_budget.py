"""编码读取窗口共享 UTF-8 正文预算，搜索引用也计入其中。"""

import unittest

from agent.execution.executor import OperationExecutor
from agent.source.paging import read_pages
from agent.source.snapshots import SourceSnapshot
from agent.source.versions import SourceVersions
from agent.task.contracts import Operation
import test_task_runner as f


class CodingReadBudgetTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp

    def reader(self, files):
        for name, text in files.items():
            (self.repo / name).write_text(text, encoding='utf-8')
        baseline = SourceSnapshot.capture(self.repo, self.root / 'baseline')
        evidence = SourceVersions(self.env, self.root / 'evidence', baseline=baseline)
        return OperationExecutor(self.env, evidence, self.root / 'records', versioned_reads=True)

    def body_size(self, result):
        return sum(len(p['text'].encode('utf-8')) for p in read_pages(result))

    def test_only_read_defaults_to_2000_and_explicit_smaller_windows_remain_valid(self):
        self.assertEqual(Operation(id='r', op='read').limit, 2000)
        self.assertEqual(Operation(id='s', op='search', query='value').limit, 80)
        self.assertEqual(Operation(id='x', op='shell', command='echo ok').limit, 80)
        self.assertEqual(Operation(id='r', op='read', limit=17).limit, 17)
        for op, extra in [('read', {}), ('search', {'query': 'value'})]:
            with self.subTest(op=op), self.assertRaises(ValueError):
                Operation(id='r', op=op, limit=2001 if op == 'read' else 1001, **extra)

    def test_455_and_401_line_sources_are_complete_across_pages(self):
        reader = self.reader({str(n) + '.py': '# source\n' * n for n in (455, 401)})
        for n in (455, 401):
            result = reader.read_continuation(str(n) + '.py', 0, 2000)
            self.assertEqual(
                sum(b - a for p in read_pages(result) for a, b in p['visible_ranges']), n
            )
            self.assertEqual(result['continuation']['next_offset'], n)
            self.assertLessEqual(self.body_size(result), 50 * 1024)

    def test_2000_lines_maximum_and_followup_suffix(self):
        reader = self.reader({'long.py': 'x\n' * 2100})
        first = reader.read_continuation('long.py', 0, 2000)
        self.assertEqual(first['continuation']['next_offset'], 2000)
        second = reader.read_continuation('long.py', 0, 2000)
        self.assertTrue(second['already_visible'])
        third = reader.read_continuation('long.py', 2000, 2000)
        self.assertEqual(third['continuation']['next_offset'], 2100)

    def test_budget_truncates_prefix_and_followup_returns_unseen_original(self):
        reader = self.reader({'long.py': ('中' * 80 + '\n') * 650})
        first = reader.read_continuation('long.py', 0, 2000)
        pages = list(read_pages(first))
        self.assertLessEqual(self.body_size(first), 50 * 1024)
        self.assertEqual(pages[0]['visible_ranges'][0][0], 0)
        next_offset = first['continuation']['next_offset']
        self.assertLess(next_offset, 650)
        self.assertTrue(first['truncated'])
        second = reader.read_continuation('long.py', 0, 2000)
        self.assertEqual(next(read_pages(second))['offset'], next_offset)

    def test_from_step_locations_share_one_budget(self):
        reader = self.reader({'a.py': ('中' * 80 + '\n') * 200, 'b.py': ('文' * 80 + '\n') * 200})
        step = Operation(id='r', op='read', from_step='find', limit=200)
        result = reader.execute(
            step,
            {
                'find': {
                    'data': {'hits': [{'path': 'a.py', 'line': 1}, {'path': 'b.py', 'line': 1}]}
                }
            },
            self.root / 'read',
            10,
        )['data']
        self.assertLessEqual(self.body_size(result), 50 * 1024)
        self.assertEqual(len(result['reads']), 2)
        self.assertLess(result['reads'][1]['continuation']['next_offset'], 200)

    def test_long_single_line_is_explicit_and_never_marked_complete(self):
        reader = self.reader({'line.py': '中' * 20000 + '\nshort\n'})
        result = reader.read_continuation('line.py', 0, 2000)
        self.assertLessEqual(self.body_size(result), 50 * 1024)
        page = next(read_pages(result))
        self.assertEqual(page['visible_ranges'], [])
        self.assertEqual(result['continuation']['next_offset'], 0)
        self.assertEqual(page['long_line']['offset'], 0)
        self.assertTrue(result['truncated'])

    def test_multiple_missing_ranges_share_budget_and_never_drop_a_gap(self):
        reader = self.reader({'long.py': ('中' * 80 + '\n') * 650})
        reader.read_continuation('long.py', 200, 10)
        result = reader.read_continuation('long.py', 0, 650)
        self.assertLessEqual(self.body_size(result), 50 * 1024)
        self.assertGreater(result['continuation']['next_offset'], 210)
        self.assertLess(result['continuation']['next_offset'], 650)
        spans = [pair for page in read_pages(result) for pair in page['visible_ranges']]
        self.assertEqual(spans[0], [0, 200])
        self.assertEqual(spans[1][0], 210)

    def test_page_has_no_12000_character_cap(self):
        reader = self.reader({'wide.py': ('x' * 100 + '\n') * 200})
        result = reader.read_continuation('wide.py', 0, 200)
        self.assertGreater(len(result['text']), 12000)
        self.assertEqual(result['visible_ranges'], [[0, 200]])
        self.assertFalse(result['truncated'])

    def test_search_reference_unvisited_location_has_explicit_continuation(self):
        reader = self.reader({'a.py': ('x' * 300 + '\n') * 200, 'b.py': 'source\n' * 200})
        result = reader.execute(
            Operation(id='r', op='read', from_step='find', limit=200),
            {
                'find': {
                    'data': {'hits': [{'path': 'a.py', 'line': 1}, {'path': 'b.py', 'line': 1}]}
                }
            },
            self.root / 'read',
            10,
        )['data']
        self.assertLessEqual(self.body_size(result), 50 * 1024)
        last = result['reads'][1]
        # 预算可能还容得下少量短行；其余内容仍未读取。
        self.assertLess(last['continuation']['next_offset'], 200)
        self.assertTrue(last['truncated'])
