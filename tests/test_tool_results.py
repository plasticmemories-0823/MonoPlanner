"""待发送集合按自然主键保留最后项；原件、事件和读取范围均不改写。"""

from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from agent.model.tool_results import normalize_result


class ToolResultTests(unittest.TestCase):
    def test_web_sources_keep_last_item_at_its_last_position(self):
        original = {'sources': [
            {'url': 'a.com/1', 'title': 'old', 'snippet': 'old'},
            {'url': 'b.com/2', 'title': 'middle'},
            {'url': 'a.com/1', 'title': 'new', 'snippet': 'new'},
        ], 'truncated': True}
        before = deepcopy(original)
        result = normalize_result('web_search', original)
        self.assertEqual(result['sources'], original['sources'][1:])
        self.assertTrue(result['truncated'])
        self.assertEqual(original, before)
        self.assertIsNot(result, original)
        self.assertIsNot(result['sources'][0], original['sources'][1])

    def test_reliable_primary_key_does_not_hash_content(self):
        value = {'sources': [{'url': 'a', 'text': 'different old'}, {'url': 'a', 'text': 'new'}]}
        with patch('agent.model.tool_results.hashlib.sha256', side_effect=AssertionError('no fallback')):
            self.assertEqual(normalize_result('web_search', value)['sources'], value['sources'][1:])

    def test_url_is_exact_nonempty_string_not_network_validation_or_normalization(self):
        urls = ['a.com/1', 'https://x/a?q=1', 'https://x/a?q=2',
                'HTTPS://X/a', 'https://x/a', ' https://x/a ']
        value = {'sources': [{'url': url} for url in urls]}
        self.assertEqual(normalize_result('web_search', value), value)

    def test_absent_blank_or_wrong_type_url_falls_back_to_single_item_hash(self):
        first = {'url': '', 'title': 'first'}
        duplicate = {'title': 'first', 'url': ''}
        value = {'sources': [first, {'url': ' ', 'title': 'second'}, duplicate,
                             {'url': 1, 'title': 'a'}, {'url': 1, 'title': 'b'}]}
        self.assertEqual(normalize_result('web_search', value)['sources'], value['sources'][1:])

    def test_hash_fallback_preserves_json_type_distinctions(self):
        value = {'sources': [True, 1, 1.0, '1', None, [1], {'value': 1},
                             {'value': 1.0}, None, [1]]}
        result = normalize_result('web_search', value)['sources']
        self.assertEqual([type(item) for item in result],
                         [bool, int, float, str, dict, dict, type(None), list])
        self.assertEqual(result[-1], [1])

    def test_non_json_unknown_objects_are_preserved_instead_of_stringified(self):
        value = {'sources': [math.nan, math.nan, {1: 'non-string key'}, {1: 'non-string key'}]}
        self.assertEqual(len(normalize_result('web_search', value)['sources']), 4)

    def test_directory_key_normalizes_only_key_and_keeps_latest_raw_value(self):
        value = {'path': 'E:/repo', 'entries': [
            {'path': 'E:\\repo\\./a', 'directory': False, 'name': 'a'},
            {'path': 'E:/repo/B', 'directory': False, 'name': 'B'},
            {'path': 'E:/repo/a', 'directory': True, 'name': ' a unchanged '},
        ], 'next_offset': 200}
        result = normalize_result('list_files', value, project=Path('E:/repo'))
        self.assertEqual(result['entries'], value['entries'][1:])
        self.assertEqual(result['next_offset'], 200)
        self.assertEqual(result['entries'][-1]['name'], ' a unchanged ')

    def test_path_case_and_different_lines_are_not_merged(self):
        value = {'hits': [
            {'path': 'a.py', 'line': 1, 'text': 'one'},
            {'path': 'a.py', 'line': 2, 'text': 'two'},
            {'path': 'A.py', 'line': 1, 'text': 'case-sensitive file'},
        ]}
        self.assertEqual(normalize_result('search_files', value), value)

    def test_search_same_location_latest_text_wins_without_mutating_text_whitespace(self):
        value = {'hits': [
            {'path': './a.py', 'line': 5, 'text': 'old'},
            {'path': 'b.py', 'line': 2, 'text': 'middle'},
            {'path': 'a.py', 'line': 5, 'text': '\t new  \r\n'},
        ], 'truncated': True, 'files_examined': 21}
        result = normalize_result('search_files', value, source_version='s1', project='root')
        self.assertEqual(result['hits'], value['hits'][1:])
        self.assertEqual(result['files_examined'], 21)

    def test_mixed_source_versions_preserve_different_observations(self):
        value = {'hits': [
            {'source_version': 's1', 'path': 'a.py', 'line': 1, 'text': 'old'},
            {'source_version': 's2', 'path': 'a.py', 'line': 1, 'text': 'new'},
        ]}
        self.assertEqual(normalize_result('search_files', value, source_version='fallback'), value)

    def test_boolean_negative_and_string_line_are_not_valid_natural_keys(self):
        rows = [{'path': 'a.py', 'line': number, 'text': text}
                for number in (True, -1, '1') for text in ('old', 'new')]
        self.assertEqual(normalize_result('search_files', {'hits': rows})['hits'], rows)

    def test_invalid_path_key_does_not_merge_different_items(self):
        rows = [{'path': path, 'name': name} for path in ('', '../x', 'bad\npath')
                for name in ('first', 'second')]
        self.assertEqual(normalize_result('list_files', {'entries': rows})['entries'], rows)

    def test_calls_and_projects_do_not_share_a_global_seen_index(self):
        value = {'hits': [{'path': 'a.py', 'line': 1, 'text': 'same'}]}
        for project in ('one', 'two', 'one'):
            self.assertEqual(normalize_result('search_files', value, project=project), value)

    def test_submit_plan_keeps_step_envelopes_and_normalizes_each_search_separately(self):
        hits = [{'path': 'a.py', 'line': 1, 'text': text} for text in ('old', 'new')]
        value = {'plan': 'p1', 'steps': [
            {'id': 'same', 'op': 'search', 'before': version, 'status': 'succeeded',
             'data': {'hits': deepcopy(hits)}} for version in ('s1', 's2')
        ]}
        result = normalize_result('submit_plan', value, source_version='s0', project='p')
        self.assertEqual(len(result['steps']), 2)
        self.assertEqual([row['before'] for row in result['steps']], ['s1', 's2'])
        for row in result['steps']:
            self.assertEqual(row['data']['hits'], hits[1:])
        self.assertEqual(value['steps'][0]['data']['hits'], hits)

    def test_read_pages_and_acceptance_arrays_are_never_generic_deduplication_targets(self):
        page = {'path': 'a.py', 'sha256': 'a' * 64, 'visible_ranges': [[0, 1]], 'text': '1: a'}
        value = {'steps': [{'id': 'r', 'op': 'read', 'data': {'reads': [page, page]}}],
                 'verification': {'test_results': [{'id': 'x', 'status': 'failed'},
                                                   {'id': 'x', 'status': 'passed'}]}}
        self.assertEqual(normalize_result('submit_plan', value), value)
        for tool in ('read_file', 'freeze_acceptance', 'revise_acceptance', 'get_work_status',
                     'read_work_events', 'run_existing', 'unknown'):
            self.assertEqual(normalize_result(tool, value), value)

    def test_shell_output_with_incidental_hits_is_not_a_search_result(self):
        value = {'steps': [{'id': 'x', 'op': 'shell', 'data': {
            'hits': [{'path': 'a', 'line': 1, 'text': 'old'},
                     {'path': 'a', 'line': 1, 'text': 'new'}],
        }}]}
        self.assertEqual(normalize_result('submit_plan', value), value)

    def test_repository_map_keeps_incremental_updates_but_normalizes_items_inside_each(self):
        old = {'kind': 'function', 'name': 'f', 'line': 2, 'end_line': 3, 'signature': 'old'}
        new = {**old, 'end_line': 5, 'signature': 'new'}
        middle = {'kind': 'class', 'name': 'C', 'line': 10}
        dependency = {'path': 'dep.py', 'line': 1, 'evidence': 'import dep.f',
                      'certainty': 'static_candidate', 'old_only': True}
        latest_dependency = {'path': 'dep.py', 'line': 1, 'evidence': 'import dep.f',
                             'certainty': 'confirmed'}
        update = {'path': 'a.py', 'read_ranges': [[0, 5]],
                  'new_symbols': [old, middle, new],
                  'new_dependencies': [dependency, latest_dependency]}
        other = {'path': 'a.py', 'read_ranges': [[20, 30]],
                 'new_symbols': [new], 'new_dependencies': []}
        value = {'repository_map': {'source_version': 's1', 'updates': [update, other],
                                    'known_symbols': 30}}
        result = normalize_result('search_files', value)
        updates = result['repository_map']['updates']
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[0]['new_symbols'], [middle, new])
        self.assertEqual(updates[0]['new_dependencies'], [latest_dependency])
        self.assertEqual(updates[1], other)
        self.assertEqual(value['repository_map']['updates'][0], update)

    def test_dependency_identity_keeps_distinct_imports_on_the_same_line(self):
        old = {'path': 'dep.py', 'line': 1, 'evidence': 'import dep.first',
               'certainty': 'static_candidate', 'old_only': True}
        other = {**old, 'evidence': 'import dep.second'}
        latest = {'path': 'dep.py', 'line': 1, 'evidence': 'import dep.first',
                  'certainty': 'confirmed'}
        value = {'repository_map': {'source_version': 's1', 'updates': [{
            'path': 'consumer.py', 'new_dependencies': [old, other, latest],
        }]}}
        before = deepcopy(value)
        result = normalize_result('search_files', value)
        dependencies = result['repository_map']['updates'][0]['new_dependencies']
        self.assertEqual(dependencies, [other, latest])
        self.assertEqual(value, before)
        self.assertEqual(normalize_result('search_files', result), result)

    def test_dependency_without_evidence_only_deduplicates_identical_items(self):
        for evidence in (None, '', ' ', 1, []):
            with self.subTest(evidence=evidence):
                old = {'path': 'dep.py', 'line': 1, 'certainty': 'static_candidate'}
                if evidence is not None:
                    old['evidence'] = evidence
                other = {**old, 'certainty': 'confirmed'}
                value = {'repository_map': {'source_version': 's1', 'updates': [{
                    'path': 'consumer.py', 'new_dependencies': [old, other, deepcopy(old)],
                }]}}
                result = normalize_result('search_files', value)
                dependencies = result['repository_map']['updates'][0]['new_dependencies']
                self.assertEqual(dependencies, [other, old])

    def test_dependency_evidence_is_exact_and_updates_stay_separate(self):
        dependencies = [{'path': 'dep.py', 'line': 1, 'evidence': evidence}
                        for evidence in ('import dep.first', 'import dep.second',
                                         ' import dep.first ', 'import dep.First')]
        updates = [{'path': 'consumer.py', 'source_version': version,
                    'new_dependencies': deepcopy(dependencies)} for version in ('s1', 's2')]
        value = {'repository_map': {'updates': updates}}
        self.assertEqual(normalize_result('search_files', value), value)

    def test_unrecognized_nested_arrays_do_not_gain_implicit_id_keys(self):
        value = {'sources': [{'url': 'a', 'nested': [{'id': 1}, {'id': 1}]}],
                 'other': [{'url': 'a'}, {'url': 'a'}]}
        self.assertEqual(normalize_result('web_search', value), value)

    def test_non_dict_result_is_returned_as_deep_copy(self):
        value = [{'id': 1}, {'id': 1}]
        result = normalize_result('web_search', value)
        self.assertEqual(result, value)
        self.assertIsNot(result, value)
