"""统一检索的字面量、原件扫描、分页和资源边界回归。"""

from dataclasses import replace
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import pytest

from agent.rag.index import Index
from agent.rag.retrieval import Retriever
from agent.rag.types import Document, Query, RagConfig, Scope
from tests.test_rag_retrieval import result_text, tokens


class UnifiedSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index = Index(Path(self.temp.name) / 'rag.db', Scope('p', 't'), RagConfig(), tokens)
        self.addCleanup(self.index.close)
        self.documents = {}

    def add(self, source, text, path=None):
        document = Document(source, 'v', text, path=path or source)
        self.documents[source] = document
        self.index.upsert(document)
        return document

    def retrieve(self, text, **kwargs):
        query = Query(text, {source: doc.version for source, doc in self.documents.items()}, **kwargs)
        result = Retriever(self.index, tokens, read_source=lambda source, version:
                           self.documents[source].text).retrieve(query)
        return json.loads(result.message)

    def test_default_query_is_one_case_sensitive_literal_not_terms_or_path_identity(self):
        self.add('one', 'red elsewhere blue', 'red blue.txt')
        self.add('two', 'red blue and Red blue')
        result = self.retrieve('red blue')
        self.assertEqual([hit['source_id'] for hit in result['hits']], ['two'])
        self.assertEqual(result['query'], 'red blue')
        self.assertEqual(result['mode'], 'exact')
        self.assertTrue(result['scan_complete'])

    def test_quotes_operators_whitespace_and_multiline_remain_literal(self):
        text = '  "a" AND b\r\nnext  '
        self.add('one', 'prefix' + text + 'suffix\n"a" b')
        result = self.retrieve(text)
        self.assertEqual(result['query'], text)
        self.assertEqual(len(result['hits']), 1)
        self.assertIn(text, result_text(result, result['hits'][0]))

    def test_no_missing_late_matches_past_bm25_top_k_and_duplicate_line_occurrences(self):
        self.add('one', 'needle ' * 230 + '\nlate needle')
        offset, cursor, found = 0, '', []
        while True:
            result = self.retrieve('needle', limit=7, cursor=cursor)
            found.extend(hit['match_start'] for hit in result['hits'])
            cursor = result['next_cursor']
            self.assertTrue(result['scan_complete'])
            if not cursor:
                break
            self.assertFalse(result['complete'])
            offset += 7
            self.assertLess(offset, 1000)
        self.assertEqual(found, [index * 7 for index in range(230)] + [1616])
        self.assertTrue(result['complete'])
        legacy = self.retrieve('needle', limit=7, offset=224)
        self.assertEqual([hit['match_start'] for hit in legacy['hits']], found[224:231])

    def test_match_crossing_chunk_boundaries_and_late_preview_are_original(self):
        text = 'x' * 6000 + 'near-needle' + 'z' * 6000
        self.add('one', text)
        # 长查询跨越旧分块边界，必须直接从原件定位并返回完整原文。
        query = text[5000:7000]
        result = self.retrieve(query)
        self.assertEqual(len(result['hits']), 1)
        hit = result['hits'][0]
        self.assertIn(query, result_text(result, hit))
        self.assertEqual(text[hit['start']:hit['end']], result_text(result, hit))
        for item in result['results']:
            self.assertEqual(item['text'], text[item['start']:item['end']])

    def test_stale_cursor_rejects_query_scope_and_source_version_changes(self):
        self.add('one', 'a a a a')
        first = self.retrieve('a', limit=1)
        with self.assertRaisesRegex(ValueError, '[Cc]ursor'):
            self.retrieve(' a', cursor=first['next_cursor'], limit=1)
        old = self.documents['one']
        self.documents['one'] = replace(old, version='v2', text='a a a')
        self.index.upsert(self.documents['one'])
        with self.assertRaisesRegex(ValueError, '[Cc]ursor'):
            self.retrieve('a', cursor=first['next_cursor'], limit=1)

    def test_exact_preserves_results_and_rejects_semantic_without_fallback(self):
        self.add('one', 'literal evidence')
        query = Query('literal', {'one': 'v'})
        retriever = Retriever(self.index, tokens, read_source=lambda *_: self.documents['one'].text)
        self.assertTrue(retriever.retrieve(query).exact_hit)
        with self.assertRaisesRegex(ValueError, 'Search mode'):
            retriever.retrieve(replace(query, mode='semantic'))

    def test_bm25_ranks_only_literal_matching_documents_before_stable_offsets(self):
        self.add('a', 'needle ' + 'filler ' * 300)
        self.add('z', 'needle needle')
        result = self.retrieve('needle')
        self.assertEqual([hit['source_id'] for hit in result['hits']], ['z', 'z', 'a'])
        self.assertGreater(result['hits'][0]['score'], result['hits'][-1]['score'])
        self.assertEqual(result['ranking'], 'literal_bm25_k1_1.2_b_0.75')

    def test_overlapping_occurrences_are_not_lost_between_pages(self):
        self.add('one', 'aaaaa')
        cursor, starts = '', []
        for _ in range(4):
            result = self.retrieve('aa', cursor=cursor, limit=1)
            starts.extend(hit['match_start'] for hit in result['hits'])
            cursor = result['next_cursor']
        self.assertEqual(starts, [0, 1, 2, 3])
        self.assertEqual(cursor, '')

    def test_cursor_rejects_changed_original_hash_and_excluded_scope(self):
        self.add('one', 'needle needle')
        self.add('two', 'needle needle')
        cursor = self.retrieve('needle', limit=1)['next_cursor']
        with self.assertRaisesRegex(ValueError, '[Cc]ursor'):
            self.retrieve('needle', cursor=cursor, excluded_paths=('two',))
        self.documents['one'] = replace(self.documents['one'], text='changed original')
        with self.assertRaisesRegex(ValueError, '[Cc]ursor'):
            self.retrieve('needle', cursor=cursor)

    def test_whitespace_only_and_late_long_line_match_remain_visible(self):
        text = 'p' * 5000 + '  needle  '
        self.add('one', text)
        blank = self.retrieve('  ')
        self.assertEqual(len(blank['hits']), 2)
        result = self.retrieve('needle')
        hit = result['hits'][0]
        self.assertIn('needle', result_text(result, hit))
        self.assertGreater(hit['start'], 4500)
        self.assertEqual(hit['match_start'], 5002)
        self.assertEqual(hit['line'], 1)
        self.assertEqual(hit['column_start'], 5003)
        self.assertEqual(hit['column_end'], 5009)
        self.assertTrue(hit['truncated'])
        self.assertTrue(result['truncated'])


    def test_metadata_only_exact_search_reads_beyond_semantic_document_limit(self):
        text = 'unrelated original line\n' * 210000 + 'late needle\n'
        document = Document('large', 'v', text, path='large.txt')
        self.assertGreater(len(text.encode()), 4 * 1024 * 1024)
        self.documents['large'] = document
        self.index.upsert(document)
        self.assertEqual(self.index.stats()['documents'], 1)
        result = self.retrieve('needle')
        self.assertEqual(len(result['hits']), 1)
        self.assertIn('needle', result_text(result, result['hits'][0]))
        self.assertEqual(result['hits'][0]['match_start'], text.index('needle'))
        self.assertTrue(result['scan_complete'])
        self.assertTrue(result['complete'])
        for item in result['results']:
            self.assertEqual(item['text'], text[item['start']:item['end']])
        self.assertEqual(self.index.stats()['documents'], 1)

    def test_metadata_only_small_source_returns_hit_without_parent_expansion(self):
        text = 'def find():\n' + '    # preserved context\n' * 200 + '    return "needle"\n'
        document = Document('code', 'v', text, kind='code', path='find.py')
        self.documents['code'] = document
        self.index.upsert(document)
        result = self.retrieve('needle')
        self.assertEqual(len(result['hits']), 1)
        self.assertEqual(result_text(result, result['hits'][0]), '    return "needle"')
        self.assertNotIn('preserved context', json.dumps(result))
        self.assertEqual(self.documents['code'].text, text)
        self.assertEqual(self.index.stats()['documents'], 1)

    def test_many_occurrences_reuse_one_current_original_while_packing(self):
        self.add('one', 'needle ' * 1000)
        reads = []
        def reader(source, version):
            reads.append(source)
            return self.documents[source].text
        result = Retriever(self.index, tokens, read_source=reader).retrieve(
            Query('needle', {'one': 'v'}, limit=1000)
        )
        self.assertLessEqual(len(reads), 3)
        self.assertLessEqual(tokens(result.message), 8192)
        self.assertTrue(json.loads(result.message)['next_cursor'])

    def test_large_identifier_metadata_does_not_interfere_with_exact_match(self):
        text = 'def ' + 'f' * 2000 + '():\n    return 1\n\nneedle\n'
        document = Document('generated', 'v', text, kind='code', path='generated.py')
        self.documents['generated'] = document
        self.index.upsert(document)
        result = self.retrieve('needle')
        self.assertEqual(len(result['hits']), 1)
        self.assertIn('needle', result_text(result, result['hits'][0]))
        self.assertEqual(result['reasons'], [])
        self.assertTrue(result['scan_complete'])

    def test_source_memory_admission_derives_from_shared_parent_reserve(self):
        from agent.rag.types import (
            RAG_PARENT_MEMORY_BYTES, SourceReadBudgetExceeded,
            guard_source_read, source_read_limit,
        )
        self.assertEqual(RAG_PARENT_MEMORY_BYTES, 256 * 1024 * 1024)
        limit = source_read_limit()
        self.assertGreater(limit, 4 * 1024 * 1024)
        self.assertLessEqual(guard_source_read(limit), RAG_PARENT_MEMORY_BYTES)
        with self.assertRaises(SourceReadBudgetExceeded) as failed:
            guard_source_read(limit + 1, source_id='source:large', path='large.txt')
        details = failed.exception.details
        self.assertEqual(details['source_id'], 'source:large')
        self.assertEqual(details['path'], 'large.txt')
        self.assertEqual(details['reason'], 'source_read_memory_budget')
        self.assertEqual(details['unread_from_byte'], 0)
        self.assertGreater(details['estimated_bytes'], details['parent_budget_bytes'])

    def test_resource_refusal_is_explicit_and_never_means_no_matches(self):
        from agent.rag.types import SOURCE_FIXED_MEMORY_BYTES, SOURCE_TRANSIENT_MULTIPLIER
        text = 'needle ' + '\u4e2d' * 100
        document = Document('one', 'v', text, path='large.txt')
        self.documents['one'] = document
        self.index.upsert(document)
        budget = SOURCE_FIXED_MEMORY_BYTES + SOURCE_TRANSIENT_MULTIPLIER * 150
        with patch('agent.rag.types.RAG_PARENT_MEMORY_BYTES', budget):
            result = self.retrieve('needle')
        self.assertEqual(result['hits'], [])
        self.assertFalse(result['scan_complete'])
        self.assertFalse(result['complete'])
        self.assertTrue(result['retrieval_incomplete'])
        self.assertIn('source_read_memory_budget', result['reasons'])
        refusal = result['unavailable_sources'][0]
        self.assertEqual(refusal['source_id'], 'one')
        self.assertEqual(refusal['path'], 'large.txt')
        self.assertEqual(refusal['size_bytes'], len(text.encode('utf-8')))
        self.assertEqual(refusal['unread_from_byte'], 0)

    def test_candidate_memory_pages_remain_a_contiguous_ranked_prefix(self):
        self.add('one', 'needle ' * 8)
        self.add('two', 'needle ' * 8)
        expected = [(hit['source_id'], hit['match_start'])
                    for hit in self.retrieve('needle')['hits']]
        cursor, actual = '', []
        with patch('agent.rag.retrieval.CANDIDATE_MEMORY_BYTES', 8000):
            for _ in range(20):
                result = self.retrieve('needle', cursor=cursor)
                actual.extend((hit['source_id'], hit['match_start']) for hit in result['hits'])
                cursor = result['next_cursor']
                if not cursor:
                    break
        self.assertFalse(cursor)
        self.assertEqual(actual, expected)


# 由退休的 rg 专属模块迁移：保留每个独立字面边界，不合并测试计数。
@pytest.mark.parametrize("word,text,count", [
    ("signal", "signal signaling _signal signal_extra SIGNAL", 4),
    ("中文", "中文 中文资料", 2),
    ("e\u0301", "e\u0301 e\u0301x", 2),
    ("🙂", "🙂a🙂", 2),
    ("a-a", "a-a-a", 2),
    ("aba", "ababa", 2),
    ("b\r\nc", "ab\r\ncd b\r\nc", 2),
    ("x + y", "x + y ax + y x + yz x+y", 3),
    ("echo echo", "echo echo echo ECHO echo", 2),
])
def test_literal_substring_boundaries(tmp_path, word, text, count):
    document = Document('source', 'v1', text, path='source.txt')
    with Index(tmp_path / 'catalog.db', Scope('p', 't'), RagConfig(), tokens) as index:
        index.upsert(document)
        result = Retriever(index, tokens, read_source=lambda *_: text).retrieve(
            Query(word, {'source': 'v1'})
        )
    body = json.loads(result.message)
    assert body['total_matches'] == count and body['scan_complete']
    assert body['query'] == word
    for hit in body['hits']:
        assert text[hit['match_start']:hit['match_end']] == word


def test_cursor_rejects_rg_version_even_with_matching_binding(tmp_path):
    with Index(tmp_path / 'catalog.db', Scope('p', 't'), RagConfig(), tokens) as index:
        index.upsert(Document('source', 'v1', 'needle needle'))
        engine = Retriever(index, tokens, read_source=lambda *_: 'needle needle')
        query = Query('needle', {'source': 'v1'}, limit=1)
        cursor = json.loads(engine.retrieve(query).message)['next_cursor']
        decoded = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)))
        assert decoded['schema'] == 5
        decoded['schema'] = 4
        retired = base64.urlsafe_b64encode(json.dumps(decoded).encode()).decode()
        with pytest.raises(ValueError, match='[Cc]ursor'):
            engine.retrieve(replace(query, cursor=retired))
        with pytest.raises(ValueError, match='[Cc]ursor'):
            engine.retrieve(replace(query, cursor=cursor, source_priorities={'source': 1}))


def test_literal_timeout_is_incomplete_not_zero_match_proof(tmp_path, monkeypatch):
    with Index(tmp_path / 'catalog.db', Scope('p', 't'), RagConfig(), tokens) as index:
        index.upsert(Document('source', 'v1', 'needle'))
        clock = [0]
        monkeypatch.setattr('agent.rag.retrieval.time.monotonic', lambda: clock[0])

        def slow_read(*_):
            clock[0] = 31
            return 'needle'

        body = json.loads(Retriever(index, tokens, read_source=slow_read).retrieve(
            Query('needle', {'source': 'v1'})
        ).message)
    assert not body['scan_complete'] and not body['complete']
    assert 'source_scan_timeout' in body['reasons'] and not body['hits']


def test_binary_original_is_explicitly_unavailable(tmp_path):
    with Index(tmp_path / 'catalog.db', Scope('p', 't'), RagConfig(), tokens) as index:
        index.upsert(Document('source', 'v1', 'needle\0private'))
        body = json.loads(Retriever(index, tokens, read_source=lambda *_: 'needle\0private')
                          .retrieve(Query('needle', {'source': 'v1'})).message)
    assert 'binary_source_skipped' in body['reasons']
    assert not body['scan_complete'] and not body['hits']
    assert 'private' not in json.dumps(body)


def test_literal_search_never_starts_process_or_retrieval_model(tmp_path, monkeypatch):
    import subprocess
    from agent.model import client

    def forbidden(*args, **kwargs):
        pytest.fail('Text retrieval must not invoke a process or a model')

    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(client, 'create_model', forbidden)
    with Index(tmp_path / 'catalog.db', Scope('p', 't'), RagConfig(), tokens) as index:
        index.upsert(Document('source', 'v1', 'needle'))
        result = Retriever(index, tokens, read_source=lambda *_: 'needle').retrieve(
            Query('needle', {'source': 'v1'})
        )
    assert result.exact_hit
