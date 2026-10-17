"""无模型、无网络地验证 RAG 的原文边界、权限、版本、预算与降级。"""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent.model.errors import ModelInterrupted
from agent.rag.chunks import chunk_document
from agent.rag.index import Index
from agent.rag.retrieval import Retriever
from agent.rag.types import Document, Query, RagConfig, Scope
from agent.task.control import Signal, check_interrupted, current, using


def tokens(value):
    # 测试用确定性计数器；不宣称等同于任一服务商分词器。
    return math.ceil(len(value.encode("utf-8")) / 4) + 2


class ChunkTests(unittest.TestCase):
    def test_python_preserves_functions_and_chinese_crlf_exactly(self):
        text = '# 中文\r\n@cache\r\ndef load():\r\n    return "原文"\r\n\r\ndef save():\r\n    pass\r\n'
        document = Document("source:a.py", "commit-1", text, "code", "a.py")
        chunks = chunk_document(document, RagConfig(), tokens)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        functions = {chunk.symbol: chunk for chunk in chunks if chunk.symbol}
        self.assertEqual(set(functions), {"load", "save"})
        self.assertTrue(functions["load"].text.startswith("@cache\r\n"))
        for chunk in chunks:
            self.assertEqual(chunk.text, text[chunk.start:chunk.end])
            self.assertEqual(chunk.source_sha256, document.sha256)
            self.assertEqual(chunk.payload()["range_unit"], "unicode_codepoints")

    def test_large_structure_windows_have_no_gaps_and_bounded_overlap(self):
        text = "def example():\n" + "    # failure evidence " * 1000 + "\n    return 1\n"
        document = Document("source:a.py", "v", text, "code", "a.py")
        chunks = chunk_document(document, RagConfig(), tokens)
        self.assertGreater(len(chunks), 2)
        coverage = [False] * len(text)
        for chunk in chunks:
            self.assertLessEqual(tokens(chunk.render()), 512)
            self.assertEqual(chunk.text, text[chunk.start:chunk.end])
            coverage[chunk.start:chunk.end] = [True] * (chunk.end - chunk.start)
        self.assertTrue(all(coverage))
        for left, right in zip(chunks, chunks[1:]):
            self.assertGreater(right.start, left.start)
            if right.start < left.end:
                self.assertLessEqual(tokens(text[right.start:left.end]), 64)

    def test_expensive_metadata_does_not_consume_the_body_target(self):
        # 模拟真实分词器把来源头计为约 250 Token 的情况。
        def counter(value):
            try:
                payload = json.loads(value)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and "source_sha256" in payload:
                return 250 + math.ceil(len(payload["text"]) / 4)
            return math.ceil(len(value) / 4)

        text = "def authorize():\n" + "    # preserve original evidence\n" * 160
        document = Document("source:auth.py", "commit", text, "code", "auth.py")
        chunks = chunk_document(document, RagConfig(), counter)
        self.assertGreater(len(text), 5000)
        self.assertLessEqual(len(chunks), 10)
        self.assertTrue(all(counter(chunk.render()) <= 512 for chunk in chunks))
        self.assertTrue(all(counter(chunk.text) <= 256 for chunk in chunks))
        self.assertTrue(all(counter(chunk.text) >= 200 for chunk in chunks[:-1]))
        self.assertEqual((chunks[0].start, chunks[-1].end), (0, len(text)))
        for chunk in chunks:
            self.assertEqual(chunk.text, text[chunk.start:chunk.end])
            self.assertEqual((chunk.parent_start, chunk.parent_end), (0, len(text)))
        for previous, following in zip(chunks, chunks[1:]):
            self.assertLessEqual(following.start, previous.end)
            self.assertLessEqual(counter(text[following.start:previous.end]), 64)

    def test_body_target_never_overrides_the_complete_payload_hard_limit(self):
        def counter(value):
            try:
                payload = json.loads(value)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and "source_sha256" in payload:
                return 400 + math.ceil(len(payload["text"]) / 4)
            return math.ceil(len(value) / 4)

        document = Document("source:a", "v", "original text " * 500)
        chunks = chunk_document(document, RagConfig(), counter)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(counter(chunk.render()) <= 512 for chunk in chunks))
        self.assertTrue(all(counter(chunk.text) <= 112 for chunk in chunks))

    def test_complete_structure_is_kept_when_only_the_soft_body_target_is_exceeded(self):
        text = "def one():\n" + "    # evidence\n" * 80
        chunks = chunk_document(Document("s", "v", text, "code", "a.py"), RagConfig(), tokens)
        self.assertGreater(tokens(text), 256)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertLessEqual(tokens(chunks[0].render()), 512)

    def test_shorter_newline_candidate_can_cost_more_tokens_and_must_be_rechecked(self):
        # 改变结束位置会改变来源哈希和数字字段，完整载荷的 BPE 计数并不单调。
        def counter(value):
            try:
                payload = json.loads(value)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and "source_sha256" in payload:
                return 250 + math.ceil(len(payload["text"]) / 4) + (
                    32 if payload["text"].endswith("\n") else 0
                )
            return math.ceil(len(value) / 4)

        text = ("original evidence on one line\n" * 180) + "tail"
        document = Document("source:auth.py", "commit", text)
        chunks = chunk_document(document, RagConfig(), counter)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any(not chunk.text.endswith("\n") for chunk in chunks[:-1]))
        for chunk in chunks:
            self.assertLessEqual(counter(chunk.render()), 512)
            self.assertLessEqual(counter(chunk.text), 256)
            self.assertEqual(chunk.text, text[chunk.start:chunk.end])
        self.assertEqual((chunks[0].start, chunks[-1].end), (0, len(text)))
        for previous, following in zip(chunks, chunks[1:]):
            self.assertLessEqual(following.start, previous.end)

    def test_hash_dependent_nonmonotonic_counts_never_escape_final_validation(self):
        def counter(value):
            try:
                payload = json.loads(value)
            except ValueError:
                payload = None
            if isinstance(payload, dict) and "source_sha256" in payload:
                overhead = 250 + int(payload["id"][:2], 16) % 24
                return overhead + math.ceil(len(payload["text"]) / 4)
            return math.ceil(len(value) / 4)

        text = "def auth():\n" + "    # 中文与正则 [a-z]+\\d{3} 保留原文\n" * 180
        chunks = chunk_document(Document("s", "v", text, "code", "auth.py"), RagConfig(), counter)
        self.assertTrue(all(counter(chunk.render()) <= 512 for chunk in chunks))
        self.assertTrue(all(counter(chunk.text) <= 256 for chunk in chunks))
        self.assertTrue(all(chunk.text == text[chunk.start:chunk.end] for chunk in chunks))

    def test_markdown_json_and_log_keep_structural_originals(self):
        fixtures = [
            ("document", "readme.md", "# 标题\n\nfirst paragraph\n\nsecond paragraph\n"),
            ("tool_result", "", '[ {"value": "a\\nb"}, {"value": "中文"} ]\r\n'),
            ("tool_result", "", '{"one": [1, 2], "two": {"x": "  text  "}}'),
            ("log", "", "2026-10-17 ERROR failed\nTraceback:\n  frame 1\nValueError: x\n2026-10-17 INFO ok\n"),
        ]
        for kind, path, text in fixtures:
            with self.subTest(kind=kind, path=path):
                chunks = chunk_document(Document("s", "v", text, kind, path), RagConfig(), tokens)
                self.assertEqual("".join(item.text for item in chunks), text)
                for item in chunks:
                    self.assertEqual(text[item.start:item.end], item.text)
        logs = chunk_document(Document("s", "v", fixtures[-1][2], "log"), RagConfig(), tokens)
        self.assertEqual(len(logs), 2)
        self.assertIn("ValueError: x", logs[0].text)

    def test_unparseable_source_falls_back_without_loss(self):
        text = "def broken(\n" + "raw evidence " * 700
        chunks = chunk_document(Document("x", "v", text, "code", "a.py"), RagConfig(), tokens)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[-1].end, len(text))
        self.assertTrue(all(item.text == text[item.start:item.end] for item in chunks))

    def test_tool_json_collection_splits_at_entry_boundaries(self):
        text = '{"results": [{"value": "first"}, {"value": "second"}], "status": "ok"}'
        chunks = chunk_document(Document("s", "v", text, "tool_result"), RagConfig(), tokens)
        self.assertEqual("".join(item.text for item in chunks), text)
        self.assertTrue(any("first" in item.text and "second" not in item.text for item in chunks))
        self.assertTrue(any("second" in item.text and "first" not in item.text for item in chunks))

    def test_overlong_provenance_is_rejected_not_truncated(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            chunk_document(Document("x" * 10000, "v", "data"), RagConfig(), tokens)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.index = Index(Path(self.temp.name) / "rag.db", Scope("project", "task"), RagConfig(), tokens)
        self.addCleanup(self.index.close)
        self.documents = {}

    def add(self, document):
        self.documents[(document.source_id, document.version)] = document.text
        self.index.upsert(document)
        return document

    def reader(self, source, version):
        return self.documents[(source, version)]

    def query(self, text="failure", **kwargs):
        return Query(text, {source: version for source, version in self.documents}, **kwargs)

    def retrieve(self, query, **kwargs):
        return Retriever(self.index, tokens, read_source=self.reader, **kwargs).retrieve(query)

    def test_incremental_index_is_idempotent_and_immutable_version_checked(self):
        doc = self.add(Document("source:one", "v1", "failure evidence", "code", "one.py"))
        before = self.index.stats()
        self.assertFalse(self.index.upsert(doc))
        self.assertEqual(self.index.stats()["chunks"], before["chunks"])
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.index.upsert(replace(doc, text="different failure"))
        self.assertEqual(self.retrieve(self.query()).chunks[0].text, doc.text)

    def test_scope_version_and_exclusions_apply_before_top_k(self):
        self.add(Document("source:a", "old", "failure secret", path="secret/a.py"))
        self.add(Document("source:a", "new", "failure current", path="public/a.py"))
        self.add(Document("record:one", "old-run", "failure historical", "log", "logs/one"))
        query = Query("failure", {"source:a": "new", "record:one": "old-run"})
        hits = self.index.chunks(query)
        self.assertEqual({chunk.version for chunk in hits}, {"new", "old-run"})
        self.assertEqual(len(self.index.chunks(replace(query, excluded_paths=("public",)))), 1)
        with Index(self.index.path, Scope("different-project", "task"), RagConfig(), tokens) as other:
            self.assertEqual(other.chunks(query), [])
        self.assertEqual(self.index.chunks(Query("failure", {})), [])

    def test_exact_bm25_is_complete_without_model_calls(self):
        self.add(Document("source:a", "v", "socket failure timeout timeout", path="a.txt"))
        self.add(Document("source:b", "v", "socket success", path="b.txt"))
        result = self.retrieve(self.query("timeout"))
        self.assertFalse(result.incomplete)
        self.assertEqual(result.reasons, ())
        self.assertEqual(result.chunks[0].source_id, "source:a")
        self.assertLessEqual(tokens(result.message), 8192)

    def test_exact_current_version_short_circuits_both_models(self):
        self.add(Document("source:a", "v", "def target():\n    pass\n", "code", "a.py"))
        class Forbidden:
            def search(self, *_):
                raise AssertionError("Vector must not be called.")
            def score(self, *_):
                raise AssertionError("Reranker must not be called.")
        query = self.query("target", intent="locate")
        result = self.retrieve(query, vector=Forbidden(), reranker=Forbidden())
        self.assertTrue(result.exact_hit)
        self.assertFalse(result.incomplete)
        self.assertEqual(len(result.chunks), 1)

    def test_exact_original_scan_is_not_limited_by_unrelated_chunk_candidates(self):
        self.add(Document("source:target", "current", "def target():\n    pass\n",
                          "code", "src/target.py"))
        self.add(Document("record:large", "v", "unrelated log output " * 2000,
                          "log", "logs/large.txt"))
        query = self.query("target", intent="locate")
        with patch("agent.rag.index.CANDIDATE_MEMORY_BYTES", 6000):
            with self.assertRaisesRegex(ValueError, "memory"):
                self.index.chunks(query)
            matches = self.retrieve(query)
            self.assertEqual([item.source_id for item in matches.chunks], ["source:target"])
            self.assertEqual(self.retrieve(replace(query, source_versions={})).chunks, ())
            self.assertEqual(self.retrieve(replace(query, excluded_paths=("src",))).chunks, ())

    def test_required_file_identity_does_not_trigger_missing_evidence(self):
        document = self.add(Document("source:a", "v", "failure evidence", "code", "auth/login.py"))
        chunk_id = self.index.chunks(self.query())[0].id
        class Vector:
            def search(self, *_):
                return [(chunk_id, 0.9)]
        class Ranker:
            def score(self, _, chunks):
                return [0.9] * len(chunks)
        result = self.retrieve(
            self.query(required_entities=("auth/login.py", "auth\\login.py")),
            vector=Vector(), reranker=Ranker(),
        )
        self.assertFalse(result.incomplete)
        self.assertEqual(result.missing_entities, ())
        from agent.rag.rewrite import assess

        signals = assess(intent="diagnostic", missing_entities=result.missing_entities)
        self.assertNotIn("required_evidence_missing", signals["triggered"])
        result = self.retrieve(self.query(required_entities=(document.sha256, "auth/login")))
        self.assertEqual(result.missing_entities, (document.sha256, "auth/login"))

    def test_required_symbol_uses_exact_parent_identity_when_body_is_partial(self):
        text = "def authenticate():\n" + "    # unrelated filler\n" * 1800
        text += "    # unique_failure_marker\n    return False\n"
        self.add(Document("source:a", "v", text, "code", "auth/login.py"))
        result = self.retrieve(self.query(
            "unique_failure_marker", required_entities=("authenticate", "authentic")
        ))
        self.assertTrue(result.continuation)
        self.assertNotIn("def authenticate", "".join(item.text for item in result.chunks))
        self.assertEqual(result.missing_entities, ("authentic",))

    def test_vector_ids_rechecked_and_reranker_orders_authorized_candidates(self):
        self.add(Document("a", "v", "failure alpha"))
        self.add(Document("b", "v", "failure beta"))
        self.add(Document("private", "v", "failure secret"))
        query = Query("failure", {"a": "v", "b": "v"}, mode="semantic")
        all_chunks = self.index.chunks(self.query())
        by_source = {item.source_id: item for item in all_chunks}
        class Vector:
            def search(self, *_):
                return [(by_source["private"].id, 1.0), (by_source["b"].id, 0.9)]
        class Ranker:
            def score(self, query, chunks):
                return [1.0 if item.source_id == "b" else 0.0 for item in chunks]
        result = self.retrieve(query, vector=Vector(), reranker=Ranker())
        self.assertFalse(result.incomplete)
        self.assertEqual(result.chunks[0].source_id, "b")
        self.assertNotIn("private", result.message)

    def test_failed_rerank_preserves_vector_evidence_and_reports_reason(self):
        self.add(Document("a", "v", "failure evidence"))
        class Broken:
            def score(self, *_):
                return [float("nan")]
        chunk_id = self.index.chunks(self.query())[0].id
        class Vector:
            def search(self, *_):
                return [(chunk_id, 0.9)]
        result = self.retrieve(self.query(mode="semantic"), vector=Vector(), reranker=Broken())
        self.assertIn("failure evidence", result.message)
        self.assertIn("reranker_unavailable:ValueError", result.reasons)

    def test_changed_source_or_revoked_reader_never_returns_cached_body(self):
        self.add(Document("a", "v", "failure confidential"))
        self.documents[("a", "v")] = "changed content"
        result = self.retrieve(self.query())
        self.assertEqual(result.chunks, ())
        self.assertNotIn("confidential", result.message)
        self.assertIn("source_unavailable:ValueError", result.reasons)
        def revoked(*_):
            raise PermissionError("revoked")
        result = Retriever(self.index, tokens, read_source=revoked).retrieve(self.query())
        self.assertEqual(result.chunks, ())
        self.assertIn("source_unavailable:PermissionError", result.reasons)

    def test_parent_expansion_merges_ranges_and_respects_whole_message_budget(self):
        text = "def function():\n" + "    # failure evidence\n" * 400 + "    pass\n"
        self.add(Document("a", "v", text, "code", "a.py"))
        result = self.retrieve(self.query("function"))
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, text)
        self.assertLessEqual(tokens(result.message), 8192)
        self.assertEqual(json.loads(result.message)["results"][0]["text"], text)

    def test_over_budget_parent_retains_hit_and_continuation_not_hidden_truncation(self):
        text = "def function():\n" + "    # failure evidence\n" * 1800 + "    pass\n"
        self.add(Document("a", "v", text, "code", "a.py"))
        result = self.retrieve(self.query("function"))
        self.assertLessEqual(tokens(result.message), 8192)
        self.assertTrue(result.continuation)
        self.assertTrue(result.incomplete)
        for chunk in result.chunks:
            self.assertEqual(chunk.text, text[chunk.start:chunk.end])
        self.assertTrue(any(item["reason"] == "parent_exceeds_message_budget" for item in result.continuation))
        for marker in result.continuation:
            self.assertEqual(marker["range_unit"], "unicode_codepoints")
            self.assertEqual(marker["path"], "a.py")
            self.assertGreaterEqual(marker["read_line_offset"], 0)

    def test_markdown_match_does_not_expand_a_whole_heading_section(self):
        text = "# Heading\n\nmatched failure paragraph\n\nUnrelated deployment details.\n"
        self.add(Document("doc", "v", text, "document", "readme.md"))
        result = self.retrieve(self.query("failure"))
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, "matched failure paragraph\n\n")
        self.assertNotIn("Unrelated deployment", result.message)

    def test_overlap_bridging_multiple_ranges_produces_one_exact_union(self):
        document = Document("a", "v", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        base = chunk_document(document, RagConfig(), tokens)[0]
        def part(start, end):
            return replace(base, start=start, end=end, text=document.text[start:end])
        result = Retriever._merge([part(0, 10), part(20, 30)], part(5, 25), {})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, document.text[:30])

    def test_vector_cache_restart_and_delete_do_not_touch_original(self):
        doc = self.add(Document("a", "v", "failure evidence"))
        chunks = self.index.chunks(self.query())
        self.index.cache_vectors("model-sha", chunks, [[0.1, 0.2]] * len(chunks))
        with Index(self.index.path, self.index.scope, RagConfig(), tokens) as reopened:
            self.assertEqual(reopened.vector_cache("model-sha", chunks)[chunks[0].id], [0.1, 0.2])
            self.assertEqual(reopened.vector_cache("other-model", chunks), {})
            reopened.delete("a")
            self.assertEqual(reopened.vector_cache("model-sha", chunks), {})
        self.assertEqual(doc.text, self.documents[("a", "v")])
        self.assertEqual(self.index.stats()["chunks"], 0)

    def test_configuration_is_frozen_in_scope(self):
        with self.assertRaisesRegex(ValueError, "configuration version"):
            Index(self.index.path, self.index.scope, replace(RagConfig(), chunk_target=384), tokens)

    def test_shared_vector_has_per_scope_references_before_cache_cleanup(self):
        document = self.add(Document("a", "v", "failure evidence"))
        chunks = self.index.chunks(self.query())
        self.index.cache_vectors("model", chunks, [[0.1, 0.2]] * len(chunks))
        with Index(self.index.path, Scope("other", "task"), RagConfig(), tokens) as other:
            self.assertEqual(other.vector_cache("model", chunks), {})
            other.upsert(document)
            self.assertTrue(other.vector_cache("model", chunks))
            self.index.delete("a")
            self.assertTrue(other.vector_cache("model", chunks))
            other.delete("a")
            self.assertEqual(other.vector_cache("model", chunks), {})

    def test_semantic_cache_reuses_equal_text_without_hash_noise_in_model_input(self):
        document = self.add(Document("a", "commit-one", "failure evidence", "code", "a.py"))
        first = self.index.chunks(Query("failure", {"a": "commit-one"}))[0]
        self.index.cache_vectors("model", [first], [[0.1, 0.2]])
        self.add(replace(document, version="commit-two"))
        second = self.index.chunks(Query("failure", {"a": "commit-two"}))[0]
        self.assertNotEqual(first.render(), second.render())
        self.assertEqual(first.embedding_text(), second.embedding_text())
        self.assertNotIn(first.source_sha256, first.embedding_text())
        self.assertEqual(self.index.vector_cache("model", [second]), {second.id: [0.1, 0.2]})

    def test_prune_sources_preserves_records_other_scope_and_shared_vectors(self):
        old = self.add(Document("source:a", "old", "failure evidence"))
        self.add(replace(old, version="new"))
        self.add(Document("source:removed", "v", "removed evidence"))
        record = self.add(Document("record:step", "original", "failure evidence", "log"))
        chunks = self.index.chunks(Query("failure", {"source:a": "old"}))
        self.index.cache_vectors("model", chunks, [[0.1, 0.2]] * len(chunks))
        with Index(self.index.path, Scope("other", "task"), RagConfig(), tokens) as other:
            other.upsert(old)
            other.vector_cache("model", chunks)
            self.assertEqual(self.index.prune_sources({"source:a": "new"}), 2)
            self.assertEqual(self.index.chunks(Query("failure", {"source:a": "old"})), [])
            self.assertTrue(other.vector_cache("model", chunks))
            self.assertTrue(self.index.chunks(Query("failure", {"record:step": "original"})))
            self.assertEqual(self.index.prune_sources({"source:a": "new"}), 0)
        self.assertEqual(self.documents[(record.source_id, record.version)], record.text)

    def test_unverifiable_exact_hit_does_not_suppress_missing_evidence(self):
        self.add(Document("source:a", "v", "def target():\n    pass\n", "code", "a.py"))
        self.documents[("source:a", "v")] = "source changed"
        result = self.retrieve(self.query("target", intent="locate"))
        self.assertFalse(result.exact_hit)
        self.assertFalse(json.loads(result.message)["exact_current_version_hit"])
        self.assertTrue(result.incomplete)

    def test_parallel_vector_inherits_pause_and_does_not_downgrade_it(self):
        self.add(Document("a", "v", "failure evidence"))
        signal = Signal()
        class Paused:
            def search(self, *_):
                if current() is not signal:
                    raise AssertionError("The worker lost the current task signal.")
                signal.interrupt("pause")
                check_interrupted(pause_only=True)
        with using(signal), self.assertRaises(ModelInterrupted):
            self.retrieve(self.query(mode="semantic"), vector=Paused())

    def test_memory_guard_stops_before_loading_all_candidates_or_vectors(self):
        self.add(Document("a", "v", "failure evidence"))
        chunks = self.index.chunks(self.query())
        self.index.cache_vectors("model", chunks, [[0.1, 0.2]] * len(chunks))
        with patch("agent.rag.index.CANDIDATE_MEMORY_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "memory"):
                self.index.chunks(self.query())
        with patch("agent.rag.index.VECTOR_MEMORY_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "memory"):
                self.index.vector_cache("model", chunks)

    def test_bm25_statistics_do_not_leak_across_project_scopes(self):
        self.add(Document("a", "v", "failure evidence"))
        before = json.loads(self.retrieve(self.query()).message)["hits"]
        with Index(self.index.path, Scope("private", "task"), RagConfig(), tokens) as other:
            other.upsert(Document("secret", "v", "failure " * 1000))
        after = json.loads(self.retrieve(self.query()).message)["hits"]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
