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
from agent.rag.index import Index
from agent.rag.retrieval import Retriever
from agent.rag.types import Document, Query, RagConfig, Scope
from agent.task.control import Signal, check_interrupted, current, using


def tokens(value):
    # 测试用确定性计数器；不宣称等同于任一服务商分词器。
    return math.ceil(len(value.encode("utf-8")) / 4) + 2


def result_text(body, hit):
    """按命中引用找到唯一原文副本，再取该命中的展示区间。"""
    item = next(item for item in body["results"] if item["id"] == hit["result_id"])
    assert (item["source_id"], item["version"], item["source_sha256"]) == (
        hit["source_id"], hit["version"], hit["source_sha256"],
    )
    assert item["start"] <= hit["start"] < hit["end"] <= item["end"]
    return item["text"][hit["start"] - item["start"]:hit["end"] - item["start"]]




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
        self.assertEqual(self.index.stats()["documents"], before["documents"])
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.index.upsert(replace(doc, text="different failure"))
        self.assertEqual(self.retrieve(self.query()).chunks[0].text, doc.text)

    def test_scope_version_and_exclusions_apply_before_top_k(self):
        self.add(Document("source:a", "old", "failure secret", path="secret/a.py"))
        self.add(Document("source:a", "new", "failure current", path="public/a.py"))
        self.add(Document("record:one", "old-run", "failure historical", "log", "logs/one"))
        query = Query("failure", {"source:a": "new", "record:one": "old-run"})
        hits = self.index.sources(query)
        self.assertEqual({chunk.version for chunk in hits}, {"new", "old-run"})
        self.assertEqual(len(self.index.sources(replace(query, excluded_paths=("public",)))), 1)
        with Index(self.index.path, Scope("different-project", "task"), RagConfig(), tokens) as other:
            self.assertEqual(other.sources(query), [])
        self.assertEqual(self.index.sources(Query("failure", {})), [])

    def test_exact_bm25_is_complete_without_model_calls(self):
        self.add(Document("source:a", "v", "socket failure timeout timeout", path="a.txt"))
        self.add(Document("source:b", "v", "socket success", path="b.txt"))
        result = self.retrieve(self.query("timeout"))
        self.assertFalse(result.incomplete)
        self.assertEqual(result.reasons, ())
        self.assertEqual(result.chunks[0].source_id, "source:a")
        self.assertLessEqual(tokens(result.message), 8192)

    def test_exact_rejects_retired_model_routes(self):
        self.add(Document("source:a", "v", "def target():\n    pass\n", "code", "a.py"))
        result = self.retrieve(self.query("target", intent="locate"))
        self.assertTrue(result.exact_hit)
        with self.assertRaisesRegex(ValueError, "Search mode"):
            self.retrieve(self.query("target", mode="semantic"))

    def test_exact_original_scan_is_not_limited_by_unrelated_chunk_candidates(self):
        self.add(Document("source:target", "current", "def target():\n    pass\n", "code", "src/target.py"))
        self.add(Document("record:large", "v", "unrelated log output " * 2000, "log", "logs/large.txt"))
        query = self.query("target", intent="locate")
        with patch("agent.rag.index.CANDIDATE_MEMORY_BYTES", 6000):
            matches = self.retrieve(query)
            self.assertEqual([item.source_id for item in matches.chunks], ["source:target"])
            self.assertEqual(self.retrieve(replace(query, source_versions={})).chunks, ())
            self.assertEqual(self.retrieve(replace(query, excluded_paths=("src",))).chunks, ())

    def test_required_file_identity_does_not_trigger_missing_evidence(self):
        doc = self.add(Document("source:a", "v", "failure evidence", "code", "auth/login.py"))
        result = self.retrieve(self.query(required_entities=("auth/login.py", "auth\\login.py")))
        self.assertFalse(result.incomplete)
        self.assertEqual(result.missing_entities, ())
        result = self.retrieve(self.query(required_entities=(doc.sha256, "auth/login")))
        self.assertEqual(result.missing_entities, (doc.sha256, "auth/login"))

    def test_required_entities_do_not_infer_unseen_parent_symbols(self):
        text = "def authenticate():\n" + "    # unrelated filler\n" * 1800
        text += "    # unique_failure_marker\n    return False\n"
        self.add(Document("source:a", "v", text, "code", "auth/login.py"))
        result = self.retrieve(self.query(
            "unique_failure_marker", required_entities=("authenticate", "authentic")
        ))
        self.assertFalse(result.continuation)
        self.assertFalse(result.incomplete)
        self.assertNotIn("def authenticate", "".join(item.text for item in result.chunks))
        self.assertEqual(result.missing_entities, ("authenticate", "authentic"))



    def test_changed_source_or_revoked_reader_never_returns_cached_body(self):
        self.add(Document("a", "v", "failure confidential"))
        self.documents[("a", "v")] = "changed content"
        result = self.retrieve(self.query())
        self.assertEqual(result.chunks, ())
        self.assertNotIn("confidential", result.message)
        self.assertIn("source_version_changed", result.reasons)
        def revoked(*_):
            raise PermissionError("revoked")
        result = Retriever(self.index, tokens, read_source=revoked).retrieve(self.query())
        self.assertEqual(result.chunks, ())
        self.assertIn("source_unavailable:PermissionError", result.reasons)

    def test_exact_function_hit_does_not_expand_parent_despite_available_budget(self):
        text = "def function():\n" + "    # failure evidence\n" * 400 + "    pass\n"
        self.add(Document("a", "v", text, "code", "a.py"))
        result = self.retrieve(self.query("function"))
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, "def function():")
        self.assertLessEqual(tokens(result.message), 8192)
        body = json.loads(result.message)
        self.assertEqual(result_text(body, body["hits"][0]), "def function():")
        self.assertNotIn("failure evidence", result.message)

    def test_large_parent_stays_readable_without_being_implicitly_returned(self):
        text = "def function():\n" + "    # failure evidence\n" * 1800 + "    pass\n"
        self.add(Document("a", "v", text, "code", "a.py"))
        result = self.retrieve(self.query("function"))
        self.assertLessEqual(tokens(result.message), 8192)
        self.assertFalse(result.continuation)
        self.assertFalse(result.incomplete)
        for chunk in result.chunks:
            self.assertEqual(chunk.text, text[chunk.start:chunk.end])
        body = json.loads(result.message)
        self.assertEqual(body["total_matches"], 1)
        hit = body["hits"][0]
        self.assertEqual((hit["path"], hit["line"], hit["version"]), ("a.py", 1, "v"))
        self.assertEqual(hit["source_sha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(result_text(body, hit), "def function():")
        self.assertIn("read the original path and line range", body["notice"])
        self.assertEqual(self.documents[("a", "v")], text)

    def test_markdown_match_does_not_expand_a_whole_heading_section(self):
        text = "# Heading\n\nmatched failure paragraph\n\nUnrelated deployment details.\n"
        self.add(Document("doc", "v", text, "document", "readme.md"))
        result = self.retrieve(self.query("failure"))
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].text, "matched failure paragraph")
        self.assertNotIn("Unrelated deployment", result.message)

    def test_overlap_bridging_multiple_ranges_produces_one_exact_union(self):
        document = Document("a", "v", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.index.upsert(document)
        base = self.index.sources(Query("0", {"a": "v"}))[0]
        def part(start, end):
            return replace(base, start=start, end=end, text=document.text[start:end])
        result = Retriever._merge([part(0, 10), part(20, 30)], part(5, 25))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, document.text[:30])


    def test_historical_vector_index_is_rejected_without_migration(self):
        import sqlite3
        from contextlib import closing
        legacy = Path(self.temp.name) / "legacy.sqlite3"
        with closing(sqlite3.connect(legacy)) as connection:
            with connection:
                connection.execute("CREATE TABLE vectors(payload TEXT)")
                connection.execute("INSERT INTO vectors VALUES('preserve historical data')")
        before = legacy.read_bytes()
        with self.assertRaisesRegex(ValueError, "frozen source"):
            Index(legacy, Scope("project", "task"), RagConfig(), tokens)
        self.assertEqual(legacy.read_bytes(), before)

    def test_configuration_is_frozen_in_scope(self):
        with self.assertRaisesRegex(ValueError, "configuration version"):
            Index(self.index.path, self.index.scope, replace(RagConfig(), message_token_limit=4096), tokens)



    def test_prune_sources_preserves_records_and_other_scope(self):
        old = self.add(Document("source:a", "old", "failure evidence"))
        self.add(replace(old, version="new"))
        self.add(Document("source:removed", "v", "removed evidence"))
        record = self.add(Document("record:step", "original", "failure evidence", "log"))
        with Index(self.index.path, Scope("other", "task"), RagConfig(), tokens) as other:
            other.upsert(old)
            self.assertEqual(self.index.prune_sources({"source:a": "new"}), 2)
            self.assertEqual(self.index.sources(Query("failure", {"source:a": "old"})), [])
            self.assertTrue(other.sources(Query("failure", {"source:a": "old"})))
            self.assertTrue(self.index.sources(Query("failure", {"record:step": "original"})))
            self.assertEqual(self.index.prune_sources({"source:a": "new"}), 0)
        self.assertEqual(self.documents[(record.source_id, record.version)], record.text)

    def test_unverifiable_exact_hit_does_not_suppress_missing_evidence(self):
        self.add(Document("source:a", "v", "def target():\n    pass\n", "code", "a.py"))
        self.documents[("source:a", "v")] = "source changed"
        result = self.retrieve(self.query("target", intent="locate"))
        self.assertFalse(result.exact_hit)
        self.assertFalse(json.loads(result.message)["exact_current_version_hit"])
        self.assertTrue(result.incomplete)


    def test_source_catalog_memory_guard_stops_before_unbounded_allocation(self):
        self.add(Document("a", "v", "failure evidence"))
        with patch("agent.rag.index.CANDIDATE_MEMORY_BYTES", 100):
            with self.assertRaisesRegex(ValueError, "memory"):
                self.index.sources(self.query())

    def test_bm25_statistics_do_not_leak_across_project_scopes(self):
        self.add(Document("a", "v", "failure evidence"))
        before = json.loads(self.retrieve(self.query()).message)["hits"]
        with Index(self.index.path, Scope("private", "task"), RagConfig(), tokens) as other:
            other.upsert(Document("secret", "v", "failure " * 1000))
        after = json.loads(self.retrieve(self.query()).message)["hits"]
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
