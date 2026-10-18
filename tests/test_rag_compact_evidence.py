"""检索交付只保存一份命中原文，不擅自展开父结构或改变查询。"""

from dataclasses import replace
import json
import math
from unittest.mock import patch

import pytest

from agent.rag.index import Index
from agent.rag.retrieval import Retriever
from agent.rag.types import Document, Query, RagConfig, Scope


def tokens(value):
    return math.ceil(len(value.encode("utf-8")) / 4) + 2


@pytest.fixture
def corpus(tmp_path):
    documents = {}
    with Index(tmp_path / "rag.db", Scope("project", "task"), RagConfig(), tokens) as index:
        def add(text, *, source="source:sample.py", path="sample.py", exact=True):
            document = Document(source, "v1", text, kind="code", path=path)
            documents[source] = document
            index.upsert(document, exact=exact)
            return document

        def query(text, **kwargs):
            return Query(text, {key: value.version for key, value in documents.items()}, **kwargs)

        def retrieve(text, **kwargs):
            result = Retriever(index, tokens, read_source=lambda source, version:
                               documents[source].text).retrieve(query(text, **kwargs))
            return json.loads(result.message)

        yield index, documents, add, query, retrieve


def checked_excerpt(body, hit, original):
    """命中仅引用已交付片段，原文范围和版本不能借压缩改变。"""
    assert "text" not in hit
    excerpt = next(item for item in body["results"] if item["id"] == hit["result_id"])
    assert excerpt["source_id"] == hit["source_id"] == original.source_id
    assert excerpt["version"] == hit["version"] == original.version
    assert excerpt["source_sha256"] == hit["source_sha256"] == original.sha256
    assert excerpt["start"] <= hit["start"] < hit["end"] <= excerpt["end"]
    assert excerpt["text"] == original.text[excerpt["start"]:excerpt["end"]]
    return excerpt


def test_exact_method_hit_excludes_unrequested_class_siblings(corpus):
    _, _, add, _, retrieve = corpus
    text = "class Catalog:\n    def wanted(self):\n        return 'needle'\n"
    text += "".join(
        f"    def unrelated_sibling_{number}(self):\n        return {number}\n"
        for number in range(100)
    )
    document = add(text)
    body = retrieve("needle")
    assert body["query"] == "needle" and body["complete"]
    assert len(body["hits"]) == len(body["results"]) == 1
    assert "unrelated_sibling" not in json.dumps(body)
    excerpt = checked_excerpt(body, body["hits"][0], document)
    assert excerpt["text"] == "        return 'needle'"
    assert not body["continuation"]


def test_exact_does_not_consult_ast_or_semantic_chunk_discovery(corpus):
    _, _, add, _, retrieve = corpus
    add("class Example:\n    def function(self):\n        return 'needle'\n")
    with patch("agent.rag.chunks._spans", side_effect=AssertionError("No structural parse")) as spans:
        with patch("ast.parse", side_effect=AssertionError("No AST parse")) as parse:
            body = retrieve("needle")
    spans.assert_not_called()
    parse.assert_not_called()
    assert len(body["hits"]) == 1


@pytest.mark.parametrize("query", ["foo.bar", "  foo_bar AND OR  ", "中文\r\n  next\t"])
def test_literal_query_and_complete_multiline_original_are_preserved(corpus, query):
    _, _, add, _, retrieve = corpus
    text = "unrelated\n" + "x" * 1200 + query + "y" * 1200 + "\n"
    document = add(text)
    body = retrieve(query)
    assert body["query"] == body["effective_query"] == query
    assert body["total_matches"] == 1
    hit = body["hits"][0]
    excerpt = checked_excerpt(body, hit, document)
    assert document.text[hit["match_start"]:hit["match_end"]] == query
    assert query in excerpt["text"]
    assert hit["start"] <= hit["match_start"] < hit["match_end"] <= hit["end"]
    assert hit["truncated"]


def test_overlapping_occurrences_page_by_hit_count_not_merged_excerpt_count(corpus):
    _, _, add, _, retrieve = corpus
    document = add("a" * 130 + "\n")
    cursor, seen = "", []
    for _ in range(40):
        body = retrieve("aa", limit=5, cursor=cursor)
        assert 1 <= len(body["hits"]) <= 5
        assert len(body["results"]) == 1
        for hit in body["hits"]:
            checked_excerpt(body, hit, document)
            seen.append(hit["match_start"])
        cursor = body["next_cursor"]
        if not cursor:
            break
    assert cursor == ""
    assert body["complete"] and body["total_matches"] == 129
    assert seen == list(range(129))


@pytest.mark.parametrize("text,query", [
    ("prefix term middle term suffix term\n", "term"),
    ("中文\r\n  foo.bar AND OR\t\n" * 3, "  foo.bar AND OR"),
    ("x" * 1200 + "needle" + "y" * 1200 + "\n", "needle"),
])
def test_model_view_preserves_matches_and_originals_without_repeating_source(corpus, text, query):
    from copy import deepcopy
    from agent.model.tool_results import model_result

    _, _, add, _, retrieve = corpus
    document = add(text)
    original = retrieve(query, limit=2)
    before = deepcopy(original)
    body = model_result("retrieve_evidence", original)
    assert original == before
    assert body["query"] == query and body["mode"] == "exact"
    assert body["next_cursor"] == original["next_cursor"]
    assert len(body["hits"]) == len(original["hits"])
    assert len(body["results"]) == len(original["results"])
    assert body["sources"] == [{"path": document.path, "sha256": document.sha256,
                                "version": "v1", "kind": "code"}]
    for hit, old in zip(body["hits"], original["hits"]):
        excerpt = body["results"][hit["result"]]
        assert excerpt["id"] == old["result_id"]
        assert excerpt["text"] == text[excerpt["start"]:excerpt["end"]]
        assert body["sources"][excerpt["source"]]["path"] == old["path"]
        for field in ("line", "line_end", "match_start", "match_end",
                      "column_start", "column_end", "truncated"):
            assert hit[field] == old[field]
    serialized = json.dumps(body)
    assert serialized.count(document.path) == serialized.count(document.sha256) == 1
    assert len(serialized) < len(json.dumps(original))
    assert "ranking" not in body and "effective_query" not in body
    assert model_result("retrieve_evidence", body) == body


@pytest.mark.parametrize("source,path", [
    ("source:sample.py", "sample.py"),
    ("record:plans/p1/p1-s1/stdout.log", "$record/plans/p1/p1-s1/stdout.log"),
    ("container:$container/system_include/x.h", "$container/system_include/x.h"),
])
def test_shared_source_metadata_still_verifies_citations(corpus, source, path):
    from langchain_core.messages import ToolMessage, AIMessage, message_to_dict
    from agent.model.tool_results import model_result
    from agent.rag.integration import _audit_markers

    index, documents, add, _, retrieve = corpus
    document = add("needle first\nneedle second\n", source=source, path=path)
    document = replace(document, version=document.sha256)
    documents[source] = document
    index.upsert(document, exact=True)
    body = model_result("retrieve_evidence", retrieve("needle"))
    assert len(body["sources"]) == 1 and "version" not in body["sources"][0]
    assert json.dumps(body).count(document.sha256) == 1
    raw = message_to_dict(AIMessage(content=f"[evidence:{body['results'][1]['id']}]"))
    tool = ToolMessage(content=json.dumps(body), name="retrieve_evidence", tool_call_id="lookup")
    observed = []

    def read(identity, version):
        observed.append((identity, version))
        return document.text

    assert _audit_markers(raw, [tool], read)[0]["status"] == "source_verified"
    assert observed == [(source, document.version)]


def test_projected_original_still_offloads_and_can_be_read_after_restart(corpus, tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace
    from agent.model.tool_results import model_result
    from agent.model import tool_payloads

    _, _, add, _, retrieve = corpus
    original = "中文" * 2500
    add(original)
    # 长查询本身不能截断；正文沿用现有按哈希卸载及字节续读。
    raw = retrieve(original[:2900], limit=1)
    view = model_result("retrieve_evidence", raw)
    loop = SimpleNamespace(root=tmp_path, state={"history": []})
    first = tool_payloads.project(loop, view)
    reference = first["results"][0]["text"]
    assert isinstance(reference, dict) and reference["path"].startswith("$record/tool-payloads/")
    before = deepcopy(first)
    loop = SimpleNamespace(root=tmp_path, state={"history": [{
        "type": "tool", "data": {"tool_call_id": "previous", "content": json.dumps(first)},
    }]})
    second = tool_payloads.project(loop, view)
    assert second["results"][0]["text"]["duplicate_of"]["tool_call_id"] == "previous"
    assert first == before
    restored = tool_payloads.read_bytes(tmp_path, reference["path"], 0)
    assert restored["text"] == raw["results"][0]["text"]
    assert second["hits"] == first["hits"] and second["sources"] == first["sources"]


def test_semantic_selected_child_does_not_expand_its_parent(corpus):
    index, documents, add, query, _ = corpus
    text = "class Catalog:\n    def unrelated_sibling(self):\n        return 0\n"
    text += "    # padding between independently useful methods\n" * 100
    text += "    def wanted(self):\n        return 'precise semantic evidence'\n"
    document = add(text, exact=False)
    request = query("the relevant method", mode="semantic")
    children = index.chunks(request)
    child = next(item for item in children if "precise semantic evidence" in item.text)
    assert (child.start, child.end) != (child.parent_start, child.parent_end)

    class Vector:
        def search(self, *_):
            return [(child.id, 0.9)]

    class Reranker:
        def score(self, _, chunks):
            return [0.9] * len(chunks)

    result = Retriever(
        index, tokens, vector=Vector(), reranker=Reranker(),
        read_source=lambda source, version: documents[source].text,
    ).retrieve(request)
    body = json.loads(result.message)
    assert len(body["results"]) == len(body["hits"]) == 1
    assert "unrelated_sibling" not in body["results"][0]["text"]
    excerpt = checked_excerpt(body, body["hits"][0], document)
    assert (excerpt["start"], excerpt["end"], excerpt["text"]) == (
        child.start, child.end, child.text,
    )
    assert "unrelated_sibling" not in excerpt["text"]


def test_changed_original_cannot_reuse_compact_search_cursor(corpus):
    index, documents, add, _, retrieve = corpus
    document = add("needle needle needle\n")
    first = retrieve("needle", limit=1)
    changed = replace(document, version="v2", text="needle changed needle\n")
    documents[changed.source_id] = changed
    index.upsert(changed, exact=True)
    with pytest.raises(ValueError, match="[Cc]ursor"):
        retrieve("needle", cursor=first["next_cursor"], limit=1)
