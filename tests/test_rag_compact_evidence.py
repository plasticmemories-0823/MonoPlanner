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


def test_same_line_distinct_matches_refer_to_one_verbatim_excerpt(corpus):
    _, _, add, _, retrieve = corpus
    document = add("prefix term middle term suffix term\n")
    body = retrieve("term")
    assert len(body["hits"]) == 3 and len(body["results"]) == 1
    assert len({hit["result_id"] for hit in body["hits"]}) == 1
    assert [hit["match_start"] for hit in body["hits"]] == [7, 19, 31]
    for hit in body["hits"]:
        checked_excerpt(body, hit, document)


def test_wire_body_contains_excerpt_text_once_and_omits_internal_parent_metadata(corpus):
    _, _, add, _, retrieve = corpus
    original = "START_MARKER needle END_MARKER"
    document = add(original + "\n")
    body = retrieve("needle")
    excerpt = checked_excerpt(body, body["hits"][0], document)
    assert json.dumps(body).count(original) == 1
    assert "parent_start" not in excerpt and "parent_end" not in excerpt
    for name in ("symbol", "run_id", "timestamp"):
        assert name not in excerpt


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
