"""真实 rg 的匹配语义、自动排序及未完成扫描边界。"""

from dataclasses import replace
import hashlib
import json
import subprocess

import pytest

from agent.rag import ripgrep
from agent.rag.index import Index
from agent.rag.retrieval import Retriever
from agent.rag.types import Query, RagConfig, Scope


@pytest.fixture
def search(tmp_path):
    paths, versions = {}, {}
    index = Index(tmp_path / "catalog.sqlite3", Scope("project", "task"), RagConfig(), len)

    def add(name, body):
        path = tmp_path / name
        raw = body.encode("utf-8") if isinstance(body, str) else body
        path.write_bytes(raw)
        version = hashlib.sha256(raw).hexdigest()
        paths[name], versions[name] = path, version
        index.register(name, version, version, path=name, kind="code")

    def find(query="", **options):
        request = Query(query, versions, **options)
        result = Retriever(index, len, source_path=lambda name, _: paths[name]).retrieve(request)
        return json.loads(result.message)

    yield add, find
    index.close()


@pytest.mark.parametrize("word,text,whole,substring", [
    ("signal", "signal signaling _signal signal_extra SIGNAL", 1, 4),
    ("中文", "中文 中文资料", 1, 2),
    ("e\u0301", "e\u0301 e\u0301x", 1, 2),
    ("🙂", "🙂a🙂", 2, 2),
    ("a-a", "a-a-a", 2, 2),
    ("aba", "ababa", 0, 2),
    ("b\r\nc", "ab\r\ncd b\r\nc", 1, 2),
])
def test_complete_words_and_explicit_substrings(search, word, text, whole, substring):
    add, find = search
    add("source.txt", text)
    assert find(word)["total_matches"] == whole
    result = find(word, match="substring")
    assert result["total_matches"] == substring
    assert result["scan_complete"]
    for hit in result["hits"]:
        assert text[hit["match_start"]:hit["match_end"]] == word


def test_keywords_union_is_complete_and_bm25_is_automatic(search):
    add, find = search
    add("a.txt", "alpha " + "filler " * 100)
    add("z.txt", "alpha beta")
    one, two = find("alpha"), find("beta")
    expected = {(x["source_id"], x["match_start"], x["match_end"])
                for body in (one, two) for x in body["hits"]}
    result = find(mode="keywords", terms=("alpha", "beta", "alpha"))
    assert result["hits"][0]["path"] == "z.txt"
    assert {(x["source_id"], x["match_start"], x["match_end"])
            for x in result["hits"]} == expected
    assert result["search_metrics"]["ranking"] == "bm25"
    assert result["search_metrics"]["ranking_seconds"] >= 0
    assert result["search_metrics"]["ranked_files"] == 2


def test_keyword_recheck_does_not_add_short_substrings_in_word_mode(search):
    add, find = search
    add("source.txt", "alphabet alpha")
    result = find(mode="keywords", terms=("alpha", "alphabet"))
    assert [(x["match_start"], x["match_end"]) for x in result["hits"]] == [(0, 8), (9, 14)]


def test_cursor_binds_match_mode_terms_and_source_priority(search):
    add, find = search
    add("a.txt", "alpha alpha")
    add("b.txt", "alpha alpha")
    cursor = find("alpha", limit=1)["next_cursor"]
    with pytest.raises(ValueError, match="[Cc]ursor"):
        find("alpha", match="substring", cursor=cursor)
    with pytest.raises(ValueError, match="[Cc]ursor"):
        find(mode="keywords", terms=("alpha",), cursor=cursor)
    with pytest.raises(ValueError, match="[Cc]ursor"):
        find("alpha", source_priorities={"b.txt": -1}, cursor=cursor)


def test_binary_skip_is_explicit_and_never_delivers_binary_body(search):
    add, find = search
    add("binary.dat", b"alpha\x00secret")
    add("normal.txt", "alpha")
    result = find("alpha")
    assert result["excluded_sources"] == [{"path": "binary.dat", "reason": "binary_source_skipped"}]
    assert [x["path"] for x in result["hits"]] == ["normal.txt"]


def test_scan_timeout_is_incomplete_not_zero_match_proof(search, monkeypatch):
    add, find = search
    add("source.txt", "alpha")

    def timeout(*args, **kwargs):
        raise ripgrep.SearchError("ripgrep_timeout")

    monkeypatch.setattr(ripgrep, "rows", timeout)
    result = find("alpha")
    assert not result["scan_complete"] and not result["complete"]
    assert "ripgrep_timeout" in result["reasons"]


def test_rg_never_inherits_user_stdin_or_shell(search, monkeypatch):
    add, find = search
    add("source.txt", "alpha")
    original = subprocess.Popen
    calls = []

    def start(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(ripgrep.subprocess, "Popen", start)
    assert find("alpha")["total_matches"] == 1
    assert calls and all(row["stdin"] == subprocess.DEVNULL and not row.get("shell") for row in calls)


def test_missing_or_damaged_runtime_is_an_explicit_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ripgrep, "ROOT", tmp_path)
    with pytest.raises(ripgrep.SearchError, match="unavailable"):
        ripgrep.executable()
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources/ripgrep.json").write_text(json.dumps({"executable_sha256": "a" * 64}))
    runtime = tmp_path / "launcher/runtime/ripgrep"
    runtime.mkdir(parents=True)
    (runtime / "rg.exe").write_bytes(b"invalid")
    with pytest.raises(ripgrep.SearchError, match="integrity"):
        ripgrep.executable()
