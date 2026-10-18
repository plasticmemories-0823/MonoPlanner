"""统一检索调用面与既有计划执行边界的离线验收。"""

from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import pytest

from agent.application.chat import Chatbot
from agent.application.store import ApplicationStore
from agent.execution.executor import OperationExecutor
from agent.model.tools import CHAT_TOOLS, CODING_TOOLS
from agent.model.tool_results import normalize_result
from agent.rag import integration
from agent.task.contracts import Operation


def test_one_public_search_tool_per_caller():
    assert 'search_files' not in {tool.name for tool in CHAT_TOOLS}
    assert 'retrieve_evidence' in {tool.name for tool in CHAT_TOOLS}
    assert 'retrieve_evidence' not in {tool.name for tool in CODING_TOOLS}


def test_work_search_adapter_passes_original_query_offset_and_limit(tmp_path, monkeypatch):
    executor = OperationExecutor(SimpleNamespace(repo_root=tmp_path), None, tmp_path)
    observed = []
    expected = {'hits': [], 'results': [], 'query': ' a.b ', 'complete': True}
    monkeypatch.setattr(integration, 'retrieve_work',
                        lambda owner, args: observed.append((owner, args)) or expected)
    step = Operation(id='search', op='search', query=' a.b ', offset=57, limit=80)
    assert executor.search(step) is expected
    assert observed[0][1]['query'] == ' a.b '
    assert observed[0][1]['offset'] == 57
    assert observed[0][1]['limit'] == 80
    assert observed[0][1]['mode'] == 'exact'


def test_exact_chat_search_works_before_binding_without_models(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'file.py').write_text('answer = 42\n', encoding='utf-8')
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(project))['thread_id']
    bot = Chatbot(store, lambda *_: pytest.fail('Unexpected work handoff'))
    monkeypatch.setattr(integration, '_models', lambda *_: pytest.fail('Exact loaded a model'))
    monkeypatch.setattr(integration, '_bind_models', lambda *_: pytest.fail('Exact bound a model'))
    result = integration.retrieve_chat(bot, conversation, 'request',
                                      {'query': 'answer', 'path': '.', 'mode': 'exact'})
    assert result['query'] == 'answer'
    assert result['hits'][0]['line'] == 1
    assert result['mode'] == 'exact' and result['complete']


def test_same_line_distinct_match_ranges_survive_deduplication():
    hits = [dict(path='file.py', line=1, version='v1', source_sha256='a' * 64,
                 start=0, end=12, match_start=start, match_end=start + 3)
            for start in (0, 4)]
    result = normalize_result('retrieve_evidence', {'hits': [hits[0], hits[1], hits[0]]})
    assert result['hits'] == [hits[1], hits[0]]


def test_from_step_refuses_changed_original(tmp_path):
    raw = b'new content\n'
    sha = hashlib.sha256(raw).hexdigest()
    executor = OperationExecutor(SimpleNamespace(repo_root=tmp_path), None, tmp_path)
    executor.inventory = lambda: {'file.py': sha}
    executor.source_bytes = lambda _: raw
    executor.read_continuation = lambda *args, **kwargs: pytest.fail('Stale source was read')
    step = Operation(id='read', op='read', from_step='find', limit=5)
    prior = {'find': {'data': {'hits': [{'path': 'file.py', 'line': 1,
                                      'version': 'old', 'source_sha256': 'b' * 64}]}}}
    with pytest.raises(ValueError, match='changed|version'):
        executor.execute(step, prior, tmp_path, 30)

@pytest.mark.parametrize('query', [' ', '  ', '\t'])
def test_exact_preserves_whitespace_queries(tmp_path, query):
    source = tmp_path / 'source.txt'
    source.write_bytes(('a' + query + 'b').encode())
    sources = integration.authorized_chat_sources(str(tmp_path), [], 'source.txt')
    result = integration._retrieve(SimpleNamespace(), sources,
        {'query': query, 'path': str(source), 'mode': 'exact'}, root=tmp_path / 'state',
        cache_root=tmp_path / 'cache', repo=tmp_path, task_id='whitespace', request_id='request',
        config={'version': 'exact-v1', 'values': __import__('dataclasses').asdict(integration.RagConfig())},
        reader_scope='authorized')
    assert result['query'] == query and result['hits']
    assert result['hits'][0]['match_start'] == 1
    assert Operation(id='find', op='search', query=query).query == query
    with pytest.raises(ValueError):
        integration.validate_arguments({'query': '', 'path': '.'})


def test_exact_searches_beyond_previous_single_source_limit(tmp_path, monkeypatch):
    source = tmp_path / 'large.txt'
    source.write_bytes(b'x' * 2_000_100 + b'UNIQUE_TAIL')
    sources = integration.authorized_chat_sources(str(tmp_path), [], 'large.txt')
    monkeypatch.setattr(integration, '_models', lambda *_: pytest.fail('Exact loaded models'))
    result = integration._retrieve(SimpleNamespace(), sources,
        {'query': 'UNIQUE_TAIL', 'path': str(source)}, root=tmp_path / 'state',
        cache_root=tmp_path / 'cache', repo=tmp_path, task_id='large', request_id='request',
        config={'version': 'exact-v1', 'values': __import__('dataclasses').asdict(integration.RagConfig())},
        reader_scope='authorized')
    assert result['complete'] and result['hits'][0]['match_start'] == 2_000_100


def test_exact_unreadable_authorized_source_is_explicitly_incomplete(tmp_path, monkeypatch):
    from pathlib import Path
    source = tmp_path / 'locked.txt'
    source.write_bytes(b'needle')
    (tmp_path / 'open.txt').write_bytes(b'other')
    original = Path.open
    def locked(path, *args, **kwargs):
        if path == source:
            raise PermissionError('sharing violation')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', locked)
    sources = integration.authorized_chat_sources(str(tmp_path), [], '.')
    result = integration._retrieve(SimpleNamespace(), sources,
        {'query': 'needle', 'path': str(tmp_path)}, root=tmp_path / 'state',
        cache_root=tmp_path / 'cache', repo=tmp_path, task_id='locked', request_id='request',
        config={'version': 'exact-v1', 'values': __import__('dataclasses').asdict(integration.RagConfig())},
        reader_scope='authorized')
    assert not result['complete'] and not result['scan_complete']
    assert result['retrieval_incomplete'] and result['reasons']
    assert result['unavailable_sources'][0]['path'] == str(source)


def test_semantic_chat_file_scope_excludes_unrelated_operation_records(tmp_path):
    import sqlite3
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'ok.txt').write_bytes(b'ordinary')
    task = tmp_path / 'state/tasks/current'
    log = task / 'plans/p1/s1/stdout.log'
    log.parent.mkdir(parents=True)
    raw = b'needle unrelated operation log'
    log.write_bytes(raw)
    with sqlite3.connect(task / 'history.sqlite3') as db:
        db.execute('CREATE TABLE artifacts(path TEXT,sha256 TEXT,kind TEXT)')
        db.execute('INSERT INTO artifacts VALUES(?,?,?)',
                   ('plans/p1/s1/stdout.log', hashlib.sha256(raw).hexdigest(), 'shell'))
    sources = integration.chat_sources(SimpleNamespace(root=tmp_path / 'state'),
        {'repo': str(project), 'task_id': 'current'}, 'ok.txt')
    assert [doc.path for doc in sources.documents()] == ['ok.txt']
    assert all(not key.startswith('record:') for key in sources.entries)


def test_semantic_rewrite_cursor_preserves_original_and_does_not_rewrite_again(tmp_path, monkeypatch):
    class Vector:
        def search(self, query, index, limit):
            chunks = sorted(index.chunks(query), key=lambda chunk: chunk.path)
            if query.text == 'answer':
                chunks = chunks[:1]
            return [(chunk.id, 1.0) for chunk in chunks[:limit]]
    class Reranker:
        def score(self, query, chunks):
            return [float(len(chunks) - index) for index, _ in enumerate(chunks)]
    monkeypatch.setattr(integration, '_models', lambda _: SimpleNamespace(
        vector=Vector(), reranker=Reranker()))
    sources = integration.Sources()
    for index in range(3):
        text = 'source content ' + str(index)
        sha = hashlib.sha256(text.encode()).hexdigest()
        sources.add('source:' + str(index), sha, len(text), lambda text=text: text,
                    kind='code', path=str(index) + '.py')
    arguments = {'query': 'answer', 'path': '.', 'mode': 'semantic', 'intent': 'diagnostic',
                 'required_entities': ['missing'], 'limit': 1}
    rewrites = []
    def retrieve(args):
        return integration._retrieve(SimpleNamespace(), sources, args,
            root=tmp_path, cache_root=tmp_path / 'tasks/task', repo=tmp_path / 'repo',
            task_id='task', request_id='same-request',
            config=integration.retrieval_configuration(),
            rewrite=lambda query, reasons: rewrites.append((query, reasons)) or 'expanded')
    first = retrieve(arguments)
    assert first['query'] == 'answer' and first['effective_query'] == 'expanded'
    assert first['next_cursor'].startswith('retrieval-v1:')
    second = retrieve({**arguments, 'cursor': first['next_cursor']})
    assert second['query'] == 'answer' and second['effective_query'] == 'expanded'
    assert second['hits'][0]['path'] != first['hits'][0]['path']
    assert len(rewrites) == 1
    with pytest.raises(ValueError, match='original query'):
        retrieve({**arguments, 'query': 'other', 'cursor': first['next_cursor']})


def test_from_step_long_utf8_line_reads_near_verified_match(tmp_path):
    raw = ('\ufeffhead\r\n' + '中🙂' * 20000 + 'NEEDLE' + '尾' * 20000 + '\r\n').encode('utf-8')
    text = raw.decode('utf-8')
    start = text.index('NEEDLE')
    sha = hashlib.sha256(raw).hexdigest()
    executor = OperationExecutor(SimpleNamespace(repo_root=tmp_path), None, tmp_path)
    executor.inventory = lambda: {'file.py': sha}
    executor.source_bytes = lambda _: raw
    hit = {'path': 'file.py', 'line': 2, 'version': sha, 'source_sha256': sha,
           'match_start': start, 'match_end': start + 6}
    result = executor.execute(Operation(id='read', op='read', from_step='find', limit=5),
        {'find': {'data': {'hits': [hit]}}}, tmp_path, 30)['data']
    page = result['reads'][0]
    assert 'NEEDLE' in page['text'] and page['visible_ranges'] == []
    assert page['byte_range'][0] > 100000
    assert len(page['text'].encode('utf-8')) <= 50 * 1024
    assert page['next_cursor'] and page['truncated']
    assert raw[slice(*page['byte_range'])].decode('utf-8') == page['text']


def test_exact_enumeration_does_not_stop_at_semantic_source_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(integration, 'SOURCE_LIMIT', 1)
    monkeypatch.setattr(integration, 'SOURCE_BYTES', 1)
    monkeypatch.setattr(integration, 'SOURCE_SECONDS', -1)
    (tmp_path / 'a.txt').write_bytes(b'ordinary')
    (tmp_path / 'z.txt').write_bytes(b'needle')
    sources = integration.authorized_chat_sources(str(tmp_path), [], '.')
    assert len(sources.entries) == 2 and sources.reasons == []


def test_exact_chat_uses_explicit_extra_file_authorization_without_work(tmp_path, monkeypatch):
    from agent.storage.read_sources import ReadSources
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    extra = tmp_path / 'approved.txt'
    extra.write_bytes(b'needle in approved source')
    denied = tmp_path / 'other.txt'
    denied.write_bytes(b'needle in unauthorized source')
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(workspace))['thread_id']
    ReadSources(store.root).set(conversation, [{'path': str(extra), 'kind': 'file'}])
    bot = Chatbot(store, lambda *_: pytest.fail('Unexpected work handoff'))
    monkeypatch.setattr(integration, '_models', lambda *_: pytest.fail('Exact loaded models'))
    found = integration.retrieve_chat(bot, conversation, 'request',
                                     {'query': 'needle', 'path': str(extra)})
    assert found['hits'][0]['path'] == str(extra) and found['complete']
    with pytest.raises(ValueError):
        integration.retrieve_chat(bot, conversation, 'request',
                                  {'query': 'needle', 'path': str(denied)})


def test_exact_memory_refusal_reports_unread_source_before_allocating(tmp_path, monkeypatch):
    from pathlib import Path
    from agent.rag.types import source_read_limit
    source = tmp_path / 'large.txt'
    size = source_read_limit() + 1
    with source.open('wb') as stream:
        stream.truncate(size)
    original = Path.open
    def forbid_read(path, *args, **kwargs):
        if path == source:
            pytest.fail('Over-budget source was opened before allocation guard')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', forbid_read)
    sources = integration.authorized_chat_sources(str(tmp_path), [], 'large.txt')
    assert sources.entries == {}
    assert sources.reasons == ['source_read_memory_budget']
    problem = sources.unavailable[0]
    assert problem['path'] == str(source) and problem['size_bytes'] == size
    assert problem['unread_from_byte'] == 0
    result = integration._retrieve(SimpleNamespace(), sources,
        {'query': 'needle', 'path': str(source)}, root=tmp_path / 'state',
        cache_root=tmp_path / 'cache', repo=tmp_path, task_id='large', request_id='request',
        config={'version': 'exact-v1', 'values': __import__('dataclasses').asdict(integration.RagConfig())},
        reader_scope='authorized')
    assert result['retrieval_incomplete'] and not result['scan_complete'] and not result['complete']
    assert result['unavailable_sources'][0] == problem


def test_exact_work_source_memory_refusal_never_reads_archived_bytes(tmp_path):
    from agent.rag.types import source_read_limit
    digest = 'a' * 64
    root = tmp_path / 'source-records'
    (root / 'blobs').mkdir(parents=True)
    with (root / 'blobs' / digest).open('wb') as stream:
        stream.truncate(source_read_limit() + 1)
    view = SimpleNamespace(root=root, baseline=None, files={'large.txt': digest},
        read_bytes=lambda *_: pytest.fail('Over-budget archived source was allocated'))
    loop = SimpleNamespace(evidence=SimpleNamespace(view=lambda: view), root=tmp_path / 'task')
    sources = integration.work_sources(loop, '.', bounded=False, include_records=False)
    assert sources.entries == {} and sources.reasons == ['source_read_memory_budget']
    assert sources.unavailable[0]['path'] == 'large.txt'
    assert sources.unavailable[0]['unread_from_byte'] == 0


@pytest.mark.parametrize('reader_scope,path,reason,removed', [
    ('work', 'sub/a.py', '', False),
    ('work', 'sub', '', False),
    ('public', 'sub/a.py', '', False),
    ('public', 'sub', '', False),
    ('authorized', '.', '', False),
    ('work', '.', 'source_scan_budget', False),
    ('public', '.', 'source_unavailable', False),
    ('work', '.', '', True),
    ('public', '.', '', True),
])
def test_only_complete_project_scope_prunes_other_vector_cache(
    tmp_path, monkeypatch, reader_scope, path, reason, removed,
):
    class Vector:
        def __init__(self):
            self.misses = []

        def search(self, query, index, limit):
            chunks = index.chunks(query)
            cached = index.vector_cache('scope-regression-vector', chunks)
            missing = [chunk for chunk in chunks if chunk.id not in cached]
            self.misses.append([chunk.path for chunk in missing])
            if missing:
                index.cache_vectors(
                    'scope-regression-vector', missing, [[1.0, 0.0] for _ in missing],
                )
            return [(chunk.id, 1.0) for chunk in chunks]

    class Ranker:
        def score(self, query, chunks):
            return [float(len(chunks) - number) for number, _ in enumerate(chunks)]

    vector = Vector()
    monkeypatch.setattr(integration, '_models', lambda _: SimpleNamespace(
        vector=vector, reranker=Ranker(),
    ))
    monkeypatch.setattr(integration, '_model_configuration', lambda: {'hash': 'scope-regression'})

    def retrieve(names, path, reason=''):
        sources = integration.Sources()
        for name in names:
            text = 'needle ' + name + '\n'
            sha = hashlib.sha256(text.encode()).hexdigest()
            sources.add('source:' + name, sha, len(text), lambda text=text: text,
                        kind='code', path=name)
        if reason:
            sources.reasons.append(reason)
        return integration._retrieve(
            SimpleNamespace(), sources, {'query': 'needle', 'path': path, 'mode': 'semantic'},
            root=tmp_path, cache_root=tmp_path / 'cache', repo=tmp_path / 'project',
            task_id='scope-regression', request_id='same-request',
            config={'version': 'scope-regression', 'values': asdict(integration.RagConfig())},
            reader_scope=reader_scope,
        )

    retrieve(['sub/a.py', 'b.py'], '.')
    assert set(vector.misses[-1]) == {'sub/a.py', 'b.py'}
    narrowed = retrieve(['sub/a.py'], path, reason)
    assert [hit['path'] for hit in narrowed['hits']] == ['sub/a.py']
    assert vector.misses[-1] == []
    retrieve(['sub/a.py', 'b.py'], '.')
    assert vector.misses[-1] == (['b.py'] if removed else [])


def test_authorized_search_prunes_private_state_and_rejects_hardlinks(tmp_path, monkeypatch):
    import os

    project = tmp_path / 'project'
    project.mkdir()
    state = project / 'private-state'
    state.mkdir()
    (state / 'secret.txt').write_text('secret evidence', encoding='utf-8')
    ordinary = project / 'ordinary.txt'
    ordinary.write_text('ordinary evidence', encoding='utf-8')
    linked = project / 'linked.txt'
    os.link(ordinary, linked)
    (project / 'allowed.txt').write_text('allowed evidence', encoding='utf-8')
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(state))
    walk = os.walk
    visited = []

    def observed_walk(*args, **kwargs):
        for row in walk(*args, **kwargs):
            visited.append(Path(row[0]))
            yield row

    from pathlib import Path
    monkeypatch.setattr(integration.os, 'walk', observed_walk)
    sources = integration.authorized_chat_sources(str(project), [], '.')
    assert state not in visited
    assert len(sources.entries) == 1
    assert next(iter(sources.entries.values()))[3].endswith('allowed.txt')
