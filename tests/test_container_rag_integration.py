"""真实统一检索、读取与引用核验的容器资料链路；仅替换容器 I/O。"""

import base64
import hashlib
import json
from pathlib import PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.execution.executor import OperationExecutor
from agent.rag import container_sources, integration
from agent.task.contracts import Operation


class FakeContainerBackend:
    """只提供已有容器的固定只读协议，不运行 Docker 或模型。"""

    def __init__(self, environment):
        self.environment = environment
        environment.container_queries.append(('connect', None))

    def binding(self):
        return {
            'container_id': self.environment.container_identity,
            'image_id': 'sha256:' + 'b' * 64,
            'repo_digests': ['fixture@sha256:' + 'c' * 64],
            'mounts': [],
            'toolchain_versions': {'status': 'recorded', 'source': 'bound_runtime_facts',
                                   'values': {'gcc': '13.2.0'}},
        }

    def call(self, operation, payload):
        self.environment.container_queries.append((operation, dict(payload)))
        path, root = payload['path'], payload['allowed_root'].rstrip('/')
        assert path == root or path.startswith(root + '/')
        assert all(path != denied and not path.startswith(denied + '/')
                   for denied in payload['denied_roots'])
        files = self.environment.container_files
        directories = {str(parent) for name in files for parent in PurePosixPath(name).parents}
        if path not in files and path not in directories:
            raise ValueError('Fixture source not found: ' + path)
        kind = 'file' if path in files else 'directory'
        if operation == 'stat':
            return {'path': path, 'realpath': path, 'type': kind, 'mtime_ns': 1,
                    'size': len(files[path]) if kind == 'file' else 0}
        if operation == 'list':
            assert kind == 'directory'
            children = sorted(name for name in files.keys() | directories
                              if str(PurePosixPath(name).parent) == path and name != path)
            selected = [name for name in children if PurePosixPath(name).name > payload['after']]
            limited = selected[:payload['limit']]
            return {
                'entries': [{'path': name, 'realpath': name,
                             'type': 'file' if name in files else 'directory'} for name in limited],
                'complete': len(limited) == len(selected),
                'next_after': PurePosixPath(limited[-1]).name if limited else '',
            }
        assert operation == 'read' and kind == 'file'
        raw = files[path]
        assert len(raw) <= payload['max_bytes']
        return {'path': path, 'realpath': path, 'sha256': hashlib.sha256(raw).hexdigest(),
                'size': len(raw), 'data_base64': base64.b64encode(raw).decode()}


@pytest.fixture
def container_work(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    original = b'project-only NEEDLE\n'
    (repo / 'main.c').write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    source_root = tmp_path / 'source-records'
    (source_root / 'blobs').mkdir(parents=True)
    (source_root / 'blobs' / digest).write_bytes(original)
    view = SimpleNamespace(root=source_root, baseline=None, files={'main.c': digest},
                           read_bytes=lambda name: original)
    environment = SimpleNamespace(
        repo_root=repo, container_files={}, container_queries=[], container_identity='a' * 64,
    )
    root = tmp_path / 'tasks' / 'test-container'
    root.mkdir(parents=True)
    loop = SimpleNamespace(
        root=root, environment=environment, evidence=SimpleNamespace(view=lambda: view),
        runtime_settings={}, state={'status': 'running'}, history_store=None, persist=Mock(),
        request_model=Mock(side_effect=AssertionError('Unexpected paid query rewrite')),
    )
    loop.executor = OperationExecutor(environment, loop.evidence, root, versioned_reads=True)
    loop.executor.retrieval_owner = loop
    monkeypatch.setattr(container_sources, 'ContainerBackend', FakeContainerBackend)
    monkeypatch.setattr(integration, '_models',
                        Mock(side_effect=AssertionError('Exact search loaded a model')))
    monkeypatch.delenv('MONOPLANER_RAG_MODELS', raising=False)
    return loop


def search(loop, query, path='$container/system_include/usr/include', **options):
    return integration.retrieve_work(loop, {'query': query, 'path': path, **options})


def test_current_project_scope_does_not_open_or_scan_container(container_work):
    result = search(container_work, 'NEEDLE', '.')
    assert result['complete'] and result['mode'] == 'exact'
    assert result['hits'][0]['path'] == 'main.c'
    assert container_work.environment.container_queries == []
    container_work.request_model.assert_not_called()


@pytest.mark.parametrize('query', ['Foo.Bar', 'foo_bar', '中文', ' x ', 'OR', 'AND', 'a+b'])
def test_container_literal_query_and_original_coordinates_are_unchanged(container_work, query):
    text = 'prefix\r\n' + query + '\r\nsuffix\r\n'
    container_work.environment.container_files['/usr/include/source.h'] = text.encode()
    result = search(container_work, query)
    assert result['query'] == result['effective_query'] == query
    assert result['mode'] == 'exact' and result['complete'] and result['scan_complete']
    hit = result['hits'][0]
    assert text[hit['match_start']:hit['match_end']] == query
    assert hit['actual_path'] == '/usr/include/source.h'
    assert hit['origin'] == 'container/toolchain'
    assert result['rewrite']['status'] == 'not_triggered'
    container_work.request_model.assert_not_called()


def test_container_exact_case_mismatch_has_no_implicit_expansion(container_work):
    container_work.environment.container_files['/usr/include/source.h'] = b'Foo.Bar\n'
    result = search(container_work, 'foo.bar')
    assert result['complete'] and not result['hits']
    assert result['query'] == result['effective_query'] == 'foo.bar'
    container_work.request_model.assert_not_called()


def test_container_search_reads_from_step_and_verifies_delivered_citation(container_work):
    from agent.model.tool_results import model_result

    raw = '\ufeffprefix\r\n函数 NEEDLE\r\n尾行\r\n'.encode()
    container_work.environment.container_files['/usr/include/source.h'] = raw
    result = search(container_work, 'NEEDLE')
    assert result['results'] and result['hits']
    read = container_work.executor.execute(
        Operation(id='read', op='read', from_step='search', limit=3),
        {'search': {'data': result}}, container_work.root, 30,
    )
    assert read['status'] == 'succeeded'
    page = read['data']['reads'][0]
    assert page['text'] == '1: prefix\n2: 函数 NEEDLE\n3: 尾行'
    assert page['version'] == result['hits'][0]['version']
    chunk = result['results'][0]
    delivered = model_result('submit_plan', {'steps': [{'op': 'search', 'data': result}]})
    messages = [{'type': 'tool', 'data': {
        'name': 'submit_plan', 'tool_call_id': 'find',
        'content': json.dumps(delivered, ensure_ascii=False),
    }}]
    response = {'data': {'content': '[evidence:' + chunk['id'] + ']'}}
    folder = container_work.root / 'checks'
    integration.audit_work_citations(container_work, response, messages, folder)
    report = json.loads((folder / 'citation-checks.json').read_bytes())
    assert report['checks'][0]['status'] == 'source_verified'
    originals = container_work.root / 'rag/container/originals'
    assert (originals / hashlib.sha256(raw).hexdigest()).read_bytes() == raw
    container_work.environment.container_files['/usr/include/source.h'] = b'changed NEEDLE\n'
    with pytest.raises(ValueError, match='version_changed'):
        container_work.executor.execute(
            Operation(id='read-again', op='read', from_step='search', limit=3),
            {'search': {'data': result}}, container_work.root, 30,
        )
    integration.audit_work_citations(container_work, response, messages, folder)
    changed = json.loads((folder / 'citation-checks.json').read_bytes())
    assert changed['checks'][0]['status'] == 'unverified'
    assert 'version_changed' in changed['checks'][0]['reason']


def test_container_exact_pagination_advances_without_duplicates(container_work):
    container_work.environment.container_files['/usr/include/source.h'] = b'NEEDLE\n' * 7
    result = search(container_work, 'NEEDLE', limit=2)
    positions = []
    for _ in range(6):
        positions.extend(hit['match_start'] for hit in result['hits'])
        if not result['next_cursor']:
            break
        result = search(container_work, 'NEEDLE', limit=2, cursor=result['next_cursor'])
    assert positions == list(range(0, 7 * 7, 7))
    assert result['complete'] and not result['next_cursor']
    container_work.request_model.assert_not_called()


def test_container_hit_cursor_cannot_be_used_for_another_scope(container_work):
    container_work.environment.container_files.update({
        '/usr/include/one.h': b'NEEDLE\nNEEDLE\n',
        '/usr/include/two.h': b'NEEDLE\nNEEDLE\n',
    })
    first = search(container_work, 'NEEDLE',
                   '$container/system_include/usr/include/one.h', limit=1)
    assert first['next_cursor']
    with pytest.raises(ValueError, match='[Cc]ursor|[Ss]cope|stale'):
        search(container_work, 'NEEDLE', '$container/system_include/usr/include/two.h',
               limit=1, cursor=first['next_cursor'])


def test_container_enumeration_cursor_does_not_report_early_zero_or_cross_scope(
    container_work, monkeypatch,
):
    container_work.environment.container_files.update({
        '/usr/include/a.h': b'ordinary\n', '/usr/include/z.h': b'NEEDLE\n',
        '/usr/local/include/z.h': b'NEEDLE\n',
    })
    monkeypatch.setattr(container_sources, 'SCAN_BATCH', 1)
    first = search(container_work, 'NEEDLE')
    assert not first['complete'] and not first['scan_complete']
    assert not first['hits'] and first['next_cursor'].startswith('container-scan-v1:')
    mismatch = search(container_work, 'NEEDLE', '$container/system_include/usr/local/include',
                      cursor=first['next_cursor'])
    assert not mismatch['complete'] and not mismatch['hits']
    assert 'container_scan_cursor_stale' in mismatch['reasons']
    result, cursors = first, {first['next_cursor']}
    for _ in range(10):
        result = search(container_work, 'NEEDLE', cursor=result['next_cursor'])
        if result['complete']:
            break
        assert not result['hits'] and not result['scan_complete']
        assert result['next_cursor'] not in cursors
        cursors.add(result['next_cursor'])
    assert result['complete'] and result['hits'][0]['actual_path'] == '/usr/include/z.h'
    container_work.request_model.assert_not_called()


def test_container_semantic_unavailable_does_not_fall_back_to_literal_or_rewrite(
    container_work, monkeypatch,
):
    container_work.environment.container_files['/usr/include/source.h'] = b'NEEDLE exact\n'
    models = Mock(return_value=SimpleNamespace(
        vector=None, reranker=None, unavailable_reason='local_models_unavailable:fixture',
    ))
    monkeypatch.setattr(integration, '_models', models)
    result = search(container_work, 'NEEDLE', mode='semantic',
                    intent='diagnostic', required_entities=['NEEDLE'])
    assert result['mode'] == 'semantic' and result['retrieval_incomplete']
    assert result['hits'] == result['results'] == []
    assert 'vector_unavailable' in result['reasons']
    assert result['rewrite']['status'] == 'not_triggered'
    assert models.call_count == 1
    container_work.request_model.assert_not_called()
