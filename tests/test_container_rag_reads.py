"""容器原文读取接入既有执行入口，不污染项目源码或扩大共享展示预算。"""

import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.execution.executor import OperationExecutor
from agent.source import lifecycle
from agent.source.paging import OUTPUT_BYTES, ReadBudget, read_pages, visible_reads
from agent.task.contracts import Operation


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    """隔离容器传输，只测试执行器与只读适配器的接口契约。"""
    module = ModuleType('agent.rag.container_sources')
    module.is_container_path = lambda path: path.startswith('$container/')
    module.read_container = Mock()
    module.container_original = Mock()
    module.container_read_cursor = Mock(return_value='binding-verified-next-cursor')
    monkeypatch.setitem(sys.modules, module.__name__, module)
    executor = OperationExecutor(SimpleNamespace(repo_root=tmp_path), None, tmp_path)
    executor.inventory = Mock(side_effect=AssertionError('Container lookup used project inventory'))
    executor.source_bytes = Mock(side_effect=AssertionError('Container lookup used project files'))
    executor.path = Mock(side_effect=AssertionError('Container lookup used host path'))
    owner = SimpleNamespace(executor=executor)
    executor.retrieval_owner = owner
    return executor, owner, module


def source_hit(raw, path='$container/system_include/usr/include/stdlib.h', line=1):
    return {'path': path, 'line': line, 'version': 'binding-file-version',
            'source_sha256': hashlib.sha256(raw).hexdigest()}


def source_meta(hit):
    return {'path': hit['path'], 'version': hit['version'],
            'sha256': hit['source_sha256'], 'source_type': 'container',
            'image_id': 'sha256:image', 'container_id': 'owned-container',
            'original_record': '$record/container-source/immutable-original'}


def test_direct_read_uses_owner_and_shared_budget_without_project_inventory(bridge):
    executor, owner, module = bridge
    expected = {'path': '$container/system_include/usr/include/stdlib.h',
                'sha256': 'a' * 64, 'text': '1: original', 'visible_ranges': [[0, 1]]}
    module.read_container.return_value = expected
    budget = ReadBudget()
    result = executor.read_continuation(expected['path'], 0, 10, budget=budget)
    assert result is expected
    module.read_container.assert_called_once_with(
        owner, expected['path'], 0, 10, budget=budget, cursor=None, repeat=False,
    )
    assert executor.visible_reads == {}


def test_low_level_read_routes_container_path_and_preserves_cursor(bridge):
    executor, owner, module = bridge
    budget = ReadBudget()
    path = '$container/system_include/usr/include/stdlib.h'
    executor.read(path, 2, 3, budget=budget, cursor='verified-container-cursor')
    module.read_container.assert_called_once_with(
        owner, path, 2, 3, budget=budget, cursor='verified-container-cursor',
    )


def test_explicit_reread_and_partial_byte_cursor_do_not_mark_lines_read(bridge):
    executor, owner, module = bridge
    path = '$container/toolchain_src/build/gcc/src/large.cc'
    partial = {'path': path, 'sha256': 'b' * 64, 'text': 'partial',
               'visible_ranges': [], 'next_cursor': 'next', 'truncated': True}
    module.read_container.return_value = partial
    result = executor.read_continuation(path, 0, 10, True, cursor='previous')
    assert result == partial and executor.visible_reads == {}
    assert module.read_container.call_args.args[:4] == (owner, path, 0, 10)
    assert module.read_container.call_args.kwargs['repeat'] is True
    assert module.read_container.call_args.kwargs['cursor'] == 'previous'


def test_from_step_verifies_binding_and_hash_on_both_original_and_read(bridge):
    executor, owner, module = bridge
    raw = '头文件\r\n原文保持\r\n'.encode()
    hit = source_hit(raw, line=2)
    metadata = source_meta(hit)
    module.container_original.return_value = (raw, metadata)
    module.read_container.side_effect = lambda *args, **kw: {
        **metadata, **kw['budget'].source(raw, args[2], args[3]),
    }
    step = Operation(id='read', op='read', from_step='find', limit=2)
    result = executor.execute(step, {'find': {'data': {'hits': [hit]}}}, executor.root, 30)
    assert result['status'] == 'succeeded'
    assert result['data']['reads'][0]['text'] == '1: 头文件\n2: 原文保持'
    module.container_original.assert_called_once_with(
        owner, hit['path'], expected_version=hit['version'], expected_hash=hit['source_sha256'],
    )
    call = module.read_container.call_args
    assert call.kwargs['expected_version'] == hit['version']
    assert call.kwargs['expected_hash'] == hit['source_sha256']
    assert call.kwargs['repeat'] is False


@pytest.mark.parametrize('message', ['Source binding changed', 'Original source hash changed'])
def test_from_step_stale_source_stops_before_reading(bridge, message):
    executor, _, module = bridge
    hit = source_hit(b'old content')
    module.container_original.side_effect = ValueError(message)
    budget = ReadBudget()
    with pytest.raises(ValueError, match=message):
        executor.read_hit(hit, Operation(id='read', op='read'), budget)
    module.read_container.assert_not_called()
    assert budget.remaining == OUTPUT_BYTES


def test_from_step_rechecks_original_bytes_even_if_adapter_metadata_is_wrong(bridge):
    executor, _, module = bridge
    hit = source_hit(b'old content')
    module.container_original.return_value = (b'changed content', source_meta(hit))
    with pytest.raises(ValueError, match='source changed'):
        executor.read_hit(hit, Operation(id='read', op='read'), ReadBudget())
    module.read_container.assert_not_called()


def test_long_utf8_hit_reads_match_window_without_claiming_complete_lines(bridge):
    executor, _, module = bridge
    raw = ('\ufeffhead\r\n' + '中🙂' * 20000 + 'NEEDLE' + '尾' * 20000 + '\r\n').encode()
    text = raw.decode('utf-8')
    hit = source_hit(raw, line=2)
    hit.update(match_start=text.index('NEEDLE'), match_end=text.index('NEEDLE') + 6)
    metadata = {**source_meta(hit), 'binding_key': 'current-environment'}
    module.container_original.return_value = raw, metadata
    budget = ReadBudget()
    page = executor.read_hit(hit, Operation(id='read', op='read', limit=5), budget)
    assert 'NEEDLE' in page['text'] and page['visible_ranges'] == []
    assert page['byte_range'][0] > 100000
    assert page['text'] == raw[slice(*page['byte_range'])].decode()
    assert len(page['text'].encode()) + budget.remaining == OUTPUT_BYTES
    assert page['next_cursor'] == 'binding-verified-next-cursor'
    assert page['version'] == hit['version'] and page['binding_key'] == 'current-environment'
    assert page['original_record'] == metadata['original_record']
    module.container_read_cursor.assert_called_once()
    assert module.container_read_cursor.call_args.args[0] == metadata
    assert module.container_read_cursor.call_args.args[1].startswith('source-bytes-v1:')
    module.read_container.assert_not_called()
    assert executor.visible_reads == {}


def test_container_lookup_cancellation_is_not_recast_as_success(bridge):
    executor, _, module = bridge
    module.read_container.side_effect = InterruptedError('Task paused')
    with pytest.raises(InterruptedError, match='Task paused'):
        executor.execute(Operation(id='read', op='read',
                         path='$container/system_include/usr/include/stdlib.h'),
                         {}, executor.root, 30)


def test_from_step_many_sources_share_the_existing_read_budget(bridge):
    executor, _, module = bridge
    raw = (('line' * 80 + '\n') * 100).encode()
    hits = [source_hit(raw, f'$container/system_include/usr/include/header{n}.h')
            for n in range(3)]
    module.container_original.side_effect = lambda _, path, **kw: (
        raw, source_meta(next(hit for hit in hits if hit['path'] == path)),
    )
    budgets = []

    def read(_, path, offset, limit, *, budget, **kwargs):
        budgets.append(budget)
        return {'path': path, **budget.source(raw, offset, limit)}

    module.read_container.side_effect = read
    data = executor.execute(Operation(id='read', op='read', from_step='find', limit=200),
                            {'find': {'data': {'hits': hits}}}, executor.root, 30)['data']
    assert len({id(budget) for budget in budgets}) == 1
    assert sum(len(page['text'].encode()) for page in read_pages(data)) <= OUTPUT_BYTES
    assert data['output_budget']['used_bytes'] <= OUTPUT_BYTES
    assert data['truncated']


def test_container_only_results_skip_project_map_but_keep_original_reference(bridge, tmp_path):
    executor, owner, module = bridge
    raw = b'header text\n'
    hit = source_hit(raw)
    metadata = source_meta(hit)
    module.read_container.return_value = {**metadata, 'text': '1: header text',
                                          'visible_ranges': [[0, 1]], 'truncated': False}
    row = executor.execute(Operation(id='read', op='read', path=hit['path']), {}, tmp_path, 30)
    executor.exploration = SimpleNamespace(observe=Mock(
        side_effect=AssertionError('Container result entered project exploration map')))
    owner.root, owner.state, owner.history_store = tmp_path, {'source_map': {}}, None
    lifecycle.record_lookup(owner, SimpleNamespace(op='read'), row, tmp_path)
    assert 'repository_map' not in row['data']
    assert row['data']['original_record'] == metadata['original_record']
    assert not (tmp_path / 'repository-map.json').exists()


def test_container_search_results_skip_project_map(bridge, tmp_path):
    executor, owner, _ = bridge
    hit = source_hit(b'header text')
    executor.exploration = SimpleNamespace(observe=Mock(
        side_effect=AssertionError('Container hit entered project exploration map')))
    owner.root, owner.state, owner.history_store = tmp_path, {'source_map': {}}, None
    row = {'status': 'succeeded', 'data': {'hits': [hit]}}
    lifecycle.record_lookup(owner, SimpleNamespace(op='search'), row, tmp_path)
    assert 'repository_map' not in row['data']


def test_empty_container_scope_does_not_claim_a_project_source_version(bridge, tmp_path):
    executor, owner, _ = bridge
    executor.exploration = SimpleNamespace(observe=Mock(
        side_effect=AssertionError('Empty container scope entered project map')))
    owner.root, owner.state, owner.history_store = tmp_path, {'source_map': {}}, None
    row = {'status': 'succeeded', 'data': {'hits': [], 'scan_complete': False}}
    lifecycle.record_lookup(owner, SimpleNamespace(
        op='search', path='$container/system_include/usr/include'), row, tmp_path)
    assert 'repository_map' not in row['data']


def test_mixed_source_result_keeps_existing_project_projection(bridge, tmp_path):
    executor, owner, _ = bridge
    hits = [source_hit(b'header text'), {'path': 'project.c', 'line': 2}]
    projected = {'source_version': 'project-version', 'updates': [{'path': 'project.c'}]}
    executor.exploration = SimpleNamespace(observe=Mock(return_value=projected), coverage=lambda: {})
    owner.root, owner.state, owner.history_store = tmp_path, {'source_map': {}}, None
    row = {'status': 'succeeded', 'data': {'hits': hits}}
    lifecycle.record_lookup(owner, SimpleNamespace(op='search'), row, tmp_path)
    executor.exploration.observe.assert_called_once_with(
        row['data'], queued=[], record='$record/repository-map.json')
    assert row['data']['repository_map']['updates'] == [{'path': 'project.c'}]
    assert (tmp_path / 'repository-map.json').exists()


def test_visible_reads_default_identity_remains_content_hash():
    page = {'path': 'source.py', 'sha256': 'content-sha', 'version': 'unrelated-version',
            'text': '1: source', 'visible_ranges': [[0, 1]]}
    rows = [{'op': 'read', 'status': 'succeeded', 'data': page}]
    assert visible_reads([], rows) == {('source.py', 'content-sha'): [[0, 1]]}


def test_resume_restores_only_complete_ranges_for_exact_container_version(
    bridge, tmp_path, monkeypatch,
):
    executor, owner, _ = bridge
    raw = b'header text\n'
    hit = source_hit(raw)
    base = {**source_meta(hit), 'text': '1: header text', 'visible_ranges': [[0, 1]]}
    pages = [
        {**base, 'version': 'old-container-binding'},
        {**base, 'version': 'new-container-binding', 'visible_ranges': [[2, 3]]},
        {**base, 'version': 'new-container-binding', 'visible_ranges': [],
         'text': 'partial line', 'byte_range': [12, 16]},
        {key: value for key, value in base.items() if key != 'version'},
    ]
    messages = [{'type': 'tool', 'data': {'content': json.dumps({
        'steps': [{'op': 'read', 'status': 'succeeded', 'data': page}],
    })}} for page in pages]
    # 助手声明不是实际工具结果，不能刷新来源可见区间。
    messages.append({'type': 'ai', 'data': {'content': json.dumps({'steps': [
        {'op': 'read', 'status': 'succeeded', 'data': {**base, 'visible_ranges': [[0, 99]]}},
    ]})}})
    pending = [{'op': 'read', 'status': 'succeeded',
                'data': {**base, 'version': 'new-container-binding',
                         'visible_ranges': [[3, 4]]}}]
    monkeypatch.setattr('agent.model.compaction.history', lambda *_: messages)
    executor.exploration = SimpleNamespace(sync=Mock())
    owner.environment = executor.environment
    owner.root, owner.state = tmp_path, {'pending': {'rows': pending}}
    lifecycle.configure_lookups(owner)
    assert executor.visible_reads == {
        (hit['path'], 'old-container-binding'): [[0, 1]],
        (hit['path'], 'new-container-binding'): [[2, 4]],
    }
    assert (hit['path'], hit['source_sha256']) not in executor.visible_reads
    executor.exploration.sync.assert_called_once_with()


def test_real_container_cursor_continues_long_match_and_refuses_rebuilt_binding(
    tmp_path, monkeypatch,
):
    from agent.rag import container_sources

    raw = ('\ufefffirst\r\n' + '中🙂' * 20000 + 'NEEDLE' + '末' * 40000 + '\r\n').encode()
    text = raw.decode('utf-8')
    hit = source_hit(raw, line=2)
    hit.update(match_start=text.index('NEEDLE'), match_end=text.index('NEEDLE') + 6)
    metadata = {**source_meta(hit), 'binding_key': 'original-container-binding'}
    executor = OperationExecutor(SimpleNamespace(repo_root=tmp_path), None, tmp_path)
    executor.inventory = Mock(side_effect=AssertionError('Container lookup used project inventory'))
    owner = SimpleNamespace(executor=executor)
    executor.retrieval_owner = owner

    def original(_, path, *, expected_version=None, expected_hash=None):
        assert path == hit['path']
        if expected_version and expected_version != metadata['version']:
            raise ValueError('Container source binding version changed')
        if expected_hash and expected_hash != metadata['sha256']:
            raise ValueError('Container source content hash changed')
        return raw, dict(metadata)

    monkeypatch.setattr(container_sources, 'container_original', original)
    first = executor.read_hit(hit, Operation(id='read', op='read', limit=5), ReadBudget())
    assert first['next_cursor'].startswith('container-read-v1:')
    second = executor.read_continuation(hit['path'], 1, 5, cursor=first['next_cursor'])
    assert second['byte_range'][0] == first['byte_range'][1]
    assert raw[slice(*second['byte_range'])].decode() == second['text']
    assert second['visible_ranges'] == []
    assert all(not spans for spans in executor.visible_reads.values())
    metadata['version'] = 'new-binding-same-content'
    metadata['binding_key'] = 'rebuilt-container'
    with pytest.raises(ValueError, match='binding version changed'):
        executor.read_continuation(hit['path'], 1, 5, cursor=first['next_cursor'])


def test_actual_container_read_uses_resumed_binding_ranges_without_suppressing_new_version(
    tmp_path, monkeypatch,
):
    from agent.rag import container_sources

    raw = b'old line\nnew line\n'
    hit = source_hit(raw)
    metadata = {**source_meta(hit), 'binding_key': 'current'}
    executor = OperationExecutor(SimpleNamespace(repo_root=tmp_path), None, tmp_path)
    executor.visible_reads = {(hit['path'], 'old-binding-version'): [[0, 2]],
                              (hit['path'], hit['version']): [[0, 1]]}
    monkeypatch.setattr(container_sources, 'container_original',
                        lambda *args, **kwargs: (raw, dict(metadata)))
    result = executor.read_continuation(hit['path'], 0, 2)
    assert result['text'] == '2: new line'
    assert result['visible_ranges'] == [[1, 2]]
    metadata['version'] = 'rebuilt-version'
    rebuilt = executor.read_continuation(hit['path'], 0, 2)
    assert rebuilt['visible_ranges'] == [[0, 2]]
