"""仅整理尚未发送的白名单结果集合；同键保留最后一项及其位置。"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import PurePosixPath


def _name(value):
    return value if isinstance(value, str) and value.strip() else None


def _path(value):
    if _name(value) is None or any(ord(char) < 32 for char in value):
        return None
    path = PurePosixPath(value.replace('\\', '/'))
    if '..' in path.parts or path.as_posix() == '.':
        return None
    return path.as_posix()


def _line(value):
    return value if type(value) is int and value > 0 else None


def _typed(value):
    """没有合法主键时才规范化单条数据，保留JSON数值和容器的类型差异。"""
    kind = type(value)
    if value is None:
        return ['null']
    if kind in (str, int, bool):
        return [kind.__name__, value]
    if kind is float and math.isfinite(value):
        return ['float', value]
    if kind in (list, tuple):
        return [kind.__name__, [_typed(child) for child in value]]
    if kind is dict and all(isinstance(key, str) for key in value):
        return ['object', [[key, _typed(value[key])] for key in sorted(value)]]
    raise TypeError('Not a canonical JSON value')


def _fallback(value):
    raw = json.dumps(_typed(value), ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _latest(items, identity):
    """[A旧,B,A新]变为[B,A新]，不为去重重新排列其余条目。"""
    kept, seen = [], set()
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        key = identity(item) if isinstance(item, dict) else None
        if key is not None:
            key = ('key', key)
        else:
            try:
                key = ('hash', _fallback(item))
            except (TypeError, ValueError, RecursionError, UnicodeError):
                # 非JSON对象不猜测身份，不通过repr等有损方式误合并。
                key = ('unidentified', index)
        if key not in seen:
            seen.add(key)
            kept.append(item)
    return list(reversed(kept))


def _collection(value, field, identity):
    if isinstance(value.get(field), list):
        value[field] = _latest(value[field], identity)


def _version(item, default):
    return (_name(item.get('version')) or _name(item.get('source_version'))
            or _name(default))


def _hits(value, source_version, project):
    def identity(item):
        path, line = _path(item.get('path')), _line(item.get('line'))
        if not path or not line:
            return None
        ranges = tuple(item.get(key) for key in ('start', 'end', 'match_start', 'match_end'))
        if any(value is not None and type(value) is not int for value in ranges):
            return None
        return (project, _version(item, source_version), path, line,
                _name(item.get('source_sha256')), *ranges)

    _collection(value, 'hits', identity)


def _repository_map(value, source_version, project):
    mapping = value.get('repository_map')
    if not isinstance(mapping, dict) or not isinstance(mapping.get('updates'), list):
        return
    version = _version(mapping, source_version)
    for update in mapping['updates']:
        if not isinstance(update, dict):
            continue
        scope = (project, _version(update, version), _path(update.get('path')))

        def symbol(item):
            kind, name, line = (_name(item.get('kind')), _name(item.get('name')),
                                _line(item.get('line')))
            return (*scope, kind, name, line) if kind and name and line else None

        def dependency(item):
            path, line = _path(item.get('path')), _line(item.get('line'))
            # 同一行可从同一模块导入多个符号；完整依据才标识具体依赖关系。
            evidence = _name(item.get('evidence'))
            return (*scope, path, line, evidence) if path and line and evidence else None

        _collection(update, 'new_symbols', symbol)
        _collection(update, 'new_dependencies', dependency)


def normalize_result(tool_name, value, *, source_version=None, project=None):
    """返回深拷贝；不读取历史、不访问文件、不改变调用与验收封套。"""
    result = deepcopy(value)
    if not isinstance(result, dict):
        return result
    project = _path(str(project)) if project is not None else None
    if tool_name == 'web_search':
        _collection(result, 'sources', lambda item: _name(item.get('url')))
    elif tool_name == 'list_files':
        _collection(result, 'entries', lambda item: _path(item.get('path')))
    elif tool_name == 'retrieve_evidence':
        _hits(result, source_version, project)
        _repository_map(result, source_version, project)
    elif tool_name == 'submit_plan':
        for row in result.get('steps', []) if isinstance(result.get('steps'), list) else []:
            if not isinstance(row, dict) or not isinstance(row.get('data'), dict):
                continue
            version = _name(row.get('before')) or source_version
            if row.get('op') == 'search':
                _hits(row['data'], version, project)
            if row.get('op') in {'read', 'search'}:
                _repository_map(row['data'], version, project)
    return result


def _search_view(value):
    """原件保留完整检索身份；模型只接收共享来源表、原文和命中位置。"""
    if value.get('format') not in {'monoplanner-retrieval-excerpts-v2', 'monoplanner-retrieval-excerpts-v3', 'monoplanner-retrieval-excerpts-v4', 'monoplanner-retrieval-excerpts-v5'}:
        return value
    sources, results, source_ids, result_ids = [], [], {}, {}
    for item in value['results']:
        identity = tuple(item[key] for key in ('source_id', 'version', 'source_sha256', 'path'))
        if identity not in source_ids:
            source = {'path': item['path'], 'sha256': item['source_sha256'], 'kind': item['kind']}
            if item['version'] != item['source_sha256']:
                source['version'] = item['version']
            kind, _, path = item['source_id'].partition(':')
            if item['path'] != ('$record/' + path if kind == 'record' else path):
                source['source_id'] = item['source_id']
            elif kind != 'source':
                source['source_type'] = kind
            for field in ('origin', 'namespace', 'actual_path', 'temporary',
                          'from_build_tree', 'timestamp'):
                if item.get(field):
                    source[field] = item[field]
            source_ids[identity] = len(sources)
            sources.append(source)
        result_ids[item['id']] = len(results)
        results.append({
            'id': item['id'], 'source': source_ids[identity],
            **{key: item[key] for key in ('start', 'end', 'line_start', 'line_end', 'text')},
            **({'symbol': item['symbol']} if item.get('symbol') else {}),
        })
    hits = [{
        'result': result_ids[hit['result_id']],
        **{key: hit[key] for key in ('line', 'line_end', 'match_start', 'match_end',
                                     'column_start', 'column_end', 'truncated')},
    } for hit in value['hits']]
    # 这些字段仍在操作原件中；它们不参与模型的阅读、定位或续查。
    internal = {'hits', 'results', 'configuration_version', 'scope', 'reader_scope',
                'retrieval_mode', 'token_count', 'ranking', 'search_metrics', 'exact_current_version_hit',
                'retrieval_incomplete', 'requested_limit', 'requested_offset', 'notice'}
    body = {key: item for key, item in value.items() if key not in internal}
    if body.get('effective_query') == body.get('query'):
        body.pop('effective_query', None)
    if body.get('rewrite', {}).get('status') == 'not_triggered':
        body.pop('rewrite')
    for field in ('reasons', 'missing_entities', 'missing_kinds', 'continuation',
                  'unavailable_sources', 'unavailable_source_count'):
        if not body.get(field):
            body.pop(field, None)
    body.update(
        format='monoplanner-search-view-v1', sources=sources, results=results, hits=hits,
        coordinates='Zero-based source/result indices and Unicode [start,end) ranges; '
                    'one-based lines and columns. Source version defaults to sha256.',
        notice='Excerpts only; read the source path and line range for surrounding context.',
    )
    return body


def model_result(tool_name, value, *, source_version=None, project=None):
    """只在工具消息提交处投影；from_step 和本地审计继续使用完整去重结果。"""
    result = normalize_result(tool_name, value, source_version=source_version, project=project)
    if not isinstance(result, dict):
        return result
    if tool_name == 'retrieve_evidence':
        return _search_view(result)
    if tool_name == 'submit_plan':
        for row in result.get('steps', []):
            if isinstance(row, dict) and row.get('op') == 'search' and isinstance(row.get('data'), dict):
                row['data'] = _search_view(row['data'])
    return result
