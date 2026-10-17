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
    elif tool_name in {'retrieve_evidence', 'search_files'}:
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
