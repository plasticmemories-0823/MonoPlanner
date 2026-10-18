"""工具正文按需卸载；只对新增消息生成展示值，不改写执行原件和旧历史。"""

from collections import deque
from copy import deepcopy
import codecs
import hashlib
import json
import os
from pathlib import Path
import re

from agent.model.errors import ModelRequestNotSent
from agent.source.snapshots import _safe_file


# 独立适配 DeepAgents 的文本卸载和内容寻址模式，没有引入额外运行框架。
# https://github.com/langchain-ai/deepagents/tree/c5313d7005534c8f8514004890c6f6d5174c6e2f
FORMAT = 'monoplanner-tool-payload-v1'
THRESHOLD = 8 * 1024
MIN_DUPLICATE = 1024
TEXT_FIELDS = {'stdout', 'stderr', 'error', 'details', 'traceback', 'text'}
JSON_FIELDS = {'hits', 'repository_map'}
_CAS = re.compile(r'\$record/tool-payloads/([0-9a-f]{64})\.txt')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _reference(value):
    return isinstance(value, dict) and value.get('format') == FORMAT


def _page(value):
    return isinstance(value, dict) and all(
        key in value for key in ('path', 'sha256', 'text', 'visible_ranges')
    )


def _text(value):
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2,
    )


def _entry(archive, relative):
    if archive is None:
        return None
    rows = archive.find('', limit=1, prefix=relative)
    return rows[0] if rows and rows[0]['path'] == relative else None


def readable(name, archive=None):
    """仅新增内容寻址正文和已有受管日志的流式读取入口。"""
    if _CAS.fullmatch(name):
        return True
    if not name.startswith('$record/'):
        return False
    relative = Path(name[8:])
    return (bool(relative.parts) and not relative.is_absolute()
            and '..' not in relative.parts and ':' not in name
            and relative.parts[0] in {'plans', 'feedback', 'acceptance'}
            and relative.name in {'stdout.log', 'stderr.log'}
            and _entry(archive, relative.as_posix()) is not None)


def _location(root, name, archive):
    match = _CAS.fullmatch(name)
    if match:
        expected = match[1]
    elif readable(name, archive):
        expected = _entry(archive, Path(name[8:]).as_posix())['sha256']
    else:
        raise ValueError('Payload reads require an exact content hash or a registered task log.')
    return _safe_file(Path(root).resolve(), Path(name[8:])), expected


def read(root, name, offset, limit, *, budget=None, archive=None, preview=False):
    """64KiB分块校验；内存只保留读取窗口和最多十行短预览。"""
    from agent.source.paging import ReadBudget, OUTPUT_BYTES, ranges
    from agent.storage.resources import owner_cancelled

    ranges(offset, limit)
    budget = budget if budget is not None else ReadBudget()
    path, expected = _location(root, name, archive)
    original_log = not bool(_CAS.fullmatch(name))
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder('utf-8')('surrogateescape' if original_log else 'strict')
    remaining, rows, end = budget.remaining, [], offset
    line, size, saved, total_bytes = 0, 0, 0, 0
    byte_position, line_start = 0, 0
    parts, head, tail = [], [], deque(maxlen=5)
    skip_lf, closed, long_line = False, False, None
    breaks = ('\n', '\r', '\v', '\f', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029')

    def finish():
        nonlocal line, size, saved, parts, remaining, end, closed, long_line, line_start
        text = ''.join(parts)
        if preview:
            shown = text.encode()[:200].decode('utf-8', errors='ignore')
            snippet = {'line': line, 'text': shown, 'partial': len(shown.encode()) < size}
            if len(head) < 5:
                head.append(snippet)
            tail.append(snippet)
        if not preview and offset <= line < offset + limit and not closed:
            prefix, separator = str(line + 1) + ': ', 1 if rows else 0
            count = len(prefix.encode()) + size
            if count + separator <= remaining:
                rows.append(prefix + text)
                remaining -= count + separator
                end = line + 1
            else:
                if count > OUTPUT_BYTES:
                    long_line = {'offset': line, 'utf8_bytes': count,
                                 'reason': 'Single line exceeds the 50 KiB read budget; partial text only.',
                                 'byte_offset': line_start,
                                 'next_query': 'payload_bytes:' + str(line_start)}
                    if remaining > separator:
                        partial = (prefix + text).encode()[:remaining - separator].decode(
                            'utf-8', errors='ignore',
                        )
                        rows.append(partial)
                        remaining -= separator + len(partial.encode())
                closed = True
        line += 1
        size, saved, parts = 0, 0, []
        line_start = byte_position

    def consume(text):
        nonlocal skip_lf, size, saved, byte_position, line_start
        if skip_lf and text:
            if text.startswith('\n'):
                text = text[1:]
                byte_position += 1
                line_start += 1
            skip_lf = False
        for part in text.splitlines(keepends=True):
            skip_lf = False
            ends = part.endswith(breaks)
            content = part[:-2] if part.endswith('\r\n') else part[:-1] if ends else part
            byte_position += len(part.encode('utf-8', errors='surrogateescape'))
            if original_log:
                content = content.encode('utf-8', errors='surrogateescape').decode('utf-8', errors='replace')
            raw = content.encode('utf-8')
            size += len(raw)
            cap = 200 if preview else OUTPUT_BYTES if offset <= line < offset + limit and not closed else 0
            if saved < cap:
                consumed = min(len(raw), cap - saved)
                piece = raw[:consumed].decode('utf-8', errors='ignore')
                parts.append(piece)
                saved += consumed
            if ends:
                finish()
                skip_lf = part.endswith('\r')

    with path.open('rb') as stream:
        while chunk := stream.read(64 * 1024):
            if owner_cancelled():
                raise InterruptedError('Payload reading was cancelled; no partial result was returned.')
            digest.update(chunk)
            total_bytes += len(chunk)
            consume(decoder.decode(chunk))
        consume(decoder.decode(b'', final=True))
    if size:
        finish()
    if digest.hexdigest() != expected:
        raise ValueError('Archived tool payload SHA-256 mismatch; no content was returned.')
    budget.remaining = remaining
    if preview:
        samples = {item['line']: item for item in [*head, *tail]}
        return {'sha256': expected, 'bytes': total_bytes, 'lines': line,
                'preview': [samples[key] for key in sorted(samples)], 'preview_complete': False,
                'encoding': 'utf-8', 'decoding_errors': 'replace' if original_log else 'strict'}
    result = {'path': name, 'sha256': expected, 'offset': offset, 'total_lines': line,
              'text': '\n'.join(rows), 'visible_ranges': [[offset, end]] if end > offset else [],
              'truncated': end < min(line, offset + limit), 'next_offset': end}
    if long_line:
        result['long_line'] = long_line
        result['next_query'] = long_line['next_query']
        result['byte_read_instruction'] = (
            'For this long original line, use the same path with offset=0, default limit=2000 '
            'and query=next_query. '
            'Byte-window reads return next_query and raw byte ranges; line limit does not bound bytes.'
        )
    result.update(encoding='utf-8', decoding_errors='replace' if original_log else 'strict')
    return result


def read_bytes(root, name, offset, *, budget=None, archive=None):
    """超长行按真实原字节续读；与源码读取共用有界校验窗口。"""
    from agent.source.paging import ReadBudget, read_byte_window

    budget = budget if budget is not None else ReadBudget()
    path, expected = _location(root, name, archive)
    original_log = not bool(_CAS.fullmatch(name))
    with path.open('rb') as stream:
        result = read_byte_window(
            stream, expected, offset, budget=budget,
            errors='replace' if original_log else 'strict',
        )
    end = result['next_byte_offset']
    return {
        'path': name,
        **result,
        'next_query': 'payload_bytes:' + str(end) if result['truncated'] else None,
        'continuation': {'reason': 'payload_byte_window', 'next_byte_offset': end},
    }


def _store(loop, text):
    raw = text.encode('utf-8')
    relative = 'tool-payloads/' + _sha(raw) + '.txt'
    path = Path(loop.root) / relative
    if path.parent.is_symlink() or path.parent.is_junction():
        raise ValueError('Tool payload storage cannot contain links.')
    if not path.exists():
        from agent.storage.resources import preflight_space

        preflight_space(path, len(raw))
    path.parent.mkdir(exist_ok=True)
    try:
        with path.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        # 完整核验仍由流式读取完成，旧对象绝不覆盖。
        _safe_file(Path(loop.root).resolve(), Path(relative))
    archive = getattr(loop, 'history_store', None)
    metadata = read(loop.root, '$record/' + relative, 0, 1, preview=True, archive=archive)
    if archive is not None:
        existing = _entry(archive, relative)
        if existing and existing['sha256'] != metadata['sha256']:
            raise ValueError('Registered tool payload identity differs from its content hash.')
        if existing is None:
            archive.register_derived(path, 'tool_payload', loop.state.get('source', ''))
    return '$record/' + relative, metadata


def _fields(value):
    if _reference(value):
        yield value['sha256'], False
    elif _page(value):
        return
    elif isinstance(value, dict):
        for key, child in value.items():
            if (key in TEXT_FIELDS and isinstance(child, str)) or (
                key in JSON_FIELDS and isinstance(child, (dict, list)) and not _reference(child)
            ):
                raw = _text(child).encode()
                if len(raw) >= MIN_DUPLICATE:
                    yield _sha(raw), True
            else:
                yield from _fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from _fields(child)


def _visible(loop, explicit=None):
    """派生索引只解析新增尾部；原有admit/guard_archive继续校验完整前缀。"""
    state = loop.state
    history = state.get('history', []) if explicit is None else explicit
    view = state.get('context_view') if explicit is None else None
    key = (id(history), json.dumps(view, sort_keys=True))
    cache = getattr(loop, '_tool_payload_visibility', None)
    valid = (cache and cache['history'] is history and cache['key'] == key
             and cache['cursor'] <= len(history)
             and (not cache['cursor'] or cache['last'] is history[cache['cursor'] - 1]))
    if valid:
        items = history[cache['cursor']:]
    else:
        from agent.model.compaction import history as current_history

        items = current_history(state, loop.root) if view else history
        cache = {'key': key, 'history': history, 'cursor': 0, 'seen': {}, 'last': None}
    for message in items:
        if message.get('type') != 'tool':
            continue
        data = message.get('data', {})
        try:
            body = json.loads(data.get('content', ''))
        except (TypeError, ValueError):
            continue
        for sha, complete in _fields(body):
            cache['seen'][sha] = {'tool_call_id': data.get('tool_call_id'), 'body_was_inline': complete}
    cache.update(cursor=len(history), last=history[-1] if history else None)
    loop._tool_payload_visibility = cache
    return dict(cache['seen'])


def project(loop, value, *, history=None):
    """原件先保存，再产生模型展示；失败会停止，不交付缺失正文。"""
    try:
        seen = _visible(loop, history)
        archive = getattr(loop, 'history_store', None)

        def payload(value, field, parent):
            text = _text(value)
            raw, path, metadata = text.encode(), None, None
            recorded = parent.get(field + '_record') if field in {'stdout', 'stderr'} else None
            if isinstance(recorded, str):
                candidate = recorded if recorded.startswith('$record/') else '$record/' + recorded
                if readable(candidate, archive):
                    source, unused = _location(loop.root, candidate, archive)
                    if source.stat().st_size > THRESHOLD:
                        path = candidate
                        metadata = read(loop.root, path, 0, 1, preview=True, archive=archive)
            sha = metadata['sha256'] if metadata else _sha(raw)
            duplicate = seen.get(sha)
            if path is None and len(raw) <= THRESHOLD and not (duplicate and len(raw) >= MIN_DUPLICATE):
                if len(raw) >= MIN_DUPLICATE:
                    seen[sha] = {'location': 'earlier_in_this_result', 'body_was_inline': True}
                return deepcopy(value)
            if path is None:
                path, metadata = _store(loop, text)
            result = {'format': FORMAT, 'path': path, **metadata,
                      'read_instruction': 'Read this exact $record path with a read step, offset and limit. '
                      'This reference and preview do not mean the complete body was read.'}
            if duplicate:
                result.update(duplicate_of=duplicate, preview=[])
            if field == 'error':
                result['summary'] = text.splitlines()[0][:512] if text.splitlines() else ''
            if duplicate is None:
                seen[sha] = {'location': 'earlier_in_this_result', 'body_was_inline': False}
            return result

        def visit(item):
            if _reference(item) or _page(item):
                return deepcopy(item)
            if isinstance(item, list):
                return [visit(child) for child in item]
            if not isinstance(item, dict):
                return deepcopy(item)
            return {key: payload(child, key, item) if (
                key in TEXT_FIELDS and isinstance(child, str)
                or key in JSON_FIELDS and isinstance(child, (dict, list)) and not _reference(child)
            ) else visit(child) for key, child in item.items()}

        return visit(value)
    except InterruptedError as exc:
        from agent.task.control import check_interrupted

        check_interrupted(pause_only=True)
        raise ModelRequestNotSent(
            'state_conflict', 'Tool payload processing was interrupted without a confirmed '
            'pause signal. Original records were preserved.',
        ) from exc
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise ModelRequestNotSent(
            'state_conflict', 'Tool payload could not be archived or verified. Original records '
            'were preserved; no incomplete result was delivered. ' + str(exc),
        ) from exc
