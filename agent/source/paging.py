"""限制请求范围后自动分页，无需规划器再次决策。"""

from bisect import bisect_right
import codecs
import hashlib
from io import BytesIO
import json
import re

PAGE_LINES = 200
MAX_REQUEST_LINES = 2000
OUTPUT_BYTES = 50 * 1024
_SOURCE_CURSOR = re.compile(r'source-bytes-v1:([0-9a-f]{64}):(0|[1-9][0-9]*):(0|[1-9][0-9]*)')


def source_cursor(sha, offset, stop):
    return f'source-bytes-v1:{sha}:{offset}:{stop}' if offset < stop else None


def read_byte_window(stream, expected, offset, *, budget, stop=None, errors='strict'):
    """分块核验完整原件，仅保留有界窗口；源码和归档正文共用字节边界规则。"""
    from agent.storage.resources import owner_cancelled

    if type(offset) is not int or offset < 0:
        raise ValueError('Byte offset must be a nonnegative integer.')
    if stop is not None and (type(stop) is not int or stop < offset):
        raise ValueError('Byte window end must not precede its offset.')
    if errors not in {'strict', 'replace'}:
        raise ValueError('Byte windows support strict or replacement UTF-8 decoding.')
    count = budget.remaining * 4 if budget.unit == 'unicode_characters' else budget.remaining
    if errors == 'replace':
        # 替代字符最多使用三个展示字节，原始日志仍按实际字节推进。
        count //= 3
    if count < 1:
        raise ValueError('Insufficient shared read budget for a complete UTF-8 byte window.')
    requested_end = offset + count if stop is None else min(stop, offset + count)
    digest, selected, total = hashlib.sha256(), bytearray(), 0
    decoder = codecs.getincrementaldecoder('utf-8')(errors)
    while chunk := stream.read(64 * 1024):
        if owner_cancelled():
            raise InterruptedError('Payload reading was cancelled; no partial result was returned.')
        digest.update(chunk)
        decoder.decode(chunk)
        start, end = max(0, offset - total), min(len(chunk), requested_end - total)
        if start < end:
            selected.extend(chunk[start:end])
        total += len(chunk)
    decoder.decode(b'', final=True)
    if digest.hexdigest() != expected:
        raise ValueError('Source or archived payload SHA-256 mismatch; no content was returned.')
    stop = total if stop is None else stop
    if offset > total or stop > total:
        raise ValueError('Byte window exceeds the original byte length.')
    if selected and errors == 'strict' and 0x80 <= selected[0] < 0xC0:
        raise ValueError('Byte offset splits a UTF-8 character; use the returned continuation.')
    selected_decoder = codecs.getincrementaldecoder('utf-8')(errors)
    text = selected_decoder.decode(bytes(selected), final=offset + len(selected) == stop)
    end = offset + len(selected) - len(selected_decoder.getstate()[0])
    if budget.unit == 'unicode_characters' and len(text) > budget.remaining:
        text = text[:budget.remaining]
        end = offset + len(text.encode('utf-8'))
    used = budget.measure(text)
    if used > budget.remaining or (end == offset and offset < stop):
        raise ValueError('The byte window cannot fit in the remaining shared read budget.')
    budget.remaining -= used
    return {
        'sha256': expected,
        'text': text,
        'visible_ranges': [],
        'byte_range': [offset, end],
        'byte_offset_unit': 'raw_utf8_bytes',
        'total_bytes': total,
        'shown_utf8_bytes': len(text.encode('utf-8')),
        'next_byte_offset': end,
        'truncated': end < stop,
        'encoding': 'utf-8',
        'decoding_errors': errors,
        'budget_unit': budget.unit,
    }


class ReadBudget:
    """一次读取的页面和位置共享预算，工作按 UTF-8 字节、聊天按字符计数。"""

    def __init__(self, maximum=OUTPUT_BYTES, *, unit='utf8_bytes'):
        if type(maximum) is not int or maximum < 1:
            raise ValueError('Read budget must be a positive integer.')
        if unit not in {'utf8_bytes', 'unicode_characters'}:
            raise ValueError('Unknown read budget unit.')
        self.maximum = self.remaining = maximum
        self.unit = unit

    def measure(self, text):
        return len(text) if self.unit == 'unicode_characters' else len(text.encode('utf-8'))

    def source(self, raw, offset, limit, *, cursor=None):
        """行参数保持原义；显式游标按绑定版本的原始字节窗口续读。"""
        ranges(offset, limit)
        text = raw.decode('utf-8-sig')
        lines = text.splitlines()
        sha = hashlib.sha256(raw).hexdigest()
        starts, position = [], 3 if raw.startswith(codecs.BOM_UTF8) else 0
        for part in text.splitlines(keepends=True):
            starts.append(position)
            position += len(part.encode('utf-8'))
        starts.append(len(raw))
        if cursor:
            match = _SOURCE_CURSOR.fullmatch(cursor) if isinstance(cursor, str) else None
            if match is None:
                raise ValueError('Invalid source byte continuation cursor.')
            if match[1] != sha:
                raise ValueError('Source version changed; continuation SHA-256 does not match.')
            start, stop = int(match[2]), int(match[3])
            result = read_byte_window(BytesIO(raw), sha, start, budget=self, stop=stop)
            result.update(
                next_cursor=source_cursor(sha, result['next_byte_offset'], stop),
                total_lines=len(lines),
                offset=offset,
                offset_unit='lines',
                read_mode='source_byte_window',
                next_offset=max(0, bisect_right(starts, result['next_byte_offset']) - 1),
            )
            return result
        stop_line = min(len(lines), offset + limit)
        stop_byte = starts[stop_line]
        rows, end, long_line, partial = [], offset, None, None
        next_byte = None
        for index in range(offset, stop_line):
            prefix = f'{index + 1}: '
            row = prefix + lines[index]
            separator = 1 if rows else 0
            count = self.measure(row)
            if count + separator > self.remaining:
                next_byte = starts[index]
                if count > self.maximum:
                    long_line = {
                        'offset': index,
                        'utf8_bytes': len(row.encode('utf-8')),
                        'reason': 'Single line exceeds the read budget; partial text only.',
                        'byte_offset': next_byte,
                    }
                    available = self.remaining - separator - self.measure(prefix)
                    if available > 0 and self.measure(lines[index][0]) <= available:
                        fragment = read_byte_window(
                            BytesIO(raw), sha, next_byte,
                            budget=ReadBudget(available, unit=self.unit),
                            stop=next_byte + len(lines[index].encode('utf-8')),
                        )
                        rows.append(prefix + fragment['text'])
                        self.remaining -= separator + self.measure(rows[-1])
                        partial = fragment['byte_range']
                        next_byte = fragment['next_byte_offset']
                break
            rows.append(row)
            self.remaining -= count + separator
            end = index + 1
        result = {
            'text': '\n'.join(rows),
            'visible_ranges': [[offset, end]] if end > offset else [],
            'sha256': sha,
            'total_lines': len(lines),
            'offset': offset,
            'offset_unit': 'lines',
            'budget_unit': self.unit,
            'truncated': end < stop_line,
            'next_offset': end,
            'next_cursor': (
                source_cursor(sha, next_byte, stop_byte) if next_byte is not None else None
            ),
        }
        if long_line:
            result['long_line'] = long_line
        if partial:
            result['partial_byte_range'] = partial
            result['byte_offset_unit'] = 'raw_utf8_bytes'
        return result


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def unread(offset, end, seen):
    result, cursor = [], offset
    for start, stop in merged(seen):
        if start > cursor:
            result.append([cursor, min(start, end)])
        cursor = max(cursor, stop)
        if cursor >= end:
            break
    if cursor < end:
        result.append([cursor, end])
    return [pair for pair in result if pair[0] < pair[1]]


def read_pages(value):
    if isinstance(value, dict):
        if all(key in value for key in ('path', 'sha256', 'text', 'visible_ranges')):
            yield value
        for item in value.get('reads', []):
            yield from read_pages(item)


def visible_reads(history, rows, *, key=None):
    """仅采纳实际工具结果和初始材料，不采纳助手声明或本地文件。"""
    values = []
    for index, message in enumerate(history):
        data = message.get('data', {})
        if message.get('type') != 'tool' and not (index == 0 and message.get('type') == 'human'):
            continue
        try:
            body = json.loads(data.get('content', ''))
        except (ValueError, TypeError):
            continue
        if not isinstance(body, dict):
            continue
        if index == 0:
            values.extend(body.get('initial_evidence', {}).get('readme', []))
        if data.get('name') == 'read_file':
            values.append(body)
        values.extend(
            row['data']
            for row in body.get('steps', [])
            if row.get('op') == 'read' and row.get('status') == 'succeeded'
        )
    values.extend(
        row['data'] for row in rows if row.get('op') == 'read' and row.get('status') == 'succeeded'
    )
    seen = {}
    for value in values:
        for page in read_pages(value):
            identity = key(page) if key else (page['path'], page['sha256'])
            if identity is not None:
                seen[identity] = merged([*seen.get(identity, []), *page['visible_ranges']])
    return seen


def ranges(offset, limit):
    if type(offset) is not int or offset < 0:
        raise ValueError('offset must be a nonnegative integer')
    if type(limit) is not int or not 1 <= limit <= MAX_REQUEST_LINES:
        raise ValueError('Request 1..2000 lines; pages contain at most 200 lines')
    return [
        (start, min(PAGE_LINES, offset + limit - start))
        for start in range(offset, offset + limit, PAGE_LINES)
    ]


def read_range(executor, path, offset, limit, budget=None, *, cursor=None):
    budget = budget if budget is not None else ReadBudget()
    if cursor:
        return executor.read(path, offset, limit, budget=budget, cursor=cursor)
    pages = []
    for start, count in ranges(offset, limit):
        if pages and start >= pages[-1]['total_lines']:
            break
        pages.append(executor.read(path, start, count, budget=budget))
        if pages[-1]['truncated']:
            break
    if len(pages) == 1:
        return pages[0]
    return {
        'reads': pages,
        'truncated': any(page['truncated'] for page in pages),
        'next_cursor': pages[-1].get('next_cursor'),
        'pagination': {
            'requested_offset': offset,
            'requested_lines': limit,
            'page_line_limit': PAGE_LINES,
            'returned_pages': len(pages),
        },
    }
