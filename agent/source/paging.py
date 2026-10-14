"""限制请求范围后自动分页，无需规划器再次决策。"""

import json

PAGE_LINES = 200
MAX_REQUEST_LINES = 2000
OUTPUT_BYTES = 50 * 1024


class ReadBudget:
    """一次读取步骤的所有页面和位置共用同一 UTF-8 文本预算。"""

    def __init__(self):
        self.remaining = OUTPUT_BYTES

    def excerpt(self, lines, offset, limit):
        rows, end, long_line = [], offset, None
        for index in range(offset, min(len(lines), offset + limit)):
            row = f'{index + 1}: {lines[index]}'
            raw = row.encode('utf-8')
            separator = 1 if rows else 0
            if len(raw) + separator > self.remaining:
                if len(raw) > OUTPUT_BYTES:
                    long_line = {
                        'offset': index,
                        'utf8_bytes': len(raw),
                        'reason': 'Single line exceeds the 50 KiB read budget; partial text only.',
                    }
                    if self.remaining > separator:
                        part = raw[: self.remaining - separator].decode('utf-8', errors='ignore')
                        rows.append(part)
                        self.remaining -= separator + len(part.encode('utf-8'))
                break
            rows.append(row)
            self.remaining -= len(raw) + separator
            end = index + 1
        result = {
            'text': '\n'.join(rows),
            'visible_ranges': [[offset, end]] if end > offset else [],
            'truncated': end < min(len(lines), offset + limit),
            'next_offset': end,
        }
        if long_line:
            result['long_line'] = long_line
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


def visible_reads(history, rows):
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
            key = (page['path'], page['sha256'])
            seen[key] = merged([*seen.get(key, []), *page['visible_ranges']])
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


def read_range(executor, path, offset, limit, budget=None):
    budget = budget if budget is not None else ReadBudget()
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
        'pagination': {
            'requested_offset': offset,
            'requested_lines': limit,
            'page_line_limit': PAGE_LINES,
            'returned_pages': len(pages),
        },
    }
