"""超长源码行必须能按原始 UTF-8 字节逐页读取，游标绑定原件版本。"""

import hashlib
import json

import pytest

from agent.execution.assistant_files import AssistantFiles
from agent.source.paging import OUTPUT_BYTES, ReadBudget


@pytest.mark.parametrize('prefix,body', [
    (b'', ('a' * 100003 + '\nlast\n').encode()),
    (b'', ('🙂中' * 40001 + '\nlast\n').encode()),
    (b'\xef\xbb\xbf', ('中🙂' * 40001 + '\r\nlast\r\n').encode()),
], ids=['ascii', 'utf8', 'bom-crlf'])
def test_chat_long_line_is_nonempty_and_cursor_reads_original_without_gaps(tmp_path, prefix, body):
    raw = prefix + body
    (tmp_path / 'long.txt').write_bytes(raw)
    files = AssistantFiles(str(tmp_path))
    first = files.read('long.txt', 0, 2, [])
    assert first['text'].startswith('1: ')
    assert len(first['text']) <= 24000
    assert first['visible_ranges'] == []
    assert first['sha256'] == hashlib.sha256(raw).hexdigest()
    assert first['budget_unit'] == 'unicode_characters'
    if prefix:
        assert len(first['text'].encode()) > OUTPUT_BYTES
    start, end = first['partial_byte_range']
    assert start == len(prefix)
    recovered = first['text'][3:].encode()
    assert recovered == raw[start:end]
    cursor = first['next_cursor']
    history = [{'type': 'tool', 'data': {'content': json.dumps(first)}}]
    pages = 0
    while cursor:
        page = files.read('long.txt', 0, 2, history, cursor=cursor)
        assert page['visible_ranges'] == []
        assert page['byte_range'][0] == end
        assert page['byte_range'][1] > end
        assert len(page['text']) <= 24000
        assert page['text'].encode() == raw[end:page['byte_range'][1]]
        assert page['byte_offset_unit'] == 'raw_utf8_bytes'
        recovered += page['text'].encode()
        end, cursor = page['byte_range'][1], page['next_cursor']
        pages += 1
        assert pages < 30
    assert pages >= 2
    assert recovered == body
    assert end == len(raw)


def test_chat_cursor_rejects_changed_source_and_current_permission_removal(tmp_path):
    path = tmp_path / 'long.txt'
    original = ('中' * 40000).encode()
    path.write_bytes(original)
    files = AssistantFiles(str(tmp_path))
    cursor = files.read('long.txt', 0, 1, [])['next_cursor']
    path.write_bytes(b'x' + original)
    with pytest.raises(ValueError, match='SHA-256|changed|version'):
        files.read('long.txt', 0, 1, [], cursor=cursor)
    path.write_bytes(original)
    with pytest.raises(ValueError, match='授权'):
        AssistantFiles(None).read(str(path), 0, 1, [], cursor=cursor)


def test_work_long_line_cursor_advances_with_shared_utf8_budget():
    raw = b'\xef\xbb\xbf' + ('中🙂' * 40001 + '\r\ntail\n').encode()
    first = ReadBudget().source(raw, 0, 2)
    assert first['visible_ranges'] == []
    assert first['next_offset'] == 0
    assert first['long_line']['offset'] == 0
    assert first['budget_unit'] == 'utf8_bytes'
    assert len(first['text'].encode()) <= OUTPUT_BYTES
    end = first['partial_byte_range'][1]
    recovered = first['text'][3:].encode()
    cursor = first['next_cursor']
    while cursor:
        budget = ReadBudget()
        page = budget.source(raw, 0, 2, cursor=cursor)
        assert page['byte_range'][0] == end
        assert page['byte_range'][1] > end
        assert len(page['text'].encode()) + budget.remaining == OUTPUT_BYTES
        assert page['visible_ranges'] == []
        recovered += page['text'].encode()
        end, cursor = page['byte_range'][1], page['next_cursor']
    assert recovered == raw[3:]


def test_source_cursor_rejects_split_utf8_and_keeps_line_offset_units():
    raw = ('short\n' + '中' * 40000 + '\n').encode()
    first = ReadBudget().source(raw, 1, 1)
    assert first['text'].startswith('2: 中')
    assert first['offset_unit'] == 'lines'
    assert first['partial_byte_range'][0] == len(b'short\n')
    cursor = first['next_cursor']
    fields = cursor.split(':')
    fields[-2] = str(int(fields[-2]) + 1)
    with pytest.raises(ValueError, match='UTF-8'):
        ReadBudget().source(raw, 1, 1, cursor=':'.join(fields))
    with pytest.raises(ValueError, match='SHA-256|changed|version'):
        ReadBudget().source(raw + b'x', 1, 1, cursor=cursor)


def test_partial_after_complete_lines_preserves_shared_budget_and_requested_end():
    raw = ('first\r\n' + '中' * 40000 + '\r\nexcluded\n').encode()
    budget = ReadBudget()
    budget.remaining = 100
    first = budget.source(raw, 0, 2)
    assert first['visible_ranges'] == [[0, 1]]
    assert first['next_offset'] == 1
    assert len(first['text'].encode()) + budget.remaining == 100
    cursor = first['next_cursor']
    recovered = first['text'].split('\n2: ', 1)[1].encode()
    while cursor:
        page = ReadBudget().source(raw, 0, 2, cursor=cursor)
        recovered += page['text'].encode()
        cursor = page['next_cursor']
    assert recovered == ('中' * 40000 + '\r\n').encode()
    assert page['next_offset'] == 2
    assert not page['truncated']


def test_byte_window_verifies_unselected_tail_before_charging_budget():
    from io import BytesIO
    from agent.source.paging import read_byte_window

    raw = b'x' * 200000
    expected = hashlib.sha256(raw).hexdigest()
    budget = ReadBudget()
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        read_byte_window(BytesIO(raw[:-1] + b'y'), expected, 0, budget=budget)
    assert budget.remaining == OUTPUT_BYTES
