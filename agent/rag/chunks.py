"""按确定性结构切分原文，偏移量始终是 Python 字符区间而非重写后的文本。

完整结构若含来源信息仍不超过硬上限，就保留原样。超长结构的 chunk_target
表示正文目标长度（默认 256 Token）；完整来源头与正文仍共同受 512 Token
硬上限约束，重叠原文最多 64 Token。目标长度不再被来源元数据先行耗尽。
"""

import ast
import bisect
from dataclasses import dataclass
import hashlib
import json
import re

from agent.rag.types import Chunk, Document, RagConfig, TokenCount


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    parent_start: int
    parent_end: int
    symbol: str = ""


def _line_offsets(text):
    offsets = [0]
    offsets.extend(match.end() for match in re.finditer("\n", text))
    if offsets[-1] != len(text):
        offsets.append(len(text))
    return offsets


def _python_spans(text):
    lines = _line_offsets(text)
    tree = ast.parse(text)
    boundaries = [0]
    names = {}
    for node in tree.body:
        start_line = min(
            [node.lineno] + [item.lineno for item in getattr(node, "decorator_list", [])]
        )
        start = lines[start_line - 1]
        end = lines[min(node.end_lineno, len(lines) - 1)]
        boundaries.extend((start, end))
        names[(start, end)] = getattr(node, "name", "")
    points = sorted(set(boundaries + [len(text)]))
    return [
        _Span(start, end, start, end, names.get((start, end), ""))
        for start, end in zip(points, points[1:]) if end > start
    ]


def _markdown_spans(text):
    headings = [match.start() for match in re.finditer(r"(?m)^#{1,6}\s+", text)]
    sections = sorted(set([0, *headings, len(text)]))
    spans = []
    for begin, finish in zip(sections, sections[1:]):
        ends = [begin]
        ends.extend(
            begin + match.end() for match in re.finditer(r"\r?\n[ \t]*\r?\n", text[begin:finish])
        )
        ends.append(finish)
        spans.extend(
            _Span(start, end, start, end)
            for start, end in zip(ends, ends[1:]) if end > start
        )
    return spans


def _json_spans(text):
    # raw_decode 只定位原件区间，不重新序列化源码、错误栈或测试样本。
    decoder = json.JSONDecoder()
    value = json.loads(text)
    if isinstance(value, (list, dict)) and len(value) > 4096:
        return [_Span(0, len(text), 0, len(text))]
    start = len(text) - len(text.lstrip())
    positions = [0]
    cursor = start + 1

    def whitespace(position):
        while position < len(text) and text[position].isspace():
            position += 1
        return position

    if isinstance(value, list):
        for _ in value:
            cursor = whitespace(cursor)
            _, cursor = decoder.raw_decode(text, cursor)
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == ",":
                cursor += 1
            positions.append(cursor)
    elif isinstance(value, dict):
        for _ in value:
            cursor = whitespace(cursor)
            _, cursor = decoder.raw_decode(text, cursor)
            cursor = whitespace(whitespace(cursor) + 1)
            value_start = cursor
            entry, cursor = decoder.raw_decode(text, cursor)
            if isinstance(entry, list) and 1 < len(entry) <= 4096:
                nested = _json_spans(text[value_start:cursor])
                positions.extend(value_start + span.end for span in nested[:-1])
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] == ",":
                cursor += 1
            positions.append(cursor)
    else:
        return [_Span(0, len(text), 0, len(text))]
    positions.append(len(text))
    # 最后一项包含关闭括号，避免仅标点生成一个无用片段。
    if len(positions) > 2:
        positions.pop(-2)
    return [_Span(a, b, a, b) for a, b in zip(positions, positions[1:]) if b > a]


def _log_spans(text):
    markers = list(re.finditer(
        r"(?m)^(?:\[?\d{4}-\d\d-\d\d[ T]|(?:INFO|WARN|ERROR|DEBUG|TRACE)\b)", text
    ))
    points = sorted(set([0, *(item.start() for item in markers), len(text)]))
    return [_Span(a, b, a, b) for a, b in zip(points, points[1:]) if b > a]


def _spans(document):
    # 复杂解析的工作集单独受限；大文件仍可用无损窗口索引，不静默截断。
    if len(document.text.encode("utf-8")) > 512 * 1024:
        return [_Span(0, len(document.text), 0, len(document.text))]
    try:
        if document.kind == "python" or document.path.endswith(".py"):
            return _python_spans(document.text)
        if (
            document.kind in {"markdown", "documentation", "document"}
            or document.path.endswith(".md")
        ):
            return _markdown_spans(document.text)
        if document.kind in {"json", "tool_result"}:
            return _json_spans(document.text)
        if document.kind in {"log", "test_log", "compile_error"}:
            return _log_spans(document.text)
    except (SyntaxError, ValueError, RecursionError):
        pass
    return [_Span(0, len(document.text), 0, len(document.text))]


def chunk_document(document: Document, config: RagConfig, token_count: TokenCount):
    """生成包含来源信息也不超过硬上限的片段；不读取任何外部文件。"""
    if len(document.text.encode("utf-8")) > config.max_document_bytes:
        raise ValueError("Document exceeds the bounded indexing input size.")
    if not document.source_id or not document.version:
        raise ValueError("Source identity and version are required.")
    digest = document.sha256
    line_starts = _line_offsets(document.text)

    def make(start, end, span):
        identity = f"{document.source_id}\0{document.version}\0{digest}\0{start}:{end}"
        return Chunk(
            hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            document.source_id, document.version, digest, document.path, document.kind,
            start, end, span.parent_start, span.parent_end,
            bisect.bisect_right(line_starts, start),
            bisect.bisect_right(line_starts, max(start, end - 1)),
            document.text[start:end], span.symbol, document.run_id, document.timestamp,
        )

    def fits(start, end, span):
        return (
            token_count(document.text[start:end]) <= config.chunk_target
            and token_count(make(start, end, span).render()) <= config.chunk_hard_limit
        )

    def fitting_end(start, end, span):
        # 原文边界改变会更新哈希与行号，计数不严格单调；二分只寻找候选，
        # 返回前必须重新核验，不能把“更短”视为自动符合预算。
        low, high = start, end
        while low < high:
            middle = (low + high + 1) // 2
            if fits(start, middle, span):
                low = middle
            else:
                high = middle - 1
        while low > start and not fits(start, low, span):
            low -= 1
        return low

    chunks = []
    for span in _spans(document):
        if span.start == span.end:
            continue
        complete = make(span.start, span.end, span)
        if token_count(complete.render()) <= config.chunk_hard_limit:
            chunks.append(complete)
            continue
        position = span.start
        while position < span.end:
            end = fitting_end(position, span.end, span)
            if end <= position:
                raise ValueError("Source metadata cannot fit the embedding input limit.")
            # 尽可能在换行处分片，超长单行则保留精确字符区间。
            newline = document.text.rfind("\n", position + 1, end)
            if (
                newline >= position + (end - position) // 2
                and fits(position, newline + 1, span)
            ):
                end = newline + 1
            chunk = make(position, end, span)
            if not fits(position, end, span):
                raise ValueError("Chunk exceeds the embedding input limit.")
            chunks.append(chunk)
            if len(chunks) > 20000:
                raise ValueError("Document exceeds the bounded chunk count.")
            if end == span.end:
                break
            # 重叠只计算原文，不重复计算来源头；必须保证下一片向前推进。
            low, high = position + max(1, (end - position) // 2), end
            while low < high:
                middle = (low + high) // 2
                if token_count(document.text[middle:end]) <= config.chunk_overlap:
                    high = middle
                else:
                    low = middle + 1
            position = low
    return chunks
