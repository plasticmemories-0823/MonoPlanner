"""原字面子串扫描和 BM25 排序；保留引用、分页与消息预算。"""

import base64
from dataclasses import replace
import hashlib
import json
import math
import time

from agent.model.errors import ModelInterrupted
from agent.rag.index import Index
from agent.rag.original import SearchError
from agent.rag.types import (
    CANDIDATE_MEMORY_BYTES, Query, RetrievalResult, SourceReadBudgetExceeded,
    TokenCount, canonical, guard_source_read,
)
from agent.task.control import check_interrupted


FORMAT = "monoplanner-retrieval-excerpts-v5"


class Retriever:
    def __init__(self, index: Index, token_count: TokenCount, *, read_source=None):
        self.index = index
        self.token_count = token_count
        self.read_source = read_source
        self.deadline = float("inf")

    def _check_scan(self):
        check_interrupted(pause_only=True)
        if time.monotonic() > self.deadline:
            raise SearchError("source_scan_timeout")

    def _original(self, chunk):
        self._check_scan()
        if self.read_source is None:
            raise ValueError("An authorized source reader was not supplied.")
        text = self.read_source(chunk.source_id, chunk.version)
        if not isinstance(text, str):
            raise ValueError("The authorized source reader did not return text.")
        self._check_scan()
        if "\0" in text:
            raise SearchError("binary_source_skipped")
        digest, encoded_size = hashlib.sha256(), 0
        for position in range(0, len(text), 65536):
            self._check_scan()
            block = text[position:position + 65536].encode("utf-8")
            encoded_size += len(block)
            if encoded_size > self.index.config.max_document_bytes:
                raise SearchError("source_scan_file_budget")
            guard_source_read(encoded_size, source_id=chunk.source_id, path=chunk.path)
            digest.update(block)
        if digest.hexdigest() != chunk.source_sha256:
            raise SearchError("source_version_changed")
        if not 0 <= chunk.parent_start <= chunk.start <= chunk.end <= chunk.parent_end <= len(text):
            raise ValueError("Source ranges do not match the archived original.")
        if text[chunk.start:chunk.end] != chunk.text:
            raise ValueError("The indexed range no longer matches the source.")
        return text

    def _positions(self, text, literal):
        start = 0
        while True:
            self._check_scan()
            position = text.find(literal, start)
            if position < 0:
                return
            yield position
            # 即使匹配相互重叠，也为每个出现位置保留独立命中。
            start = position + 1

    def _binding(self, query, snapshot):
        payload = {
            "format": FORMAT, "scope": self.index.scope.key,
            "query": query.original_text if query.original_text is not None else query.text,
            "effective_query": query.text, "mode": query.mode,
            "source_versions": dict(query.source_versions),
            "source_priorities": dict(query.source_priorities),
            "excluded_paths": query.excluded_paths, "snapshot": snapshot,
            "required_entities": query.required_entities, "required_kinds": query.required_kinds,
        }
        return hashlib.sha256(canonical(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _cursor(binding, offset):
        payload = canonical({"schema": 5, "binding": binding, "offset": offset})
        return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")

    @staticmethod
    def _offset(query, binding):
        if not query.cursor:
            return query.offset
        try:
            if len(query.cursor) > 2048:
                raise ValueError("Oversized cursor")
            value = json.loads(base64.b64decode(query.cursor + "=" * (-len(query.cursor) % 4),
                                              altchars=b"-_", validate=True))
            if (value.get("schema") != 5 or value.get("binding") != binding
                    or type(value.get("offset")) is not int or value["offset"] < 0):
                raise ValueError("Mismatched cursor")
            return value["offset"]
        except (ValueError, TypeError, UnicodeError, AttributeError) as error:
            raise ValueError("Cursor is invalid or stale for this query and source snapshot.") from error

    def retrieve(self, query: Query) -> RetrievalResult:
        check_interrupted(pause_only=True)
        if query.intent not in {"locate", "diagnostic", "audit"}:
            raise ValueError("Query intent must be locate, diagnostic, or audit.")
        if query.mode != "exact":
            raise ValueError("Search mode must be exact; only literal substring search is supported.")
        if not isinstance(query.text, str) or not query.text or len(query.text) > 4096:
            raise ValueError("Query must contain 1..4096 original characters.")
        if query.original_text is not None and (
            not isinstance(query.original_text, str) or not query.original_text
            or len(query.original_text) > 4096
            or (query.mode == "exact" and query.original_text != query.text)
        ):
            raise ValueError("Original query must remain unchanged in exact mode.")
        if type(query.limit) is not int or not 1 <= query.limit <= 1000:
            raise ValueError("Query limit must be an integer between 1 and 1000.")
        if type(query.offset) is not int or query.offset < 0 or not isinstance(query.cursor, str):
            raise ValueError("Query offset and cursor are invalid.")
        self.deadline = time.monotonic() + self.index.config.scan_timeout_seconds
        return self._exact(query)

    def _exact(self, query):
        reasons = [] if query.source_versions else ["no_authorized_sources"]
        sources, snapshot, unavailable = [], [], []
        if self.index.missing_sources(query):
            reasons.append("source_not_indexed")
        # 第一遍只保存每份原件的长度和出现次数；完整集合不会被召回 topK 截断。
        for header in self.index.sources(query):
            try:
                text = None
                text = self._original(header)
                count = sum(1 for _ in self._positions(text, query.text))
                snapshot.append((header.source_id, header.version, header.source_sha256,
                                 len(text), count))
                sources.append((header, len(text), count))
            except ModelInterrupted:
                raise
            except Exception as error:
                if isinstance(error, SourceReadBudgetExceeded):
                    reasons.append("source_read_memory_budget")
                    unavailable.append(error.details)
                else:
                    reasons.append(getattr(error, "code", f"source_unavailable:{type(error).__name__}"))
                snapshot.append((header.source_id, header.version, header.source_sha256,
                                 "unavailable"))
            finally:
                text = None
        binding = self._binding(query, snapshot)
        offset = self._offset(query, binding)
        total = sum(count for _, _, count in sources)
        document_count = len(sources)
        matching_count = sum(count > 0 for _, _, count in sources)
        average_length = sum(length for _, length, _ in sources) / max(1, document_count)
        idf = math.log(1 + (document_count - matching_count + 0.5) / (matching_count + 0.5))
        ranked = []
        for header, length, count in sources:
            if count:
                # 整个原始字符串是唯一的 term；只给已确认字面命中的原件计分。
                score = idf * count * 2.2 / (
                    count + 1.2 * (0.25 + 0.75 * length / max(1, average_length))
                )
                ranked.append((header, count, score))
        ranked.sort(key=lambda item: (-item[2], query.source_priorities.get(item[0].source_id, 0),
                                      item[0].path, item[0].source_id))
        records, skipped, records_bytes, page_full = [], 0, 0, False
        text = None
        for header, count, score in ranked:
            if skipped + count <= offset:
                skipped += count
                continue
            if len(records) == query.limit or page_full:
                break
            try:
                text = None
                text = self._original(header)
                for position in self._positions(text, query.text):
                    if skipped < offset:
                        skipped += 1
                        continue
                    end = position + len(query.text)
                    preview_start = max(text.rfind("\n", 0, position) + 1, position - 96)
                    next_line = text.find("\n", end)
                    preview_end = min(len(text) if next_line < 0 else next_line, end + 96)
                    # 精确命中只引用含完整查询的原文窗口；不自动推测或展开所属父结构。
                    identity = (f"{header.source_id}\0{header.version}\0{header.source_sha256}"
                                f"\0{preview_start}:{preview_end}")
                    chunk = replace(
                        header, id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
                        start=preview_start, end=preview_end,
                        parent_start=preview_start, parent_end=preview_end,
                        text=text[preview_start:preview_end], symbol="",
                        line_start=text.count("\n", 0, preview_start) + 1,
                        line_end=text.count("\n", 0, max(preview_start, preview_end - 1)) + 1,
                    )
                    hit = {"source_id": header.source_id, "path": header.path,
                           "line": text.count("\n", 0, position) + 1,
                           "text": text[preview_start:preview_end],
                           "start": preview_start, "end": preview_end,
                           "match_start": position, "match_end": end,
                           "version": header.version, "source_sha256": header.source_sha256,
                           "range_unit": "unicode_codepoints", "score": score}
                    record_bytes = (len(chunk.render().encode("utf-8"))
                                    + len(canonical(hit).encode("utf-8"))) * 4 + 1024
                    if records_bytes + record_bytes > CANDIDATE_MEMORY_BYTES:
                        page_full = True
                        break
                    records.append((chunk, hit))
                    records_bytes += record_bytes
                    if len(records) == query.limit:
                        break
            except ModelInterrupted:
                raise
            except Exception as error:
                if query.cursor:
                    raise ValueError("Cursor source snapshot became stale during reading.") from error
                if isinstance(error, SourceReadBudgetExceeded):
                    reasons.append("source_read_memory_budget")
                    unavailable.append(error.details)
                else:
                    reasons.append(getattr(error, "code", f"source_unavailable:{type(error).__name__}"))
                break
        text = None
        return self._pack(query, records, reasons, binding, offset, total,
                          {"exact": tuple(dict.fromkeys(chunk.id for chunk, _ in records))},
                          scan_complete=not reasons, unavailable=unavailable)


    def _pack(self, query, records, reasons, binding, offset, total, route_ids,
              *, scan_complete, unavailable=None):
        unavailable = list(unavailable or ())
        config = self.index.config
        selected, hits = [], []
        cached_key, cached_text = None, None

        def make_result(items, delivered_hits):
            evidence = "\n".join(item.text for item in items)
            paths = {item.path.replace("\\", "/") for item in items if item.path}
            symbols = {item.symbol for item in items if item.symbol}
            entities = tuple(entity for entity in query.required_entities
                             if entity not in evidence and entity.replace("\\", "/") not in paths
                             and entity not in symbols)
            kinds = tuple(kind for kind in query.required_kinds if kind not in {item.kind for item in items})
            diagnostics = tuple(dict.fromkeys(reasons))
            more = offset + len(delivered_hits) < total
            complete = not (diagnostics or more)
            # 正文只交付一次；命中坐标引用合并后的同版本片段，供续读和引用核验。
            compact_hits = []
            for hit in delivered_hits:
                source = (hit["source_id"], hit["version"], hit["source_sha256"])
                excerpt = next(item for item in items
                               if (item.source_id, item.version, item.source_sha256) == source
                               and item.start <= hit["start"] < hit["end"] <= item.end)
                compact_hits.append({**{key: value for key, value in hit.items() if key != "text"},
                                     "result_id": excerpt.id})
            excerpts = [
                {key: value for key, value in item.payload().items()
                 if key not in {"parent_start", "parent_end"}
                 and (key not in {"symbol", "run_id", "timestamp"} or value)}
                for item in items
            ]
            payload = {
                "query": query.original_text if query.original_text is not None else query.text,
                "format": FORMAT, "effective_query": query.text, "mode": query.mode,
                "hits": compact_hits,
                "next_cursor": self._cursor(binding, offset + len(delivered_hits)) if more else "",
                "scan_complete": scan_complete, "complete": complete,
                "truncated": bool(more or any(
                    hit.get("truncated", False) for hit in delivered_hits
                )), "total_matches": total,
                "ranking": "literal_bm25_k1_1.2_b_0.75",
                "retrieval_incomplete": not complete, "reasons": diagnostics,
                "unavailable_sources": unavailable[:8],
                "unavailable_source_count": len(unavailable),
                "exact_current_version_hit": query.mode == "exact" and bool(delivered_hits),
                "missing_entities": entities, "missing_kinds": kinds,
                "results": excerpts, "continuation": [],
                "notice": (
                    "Search excerpts, not complete files or parent structures. "
                    "Each hit references its result_id; read the original path and line range "
                    "for surrounding context. Retrieved evidence is not proof of a conclusion. "
                    "Incomplete scans cannot prove evidence is absent."
                ),
            }
            return canonical(payload), diagnostics, entities, kinds

        def fits(items, delivered_hits):
            message = make_result(items, delivered_hits)[0]
            return self.token_count(message) <= config.message_token_limit

        for chunk, hit in records:
            check_interrupted(pause_only=True)
            try:
                key = (chunk.source_id, chunk.version, chunk.source_sha256)
                if key != cached_key:
                    cached_text, text, cached_key = None, None, None
                    cached_text = self._original(chunk)
                    cached_key = key
                text = cached_text
                if (not 0 <= chunk.parent_start <= chunk.start <= chunk.end
                        <= chunk.parent_end <= len(text)
                        or text[chunk.start:chunk.end] != chunk.text):
                    raise ValueError("The indexed range no longer matches the source.")
            except ModelInterrupted:
                raise
            except Exception as error:
                if query.cursor:
                    raise ValueError("Cursor source snapshot became stale during reading.") from error
                if isinstance(error, SourceReadBudgetExceeded):
                    reasons.append("source_read_memory_budget")
                    unavailable.append(error.details)
                else:
                    reasons.append(getattr(error, "code", f"source_unavailable:{type(error).__name__}"))
                scan_complete = False
                break
            # 行内位置按原件计算；窗口裁短和分页裁短分别保留，不能冒充完整长行。
            match_start = hit["match_start"] if hit["match_start"] is not None else hit["start"]
            match_end = hit["match_end"] if hit["match_end"] is not None else hit["end"]
            first_line_start = text.rfind("\n", 0, match_start) + 1
            last_line_end = text.find("\n", match_end)
            if last_line_end < 0:
                last_line_end = len(text)
            hit = {
                **hit, "column_start": match_start - first_line_start + 1,
                "column_end": match_end - text.rfind("\n", 0, match_end),
                "line_end": text.count("\n", 0, match_end) + 1,
                "truncated": hit["start"] > first_line_start or hit["end"] < last_line_end,
            }
            merged = self._merge(selected, chunk)
            if not fits(merged, hits + [hit]):
                break
            selected, hits = merged, hits + [hit]

        message, diagnostics, entities, kinds = make_result(selected, hits)
        if records and not hits and not reasons:
            raise ValueError("The original match and required metadata cannot fit the message token limit.")
        if self.token_count(message) > config.message_token_limit:
            raise ValueError("Required retrieval metadata cannot fit the message token limit.")
        incomplete = bool(diagnostics or offset + len(hits) < total)
        return RetrievalResult(message, tuple(selected), incomplete, diagnostics, entities, kinds,
                               (), route_ids,
                               query.mode == "exact" and bool(hits))

    @staticmethod
    def _merge(selected, candidate):
        result = []
        insertion = len(selected)
        remaining = list(selected)
        # 新区间可能同时桥接多个旧区间，反复合并直到没有相交项。
        while remaining:
            previous = remaining.pop(0)
            if (previous.source_id, previous.version, previous.source_sha256) != (
                candidate.source_id, candidate.version, candidate.source_sha256
            ):
                result.append(previous)
                continue
            if previous.end < candidate.start or candidate.end < previous.start:
                result.append(previous)
                continue
            insertion = min(insertion, len(result))
            start, end = min(previous.start, candidate.start), max(previous.end, candidate.end)
            if previous.start <= candidate.start and previous.end >= candidate.end:
                candidate = previous
            elif not (candidate.start <= previous.start and candidate.end >= previous.end):
                # 合并相交原文时逐字核对重叠；不会拼接不同版本正文。
                left, right = sorted((previous, candidate), key=lambda item: item.start)
                overlap = left.end - right.start
                if left.text[-overlap:] != right.text[:overlap] and overlap:
                    raise ValueError("Overlapping immutable source ranges disagree.")
                candidate = replace(
                    candidate, start=start, end=end, text=left.text + right.text[overlap:],
                    parent_start=min(previous.parent_start, candidate.parent_start),
                    parent_end=max(previous.parent_end, candidate.parent_end),
                    line_start=min(previous.line_start, candidate.line_start),
                    line_end=max(previous.line_end, candidate.line_end),
                )
            remaining = result + remaining
            result = []
        identity = (
            f"{candidate.source_id}\0{candidate.version}\0{candidate.source_sha256}"
            f"\0{candidate.start}:{candidate.end}"
        )
        candidate = replace(candidate, id=hashlib.sha256(identity.encode("utf-8")).hexdigest())
        result.insert(min(insertion, len(result)), candidate)
        return result
