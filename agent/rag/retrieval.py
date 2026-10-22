"""rg 枚举授权原文匹配，BM25 自动排列同一批候选。"""

import base64
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

from agent.model.errors import ModelInterrupted
from agent.rag import ripgrep
from agent.rag.types import RetrievalResult, canonical, CANDIDATE_MEMORY_BYTES, guard_source_read, SourceReadBudgetExceeded
from agent.task.control import check_interrupted


FORMAT = "monoplanner-retrieval-excerpts-v4"


class Retriever:
    def __init__(self, index, token_count, *, read_source=None, source_path=None):
        self.index, self.token_count = index, token_count
        self.read_source, self.source_path = read_source, source_path

    @staticmethod
    def _cursor(binding, offset):
        raw = canonical({"schema": 4, "binding": binding, "offset": offset}).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def _offset(query, binding):
        if not query.cursor:
            return query.offset
        try:
            if len(query.cursor) > 2048:
                raise ValueError("Oversized cursor")
            value = json.loads(base64.b64decode(
                query.cursor + "=" * (-len(query.cursor) % 4), altchars=b"-_", validate=True
            ))
            if (value.get("schema") != 4 or value.get("binding") != binding
                    or type(value.get("offset")) is not int or value["offset"] < 0):
                raise ValueError("Mismatched cursor")
            return value["offset"]
        except (ValueError, TypeError, UnicodeError, AttributeError) as error:
            raise ValueError("Cursor is invalid or stale for this query and source snapshot.") from error

    def retrieve(self, query):
        check_interrupted(pause_only=True)
        if query.intent not in {"locate", "diagnostic", "audit"}:
            raise ValueError("Invalid search intent")
        if query.mode not in {"exact", "keywords"} or query.match not in {"whole_word", "substring"}:
            raise ValueError("Search mode or match rule is invalid.")
        if not isinstance(query.text, str) or len(query.text) > 4096:
            raise ValueError("Query must contain at most 4096 original characters.")
        if query.mode == "keywords":
            if (query.text or not 1 <= len(query.terms) <= 16
                    or any(not isinstance(t, str) or not 1 <= len(t) <= 2000 for t in query.terms)
                    or sum(map(len, query.terms)) > 2000):
                raise ValueError("keywords requires explicit bounded terms and an empty query.")
        elif not query.text or query.terms:
            raise ValueError("exact requires a query and no keyword terms.")
        if query.original_text is not None and query.original_text != query.text:
            raise ValueError("Original query must remain unchanged.")
        if (type(query.limit) is not int or not 1 <= query.limit <= 1000
                or type(query.offset) is not int or query.offset < 0):
            raise ValueError("Search limit or offset is invalid.")
        program = ripgrep.executable()
        with tempfile.TemporaryDirectory(prefix="monoplanner-search-") as directory:
            database = sqlite3.connect(Path(directory) / "matches.sqlite3")
            try:
                database.execute("PRAGMA cache_size=-4096")
                database.execute("PRAGMA temp_store=FILE")
                database.execute(f"PRAGMA max_page_count={self.index.config.max_index_bytes // 4096}")
                database.execute(
                    "CREATE TABLE hits (source TEXT, start INTEGER, end INTEGER, "
                    "PRIMARY KEY(source,start,end)) WITHOUT ROWID"
                )
                return self._search(program, Path(directory), database, query)
            finally:
                database.close()

    def _search(self, program, directory, database, query):
        started = time.monotonic()
        headers = self.index.sources(query)
        originals, paths, frequencies = {}, defaultdict(list), defaultdict(Counter)
        unavailable, excluded, reasons = [], [], []
        if self.index.missing_sources(query):
            reasons.append("source_not_indexed")
        if not query.source_versions:
            reasons.append("no_authorized_sources")

        def failed(header, error):
            code = ("source_read_memory_budget" if isinstance(error, SourceReadBudgetExceeded)
                    else getattr(error, "code", "source_unavailable:" + type(error).__name__))
            reasons.append(code)
            unavailable.append({**getattr(error, "details", {}), "path": header.path,
                                "source_id": header.source_id, "reason": code})

        for number, header in enumerate(headers):
            check_interrupted(pause_only=True)
            try:
                path = self.source_path(header.source_id, header.version) if self.source_path else None
                if path is None:
                    if self.read_source is None:
                        raise ValueError("An authorized source reader was not supplied.")
                    text = self.read_source(header.source_id, header.version)
                    if not isinstance(text, str):
                        raise ValueError("The authorized source reader did not return text.")
                    guard_source_read(len(text.encode("utf-8")), source_id=header.source_id, path=header.path)
                    path = directory / str(number)
                    path.write_bytes(text.encode("utf-8"))
                    del text
                original = ripgrep.Original(
                    path, header.source_sha256, max_bytes=self.index.config.max_document_bytes
                )
                if original.binary:
                    excluded.append({"path": header.path, "reason": "binary_source_skipped"})
                    continue
                originals[header.source_id] = (header, original)
                paths[os.path.normcase(str(Path(path).resolve()))].append(header.source_id)
            except ModelInterrupted:
                raise
            except (OSError, ValueError) as error:
                failed(header, error)

        terms = tuple(dict.fromkeys(query.terms)) if query.mode == "keywords" else (query.text,)
        encoded = [term.encode("utf-8") for term in terms]
        batch, size, batches = [], 0, []
        # Windows 命令行有长度上限；只传入授权清单中的显式文件，不让 rg 自行递归。
        for path in sorted(paths):
            if batch and (size + len(path) > 20000 or len(batch) >= 64):
                batches.append(batch)
                batch, size = [], 0
            batch.append(path)
            size += len(path) + 3
        if batch:
            batches.append(batch)
        try:
            for batch in batches:
                for row in ripgrep.rows(
                    program, batch, query, timeout=self.index.config.scan_timeout_seconds
                ):
                    if row.get("type") != "match":
                        continue
                    data = row["data"]
                    name = os.path.normcase(str(Path(data["path"]["text"]).resolve()))
                    if name not in paths:
                        raise ripgrep.SearchError("ripgrep_unauthorized_source")
                    block = data["lines"]
                    raw = (block["text"].encode("utf-8") if "text" in block
                           else base64.b64decode(block["bytes"], validate=True))
                    for match in data["submatches"]:
                        relative = match["start"]
                        start = data["absolute_offset"] + relative
                        for identity in paths[name]:
                            original = originals[identity][1]
                            tail = raw[relative:relative + max(map(len, encoded))]
                            if len(tail) < max(map(len, encoded)):
                                tail = original.bytes_at(start, max(map(len, encoded)))
                            found = [
                                (i, start + len(term)) for i, term in enumerate(encoded)
                                if tail.startswith(term) and (
                                    query.match == "substring" or ripgrep.literal_boundaries(
                                        original, start, start + len(term), terms[i]
                                    )
                                )
                            ]
                            for term, end in found:
                                if not 0 <= start <= end <= original.size:
                                    raise ripgrep.SearchError("ripgrep_invalid_range")
                                frequencies[identity][term] += 1
                                database.execute(
                                    "INSERT OR IGNORE INTO hits VALUES(?,?,?)", (identity, start, end)
                                )
                database.commit()
        except ModelInterrupted:
            raise
        except (OSError, ValueError, sqlite3.Error) as error:
            reasons.append(getattr(error, "code", "ripgrep_scan_failed:" + type(error).__name__))
            unavailable.append({"reason": reasons[-1], "detail": str(error)[:4096]})

        # rg 退出不代表原件未被更换；交付前重新核验实际字节。
        for identity, (header, original) in list(originals.items()):
            try:
                original.verify()
            except ModelInterrupted:
                raise
            except (OSError, ValueError) as error:
                failed(header, error)
                database.execute("DELETE FROM hits WHERE source=?", (identity,))
                del originals[identity]
                frequencies.pop(identity, None)
        snapshot = [(h.source_id, h.version, h.source_sha256) for h in headers]
        binding = hashlib.sha256(canonical({
            "format": FORMAT, "scope": self.index.scope.key, "query": query.text,
            "terms": terms, "mode": query.mode, "match": query.match,
            "required_entities": query.required_entities, "required_kinds": query.required_kinds,
            "source_versions": dict(query.source_versions), "snapshot": snapshot,
            "source_priorities": dict(query.source_priorities),
            "excluded_paths": query.excluded_paths, "reasons": reasons, "excluded": excluded,
        }).encode("utf-8")).hexdigest()
        offset = self._offset(query, binding)
        total = database.execute("SELECT count(*) FROM hits").fetchone()[0]
        scan_seconds = time.monotonic() - started
        ranking_started = time.monotonic()
        count = len(originals)
        average = sum(o.chars for _, o in originals.values()) / max(1, count)
        document_frequency = Counter()
        for values in frequencies.values():
            document_frequency.update(values.keys())
        scores = {}
        for identity, (_, original) in originals.items():
            scores[identity] = sum(
                math.log(1 + (count - document_frequency[t] + 0.5)
                         / (document_frequency[t] + 0.5))
                * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * original.chars / max(1, average)))
                for t, tf in frequencies[identity].items()
            )
        ordered = sorted(
            originals, key=lambda identity: (
                -scores[identity], query.source_priorities.get(identity, 0), originals[identity][0].path.replace("\\", "/"), identity
            )
        )
        selected, hits, skipped = [], [], 0
        timing = {
            "engine": "ripgrep", "ranking": "bm25", "scan_seconds": scan_seconds,
            "ranking_seconds": time.monotonic() - ranking_started,
            "ranked_files": sum(bool(frequencies[identity]) for identity in ordered),
            "ranked_file_preview": [
                {"path": originals[identity][0].path, "score": scores[identity]}
                for identity in ordered if frequencies[identity]
            ][:16],
        }

        def pack():
            return self._payload(
                query, selected, hits, reasons, unavailable, excluded, binding, offset, total, timing
            )

        full = False
        for identity in ordered:
            header, original = originals[identity]
            for start, end in database.execute(
                "SELECT start,end FROM hits WHERE source=? ORDER BY start,end", (identity,)
            ):
                if skipped < offset:
                    skipped += 1
                    continue
                if len(hits) >= query.limit:
                    full = True
                    break
                check_interrupted(pause_only=True)
                try:
                    window = original.window(start, end)
                except (OSError, ValueError) as error:
                    failed(header, error)
                    full = True
                    break
                hit = {
                    **window, "source_id": identity, "path": header.path, "version": header.version,
                    "source_sha256": header.source_sha256, "range_unit": "unicode_codepoints",
                    "score": scores[identity],
                }
                chunk = replace(
                    header, start=hit["start"], end=hit["end"], parent_start=hit["start"],
                    parent_end=hit["end"], text=hit["text"], line_start=hit["line"],
                    line_end=hit["line_end"],
                )
                previous = selected
                selected = self._merge(selected, chunk)
                hits.append(hit)
                packed = canonical(pack())
                if (self.token_count(packed) > self.index.config.message_token_limit
                        or len(packed.encode("utf-8")) * 4 > CANDIDATE_MEMORY_BYTES):
                    selected = previous
                    hits.pop()
                    if not hits:
                        raise ValueError("The original match cannot fit the message token limit.")
                    full = True
                    break
            if full:
                break
        payload = pack()
        if self.token_count(canonical(payload)) > self.index.config.message_token_limit:
            raise ValueError("Required retrieval metadata cannot fit the message token limit.")
        return RetrievalResult(
            canonical(payload), tuple(selected), not payload["complete"],
            tuple(dict.fromkeys(reasons)), tuple(payload["missing_entities"]),
            tuple(payload["missing_kinds"]), (),
            {"rg": tuple(item.id for item in selected)}, query.mode == "exact" and bool(hits),
        )

    def _payload(self, query, selected, hits, reasons, unavailable, excluded, binding, offset,
                 total, timing):
        compact = []
        for hit in hits:
            excerpt = next(item for item in selected
                           if item.source_id == hit["source_id"]
                           and item.start <= hit["start"] <= hit["end"] <= item.end)
            compact.append({**{k: v for k, v in hit.items() if k != "text"},
                            "result_id": excerpt.id})
        evidence = "\n".join(item.text for item in selected)
        paths = {item.path.replace("\\", "/") for item in selected}
        more = offset + len(hits) < total
        return {
            "format": FORMAT, "query": query.text, "effective_query": query.text,
            "mode": query.mode, "terms": query.terms, "match": query.match,
            "ranking": "literal_bm25_k1_1.2_b_0.75",
            "hits": compact, "results": [
                {k: v for k, v in item.payload().items()
                 if k not in {"parent_start", "parent_end"}
                 and (k not in {"symbol", "run_id", "timestamp"} or v)}
                for item in selected
            ],
            "total_matches": total, "scan_complete": not reasons,
            "complete": not reasons and not more, "retrieval_incomplete": bool(reasons or more),
            "truncated": more or any(hit["truncated"] for hit in hits),
            "next_cursor": self._cursor(binding, offset + len(hits)) if more else "",
            "reasons": list(dict.fromkeys(reasons)), "continuation": [],
            "unavailable_sources": unavailable[:8], "unavailable_source_count": len(unavailable),
            "excluded_sources": excluded[:8], "excluded_source_count": len(excluded),
            "exact_current_version_hit": query.mode == "exact" and bool(hits),
            "missing_entities": [e for e in query.required_entities
                                 if e not in evidence and e.replace("\\", "/") not in paths],
            "missing_kinds": [k for k in query.required_kinds
                              if k not in {item.kind for item in selected}],
            "search_metrics": timing,
            "notice": "Search excerpts, not complete files. Each hit references its result_id; "
                      "read the original path and line range for context. Incomplete scans do not prove absence.",
        }

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
