"""双路召回、有限重排与原文补读；模型不可用时明确返回降级状态。"""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import replace
import hashlib
import math

from agent.model.errors import ModelInterrupted
from agent.rag.index import Index
from agent.rag.types import Query, RetrievalResult, TokenCount, canonical
from agent.task.control import check_interrupted


class Retriever:
    def __init__(self, index: Index, token_count: TokenCount, *, vector=None,
                 reranker=None, read_source=None):
        self.index = index
        self.token_count = token_count
        self.vector = vector
        self.reranker = reranker
        self.read_source = read_source

    def retrieve(self, query: Query) -> RetrievalResult:
        check_interrupted(pause_only=True)
        if query.intent not in {"locate", "diagnostic", "audit"}:
            raise ValueError("Query intent must be locate, diagnostic, or audit.")
        config = self.index.config
        reasons = []
        if not query.source_versions:
            reasons.append("no_authorized_sources")
        route_ids = {}
        rerank_scores = ()
        exact = self.index.exact(query) if query.intent == "locate" else []

        if exact:
            ordered = exact
            route_ids["exact"] = tuple(chunk.id for chunk in exact)
        else:
            # 两路可以并行；本地模型自己的共享资源槽控制加载和推理。
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-recall") as pool:
                lexical_job = pool.submit(copy_context().run, self.index.search, query)
                vector_job = (
                    pool.submit(
                        copy_context().run, self.vector.search,
                        query, self.index, config.recall_top_k,
                    )
                    if self.vector is not None and query.source_versions else None
                )
                lexical = lexical_job.result()
                vector_pairs = []
                if vector_job is None:
                    reasons.append("vector_unavailable")
                else:
                    try:
                        received = list(vector_job.result())
                        if len(received) > config.recall_top_k or not all(
                            isinstance(item, (tuple, list)) and len(item) == 2
                            and isinstance(item[0], str) and math.isfinite(item[1])
                            for item in received
                        ):
                            raise ValueError("Invalid or unbounded vector retrieval response.")
                        vector_pairs = sorted(received, key=lambda item: (-item[1], item[0]))
                    except ModelInterrupted:
                        raise
                    except Exception as error:
                        reasons.append(f"vector_unavailable:{type(error).__name__}")

            vector_chunks = self.index.get([item[0] for item in vector_pairs], query)
            lexical_chunks = [item[0] for item in lexical]
            route_ids["bm25"] = tuple(chunk.id for chunk in lexical_chunks)
            route_ids["vector"] = tuple(chunk.id for chunk in vector_chunks)
            all_chunks = {chunk.id: chunk for chunk in [*lexical_chunks, *vector_chunks]}
            weights = {"bm25": config.bm25_weight, "vector": config.vector_weight}
            if any(reason.startswith("vector_unavailable") for reason in reasons):
                weights["bm25"] = 1.0
            scores = {}
            for route, ids in route_ids.items():
                for rank, chunk_id in enumerate(dict.fromkeys(ids), 1):
                    scores[chunk_id] = scores.get(chunk_id, 0) + weights[route] / (60 + rank)
            ordered = [all_chunks[item] for item in sorted(scores, key=lambda x: (-scores[x], x))]
            ordered = ordered[:config.recall_top_k]

            if ordered and query.intent in {"diagnostic", "audit"}:
                if self.reranker is None:
                    reasons.append("reranker_unavailable")
                else:
                    try:
                        candidates = ordered[:config.rerank_k]
                        values = tuple(self.reranker.score(query.text, candidates))
                        if len(values) != len(candidates) or not all(
                            math.isfinite(value) for value in values
                        ):
                            raise ValueError("Reranker returned invalid scores.")
                        ranked = sorted(zip(candidates, values), key=lambda x: (-x[1], x[0].id))
                        ordered = [item[0] for item in ranked] + ordered[config.rerank_k:]
                        rerank_scores = tuple(item[1] for item in ranked)
                    except ModelInterrupted:
                        raise
                    except Exception as error:
                        reasons.append(f"reranker_unavailable:{type(error).__name__}")

        return self._pack(query, ordered, reasons, route_ids, bool(exact), rerank_scores)

    def _pack(self, query, ordered, reasons, route_ids, exact_hit, rerank_scores):
        config = self.index.config
        selected = []
        continuation = []
        sources = {}
        source_bytes = 0

        def original(chunk):
            nonlocal source_bytes

            key = (chunk.source_id, chunk.version, chunk.source_sha256)
            if key not in sources:
                if self.read_source is None:
                    raise ValueError("An authorized source reader was not supplied.")
                text = self.read_source(chunk.source_id, chunk.version)
                if not isinstance(text, str):
                    raise ValueError("The authorized source reader did not return text.")
                encoded_size = len(text.encode("utf-8"))
                if encoded_size > config.max_document_bytes:
                    raise ValueError("The source exceeds the bounded indexing input size.")
                if (
                    hashlib.sha256(text.encode("utf-8")).hexdigest() != key[2]
                ):
                    raise ValueError("The authorized source version no longer matches its hash.")
                if not (
                    0 <= chunk.parent_start <= chunk.start
                    <= chunk.end <= chunk.parent_end <= len(text)
                ):
                    raise ValueError("Source ranges do not match the archived original.")
                if text[chunk.start:chunk.end] != chunk.text:
                    raise ValueError("The indexed range no longer matches the source.")
                if source_bytes + encoded_size * 4 > 32 * 1024 * 1024:
                    sources.clear()
                    source_bytes = 0
                sources[key] = text
                source_bytes += encoded_size * 4
            return sources[key]

        def make_result(items, pending, extra_reasons=()):
            evidence = "\n".join(item.text for item in items)
            paths = {item.path.replace("\\", "/") for item in items if item.path}
            symbols = {item.symbol for item in items if item.symbol}
            # 来源路径和所属符号也是已交付的精确身份；哈希、版本及任意元数据不充当证据。
            missing_entities = tuple(
                entity for entity in query.required_entities
                if entity not in evidence
                and entity.replace("\\", "/") not in paths
                and entity not in symbols
            )
            kinds = {item.kind for item in items}
            missing_kinds = tuple(kind for kind in query.required_kinds if kind not in kinds)
            diagnostics = tuple(dict.fromkeys([*reasons, *extra_reasons]))
            payload = {
                "retrieval_incomplete": bool(diagnostics or pending),
                "reasons": diagnostics,
                "exact_current_version_hit": exact_hit and bool(items),
                "missing_entities": missing_entities,
                "missing_kinds": missing_kinds,
                "results": [item.payload() for item in items],
                "continuation": pending,
                "notice": "Retrieved evidence is not proof of a conclusion or a complete corpus scan.",
            }
            return canonical(payload), diagnostics, missing_entities, missing_kinds

        def fits(items, pending, extra_reasons=()):
            message = make_result(items, pending, extra_reasons)[0]
            return self.token_count(message) <= config.message_token_limit

        def marker(chunk, reason, start=None, end=None):
            return {
                "source_id": chunk.source_id, "version": chunk.version,
                "source_sha256": chunk.source_sha256,
                "path": chunk.path, "range_unit": "unicode_codepoints",
                "read_line_offset": max(0, chunk.line_start - 1),
                "start": chunk.start if start is None else start,
                "end": chunk.end if end is None else end, "reason": reason,
            }

        for position, chunk in enumerate(ordered):
            check_interrupted(pause_only=True)
            if query.source_versions.get(chunk.source_id) != chunk.version:
                continue
            candidate = chunk
            extra = []
            if self.read_source is not None:
                try:
                    original(chunk)
                except ModelInterrupted:
                    raise
                except Exception as error:
                    reasons.append(f"source_unavailable:{type(error).__name__}")
                    continuation.append(marker(chunk, "source_unavailable"))
                    continue
            if (chunk.parent_start, chunk.parent_end) != (chunk.start, chunk.end):
                try:
                    text = original(chunk)
                    # 无结构文本的 parent 是整个输入，不自动扩展为整份文件。
                    structured = bool(chunk.symbol) or chunk.kind in {
                        "markdown", "documentation", "document", "log", "test_log",
                        "compile_error", "tool_result", "json",
                    }
                    if structured:
                        candidate = replace(
                            chunk, start=chunk.parent_start, end=chunk.parent_end,
                            text=text[chunk.parent_start:chunk.parent_end],
                            line_start=text.count("\n", 0, chunk.parent_start) + 1,
                            line_end=text.count(
                                "\n", 0, max(chunk.parent_start, chunk.parent_end - 1)
                            ) + 1,
                        )
                    else:
                        extra.append(marker(
                            chunk, "parent_not_expanded", chunk.parent_start, chunk.parent_end
                        ))
                except ModelInterrupted:
                    raise
                except Exception as error:
                    # 损坏/失效原件不交付其旧缓存；缺失 reader 仍可交付已归档片段。
                    if self.read_source is not None:
                        reasons.append(f"source_unavailable:{type(error).__name__}")
                        continuation.append(marker(chunk, "source_unavailable"))
                        continue
                    extra.append(marker(
                        chunk, "parent_reader_unavailable", chunk.parent_start, chunk.parent_end
                    ))

            merged = self._merge(selected, candidate, sources)
            if (candidate.start, candidate.end) != (chunk.start, chunk.end) and not fits(
                merged, [*continuation, *extra]
            ):
                candidate = chunk
                extra = [marker(
                    chunk, "parent_exceeds_message_budget", chunk.parent_start, chunk.parent_end
                )]
                merged = self._merge(selected, candidate, sources)
            if fits(merged, [*continuation, *extra]):
                selected = merged
                continuation.extend(extra)
                continue

            # 保留一个确定性的续读位置和剩余数量，而非为每个未返回块堆积元数据。
            pending = [*continuation, {
                **marker(chunk, "message_budget"),
                "remaining_candidates": len(ordered) - position,
            }]
            while selected and not fits(selected, pending):
                removed = selected.pop()
                pending[-1] = {
                    **marker(removed, "message_budget"),
                    "remaining_candidates": len(ordered) - position + 1,
                }
            continuation = pending
            break

        message, diagnostics, entities, kinds = make_result(selected, continuation)
        if self.token_count(message) > config.message_token_limit:
            raise ValueError("Required retrieval metadata cannot fit the message token limit.")
        return RetrievalResult(
            message, tuple(selected), bool(diagnostics or continuation), diagnostics,
            entities, kinds, tuple(continuation), route_ids,
            exact_hit and bool(selected), rerank_scores,
        )

    @staticmethod
    def _merge(selected, candidate, sources):
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
