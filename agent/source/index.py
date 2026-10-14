"""有界的 Python 词法导航；只解析源码，不导入或执行。"""

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time
import uuid

from agent.source import excerpts
from agent.source.syntax import _module, _tokens, _is_test, _parse, query_identifiers, query_focus
from agent.source.snapshots import _link, _reason


SCHEMA = 1
MAX_FILES = 3000
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024


class RepositoryNavigator:
    def __init__(self, repo_root, cache_dir):
        self.root = Path(repo_root).resolve()
        cache = Path(cache_dir).resolve()
        if cache == self.root or self.root in cache.parents:
            raise ValueError("导航缓存必须位于被导航仓库外。")
        identity = hashlib.sha256(str(self.root).encode()).hexdigest()[:24]
        self.cache_dir = cache / identity
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "index.json"
        self.events_path = self.cache_dir / "queries.jsonl"
        self._lock = threading.RLock()
        self.entries, self.stats, self.version = {}, {}, ""
        try:
            saved = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if (
                isinstance(saved, dict)
                and isinstance(saved.get("entries"), dict)
                and saved.get("schema") == SCHEMA
                and saved.get("python") == list(sys.version_info[:2])
                and saved.get("root") == str(self.root)
            ):
                self.entries = saved["entries"]
        except (OSError, ValueError, KeyError, TypeError):
            pass

    @classmethod
    def from_prepared(cls, view, prepared, cache_dir):
        """使用归档版本及已有解析结果，不实时刷新。"""
        if prepared['version'] != view.version or any(
            view.files.get(name) != entry['digest'] for name, entry in prepared['entries'].items()
        ):
            raise ValueError('Prepared navigation does not match the recorded source')
        instance = cls(view.origin_root, cache_dir)
        instance._frozen_view = view
        instance.entries = prepared['entries']
        instance.version = view.version
        instance.stats = {k: prepared[k] for k in ('parsed_files', 'reused_files', 'incomplete')}
        instance.stats.update(indexed_files=len(instance.entries), source='SourceVersions')
        return instance

    # ========================================================================
    # 刷新文件版本与静态语法索引
    # ========================================================================
    def refresh(self):
        """重新计算实时文件字节的哈希，或保留明确固定的归档版本。"""
        if getattr(self, "_frozen_view", None) is not None:
            return dict(self.stats)
        with self._lock:
            start = time.monotonic()
            entries, skipped, reused, parsed, total_bytes = {}, [], 0, 0, 0
            stopped = False
            for directory, dirs, names in os.walk(self.root, followlinks=False):
                base = Path(directory)
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if not _reason((base / d).relative_to(self.root) / "_") and not _link(base / d)
                )
                for name in sorted(names):
                    if not name.endswith(".py"):
                        continue
                    path = base / name
                    relative = path.relative_to(self.root)
                    if _reason(relative) or _link(path):
                        continue
                    if len(entries) >= MAX_FILES or total_bytes >= MAX_TOTAL_BYTES:
                        stopped = True
                        break
                    try:
                        before = path.stat()
                        if (
                            before.st_size > MAX_FILE_BYTES
                            or total_bytes + before.st_size > MAX_TOTAL_BYTES
                        ):
                            skipped.append(
                                {"path": relative.as_posix(), "reason": "source size limit"}
                            )
                            continue
                        with path.open("rb") as stream:
                            data = stream.read(MAX_FILE_BYTES + 1)
                        after = path.stat()
                        if len(data) > MAX_FILE_BYTES or (before.st_mtime_ns, before.st_size) != (
                            after.st_mtime_ns,
                            after.st_size,
                        ):
                            raise ValueError("source changed during read")
                        total_bytes += len(data)
                        digest = hashlib.sha256(data).hexdigest()
                        key = relative.as_posix()
                        old = self.entries.get(key, {})
                        if (
                            isinstance(old, dict)
                            and old.get("digest") == digest
                            and all(
                                isinstance(old.get(k), list)
                                for k in ("symbols", "imports", "calls")
                            )
                            and "module" in old
                            and "error" in old
                        ):
                            entry = old
                            reused += 1
                        else:
                            parsed += 1
                            try:
                                entry = _parse(data, key)
                            except (SyntaxError, UnicodeError, ValueError, RecursionError) as exc:
                                entry = {
                                    "module": _module(key),
                                    "doc": "",
                                    "symbols": [],
                                    "imports": [],
                                    "calls": [],
                                    "error": f"{type(exc).__name__}: {str(exc)[:220]}",
                                }
                            entry["digest"] = digest
                        entries[key] = entry
                    except (OSError, ValueError) as exc:
                        skipped.append({"path": relative.as_posix(), "reason": str(exc)[:180]})
                if stopped:
                    break
            self.entries = entries
            self.version = hashlib.sha256(
                json.dumps([(p, e["digest"]) for p, e in sorted(entries.items())]).encode()
            ).hexdigest()
            errors = [{"path": p, "reason": e["error"]} for p, e in entries.items() if e["error"]]
            self.stats = {
                "indexed_files": len(entries),
                "parsed_files": parsed,
                "reused_files": reused,
                "hashed_bytes": total_bytes,
                "incomplete": stopped or bool(skipped) or bool(errors),
                "skipped": skipped[:4],
                "parse_errors": errors[:3],
                "elapsed_ms": round((time.monotonic() - start) * 1000, 2),
            }
            temporary = self.cache_path.with_name(uuid.uuid4().hex + ".tmp")
            try:
                temporary.write_text(
                    json.dumps(
                        {
                            "schema": SCHEMA,
                            "python": list(sys.version_info[:2]),
                            "root": str(self.root),
                            "entries": entries,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                temporary.replace(self.cache_path)
            finally:
                temporary.unlink(missing_ok=True)
            return dict(self.stats)

    # ========================================================================
    # 关联符号与查询候选
    # ========================================================================
    def _related(self, path, entry):
        targets = {entry["module"] + "." + s["name"] for s in entry["symbols"]}
        leaves = {s["name"].split(".")[-1] for s in entry["symbols"]}
        tests, callers = [], []
        for other, data in self.entries.items():
            if other == path:
                continue
            evidence = []
            for call in data["calls"]:
                if targets.intersection(call["targets"]):
                    evidence.append(
                        (
                            3,
                            call["line"],
                            "调用语法 " + call["name"] + " 指向该模块符号（静态候选）",
                        )
                    )
                elif (
                    "." not in call["name"]
                    and call["name"] in leaves
                    and not any(
                        item["binding"] == call["name"]
                        and item["target"].split(".")[0] != entry["module"].split(".")[0]
                        for item in data["imports"]
                    )
                ):
                    evidence.append(
                        (1, call["line"], "同名调用 " + call["name"] + "，绑定尚未确认")
                    )
            for item in data["imports"]:
                if item["target"] in targets or item["target"] == entry["module"]:
                    evidence.append((2, item["line"], "导入 " + item["target"]))
            if evidence:
                best = max(evidence)
                (tests if _is_test(other) else callers).append((best[0], other, best[1], best[2]))

        def compact(items, limit):
            return [
                {"path": p, "line": line, "evidence": text, "certainty": "static_candidate"}
                for _, p, line, text in sorted(items, key=lambda x: (-x[0], x[1]))[:limit]
            ]

        related_tests = compact(tests, 3)
        fixtures = set()
        for test in related_tests:
            parent = PurePosixPath(test["path"]).parent
            for directory in (parent, *parent.parents):
                candidate = (directory / "conftest.py").as_posix()
                if candidate in self.entries:
                    fixtures.add(candidate)
        modules = {e["module"]: p for p, e in self.entries.items()}
        dependencies = []
        for item in entry["imports"]:
            target = item["target"]
            while target and target not in modules:
                target = target.rpartition(".")[0]
            if target and modules[target] != path:
                candidate = {
                    "path": modules[target],
                    "evidence": "导入 " + item["target"],
                    "from_line": item["line"],
                }
                if candidate not in dependencies:
                    dependencies.append(candidate)
        return {
            "tests": related_tests,
            "callers": compact(callers, 2),
            "dependencies": dependencies[:2],
            "fixture_files": sorted(fixtures)[:3],
        }

    def _query(self, query, path, limit):
        normalized = query.replace("\\", "/").lower()
        terms = _tokens(query)
        identifiers = query_identifiers(query)
        title_identifiers, primary = query_focus(query)
        ranked = []
        for file, entry in self.entries.items():
            if path and file != path and not file.startswith(path.rstrip("/") + "/"):
                continue
            exact_path = file.lower() in normalized
            symbol_hits = [s for s in entry["symbols"] if s["name"].split(".")[-1] in identifiers]
            symbol_hits.sort(
                key=lambda s: (
                    s["name"].split(".")[-1] != primary,
                    s["name"].split(".")[-1] not in title_identifiers,
                )
            )
            title_hits = {s["name"].split(".")[-1] for s in symbol_hits} & title_identifiers
            text = (
                file
                + " "
                + entry["doc"]
                + " "
                + " ".join(s["name"] + " " + s["doc"] for s in entry["symbols"])
            )
            matched = sorted(terms & _tokens(text))
            if query and not (exact_path or symbol_hits or matched or path):
                continue
            reasons = ["路径命中 " + file] if exact_path or path else []
            reasons += ["定义 " + s["name"] for s in symbol_hits[:3]]
            if matched:
                reasons.append("路径、定义或文档词匹配：" + ", ".join(matched[:8]))
            if not reasons:
                reasons = ["概览中的模块；尚未匹配具体需求"]
            score = (
                int(exact_path),
                int(primary in title_hits),
                len(title_hits),
                len({s["name"].split(".")[-1] for s in symbol_hits}),
                len(matched),
                int(not _is_test(file)),
            )
            ranked.append((score, file, entry, reasons, symbol_hits))
        results = []
        for _, file, entry, reasons, hits in sorted(
            ranked, key=lambda x: (tuple(-v for v in x[0]), x[1])
        )[:limit]:
            # 按符号查询时，不能将同模块中无关方法纳入测试选择；
            # 按类查询仍包含该类的方法。
            children = [
                s
                for s in entry["symbols"]
                if any(
                    hit["kind"] == "class" and s["name"].startswith(hit["name"] + ".")
                    for hit in hits
                )
            ]
            scoped = hits + children if hits else entry["symbols"]
            symbols = scoped[:5]
            results.append(
                {
                    "path": file,
                    "module": entry["module"],
                    "reasons": reasons,
                    "symbols": symbols,
                    "parse_error": entry["error"],
                    "related": self._related(file, {**entry, "symbols": scoped}),
                    "read_hint": {
                        "mode": "read",
                        "path": file,
                        "offset": max(0, symbols[0]["line"] - 1) if symbols else 0,
                        "limit": 60,
                    },
                }
            )
        return {
            "source_version": self.version,
            "query": query,
            "candidates": results,
            "more_candidates": len(ranked) > len(results),
            "index": dict(self.stats),
            "note": "词法与静态结构导航，不是语义检索或完整调用图；没有命中可换代码标识符、路径，或使用 search_code。",
        }

    # ========================================================================
    # 有界查询结果与源码摘录
    # ========================================================================
    @staticmethod
    def _pack(value, maximum):
        value = copy.deepcopy(value)
        value["response_omitted"] = False
        while len(json.dumps(value, ensure_ascii=False)) > maximum:
            value["response_omitted"] = True
            # 保留候选覆盖范围，不修改底层 AST 缓存。
            compacted = False
            for candidate in sorted(
                value.get("candidates", []), key=lambda c: len(json.dumps(c)), reverse=True
            ):
                if len(candidate.get("symbols", [])) > 1:
                    candidate["symbols"].pop()
                    compacted = True
                    break
                for relations in candidate.get("related", {}).values():
                    if isinstance(relations, list) and len(relations) > 1:
                        relations.pop()
                        compacted = True
                        break
                if compacted:
                    break
            if compacted:
                continue
            if value.get("candidates"):
                value["candidates"].pop()
                value["more_candidates"] = True
            elif value.get("index", {}).get("skipped") or value.get("index", {}).get(
                "parse_errors"
            ):
                value["index"]["skipped"] = []
                value["index"]["parse_errors"] = []
            else:
                # 路径和错误元数据也可能很长；保留有效且明确的后备结果，
                # 不能从序列化 JSON 中间直接截断。
                return {
                    "source_version": value.get("source_version", ""),
                    "candidates": [],
                    "more_candidates": True,
                    "response_omitted": True,
                    "note": "导航结果超出长度限制，请指定更具体的相对路径或代码标识符。",
                }
        return value

    def query(self, query="", path="", limit=5):
        if (
            not isinstance(query, str)
            or len(query) > 1200
            or type(limit) is not int
            or not 1 <= limit <= 6
        ):
            raise ValueError("query 最多 1200 字符，limit 必须为 1 到 6。")
        if not isinstance(path, str) or len(path) > 600:
            raise ValueError("path 必须是最多 600 字符的仓库相对路径。")
        path = path.replace("\\", "/")
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or ":" in path:
            raise ValueError("path 必须是仓库内相对文件或目录，不得越界。")
        path = PurePosixPath(path).as_posix() if path else ""
        if path == ".":
            path = ""
        with self._lock:
            self.refresh()
            value = self._pack(self._query(query, path, limit), 10_000)
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "query": query,
                            "path": path,
                            "source_version": self.version,
                            "selected_paths": [c["path"] for c in value["candidates"]],
                            "index": self.stats,
                            "response_chars": len(json.dumps(value, ensure_ascii=False)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return value

    def inspect_code(self, query, path="", limit=1):
        """各角色共用同一套确定性的查找、读取和测试示例流程。"""
        if type(limit) is not int or not 1 <= limit <= 2 or not query.strip():
            raise ValueError("query 必须非空，limit 必须为 1 到 2；只需骨架时用 mode=map。")
        with self._lock:
            located = self.query(query, path, limit)
            items = []
            for item in located["candidates"]:
                line = item["read_hint"]["offset"] + 1
                related = item["related"]
                tests = related["tests"][:2]
                items.append(
                    {
                        "path": item["path"],
                        "reasons": item["reasons"][:4],
                        "symbols": item["symbols"][:2],
                        "definition": self._code_excerpt(item["path"], line, 70, 2400),
                        "related_tests": tests,
                        "test_example": (
                            self._code_excerpt(tests[0]["path"], tests[0]["line"], 45, 1600)
                            if tests
                            else None
                        ),
                        "fixture_files": related["fixture_files"][:2],
                        "dependencies": related["dependencies"][:2],
                        "callers": related["callers"][:2],
                    }
                )
            result = self._pack(
                {
                    "source_version": self.version,
                    "candidates": items,
                    "more_candidates": located["more_candidates"],
                    "note": "片段来自当前绑定仓库；静态关联不等于运行覆盖。标记 truncated 时按 read_hint 补查具体缺口。",
                },
                10_000,
            )
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "event": "inspect_code",
                            "source_version": self.version,
                            "selected_paths": [i["path"] for i in result["candidates"]],
                            "response_chars": len(json.dumps(result, ensure_ascii=False)),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return result

    def _code_excerpt(self, name, line, count, max_chars):
        return excerpts._code_excerpt(self, name, line, count, max_chars)
