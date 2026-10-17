"""版本化 Python 静态语法缓存；只解析源码，不导入或执行。"""

import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import uuid

from agent.source.syntax import _module, _parse
from agent.source.snapshots import _link, _reason, _selected_files
from agent.source.scope import project_scope as resolve_scope


SCHEMA = 1
MAX_FILES = 3000
MAX_FILE_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024


class RepositoryNavigator:
    def __init__(self, repo_root, cache_dir, *, project_scope=None):
        self.root = Path(repo_root).resolve()
        self.project_scope = project_scope
        cache = Path(cache_dir).resolve()
        if cache == self.root or self.root in cache.parents:
            raise ValueError("导航缓存必须位于被导航仓库外。")
        identity = hashlib.sha256(str(self.root).encode()).hexdigest()[:24]
        self.cache_dir = cache / identity
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "index.json"
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
            if self.project_scope is None:
                self.project_scope = resolve_scope(self.root)
            selected = {relative.as_posix() for relative, skip, tracked in
                        _selected_files(self.root, project_scope=self.project_scope)[1]
                        if not skip and (tracked or not _reason(relative))}
            for directory, dirs, names in os.walk(self.root, followlinks=False):
                base = Path(directory)
                dirs[:] = sorted(
                    d
                    for d in dirs
                    if not _reason((base / d).relative_to(self.root) / "_") and not _link(base / d)
                    and not self.project_scope.excludes((base / d).relative_to(self.root))
                )
                for name in sorted(names):
                    if not name.endswith(".py"):
                        continue
                    path = base / name
                    relative = path.relative_to(self.root)
                    if relative.as_posix() not in selected or _reason(relative) or _link(path):
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
