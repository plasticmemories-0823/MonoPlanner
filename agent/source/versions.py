"""管理不可变源码版本，并针对其原始字节记录操作日志。"""

import hashlib
import json
from pathlib import Path
from threading import RLock
import time
from agent.source.snapshots import (
    _reason,
    _safe_file,
    _selected_files,
    _inspect_file,
    _within,
    _SECRET_REASONS,
)


def _save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


class SourceVersions:
    def __init__(self, environment, directory, baseline=None):
        self.environment, self.baseline = environment, baseline
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "blobs").mkdir(exist_ok=True)
        (self.root / "states").mkdir(exist_ok=True)
        self.lock, self.generation = RLock(), 0
        self.current = None
        self.record("start")

    def _source(self):
        started = time.monotonic()
        root, files, excluded = self.environment.repo_root, {}, {}
        hashed_bytes = hashed_files = 0
        for relative, skip, tracked in _selected_files(
            root, getattr(self.environment, "runtime_files", ())
        )[1]:
            path = root / relative
            if (
                _within(path.resolve(), self.environment.artifact_root.resolve())
                or not path.exists()
            ):
                continue
            reason = skip or _reason(relative)
            if reason and not tracked and reason not in _SECRET_REASONS:
                continue
            path = _safe_file(root, relative)
            sha, private = _inspect_file(path)
            hashed_files += 1
            hashed_bytes += path.stat().st_size
            name = relative.as_posix()
            if private or reason in _SECRET_REASONS:
                excluded[name] = sha
                continue
            files[name] = sha
            blob = self.root / "blobs" / sha
            if not blob.exists() and (
                self.baseline is None or self.baseline.files.get(name) != sha
            ):
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != sha:
                    raise RuntimeError("Source changed while recording test evidence: " + name)
                blob.write_bytes(data)
        state = {"files": files, "excluded": excluded}
        digest = _digest(state)
        manifest = self.root / "states" / (digest + ".json")
        if not manifest.exists():
            _save(
                manifest,
                {
                    **state,
                    "digest": digest,
                    "baseline_manifest": (
                        str(self.baseline.manifest_path) if self.baseline else None
                    ),
                },
            )
        self.last_scan = {"hashed_files": hashed_files, "hashed_bytes": hashed_bytes}
        return digest, time.monotonic() - started

    def record(self, kind, **fields):
        with self.lock:
            digest, elapsed = self._source()
            event = {
                "time": time.time(),
                "kind": kind,
                "source": digest,
                "previous_source": self.current,
                "scan_seconds": elapsed,
                "environment_generation": self.generation,
                "scan_kind": "full",
                **self.last_scan,
                **fields,
            }
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(event, ensure_ascii=False) + "\n")
            self.current = digest
            return digest

    def view(self, version=None):
        from agent.source.view import SourceView

        version = version or self.current
        views = getattr(self, '_views', {})
        if version not in views:
            views = {version: SourceView(self, version)}
            self._views = views
        return views[version]

    def record_reference(self, kind, source, **fields):
        """记录归档字节上的操作，不将其当作实时文件状态的声明。"""
        with self.lock:
            if source != self.current:
                raise ValueError('Recorded operation source is not the active version')
            self.view(source)  # 清单只校验一次；文件字节在使用时检查。
            event = {
                'time': time.time(),
                'kind': kind,
                'source': source,
                'previous_source': source,
                'scan_seconds': 0.0,
                'scan_kind': 'reference',
                'hashed_files': 0,
                'hashed_bytes': 0,
                'environment_generation': self.generation,
                **fields,
            }
            with (self.root / 'events.jsonl').open('a', encoding='utf-8') as out:
                out.write(json.dumps(event, ensure_ascii=False) + '\n')
            return source
