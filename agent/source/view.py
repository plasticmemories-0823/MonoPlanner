"""读取 SourceVersions 中的已有版本，不读取可变工作目录。"""

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from threading import RLock
from types import MappingProxyType

from agent.source.snapshots import SourceSnapshot, _safe_file


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


class SourceView:
    """以有界内存缓存复用不可变字节；版本归归档管理。"""

    def __init__(self, evidence, version=None, max_cache_bytes=32 * 1024 * 1024):
        self.root, self.baseline = evidence.root, evidence.baseline
        self.origin_root = evidence.environment.repo_root
        self.version = version or evidence.current
        if not re.fullmatch(r'[0-9a-f]{64}', self.version or ''):
            raise ValueError('Invalid recorded source version')
        path = _safe_file(self.root, Path('states') / (self.version + '.json'))
        data = json.loads(path.read_text(encoding='utf-8'))
        value = {'files': data['files'], 'excluded': data['excluded']}
        if digest(value) != self.version or data.get('digest') != self.version:
            raise ValueError('Source manifest fingerprint mismatch')
        self.files = MappingProxyType(dict(data['files']))
        self.excluded = MappingProxyType(dict(data['excluded']))
        self._cache, self._lock = OrderedDict(), RLock()
        self.max_cache_bytes, self.cache_bytes = max_cache_bytes, 0
        self.stats = {'verified_files': 0, 'verified_bytes': 0, 'byte_cache_hits': 0}

    def _name(self, name):
        path = PurePosixPath(str(name).replace('\\', '/'))
        if path.is_absolute() or '..' in path.parts or ':' in str(name):
            raise ValueError('Source path must remain inside the recorded version')
        name = path.as_posix()
        if name not in self.files:
            raise ValueError('Path is absent or excluded from the source inventory')
        if not re.fullmatch(r'[0-9a-f]{64}', self.files[name]):
            raise ValueError('Invalid source content fingerprint')
        return name

    def read_bytes(self, name):
        name = self._name(name)
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                self.stats['byte_cache_hits'] += 1
                return self._cache[name]
        sha = self.files[name]
        blob = self.root / 'blobs' / sha
        if blob.exists() or blob.is_symlink():
            source = _safe_file(self.root, Path('blobs') / sha)
        elif self.baseline is not None and self.baseline.files.get(name) == sha:
            source = _safe_file(self.baseline.root, Path(name))
        else:
            raise ValueError('Recorded source bytes are missing: ' + name)
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError('Recorded source bytes changed: ' + name)
        with self._lock:
            self.stats['verified_files'] += 1
            self.stats['verified_bytes'] += len(raw)
            if len(raw) <= self.max_cache_bytes and name not in self._cache:
                while self._cache and self.cache_bytes + len(raw) > self.max_cache_bytes:
                    unused, old = self._cache.popitem(last=False)
                    self.cache_bytes -= len(old)
                self._cache[name] = raw
                self.cache_bytes += len(raw)
        return raw

    def export(self, destination, *, cancel=None):
        """隔离调查只使用归档字节生成副本，不使用实时文件。"""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        target = destination / 'source'
        target.mkdir()
        for name in self.files:
            if cancel is not None and cancel.is_set():
                from concurrent.futures import CancelledError

                raise CancelledError()
            raw = self.read_bytes(name)
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                for offset in range(0, len(raw), 1024 * 1024):
                    if cancel is not None and cancel.is_set():
                        from concurrent.futures import CancelledError

                        raise CancelledError()
                    stream.write(raw[offset : offset + 1024 * 1024])
            if self.baseline is not None and name in self.baseline.files:
                original = _safe_file(self.baseline.root, Path(name))
                path.chmod(original.stat().st_mode)
        payload = {'files': dict(self.files), 'excluded_files': dict(self.excluded)}
        sha = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
        manifest = destination / 'manifest.json'
        manifest.write_text(
            json.dumps(
                {
                    **payload,
                    'root': str(target.resolve()),
                    'digest': sha,
                    'status': 'captured',
                    'source_version': self.version,
                    'origin': 'SourceVersions archived bytes; no live workspace read',
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding='utf-8',
        )
        return SourceSnapshot(target, dict(self.files), sha, manifest, dict(self.excluded))
