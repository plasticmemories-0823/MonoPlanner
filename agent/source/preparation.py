"""唯一规划器等待时可进行本地准备；此处不调用模型或 SQL。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time

from agent.source.index import _parse, _module, MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES


def build_preparation(view, previous, cancel, max_seconds=8.0):
    started = time.monotonic()
    entries, offsets, lines = {}, {}, {}
    parsed = reused = warmed = total = 0
    incomplete = False
    for name, sha in sorted(view.files.items()):
        if cancel.is_set() or time.monotonic() - started >= max_seconds:
            incomplete = True
            break
        if warmed >= MAX_FILES or total >= MAX_TOTAL_BYTES:
            incomplete = True
            break
        raw = view.read_bytes(name)
        if len(raw) > 2_000_000 or total + len(raw) > MAX_TOTAL_BYTES:
            incomplete = True
            continue
        try:
            text = raw.decode('utf-8')
        except UnicodeError:
            continue
        warmed += 1
        total += len(raw)
        lines[name] = tuple(text.splitlines())
        position, starts = 0, []
        for line in text.splitlines(keepends=True):
            starts.append(position)
            position += len(line.encode('utf-8'))
        offsets[name] = {'sha256': sha, 'byte_offsets': starts}
        if not name.endswith('.py') or len(raw) > MAX_FILE_BYTES:
            continue
        key = (name, sha)
        if key in previous:
            entry = previous[key]
            reused += 1
        else:
            parsed += 1
            try:
                entry = _parse(raw, name)
            except (SyntaxError, UnicodeError, ValueError, RecursionError) as exc:
                entry = {
                    'module': _module(name),
                    'doc': '',
                    'symbols': [],
                    'imports': [],
                    'calls': [],
                    'error': type(exc).__name__,
                }
            entry = {**entry, 'digest': sha}
        entries[name] = entry
    cache = {(name, view.files[name]): entry for name, entry in entries.items()}
    return {
        'version': view.version,
        'entries': entries,
        'line_offsets': offsets,
        'lines': lines,
        'parse_cache': cache,
        'started': started,
        'ended': time.monotonic(),
        'parsed_files': parsed,
        'reused_files': reused,
        'warmed_files': warmed,
        'prepared_bytes': total,
        'incomplete': incomplete,
    }


class Preparation:
    """仅使用一个工作线程；只有主线程发布产物并更新 SQLite。"""

    def __init__(self, directory, *, prepare_probes=False):
        self.directory = directory
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='source-prepare')
        from agent.source.probe_preparation import ProbePreparation

        self.probes = ProbePreparation(self.pool, directory / 'probes') if prepare_probes else None
        self.active = None
        self.cache, self.seen, self.jobs = {}, set(), []
        self.bundle, self.closed = None, False
        self.model_intervals = []
        self.commit_seconds = self.close_seconds = 0.0

    def start(self, view):
        if self.closed:
            return
        if view.version not in self.seen:
            if self.active is not None:
                if self.active['version'] != view.version:
                    self.active['cancel'].set()
            else:
                cancel = Event()
                future = self.pool.submit(build_preparation, view, dict(self.cache), cancel)
                self.active = {'version': view.version, 'future': future, 'cancel': cancel}
                self.seen.add(view.version)
        # 保持现有索引优先级；模型等待时间较短时，可以取消排队中的副本准备，
        # 避免与常规源码准备竞争。
        if self.probes is not None:
            self.probes.start(view)

    def collect(self, version, archive=None):
        active = self.active
        if active is None or not active['future'].done():
            return self.bundle if self.bundle and self.bundle['version'] == version else None
        self.active = None
        try:
            result = active['future'].result()
        except Exception as exc:
            self.jobs.append(
                {'version': active['version'], 'status': 'error', 'error': type(exc).__name__}
            )
            if isinstance(exc, OSError):
                raise
            return None
        meta = {
            k: v
            for k, v in result.items()
            if k not in {'entries', 'line_offsets', 'lines', 'parse_cache'}
        }
        meta['status'] = 'accepted' if result['version'] == version else 'stale'
        self.jobs.append(meta)
        self.cache = result['parse_cache']
        if result['version'] != version:
            return None
        started = time.monotonic()
        from agent.storage.records import save
        from agent.model.context import fingerprint

        data = {k: result[k] for k in ('version', 'entries', 'line_offsets', 'incomplete')}
        path = self.directory / (version + '-' + fingerprint(data)[:16] + '.json')
        if not path.exists():
            save(path, data)
        if archive is not None:
            archive.register_derived(path, 'source_navigation', version)
        self.commit_seconds += time.monotonic() - started
        self.bundle = result
        return result

    def note_request(self, started, ended):
        self.model_intervals.append((started, ended))

    def close(self, version, archive=None):
        if self.closed:
            return
        started = time.monotonic()
        self.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        try:
            if self.probes is not None:
                self.probes.finish()
            self.collect(version, archive)
        finally:
            self.closed = True
            self.close_seconds = time.monotonic() - started

    def cancel(self):
        if self.active is not None:
            self.active['cancel'].set()
        if self.probes is not None:
            self.probes.cancel()

    def report(self):
        overlap = total = 0.0
        probe_jobs = self.probes.jobs if self.probes is not None else []
        for job in self.jobs + probe_jobs:
            if 'started' not in job:
                continue
            total += job['ended'] - job['started']
            overlap += sum(
                max(0.0, min(job['ended'], end) - max(job['started'], start))
                for start, end in self.model_intervals
            )
        return {
            'jobs': self.jobs,
            'probe_jobs': probe_jobs,
            'worker_seconds': total,
            'overlapped_model_seconds': overlap,
            'main_commit_seconds': self.commit_seconds,
            'cleanup_wait_seconds': self.close_seconds,
            'worker_model_calls': 0,
            'worker_sqlite_writes': 0,
        }
