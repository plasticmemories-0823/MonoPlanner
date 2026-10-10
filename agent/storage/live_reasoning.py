"""Append-only display fragments; display failures never change task execution."""
import json
import logging
from pathlib import Path
from threading import RLock
import time
from weakref import WeakValueDictionary

_active = WeakValueDictionary()
_lock = RLock()
log = logging.getLogger(__name__)


class LiveReasoning:
    def __init__(self, root, call, after_id):
        self.path = Path(root)/'display/model.jsonl'
        self.row = {'id': f'{root.name}-thinking-{call}', 'task_id': root.name,
                    'after_id': after_id, 'kind': 'thinking', 'label': '正在思考',
                    'status': 'running', 'detail': '', 'started_at': time.time(),
                    'version': call * 1_000_000}
        self.pending, self.updated, self.error = [], time.monotonic(), None
        self.lock = RLock()
        with _lock:
            _active[str(Path(root).resolve())] = self
        self._write({**self.row, 'append': False})

    def _write(self, row):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        except OSError as exc:
            if self.error is None:
                log.warning('Live display unavailable: %s', type(exc).__name__)
            self.error = '实时显示记录暂不可用；任务继续接收回复。'

    def append(self, text):
        with self.lock:
            self.pending.append(text)
            if not self.row['detail'] or time.monotonic() - self.updated >= .2:
                self.flush()

    def flush(self):
        with self.lock:
            delta = ''.join(self.pending)
            self.pending.clear()
            self.row['detail'] += delta
            self.row['version'] += 1
            if delta:
                self._write({**self.row, 'detail': delta, 'append': True})
            self.updated = time.monotonic()

    def finish(self, status, detail=None):
        with self.lock:
            self.flush()
            self.row.update(status=status, label='思考' if status == 'completed' else '回复中断',
                            seconds=max(0, time.time() - self.row['started_at']))
            if detail is not None:
                self.row['detail'] = detail
            self.row['version'] += 1
            self._write({**self.row, 'append': False})


def read_live(root, cursor=0, version=0):
    """Consume complete UTF-8 lines only, without an execution lease."""
    rows, more, error, position = {}, False, None, cursor
    try:
        path = Path(root)/'display/model.jsonl'
        if path.exists():
            from agent.source.snapshots import _safe_file
            path = _safe_file(Path(root), Path('display/model.jsonl'))
            with path.open('rb') as stream:
                stream.seek(cursor)
                for _ in range(100):
                    start = stream.tell()
                    line = stream.readline()
                    if not line.endswith(b'\n'):
                        stream.seek(start)
                        break
                    row = json.loads(line.decode('utf-8'))
                    if row['task_id'] != Path(root).name or row['kind'] != 'thinking':
                        raise ValueError('Invalid live activity identity')
                    previous = rows.get(row['id'])
                    if row.get('append') and previous:
                        row['detail'] = previous['detail'] + row['detail']
                        row['append'] = previous.get('append', False)
                    rows[row['id']] = row
                    position = stream.tell()
                more = bool(stream.read(1)) and position > cursor
    except (OSError, ValueError, KeyError, TypeError):
        error = '实时显示暂不可用；可重新连接查看已保存的结果。'
    with _lock:
        live = _active.get(str(Path(root).resolve()))
    if live:
        with live.lock:
            if live.error and live.row['version'] > version:
                rows[live.row['id']] = {**live.row, 'append': False}
            error = live.error or error
    latest = max([version, *(r['version'] for r in rows.values())])
    return list(rows.values()), position, latest, more, error
