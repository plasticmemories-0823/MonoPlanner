"""Prepare at most one immutable probe copy while a model request is pending."""
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
import time
import uuid


def prepare_copy(view, directory, cancel):
    started = time.monotonic()
    snapshot = view.export(directory, cancel=cancel)
    return {'snapshot': snapshot, 'version': view.version,
            'started': started, 'ended': time.monotonic()}


class ProbePreparation:
    """Shares the source worker; only the owner consumes the completed copy."""
    def __init__(self, pool, directory):
        self.pool, self.directory = pool, Path(directory)
        self.active = None
        self.jobs = []

    def start(self, view):
        if self.active is not None:
            if self.active['version'] == view.version and not self.active['cancel'].is_set():
                return
            self.cancel()
            if not self.active['future'].done():
                return
            self._finish('stale')
        cancel = Event()
        directory = self.directory / ('copy-' + uuid.uuid4().hex)
        self.active = {'version': view.version, 'cancel': cancel,
                       'future': self.pool.submit(prepare_copy, view, directory, cancel)}

    def _finish(self, status):
        active, self.active = self.active, None
        try:
            result = active['future'].result()
        except CancelledError:
            self.jobs.append({'version': active['version'], 'status': 'cancelled'})
            return None
        except Exception as exc:
            self.jobs.append({'version': active['version'], 'status': 'error',
                              'error': type(exc).__name__})
            if status == 'consumed':
                raise
            return None
        self.jobs.append({k: v for k, v in result.items() if k != 'snapshot'} | {'status': status})
        return result['snapshot']

    def take(self, version):
        if self.active is None:
            return None
        if not self.active['future'].done():
            # Do not compete with the synchronous fallback by copying twice.
            self.cancel()
            return None
        if self.active['version'] != version or self.active['cancel'].is_set():
            self._finish('stale')
            return None
        snapshot = self._finish('consumed')
        if snapshot is not None and not snapshot.verify_integrity()['valid']:
            raise ValueError('Prepared probe source integrity failed')
        return snapshot

    def cancel(self):
        if self.active is not None:
            self.active['cancel'].set()
            self.active['future'].cancel()

    def finish(self):
        if self.active is not None:
            self._finish('unused')
