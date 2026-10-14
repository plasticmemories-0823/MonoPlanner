"""中断当前模型等待；执行仍在操作边界提交。"""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock

_active = {}
_lock = RLock()
_current = ContextVar('conversation_interrupt', default=None)


class Signal:
    def __init__(self):
        self.kind = None
        self.sequence = 0
        self.pending = {}
        self.callbacks = set()
        self.lock = RLock()

    def interrupt(self, kind, identity=None):
        with self.lock:
            self.sequence += 1
            self.pending[self.sequence] = (kind, identity)
            self._refresh()
            callbacks = tuple(self.callbacks)
        for callback in callbacks:
            callback()
        return self.sequence

    def _refresh(self):
        kinds = [kind for kind, _ in self.pending.values()]
        self.kind = 'pause' if 'pause' in kinds else kinds[-1] if kinds else None

    def consume_input(self, identity):
        with self.lock:
            for sequence, (_, saved) in list(self.pending.items()):
                if saved == identity:
                    self.pending.pop(sequence)
            self._refresh()

    @contextmanager
    def subscribe(self, callback):
        with self.lock:
            self.callbacks.add(callback)
            interrupted = self.kind is not None
        if interrupted:
            callback()
        try:
            yield
        finally:
            with self.lock:
                self.callbacks.discard(callback)


def current():
    return _current.get()


def active_count(root):
    with _lock:
        return sum(saved_root == str(root) for saved_root, _ in _active)


def interrupt(root, thread, kind, identity=None):
    if kind not in {'pause', 'message'}:
        return
    with _lock:
        signal = _active.get((str(root), thread))
    if signal:
        signal.interrupt(kind, identity)


@contextmanager
def active(root, thread):
    key = (str(root), thread)
    signal = Signal()
    with _lock:
        if key in _active:
            raise RuntimeError('Conversation already has an active operation owner')
        _active[key] = signal
    token = _current.set(signal)
    try:
        yield signal
    finally:
        _current.reset(token)
        with _lock:
            if _active.get(key) is signal:
                del _active[key]
