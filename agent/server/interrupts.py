"""Interrupt current model waits; execution still commits at operation boundaries."""
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock

_active = {}
_lock = RLock()
_current = ContextVar('conversation_interrupt', default=None)


class Signal:
    def __init__(self):
        self.kind = None
        self.callbacks = set()
        self.lock = RLock()

    def interrupt(self, kind):
        with self.lock:
            if self.kind != 'pause':
                self.kind = kind
            callbacks = tuple(self.callbacks)
        for callback in callbacks:
            callback()

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


def interrupt(root, thread, kind):
    if kind not in {'pause', 'message'}:
        return
    with _lock:
        signal = _active.get((str(root), thread))
    if signal:
        signal.interrupt(kind)


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
