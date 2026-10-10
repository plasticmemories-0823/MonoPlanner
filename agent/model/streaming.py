"""One shared async loop lets synchronous graph nodes cancel network waits."""
import asyncio
from concurrent.futures import Future
from threading import Lock, Thread
from agent.model.errors import ModelInterrupted

_loop = None
_lock = Lock()


def run(coroutine, signal):
    global _loop
    with _lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            Thread(target=_loop.run_forever, name='model-stream', daemon=True).start()
    result = Future()
    task = None
    cancelling = False

    def completed(value):
        try:
            result.set_result(value.result())
        except asyncio.CancelledError:
            error = (ModelInterrupted(signal.kind) if signal.kind else
                     RuntimeError('Model stream cancelled without a user interruption'))
            result.set_exception(error)
        except BaseException as exc:
            result.set_exception(exc)

    def start():
        nonlocal task
        task = _loop.create_task(coroutine)
        task.add_done_callback(completed)

    def cancel():
        def request_cancel():
            nonlocal cancelling
            if task and not task.done() and not cancelling:
                cancelling = True
                task.cancel()
        _loop.call_soon_threadsafe(request_cancel)

    _loop.call_soon_threadsafe(start)
    with signal.subscribe(cancel):
        return result.result()
