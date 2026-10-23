"""桌面退出的单次停止协调；控制超时不会冒充已终止，也不重放工作。"""

from threading import Event, Lock, Thread
import time
import uuid


def watch_desktop_close(shutdown):
    """启动器的关闭事件独立于 HTTP，可在页面或服务尚未就绪时取消工作。"""
    import os

    name = os.getenv('MONOPLANER_CLOSE_SIGNAL')
    if os.name != 'nt' or not name:
        return lambda: None
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenEventW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenEventW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenEventW(0x00100000, False, name)
    if not handle:
        # 已声明的桌面控制对象丢失，不能悄悄降级成无人管理的服务。
        shutdown.start()
        return lambda: None
    heartbeat_name = os.getenv('MONOPLANER_DESKTOP_HEARTBEAT')
    heartbeat = kernel.OpenEventW(0x00100000, False, heartbeat_name) if heartbeat_name else None
    if heartbeat_name and not heartbeat:
        kernel.CloseHandle(handle)
        shutdown.start()
        return lambda: None
    # 已关闭的桌面可能早于后端启动；同步读取，不能等待监听线程调度后才闭锁。
    if kernel.WaitForSingleObject(handle, 0) == 0:
        kernel.CloseHandle(handle)
        if heartbeat:
            kernel.CloseHandle(heartbeat)
        shutdown.start()
        return lambda: None
    stopped = Event()

    def watch():
        last_pulse = time.monotonic()
        try:
            while not stopped.is_set():
                result = kernel.WaitForSingleObject(handle, 100)
                if result == 0:
                    shutdown.start()
                    return
                if result != 258:
                    shutdown.start()
                    return
                if heartbeat:
                    pulse = kernel.WaitForSingleObject(heartbeat, 0)
                    if pulse == 0:
                        last_pulse = time.monotonic()
                    elif pulse != 258 or time.monotonic() - last_pulse > 8:
                        shutdown.start()
                        return
        finally:
            kernel.CloseHandle(handle)
            if heartbeat:
                kernel.CloseHandle(heartbeat)

    thread = Thread(target=watch, name='desktop-close-event', daemon=True)
    thread.start()

    def close():
        stopped.set()
        thread.join(timeout=0.2)

    return close


class Shutdown:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.lock = Lock()
        self.identity = 'shutdown-' + uuid.uuid4().hex
        self.started = False
        self.finished = Event()
        self.pending = set()
        self.errors = []
        self.results = {}

    def start(self):
        with self.lock:
            if self.started:
                return
            self.started = True
            self.scheduler.stop.set()
            self.scheduler.store.closing.set()
            self.scheduler.wake.set()
            self.pending.add('shutdown_dispatch')
        Thread(target=self._run, name='application-shutdown', daemon=True).start()

    def _operation(self, name, callback):
        with self.lock:
            self.pending.add(name)

        def run():
            try:
                result = callback()
                with self.lock:
                    self.results[name] = result
                    if isinstance(result, dict):
                        termination = result.get('termination', [])
                        if result.get('stopped') is False or any(
                            row.get('stopped') is not True for row in termination
                        ):
                            self.errors.append({'operation': name, 'error': 'StopUnconfirmed'})
            except Exception as exc:
                with self.lock:
                    self.errors.append({'operation': name, 'error': type(exc).__name__})
                if name == 'pending_records':
                    self.scheduler.storage_failed('shutdown_record', exc)
            finally:
                with self.lock:
                    self.pending.discard(name)
                if self.finished.is_set():
                    self.scheduler._release_if_idle()

        Thread(target=run, name='shutdown-' + name.split(':')[0], daemon=True).start()

    def _run(self):
        from agent.execution.preparation import close_jobs
        from agent.task.control import interrupt

        scheduler = self.scheduler
        try:
            # 先给全部在途调用发取消，再并行停止各自容器，避免一项超时挡住其余项。
            with scheduler.lock:
                tasks = set(scheduler.active_jobs) | set(scheduler.controls_inflight.values())
                scopes = [*tasks, *('chat:' + item for item in scheduler.chats),
                          *('title:' + item for item in scheduler.titles)]
                if scheduler.report_task:
                    scopes.append('progress:' + scheduler.report_task)
            for scope in scopes:
                self._operation('model:' + scope, lambda scope=scope:
                                interrupt(scheduler.store.root, scope, 'pause', self.identity))
            for task in tasks:
                self._operation('work:' + task, lambda task=task:
                                scheduler.pause_callback(task, self.identity))
            self._operation('preparations', lambda: close_jobs(scheduler.store.root))
            self._operation('pending_records', scheduler.store.stop_pending)
        except Exception as exc:
            with self.lock:
                self.errors.append({'operation': 'shutdown_dispatch', 'error': type(exc).__name__})
        finally:
            with self.lock:
                self.pending.discard('shutdown_dispatch')

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if self.status()['complete']:
                break
            time.sleep(0.025)
        # 运行中的线程只能合作收尾；最后的进程截止期限由拥有 PID 的启动器处理。
        # 不取消排队 Future：其入口会检查停止标志并释放自己的登记。
        for pool in (scheduler.chat_pool, scheduler.work_pool, scheduler.report_pool,
                     scheduler.title_pool, scheduler.control_pool):
            pool.shutdown(wait=False, cancel_futures=False)
        self.finished.set()
        scheduler._release_if_idle()

    def controls_idle(self):
        with self.lock:
            return not self.pending

    def status(self):
        scheduler = self.scheduler
        with scheduler.lock:
            workers = [*('chat:' + item for item in scheduler.chats),
                       *('work:' + item for item in scheduler.works),
                       *('title:' + item for item in scheduler.titles),
                       *('control:' + item for item in scheduler.controls_inflight)]
            if scheduler.starting:
                workers.append('startup')
            if scheduler.reporting:
                workers.append('progress')
            if scheduler.worker is not None and scheduler.worker.is_alive():
                workers.append('dispatcher')
        fault = scheduler.storage_error()
        with self.lock:
            pending = sorted(self.pending | set(workers))
            complete = self.started and not pending
            return {'shutdown': self.started, 'dispatch_suspended': self.started,
                    'complete': complete, 'termination_confirmed': complete and not self.errors,
                    'pending_operations': pending, 'errors': list(self.errors),
                    'storage_error': fault}
