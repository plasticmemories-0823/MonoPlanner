"""失联即撤销运行租约；容器内的独立看守负责最终停止命名空间。"""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from pathlib import Path
from threading import Event, Lock, Thread
import time
import uuid

from agent.settings import PROJECT_ROOT
from agent.storage.records import immutable, save
from agent.storage.preferences import atomic_write


HOST_TIMEOUT = 8.0
_host = ContextVar('runtime_host_health', default=None)


@contextmanager
def host_scope(alive):
    token = _host.set(alive)
    try:
        yield
    finally:
        _host.reset(token)


class Lease:
    """文件夹只能只读挂载；工作目录不能修改续租文件或看守脚本。"""

    def __init__(self, root, name, *, cancelled=None):
        from agent.storage.resources import current_owner, owner_cancelled, _safe_tree
        from agent.task.control import current

        self.root = Path(root).resolve()
        self.directory = self.root / 'runtime-leases' / hashlib.sha256(name.encode()).hexdigest()
        _safe_tree(self.directory, self.root)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.binding = self.directory / 'binding.json'
        script = (PROJECT_ROOT / 'resources/drivers/lifetime.sh').read_bytes().replace(b'\r\n', b'\n')
        if self.binding.exists():
            value = json.loads(self.binding.read_bytes())
            if (value.get('name') != name or value.get('script_sha256') != hashlib.sha256(script).hexdigest()
                    or set(value) != {'name', 'script_sha256', 'token'}):
                raise ValueError('Saved container lifetime binding changed')
            self.token = str(uuid.UUID(value['token']))
        else:
            self.token = str(uuid.uuid4())
            save(self.binding, {'name': name, 'token': self.token,
                                'script_sha256': hashlib.sha256(script).hexdigest()})
        immutable(self.directory / 'lifetime.sh', script)
        owner, signal, host = current_owner(), current(), _host.get()
        self.signal = signal
        self.allowed = lambda: (not (cancelled and cancelled())
                                and not (owner and owner_cancelled(owner, self.root))
                                and not (signal and signal.kind == 'pause')
                                and (host is None or host()))
        self.stopped = Event()
        self.thread = None
        self.lock = Lock()
        self.sequence = 0
        if owner:
            from agent.storage.resources import resource_catalog

            resource_catalog(self.root).register(
                'temporary:lifetime:' + name, 'temporary', str(self.directory),
                owner=owner, size_bytes=len(script) + 1024,
            )

    def _write(self, value):
        if not self.binding.is_file():
            raise OSError('Container lifetime original is missing')
        atomic_write(self.directory / 'heartbeat', (value + '\n').encode('ascii'))

    def arm(self):
        if self.thread and self.stopped.is_set():
            self.thread.join(timeout=2)
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            if not self.allowed():
                raise InterruptedError('Runtime owner is not live; no container was started')
            self.stopped.clear()
            self.sequence += 1
            self._write(self.token + ' ' + str(self.sequence))
            self.thread = Thread(target=self._renew, name='container-lease', daemon=True)
            self.thread.start()

    def _renew(self):
        try:
            while not self.stopped.wait(1):
                with self.lock:
                    if self.stopped.is_set() or not self.allowed():
                        self._write('revoked')
                        return
                    self.sequence += 1
                    self._write(self.token + ' ' + str(self.sequence))
        except Exception:
            # 无法续租时保留最后一份文件，容器按自身时间失效，不能保持运行。
            self.stopped.set()
            if self.signal:
                self.signal.interrupt('pause', 'lifetime-renewal-failed')

    def revoke(self):
        self.stopped.set()
        with self.lock:
            try:
                self._write('revoked')
            except OSError:
                pass  # 失写仍会由容器的 15 次无更新检查停止。

    def wrap(self, arguments, image):
        if ',' in str(self.directory):
            raise ValueError('Container lifetime mount path cannot contain a comma')
        values = list(arguments)
        index = values.index('--entrypoint')
        image_index = values.index(image, index + 2)
        command = [values[index + 1], *values[image_index + 1:]]
        values[index + 1] = '/bin/sh'
        values[image_index + 1:] = ['/monoplanner-lifetime/lifetime.sh', self.token, *command]
        values[1:1] = ['--mount', f'type=bind,source={self.directory},target=/monoplanner-lifetime,readonly']
        if '--init' not in values:
            values.insert(1, '--init')
        self.arm()
        return values
