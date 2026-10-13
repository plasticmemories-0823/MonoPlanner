"""Bounded background jobs shared by the UI, chat handoff and CLI.

Only disposable environment setup is retriable here. Model requests, source
modifications and acceptance execution never pass through this job manager.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from threading import Event, RLock
import time
import uuid

from agent.execution.projects import EnvironmentRequired, ProjectEnvironments, workspace_path
from agent.storage.preferences import atomic_write, state_root


class PreparationCancelled(EnvironmentRequired):
    pass


class PreparationJobs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.folder = self.root/'environment-jobs'
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='environment-setup')
        self.lock = RLock()
        self.jobs = {}
        self.closed = False

    def _path(self, repo):
        return self.folder/(hashlib.sha256(repo.encode()).hexdigest() + '.json')

    def _read(self, repo):
        path = self._path(repo)
        return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None

    def _save(self, repo, value):
        value = {**value, 'updated_at': time.time()}
        atomic_write(self._path(repo), json.dumps(value, ensure_ascii=False).encode())
        return value

    def status(self, repo):
        repo = workspace_path(repo)
        binding = ProjectEnvironments(self.root).get(repo)
        if binding.get('image'):
            return {'mode': 'manual', 'status': 'ready', 'stage': 'ready',
                    'message': '使用已保存的高级环境配置', 'environment': binding}
        with self.lock:
            value = self._read(repo)
            if value and value['status'] == 'preparing':
                if repo in self.jobs:
                    return {**value, 'mode': 'auto'}
                value = self._save(repo, {**value, 'status': 'failed', 'stage': 'interrupted',
                    'error': '上次环境准备被中断；可以重新准备，不会重放编码任务。',
                    'message': '环境准备尚未完成'})
        if value and value['status'] in {'failed', 'cancelled'}:
            return {**value, 'mode': 'auto'}
        from agent.execution.managed import ManagedEnvironments
        ready = ManagedEnvironments(self.root).status(repo)
        cached = ready.get('state') == 'ready'
        result = {**ready, 'mode': 'auto', 'status': 'ready' if cached else 'idle',
                  'stage': 'ready' if cached else 'checking'}
        if cached and ready.get('runtime'):
            result['environment'] = {key: ready[key] for key in ('backend', 'image', 'runtime', 'managed') if key in ready}
        return result

    def start(self, repo, *, background=True):
        repo = workspace_path(repo)
        if ProjectEnvironments(self.root).get(repo).get('image'):
            return self.status(repo)
        with self.lock:
            if self.closed:
                raise EnvironmentRequired('环境准备服务正在关闭，请重新打开应用。')
            if repo in self.jobs:
                self.jobs[repo]['background'] |= background
                return self._read(repo)
            # Bound both active and queued work; browsing cannot enqueue builds.
            if len(self.jobs) >= 2:
                raise EnvironmentRequired('已有两个项目正在准备环境，请稍后重试。')
            value = self._save(repo, {'mode': 'auto', 'status': 'preparing',
                'id': uuid.uuid4().hex, 'stage': 'checking', 'message': '正在检查项目环境'})
            cancel = Event()
            self.jobs[repo] = {'cancel': cancel, 'id': value['id'], 'background': background, 'waiters': 0}
            self.pool.submit(self._run, repo, value, cancel)
            return value

    def _run(self, repo, value, cancel):
        def progress(update):
            nonlocal value
            # The builder supplies human-facing stages, never raw subprocess
            # output/URLs that can contain package credentials.
            with self.lock:
                value = self._save(repo, {**value, 'stage': update['stage'],
                    'message': update['message'], 'status': 'preparing'})
        try:
            from agent.execution.managed import ManagedEnvironments
            binding = ManagedEnvironments(self.root).prepare(repo, progress=progress, cancelled=cancel.is_set)
            if cancel.is_set():
                raise PreparationCancelled('环境准备已取消')
            value = {**value, 'status': 'ready', 'stage': 'ready',
                     'message': '项目环境已就绪', 'environment': binding}
        except Exception as exc:
            from agent.execution.managed_plan import PreparationError
            cancelled = cancel.is_set() or isinstance(exc, PreparationCancelled)
            # Technical logs remain in the private managed environment record.
            value = {**value, 'status': 'cancelled' if cancelled else 'failed',
                'stage': 'cancelled' if cancelled else 'failed',
                'message': '环境准备已取消' if cancelled else '环境准备未完成',
                'error': ('环境准备已取消，项目文件未修改。' if cancelled else
                          str(exc) if isinstance(exc, (EnvironmentRequired, PreparationError)) else
                          'Docker 环境准备未完成，请查看本地准备记录并重试。')}
        finally:
            with self.lock:
                try:
                    self._save(repo, value)
                finally:
                    self.jobs.pop(repo, None)

    def cancel(self, repo):
        repo = workspace_path(repo)
        with self.lock:
            if repo in self.jobs:
                self.jobs[repo]['cancel'].set()
                value = self._read(repo)
                return self._save(repo, {**value, 'stage': 'cancelling', 'message': '正在停止环境准备'})
        return self.status(repo)

    def wait(self, repo, *, progress=None, cancelled=None):
        repo = workspace_path(repo)
        value = self.start(repo, background=False)
        if value.get('mode') == 'manual' and value.get('status') == 'ready':
            return value['environment']
        with self.lock:
            if repo in self.jobs:
                self.jobs[repo]['waiters'] += 1
        last = None
        detached = False
        try:
            while True:
                if cancelled and cancelled():
                    detached = True
                    raise PreparationCancelled('环境准备等待已中断；项目执行尚未开始。')
                with self.lock:
                    value = self._read(repo) or value
                    running = repo in self.jobs
                if progress and (value.get('stage'), value.get('message')) != last:
                    progress(value)
                    last = (value.get('stage'), value.get('message'))
                if not running:
                    if value['status'] == 'ready':
                        return value['environment']
                    if value['status'] == 'cancelled':
                        raise PreparationCancelled(value.get('error', '环境准备已取消'))
                    raise EnvironmentRequired(value.get('error', 'Docker 环境准备未完成'))
                time.sleep(.15)
        finally:
            with self.lock:
                job = self.jobs.get(repo)
                if job and job['id'] == value.get('id'):
                    job['waiters'] = max(0, job['waiters'] - 1)
                    if detached and not job['waiters'] and not job['background']:
                        job['cancel'].set()

    def close(self):
        with self.lock:
            self.closed = True
            for job in self.jobs.values():
                job['cancel'].set()
        self.pool.shutdown(wait=True, cancel_futures=False)


_services = {}
_services_lock = RLock()


def jobs(root=None):
    root = Path(root or state_root()).resolve()
    with _services_lock:
        if root not in _services:
            _services[root] = PreparationJobs(root)
        return _services[root]


def close_jobs(root):
    with _services_lock:
        service = _services.pop(Path(root).resolve(), None)
    if service:
        service.close()


def active_count(root=None):
    with _services_lock:
        service = _services.get(Path(root or state_root()).resolve())
    if service is None:
        return 0
    with service.lock:
        return len(service.jobs)


def ensure_prepared(repo, *, progress=None, cancelled=None):
    return jobs().wait(repo, progress=progress, cancelled=cancelled)
