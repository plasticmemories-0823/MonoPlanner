"""单一环境管理者：作业与等待者持久保存在资源台账，安装操作不自动恢复。"""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import hashlib
import json
from pathlib import Path
from threading import Event, RLock
import time
import uuid

from agent.application.lease import ServiceLease
from agent.execution.projects import EnvironmentRequired, ProjectEnvironments, workspace_path
from agent.storage.resources import (
    resource_catalog, current_root, current_owner, container_owner,
    register_active_container, unregister_active_container, cancel_owned_containers,
)


class PreparationCancelled(EnvironmentRequired):
    pass


class PreparationJobs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.catalog = resource_catalog(self.root)
        lease_root = self.root / 'environment-manager'
        lease_root.mkdir(exist_ok=True)
        self.lease = ServiceLease(lease_root)
        self.lock, self.workers, self.closed = RLock(), {}, False
        try:
            self.lease.acquire()
        except RuntimeError:
            self.pool = None
        else:
            self.catalog.recover_environment_jobs()
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='environment-setup')

    def _legacy(self, repo):
        # 旧 JSON 仅用于解释历史状态，不导入新队列、不恢复旧安装。
        path = self.root / 'environment-jobs' / (hashlib.sha256(repo.encode()).hexdigest() + '.json')
        if path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024 * 1024:
            return {'mode': 'auto', 'status': 'interrupted', 'stage': 'interrupted',
                    'message': '旧版环境准备记录仅供保留，请重新准备环境。'}
        return None

    def status(self, repo):
        repo = workspace_path(repo)
        binding = ProjectEnvironments(self.root).get(repo)
        if binding.get('image'):
            return {'mode': 'manual', 'status': 'ready', 'stage': 'ready',
                    'message': '使用已保存的高级环境配置', 'environment': binding}
        value = self.catalog.environment_job(repo)
        if value:
            with self.lock:
                if (value['status'] == 'preparing' and self.pool is not None
                        and value['id'] not in self.workers):
                    return {**value, 'status': 'interrupted', 'stage': 'interrupted',
                            'message': '准备记录未能正常完成，现场已保留；请核对后明确重新准备。'}
            return value
        from agent.execution.managed import ManagedEnvironments

        ready = ManagedEnvironments(self.root).status(repo)
        if ready.get('state') == 'ready':
            return {**ready, 'mode': 'auto', 'status': 'ready', 'stage': 'ready',
                    'environment': {key: ready[key] for key in ('backend', 'image', 'runtime', 'managed')
                                    if key in ready}}
        return self._legacy(repo) or {**ready, 'mode': 'auto', 'status': 'idle', 'stage': 'checking'}

    def start(self, repo, *, background=True, retry=False):
        repo = workspace_path(repo)
        if ProjectEnvironments(self.root).get(repo).get('image'):
            return self.status(repo)
        with self.lock:
            if self.closed:
                raise EnvironmentRequired('环境准备服务正在关闭，请重新打开应用。')
            if self.pool is None:
                raise EnvironmentRequired('此数据目录已有环境管理者，请连接原服务。')
            existing = self.catalog.environment_job(repo)
            if existing and existing['id'] in self.workers:
                if background and not existing.get('background'):
                    existing = self.catalog.save_environment_job({**existing, 'background': True})
                return existing
            if existing and existing['status'] in {'failed', 'cancelled', 'interrupted', 'preparing'}:
                if not retry:
                    return existing
                results = cancel_owned_containers(existing['owner'], self.root)
                if any(not result['stopped'] for result in results):
                    raise EnvironmentRequired('上次准备容器停止未确认，不能创建新环境。')
            elif not retry and self._legacy(repo):
                raise EnvironmentRequired('旧版准备记录仅供保留，请明确点击重新准备环境。')
            if any(not worker['future'].done() for worker in self.workers.values()):
                raise EnvironmentRequired('已有一个项目正在准备环境，请稍后重试。')
            identity = uuid.uuid4().hex
            value = self.catalog.save_environment_job({
                'id': identity, 'repo': repo, 'owner': 'prepare:' + identity,
                'mode': 'auto', 'status': 'preparing', 'stage': 'checking',
                'message': '正在检查项目环境', 'background': background,
            })
            cancel = Event()
            future = self.pool.submit(copy_context().run, self._run, value, cancel)
            self.workers[identity] = {'future': future, 'cancel': cancel}
            future.add_done_callback(lambda _: self._finished(identity))
            return value

    def retry(self, repo):
        """仅显式用户操作可重新准备失败或中断的环境。"""
        return self.start(repo, retry=True)

    def _finished(self, identity):
        with self.lock:
            self.workers.pop(identity, None)
            if self.closed and not self.workers:
                self.lease.close()

    def _run(self, value, cancel):
        with container_owner(value['owner'], root=self.root):
            def progress(update):
                current = self.catalog.environment_job(job_id=value['id'])
                self.catalog.save_environment_job({**current,
                    'stage': update['stage'], 'message': update['message'],
                    'projects': update.get('projects'), 'materials': update.get('materials')})
            try:
                from agent.execution.managed import ManagedEnvironments

                binding = ManagedEnvironments(self.root).prepare(
                    value['repo'], progress=progress, cancelled=cancel.is_set)
                if cancel.is_set():
                    raise PreparationCancelled('环境准备已取消')
                result = {'status': 'ready', 'stage': 'ready', 'message': '项目环境已就绪',
                          'environment': binding}
                # 完成的安装资源已经通过同一台账收尾；镜像由项目引用保留。
                self.catalog.unlink(value['owner'])
            except Exception as exc:
                from agent.execution.managed_plan import PreparationError

                stopped = cancel.is_set() or isinstance(exc, PreparationCancelled)
                result = {'status': 'cancelled' if stopped else 'failed',
                          'stage': 'cancelled' if stopped else 'failed',
                          'message': '环境准备已取消' if stopped else '环境准备未完成',
                          'error': '环境准备已取消，项目文件未修改。' if stopped else (
                              str(exc) if isinstance(exc, (EnvironmentRequired, PreparationError))
                              else 'Docker 环境准备未完成，请查看本地记录后重试。')}
            current = self.catalog.environment_job(job_id=value['id'])
            self.catalog.save_environment_job({**current, **result})

    def _stop(self, value):
        with self.lock:
            worker = self.workers.get(value['id'])
            if worker:
                worker['cancel'].set()
        results = cancel_owned_containers(value['owner'], self.root)
        if any(not result['stopped'] for result in results):
            raise EnvironmentRequired('环境准备容器停止未确认，现场已保留。')
        return results

    def cancel(self, repo):
        repo = workspace_path(repo)
        value = self.catalog.environment_job(repo)
        if value and value['status'] in {'preparing', 'interrupted', 'cancelled'}:
            self._stop(value)
        return self.status(repo)

    def _detach(self, value, token, owner, *, cancel):
        with self.lock:
            if not any(item['token'] == token for item in self.catalog.environment_waiters(value['id'])):
                return {'preparation_job': value['id'], 'stopped': True, 'detached': True}
            others = self.catalog.detach_environment_waiter(value['id'], token)
            if not cancel or others or (not owner and value.get('background')):
                return {'preparation_job': value['id'], 'stopped': True, 'detached': True}
            try:
                self._stop(value)
            except Exception:
                self.catalog.attach_environment_waiter(value['id'], owner, token=token)
                raise
            return {'preparation_job': value['id'], 'stopped': True, 'detached': False}

    def cancel_owner(self, owner):
        """服务重启后仍按持久等待关系撤销这个任务的准备，不影响其他等待任务。"""
        with self.catalog._db() as db:
            rows = list(db.execute('SELECT j.value,w.token FROM environment_waiters w '
                'JOIN environment_jobs j ON j.id=w.job WHERE w.owner=? '
                "AND j.status IN ('preparing','interrupted','cancelled','failed')", (owner,)))
        results = []
        for row in rows:
            value = json.loads(row['value'])
            try:
                results.append(self._detach(value, row['token'], owner, cancel=True))
            except Exception as exc:
                results.append({'preparation_job': value['id'], 'stopped': False,
                                'error': type(exc).__name__})
        return results

    def wait(self, repo, *, progress=None, cancelled=None):
        repo = workspace_path(repo)
        value = self.start(repo, background=False)
        if value.get('mode') == 'manual':
            return value['environment']
        owner = current_owner()
        token = self.catalog.attach_environment_waiter(value['id'], owner)
        detached, control = Event(), None
        def stop_waiting():
            detached.set()
            return self._detach(value, token, owner, cancel=True)
        if owner:
            control = register_active_container(
                owner, 'prepare-wait:' + value['id'] + ':' + token, stop_waiting, root=self.root)
        last = None
        try:
            while True:
                if detached.is_set() or (cancelled and cancelled()):
                    raise PreparationCancelled('环境准备等待已中断；项目执行尚未开始。')
                current = self.catalog.environment_job(job_id=value['id'])
                if progress and (current['stage'], current['message']) != last:
                    progress(current)
                    last = (current['stage'], current['message'])
                if current['status'] != 'preparing':
                    if current['status'] == 'ready':
                        return current['environment']
                    if current['status'] == 'cancelled':
                        raise PreparationCancelled(current.get('error', '环境准备已取消'))
                    raise EnvironmentRequired(current.get('error', '环境准备未完成'))
                with self.lock:
                    if value['id'] not in self.workers:
                        raise EnvironmentRequired('准备记录未能正常完成，现场已保留；不会自动重试。')
                time.sleep(0.05)
        finally:
            if control:
                unregister_active_container(control)
            self._detach(value, token, owner,
                         cancel=detached.is_set() or bool(cancelled and cancelled()))

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            workers = list(self.workers.items())
        for identity, worker in workers:
            worker['cancel'].set()
            cancel_owned_containers('prepare:' + identity, self.root)
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=False)
        for _, worker in workers:
            try:
                worker['future'].result(timeout=2)
            except Exception:
                pass
        with self.lock:
            if not self.workers:
                self.lease.close()


_services, _services_lock = {}, RLock()


def jobs(root=None):
    root = current_root(root)
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
        service = _services.get(current_root(root))
    if service is None:
        return 0
    with service.lock:
        return sum(not item['future'].done() for item in service.workers.values())


def ensure_prepared(repo, *, progress=None, cancelled=None, root=None):
    return jobs(root).wait(repo, progress=progress, cancelled=cancelled)
