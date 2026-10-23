"""单一环境管理者：作业与等待者持久保存在资源台账，安装操作不自动恢复。"""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import json
from pathlib import Path
from threading import Event, RLock, current_thread
import time
import uuid

from agent.application.lease import ServiceLease
from agent.execution.projects import EnvironmentRequired, ProjectEnvironments, workspace_path
from agent.source.scope import ProjectScope, ProjectScopeError, project_scope as resolve_scope
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

    def status(self, repo, *, verify=False):
        value = self._status(repo, verify=verify)
        enabled = bool(ProjectEnvironments(self.root).get(repo).get('wine_cmd'))
        return {**value, **({'wine_cmd': True} if enabled else {})}

    def _status(self, repo, *, verify=False):
        repo = workspace_path(repo)
        binding = ProjectEnvironments(self.root).get(repo)
        if binding.get('image'):
            value = {'mode': 'manual', 'status': 'ready', 'stage': 'ready',
                     'message': '使用已保存的高级环境配置', 'environment': binding}
            return self._verify(value) if verify else value
        value = self.catalog.environment_job(repo)
        if value:
            with self.lock:
                if (value['status'] == 'preparing' and self.pool is not None
                        and value['id'] not in self.workers):
                    return {**value, 'status': 'interrupted', 'stage': 'interrupted',
                            'message': '准备记录未能正常完成，现场已保留；请核对后明确重新准备。'}
            if verify and (value['status'] == 'ready' or value.get('verification')):
                return self.catalog.save_environment_job(self._verify(value))
            return value
        from agent.execution.managed import ManagedEnvironments

        ready = ManagedEnvironments(self.root).status(repo)
        if ready.get('state') == 'ready':
            value = {**ready, 'mode': 'auto', 'status': 'ready', 'stage': 'ready',
                     'environment': {key: ready[key]
                         for key in ('backend', 'image', 'runtime', 'managed') if key in ready}}
            return self._verify(value) if verify else value
        return {**ready, 'mode': 'auto', 'status': 'idle', 'stage': 'checking'}

    def _verify(self, value):
        from agent.execution.managed import ManagedEnvironments

        result = ManagedEnvironments(self.root).verify(value.get('environment') or {})
        ready = result['state'] == 'ready'
        value = {**value, 'verification': result,
                 'status': 'ready' if ready else 'failed',
                 'stage': 'ready' if ready else 'verification', 'message': result['message']}
        if ready:
            value.pop('error', None)
        else:
            value['error'] = result['message']
        return value

    def start(self, repo, *, background=True, retry=False, project_scope=None, wine_install=False):
        repo = workspace_path(repo)
        scope = resolve_scope(repo, project_scope)
        if ProjectEnvironments(self.root).get(repo).get('image'):
            return self.status(repo)
        with self.lock:
            if self.closed:
                raise EnvironmentRequired('环境准备服务正在关闭，请重新打开应用。')
            if self.pool is None:
                raise EnvironmentRequired('此数据目录已有环境管理者，请连接原服务。')
            existing = self.catalog.environment_job(repo)
            if existing and existing['id'] in self.workers:
                if existing.get('project_scope') != scope.to_record():
                    raise ProjectScopeError('同一目录的准备作业绑定了不同项目范围，未共享或替换原作业。')
                if background and not existing.get('background'):
                    existing = self.catalog.save_environment_job({**existing, 'background': True})
                return existing
            if existing and existing['status'] in {'failed', 'cancelled', 'interrupted', 'preparing'}:
                if not retry:
                    return existing
                results = cancel_owned_containers(existing['owner'], self.root)
                if any(not result['stopped'] for result in results):
                    raise EnvironmentRequired('上次准备容器停止未确认，不能创建新环境。')
            if any(not worker['future'].done() for worker in self.workers.values()):
                raise EnvironmentRequired('已有一个项目正在准备环境，请稍后重试。')
            identity = uuid.uuid4().hex
            value = self.catalog.save_environment_job({
                'id': identity, 'repo': repo, 'owner': 'prepare:' + identity,
                'mode': 'auto', 'status': 'preparing', 'stage': 'checking',
                'wine_cmd': bool(ProjectEnvironments(self.root).get(repo).get('wine_cmd')),
                'wine_install_authorized': wine_install,
                'message': '正在检查项目环境', 'background': background,
                'project_scope': scope.to_record(),
            })
            cancel = Event()
            future = self.pool.submit(copy_context().run, self._run, value, cancel)
            self.workers[identity] = {'future': future, 'cancel': cancel}
            future.add_done_callback(lambda _: self._finished(identity))
            return value

    def install_wine(self, repo):
        """只有环境设置的显式安装动作调用；不经模型或工作派发入口。"""
        repo = workspace_path(repo)
        with self.lock:
            if self.closed or self.pool is None:
                raise EnvironmentRequired('环境准备服务不可用，请重新打开应用。')
            current = self.catalog.environment_job(repo)
            if any(not worker['future'].done() for worker in self.workers.values()):
                if (current and current['id'] in self.workers
                        and ProjectEnvironments(self.root).get(repo).get('wine_cmd')):
                    return self.status(repo)
                raise EnvironmentRequired('已有环境正在准备，请等完成或取消后再安装 Wine。')
            if ProjectEnvironments(self.root).get(repo).get('image'):
                raise EnvironmentRequired('当前使用手动镜像；请先改为自动准备，或选择已有 Wine 镜像。')
            ProjectEnvironments(self.root).enable_wine(repo)
            return self.start(repo, retry=True, wine_install=True)

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

                scope = ProjectScope.from_record(value['project_scope'], root=value['repo'])
                binding = ManagedEnvironments(self.root).prepare(
                    value['repo'], progress=progress, cancelled=cancel.is_set, project_scope=scope,
                    **({'allow_wine_install': True} if value.get('wine_install_authorized') else {}))
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
                              str(exc) if isinstance(exc, (EnvironmentRequired, PreparationError, ProjectScopeError))
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
        # 旧进程已经退出时没有 worker 收尾，停止确认后同步作业，避免界面永久停在准备中。
        if worker is None and value['status'] in {'preparing', 'interrupted'}:
            self.catalog.save_environment_job({**value, 'status': 'cancelled',
                'stage': 'cancelled', 'message': '环境准备已取消，现场已保留。'})
        return results

    def cancel(self, repo):
        repo = workspace_path(repo)
        value = self.catalog.environment_job(repo)
        if value and value['status'] in {'preparing', 'interrupted', 'cancelled', 'failed'}:
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

    def wait(self, repo, *, progress=None, cancelled=None, project_scope=None, retry=False):
        repo = workspace_path(repo)
        options = {"project_scope": project_scope} if project_scope is not None else {}
        value = self.start(repo, background=False, retry=retry, **options)
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
                return getattr(self, 'close_report', {
                    'stopped': False, 'resources': [], 'errors': []})
            self.closed = True
            self.close_report = {'stopped': False, 'resources': [], 'errors': []}
            workers = list(self.workers.items())
        # 先阻止所有作业继续派发；不能因前一个 Docker 查询缓慢，延后其他作业的取消信号。
        for identity, worker in workers:
            worker['cancel'].set()
        stopped = _close_preparation_resources(
            self.root, {'prepare:' + identity for identity, _ in workers})
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=False)
        deadline = time.monotonic() + 2
        for _, worker in workers:
            try:
                worker['future'].result(timeout=max(0, deadline - time.monotonic()))
            except Exception:
                pass
        # Future 完成时，线程仍可能在执行回调或退出线程池；不能提前报告已停止。
        threads = tuple(self.pool._threads) if self.pool else ()
        for thread in threads:
            if thread is not current_thread():
                thread.join(timeout=max(0, deadline - time.monotonic()))
        threads_stopped = all(not thread.is_alive() for thread in threads)
        with self.lock:
            if not self.workers:
                self.lease.close()
            self.close_report = {
                **stopped,
                'stopped': not self.workers and threads_stopped and stopped['stopped'],
            }
            return self.close_report


_services, _services_lock, _closed_roots = {}, RLock(), set()


def jobs(root=None):
    root = current_root(root)
    with _services_lock:
        if root not in _services:
            if root in _closed_roots:
                raise EnvironmentRequired('环境准备服务正在关闭，请重新打开应用。')
            _services[root] = PreparationJobs(root)
        return _services[root]


def _close_preparation_resources(root, owners=()):
    """退出时只读取持久归属并停止原资源；不为历史准备重建管理者或工作线程。"""
    owners, resources, errors = set(owners), [], []
    if (root / 'resources.sqlite3').exists():
        try:
            with resource_catalog(root)._db() as db:
                owners.update(row[0] for row in db.execute(
                    'SELECT DISTINCT f.owner FROM resource_refs f '
                    'JOIN resources r ON r.id=f.resource '
                    "WHERE f.owner LIKE 'prepare:%' AND r.kind='container' "
                    "AND r.owned=1 AND r.status!='deleted'"))
        except Exception as exc:
            errors.append({'scope': 'persistent_preparation_ownership',
                           'error': type(exc).__name__})
    for owner in sorted(owners):
        try:
            resources.extend(cancel_owned_containers(owner, root))
        except Exception as exc:
            errors.append({'owner': owner, 'error': type(exc).__name__})
    return {'stopped': not errors and all(result['stopped'] for result in resources),
            'resources': resources, 'errors': errors}


def close_jobs(root):
    root = Path(root).resolve()
    with _services_lock:
        _closed_roots.add(root)
        service = _services.get(root)
    if service:
        return service.close()
    return _close_preparation_resources(root)


def cancel_owner(owner, root=None):
    """退出时复用管理者或只检查原等待记录，不能为了暂停重新启动准备服务。"""
    root = current_root(root)
    with _services_lock:
        service, closed = _services.get(root), root in _closed_roots
    if service:
        return service.cancel_owner(owner)
    if not closed:
        return jobs(root).cancel_owner(owner)
    if not (root / 'resources.sqlite3').is_file():
        return []
    catalog = resource_catalog(root)
    with catalog._db() as db:
        rows = list(db.execute('SELECT j.value,w.token FROM environment_waiters w '
            'JOIN environment_jobs j ON j.id=w.job WHERE w.owner=? '
            "AND j.status IN ('preparing','interrupted','cancelled','failed')", (owner,)))
    results = []
    for row in rows:
        value = json.loads(row['value'])
        try:
            if value.get('owner') != 'prepare:' + value['id']:
                raise ValueError('准备作业归属不一致。')
            resources = cancel_owned_containers(value['owner'], root)
            if any(not result['stopped'] for result in resources):
                raise EnvironmentRequired('环境准备容器停止未确认，现场已保留。')
            # 整个服务已关闭，不再有其他等待者继续安装；只解除当前任务的等待关系。
            catalog.detach_environment_waiter(value['id'], row['token'])
            if value['status'] in {'preparing', 'interrupted'}:
                catalog.save_environment_job({**value, 'status': 'cancelled',
                    'stage': 'cancelled', 'message': '环境准备已取消，现场已保留。'})
            results.append({'preparation_job': value['id'], 'stopped': True})
        except Exception as exc:
            results.append({'preparation_job': value.get('id'), 'stopped': False,
                            'error': type(exc).__name__})
    return results


def active_count(root=None):
    with _services_lock:
        service = _services.get(current_root(root))
    if service is None:
        return 0
    with service.lock:
        return sum(not item['future'].done() for item in service.workers.values())


def ensure_prepared(repo, *, progress=None, cancelled=None, root=None, project_scope=None, retry=False):
    options = {"project_scope": project_scope} if project_scope is not None else {}
    return jobs(root).wait(repo, progress=progress, cancelled=cancelled, retry=retry, **options)
