"""资源归属、引用与空间预算；不猜测用户文件或 Docker 资源的归属。"""

from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
from threading import RLock
import time
import uuid

from agent.storage.preferences import state_root

RESERVE_BYTES = 5 * 1024**3
IDLE_CACHE_BYTES = 20 * 1024**3
_owner = ContextVar('monoplanner_resource_owner', default=None)
_root = ContextVar('monoplanner_resource_root', default=None)
_active = {}
_cancelled = set()
_active_lock = RLock()


def preflight_space(destination, required_bytes, *, reserve_bytes=None):
    """复制开始前检查目标盘；这不是对 Docker 可写层的硬配额。"""
    if reserve_bytes is None:
        settings = current_root() / 'resource-limits.json'
        reserve_bytes = json.loads(settings.read_text('utf-8'))['reserve_bytes'] if settings.exists() else RESERVE_BYTES
    if required_bytes < 0 or reserve_bytes < 0:
        raise ValueError('空间预算不能为负数。')
    path = Path(destination).resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < required_bytes + reserve_bytes:
        raise OSError(
            f'磁盘空间不足：需新增 {required_bytes:,} 字节，并保留 '
            f'{reserve_bytes:,} 字节；当前可用 {free:,} 字节。'
        )
    return {'required_bytes': required_bytes, 'free_bytes': free, 'reserve_bytes': reserve_bytes}


def _safe_tree(path, root):
    path, root = Path(path), Path(root).resolve()
    if path.resolve() == root or not path.resolve().is_relative_to(root):
        raise ValueError('资源路径不在私有记录目录中。')
    for item in (path, *path.parents):
        if item == root.parent:
            break
        if item.exists() or item.is_symlink():
            info = item.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError('资源目录不能包含链接。')
    if path.exists() and path.is_dir():
        for directory, folders, files in os.walk(path, followlinks=False):
            for name in folders + files:
                info = (Path(directory) / name).lstat()
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                    raise ValueError('资源目录不能包含链接。')
    return path


def remove_private_tree(path, root):
    """只清理已核验的私有派生目录；原件归档由调用方保留。"""
    path = _safe_tree(path, root)
    if path.exists():
        def readonly(function, name, error):
            target = _safe_tree(name, root)
            if not isinstance(error, PermissionError):
                raise error
            target.chmod(target.stat().st_mode | stat.S_IWRITE)
            function(name)
        shutil.rmtree(path, onexc=readonly)


class ResourceCatalog:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'resources.sqlite3'
        from agent.storage.schema import initialize

        # 原生资源协议独立固定为 1，不跟随应用数据库版本自动迁移。
        initialize(self.path, target=1, migrations={})
        if self.path.is_symlink():
            raise ValueError('资源台账不能是链接。')
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, location TEXT NOT NULL,
                    owned INTEGER NOT NULL, size_bytes INTEGER NOT NULL,
                    metadata TEXT NOT NULL, status TEXT NOT NULL, touched REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS resource_refs (
                    owner TEXT NOT NULL, resource TEXT NOT NULL, PRIMARY KEY(owner, resource));
                CREATE TABLE IF NOT EXISTS resource_leases (
                    token TEXT NOT NULL, owner TEXT NOT NULL, resource TEXT NOT NULL,
                    PRIMARY KEY(token, resource));
                CREATE TABLE IF NOT EXISTS environment_jobs (
                    id TEXT PRIMARY KEY, repo TEXT NOT NULL, status TEXT NOT NULL,
                    value TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS environment_waiters (
                    token TEXT PRIMARY KEY, job TEXT NOT NULL, owner TEXT NOT NULL);
            ''')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, resource_id, kind, location, *, owned=True, size_bytes=None,
                 metadata=None, owner=None):
        if kind not in {'image', 'volume', 'container', 'network', 'temporary', 'evidence'}:
            raise ValueError('未知资源类型。')
        if kind in {'temporary', 'evidence'}:
            _safe_tree(location, self.root)
        if size_bytes is not None and (type(size_bytes) is not int or size_bytes < 0):
            raise ValueError('资源大小必须为非负整数。')
        metadata = dict(metadata or {})
        metadata['size_known'] = size_bytes is not None
        metadata.setdefault('estimated', kind in {'image', 'volume', 'container'})
        value = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM resources WHERE id=?', (resource_id,)).fetchone()
            if old and (old['kind'], old['location'], old['owned']) != (kind, str(location), int(owned)):
                raise ValueError('资源身份发生变化。')
            if old and old['status'] == 'deleting':
                raise ValueError('资源正在清理。')
            db.execute('INSERT OR REPLACE INTO resources VALUES(?,?,?,?,?,?,?,?)',
                       (resource_id, kind, str(location), int(owned), size_bytes or 0,
                        value, 'ready', time.time()))
            if owner:
                db.execute('INSERT OR IGNORE INTO resource_refs VALUES(?,?)', (owner, resource_id))
        return resource_id

    def link(self, owner, resource_id):
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status FROM resources WHERE id=?', (resource_id,)).fetchone()
            if not row or row['status'] != 'ready':
                raise ValueError('资源不可用，不能建立引用。')
            db.execute('INSERT OR IGNORE INTO resource_refs VALUES(?,?)', (owner, resource_id))

    def unlink(self, owner, resource_id=None):
        with self._db() as db:
            if resource_id is None:
                db.execute('DELETE FROM resource_refs WHERE owner=?', (owner,))
            else:
                db.execute('DELETE FROM resource_refs WHERE owner=? AND resource=?',
                           (owner, resource_id))

    def record_container_stop(self, owner, name, receipt):
        """只同步已核验的原容器状态，不改写任务执行结果或删除故障证据。"""
        state = receipt.get('resource_state')
        if state not in {'missing', 'stopped'} or not receipt.get('stopped'):
            return
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = list(db.execute('SELECT r.id,r.metadata FROM resources r '
                'JOIN resource_refs f ON r.id=f.resource WHERE f.owner=? AND r.location=? '
                "AND r.kind='container' AND r.owned=1 AND r.status!='deleted'", (owner, name)))
            for row in rows:
                metadata = json.loads(row['metadata'])
                identity = receipt.get('container_id')
                if metadata.get('container_id') not in {None, identity}:
                    raise ValueError('停止回执与原容器身份不一致。')
                metadata.update(container_id=identity, resource_state=state,
                                stop_confirmed_at=time.time())
                db.execute('UPDATE resources SET metadata=?,status=?,touched=? WHERE id=?', (
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    'deleted' if state == 'missing' else 'ready', time.time(), row['id']))

    @contextmanager
    def lease(self, owner, resource_ids):
        token = uuid.uuid4().hex
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            for resource_id in resource_ids:
                row = db.execute('SELECT status FROM resources WHERE id=?', (resource_id,)).fetchone()
                if not row or row['status'] != 'ready':
                    raise ValueError('资源不可用，不能开始操作。')
                db.execute('INSERT INTO resource_leases VALUES(?,?,?)', (token, owner, resource_id))
        try:
            yield token
        finally:
            with self._db() as db:
                db.execute('DELETE FROM resource_leases WHERE token=?', (token,))

    def usage(self):
        with self._db() as db:
            rows = [dict(row) for row in db.execute('''SELECT resources.*,
                (SELECT COUNT(*) FROM resource_refs WHERE resource=resources.id) AS "references",
                (SELECT COUNT(*) FROM resource_leases WHERE resource=resources.id) AS leases
                FROM resources WHERE status != 'deleted' ORDER BY touched''')]
        for row in rows:
            row['metadata'] = json.loads(row['metadata'])
            row['size_known'] = row['metadata'].get('size_known', False)
            row['estimated'] = row['metadata'].get('estimated', True)
            row['unknown_size'] = not row['size_known']
            if not row['size_known']:
                row['size_bytes'] = None
        with self._db() as db:
            task_refs = dict(db.execute("SELECT resource,COUNT(*) FROM resource_refs "
                                        "WHERE owner LIKE 'task:%' GROUP BY resource"))
            owners = {}
            for owner, identity in db.execute('SELECT owner,resource FROM resource_refs'):
                owners.setdefault(identity, set()).add(owner)
        idle_ids = {row['id'] for row in rows if row['owned'] and row['kind'] != 'evidence'
                    and not row['references'] and not row['leases']}
        # 闲置环境的依赖仍有 environment 引用；其体积不能因此漏计为零。
        while True:
            idle_owners = {'environment:' + row['location'] for row in rows
                           if row['id'] in idle_ids and row['kind'] == 'image'}
            additions = {row['id'] for row in rows if row['owned'] and row['kind'] != 'evidence'
                         and not row['leases'] and owners.get(row['id'], set()) <= idle_owners}
            if additions <= idle_ids:
                break
            idle_ids.update(additions)
        settings_path = self.root / 'resource-limits.json'
        limits = json.loads(settings_path.read_text('utf-8')) if settings_path.exists() else {
            'cache_limit_bytes': IDLE_CACHE_BYTES, 'reserve_bytes': RESERVE_BYTES}
        total = sum(row['size_bytes'] or 0 for row in rows)
        environments = [{
            'id': row['id'], 'repo': row['metadata'].get('repo', ''),
            'size_bytes': row['size_bytes'], 'status': row['status'],
            'last_used': row['touched'], 'pinned': bool(row['references']),
            'active_task_refs': task_refs.get(row['id'], 0),
            'in_use': bool(row['leases'] or task_refs.get(row['id'], 0)),
            'size_known': row['size_known'], 'unknown_size': row['unknown_size'],
            'estimated': row['estimated'],
        } for row in rows if row['kind'] == 'image']
        from agent.storage.disk_space import docker_storage_status

        return {'resources': rows, 'total_bytes': total, 'used_bytes': total,
                'unknown_size': any(row['unknown_size'] for row in rows),
                'unknown_size_count': sum(row['unknown_size'] for row in rows),
                'estimated': any(row['estimated'] for row in rows if row['size_known']),
                'idle_bytes': sum(row['size_bytes'] or 0 for row in rows if row['id'] in idle_ids),
                'idle_unknown_count': sum(row['unknown_size'] for row in rows
                                          if row['id'] in idle_ids),
                'idle_limit_bytes': limits['cache_limit_bytes'], **limits,
                'free_bytes': shutil.disk_usage(self.root).free, 'environments': environments,
                'docker_storage': docker_storage_status()}

    def set_limits(self, cache_limit_bytes, reserve_bytes):
        from agent.storage.preferences import atomic_write

        if any(type(value) is not int or value < 1024**3
               for value in (cache_limit_bytes, reserve_bytes)):
            raise ValueError('缓存上限和磁盘预留至少为 1 GiB。')
        atomic_write(self.root / 'resource-limits.json', json.dumps({
            'cache_limit_bytes': cache_limit_bytes, 'reserve_bytes': reserve_bytes,
        }).encode('utf-8'))
        return self.usage()

    def environment_job(self, repo=None, *, job_id=None):
        with self._db() as db:
            row = (db.execute('SELECT value FROM environment_jobs WHERE id=?', (job_id,)).fetchone()
                   if job_id else db.execute('SELECT value FROM environment_jobs WHERE repo=? '
                             'ORDER BY created DESC LIMIT 1', (str(repo),)).fetchone())
        return json.loads(row[0]) if row else None

    def save_environment_job(self, value):
        value = {**value, 'updated_at': time.time()}
        with self._db() as db:
            db.execute('INSERT INTO environment_jobs VALUES(?,?,?,?,?) ON CONFLICT(id) '
                       'DO UPDATE SET status=excluded.status,value=excluded.value',
                       (value['id'], value['repo'], value['status'],
                        json.dumps(value, ensure_ascii=False), time.time()))
        return value

    def recover_environment_jobs(self):
        # 只有持有环境管理租约的调用者可以标记上一进程的中断；不重放安装。
        with self._db() as db:
            rows = list(db.execute("SELECT value FROM environment_jobs WHERE status='preparing'"))
        for row in rows:
            self.save_environment_job({**json.loads(row[0]), 'status': 'interrupted',
                'stage': 'interrupted', 'message': '上次准备被中断，现场已保留。',
                'error': '重新准备前会核验并停止上次准备容器，不会恢复未知安装操作。'})

    def attach_environment_waiter(self, job, owner, *, token=None):
        token = token or uuid.uuid4().hex
        with self._db() as db:
            db.execute('INSERT OR IGNORE INTO environment_waiters VALUES(?,?,?)', (token, job, owner or ''))
        return token

    def detach_environment_waiter(self, job, token):
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('DELETE FROM environment_waiters WHERE token=? AND job=?', (token, job))
            return [row[0] for row in db.execute(
                'SELECT owner FROM environment_waiters WHERE job=?', (job,))]

    def environment_waiters(self, job):
        with self._db() as db:
            return [dict(row) for row in db.execute(
                'SELECT token,owner FROM environment_waiters WHERE job=?', (job,))]

    def finish_resources(self, owner, identities, *, completed, docker=None):
        """唯一终结入口：未知资源保留引用；确认完成才停止并清理自有资源。"""
        if not completed:
            return {'removed': [], 'preserved': True}
        for identity in identities:
            self.unlink(owner, identity)
        return self.cleanup(identities, docker=docker or (
            lambda kind, identity, metadata: delete_owned_docker_resource(
                kind, identity, metadata, stop_running=True)))

    def cleanup(self, resource_ids=None, *, docker=None, limit_bytes=None):
        """一次选择一个闲置环境，保留未选择环境的依赖引用。"""
        selected = set(resource_ids) if resource_ids is not None else None
        if limit_bytes is None:
            limit_bytes = self.usage()['cache_limit_bytes']
        report = {'removed': [], 'retained': [], 'errors': []}
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = {row['id']: dict(row) for row in db.execute(
                "SELECT * FROM resources WHERE status != 'deleted' ORDER BY touched")}
            for row in rows.values():
                row['metadata'] = json.loads(row['metadata'])

            def referenced(identity, ignored_owner=None):
                return bool(db.execute(
                    'SELECT 1 FROM resource_refs WHERE resource=? AND owner<>? UNION ALL '
                    'SELECT 1 FROM resource_leases WHERE resource=?',
                    (identity, ignored_owner or '', identity),
                ).fetchone())

            def candidates():
                values = []
                for row in rows.values():
                    if (row['status'] == 'deleted' or not row['owned']
                            or row['kind'] == 'evidence' or referenced(row['id'])):
                        continue
                    members = [row]
                    if row['kind'] == 'image':
                        owner = 'environment:' + row['location']
                        for child_id, in db.execute(
                            'SELECT resource FROM resource_refs WHERE owner=?', (owner,)
                        ):
                            child = rows.get(child_id)
                            if (child and child['status'] != 'deleted' and child['owned']
                                    and not referenced(child_id, owner)):
                                members.append(child)
                    values.append(members)
                return values

            def idle_size(groups):
                unique = {row['id']: row for group in groups for row in group}
                return sum(row['size_bytes'] for row in unique.values())

            groups = candidates()
            groups.sort(key=lambda group: (
                0 if group[0]['kind'] in {'container', 'temporary'} else
                (2 if group[0]['kind'] == 'network' else 1),
                group[0]['touched'],
            ))
            considered = set()
            for group in groups:
                primary = group[0]
                if primary['status'] == 'deleted':
                    continue
                if selected is not None and primary['id'] not in selected:
                    continue
                unknown = any(not row['metadata'].get('size_known', False) for row in group)
                if selected is None and idle_size(candidates()) <= limit_bytes and not unknown:
                    continue
                considered.add(primary['id'])
                if primary['kind'] == 'image':
                    report['retained'].extend(child_id for child_id, in db.execute(
                        'SELECT resource FROM resource_refs WHERE owner=?',
                        ('environment:' + primary['location'],))
                        if child_id in rows and referenced(
                            child_id, 'environment:' + primary['location']))
                # 镜像先删：如果它仍被 Docker 使用，不能先破坏其依赖卷。
                # 镜像删除成功后，失败的卷会明确留下为闲置资源，可单独重试清理。
                for row in group:
                    if row['status'] == 'deleted':
                        continue
                    identity = row['id']
                    ignored = ('environment:' + primary['location']
                               if row is not primary and primary['kind'] == 'image' else None)
                    if referenced(identity, ignored):
                        report['retained'].append(identity)
                        continue
                    try:
                        db.execute("UPDATE resources SET status='deleting' WHERE id=?", (identity,))
                        if row['kind'] == 'temporary':
                            remove_private_tree(row['location'], self.root)
                        elif docker is not None:
                            docker(row['kind'], row['location'], row['metadata'])
                        else:
                            raise ValueError('Docker 清理需要受信任的资源删除器。')
                        row['status'] = 'deleted'
                        db.execute("UPDATE resources SET status='deleted' WHERE id=?", (identity,))
                        db.execute('DELETE FROM resource_refs WHERE resource=?', (identity,))
                        if row['kind'] == 'image':
                            db.execute('DELETE FROM resource_refs WHERE owner=?',
                                       ('environment:' + row['location'],))
                        report['removed'].append(identity)
                    except Exception as exc:
                        db.execute("UPDATE resources SET status='ready' WHERE id=?", (identity,))
                        report['errors'].append({'id': identity, 'error': type(exc).__name__})
                        # 未能删除环境镜像时，其依赖引用保持原状。
                        if row is primary:
                            break
            if selected is not None:
                report['retained'].extend(identity for identity in selected
                    if identity not in considered and identity in rows
                    and rows[identity]['status'] != 'deleted')
        report['retained'] = list(dict.fromkeys(report['retained']))
        return report


def resource_catalog(root=None):
    return ResourceCatalog(current_root(root))


def current_root(root=None):
    return Path(root or _root.get() or state_root()).resolve()


def current_owner(default=None):
    return _owner.get() or default


def ownership_labels(owner, root=None):
    return {'com.monoplanner.owner': owner,
            'com.monoplanner.state': hashlib.sha256(str(current_root(root)).encode()).hexdigest()}


@contextmanager
def container_owner(owner, root=None):
    directory = str(current_root(root))
    nested = _owner.get() == owner and _root.get() == directory
    token = _owner.set(owner)
    root_token = _root.set(directory)
    if not nested:
        with _active_lock:
            _cancelled.discard((directory, owner))
    try:
        yield owner
    finally:
        _owner.reset(token)
        _root.reset(root_token)


def owner_cancelled(owner=None, root=None):
    with _active_lock:
        return (str(current_root(root)), owner or current_owner()) in _cancelled


def register_active_container(owner, container_id, kill_callback, *, root=None):
    """暂停只终止本任务已登记容器；不通过结束 Docker CLI 假装容器已停止。"""
    key = (str(current_root(root)), owner, container_id)
    with _active_lock:
        _active[key] = kill_callback
    return key


def unregister_active_container(key):
    with _active_lock:
        _active.pop(key, None)


def cancel_owned_containers(owner, root=None):
    directory = str(current_root(root))
    with _active_lock:
        _cancelled.add((directory, owner))
        selected = [(key, stop) for key, stop in _active.items()
                    if key[:2] == (directory, owner)]
    results = []
    for key, stop in selected:
        try:
            detail = stop()
            results.append({'container': key[2], 'stopped': True,
                            **(detail if isinstance(detail, dict) else {})})
        except Exception as exc:
            results.append({'container': key[2], 'stopped': False, 'error': type(exc).__name__})
    # 服务重启后内存回调消失，仍可凭持久的原 ID 和标签终止这个任务自己的容器。
    path = Path(directory) / 'resources.sqlite3'
    active_names = {key[2] for key, _ in selected}
    if path.is_file():
        try:
            catalog = resource_catalog(directory)
            for result in results:
                catalog.record_container_stop(owner, result['container'], result)
            with catalog._db() as db:
                saved = list(db.execute('SELECT r.location,r.metadata FROM resources r '
                    'JOIN resource_refs f ON r.id=f.resource WHERE f.owner=? '
                    "AND r.kind='container' AND r.owned=1 AND r.status!='deleted'", (owner,)))
        except Exception as exc:
            # 当前进程已执行的终止回执不能因台账故障丢失；没有身份就不能猜杀别的容器。
            results.append({'stopped': False, 'error': type(exc).__name__,
                            'recovered': True, 'scope': 'persistent_container_ownership'})
            return results
        for row in saved:
            if row['location'] in active_names:
                continue
            try:
                receipt = stop_recorded_container(
                    row['location'], json.loads(row['metadata']), owner, directory)
                result = {'container': row['location'], 'stopped': True, 'recovered': True,
                          **(receipt or {})}
                results.append(result)
                catalog.record_container_stop(owner, row['location'], result)
            except Exception as exc:
                results.append({'container': row['location'], 'stopped': False,
                                'error': type(exc).__name__, 'recovered': True})
    return results


def stop_recorded_container(name, metadata, owner, root):
    import re
    import subprocess
    from agent.execution.docker import docker_executable

    identity = metadata.get('container_id')
    labels = ownership_labels(owner, root)
    if (not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name)
            or (identity is not None and (
                not isinstance(identity, str) or not re.fullmatch(r'[0-9a-f]{64}', identity)))
            or metadata.get('labels') != labels
            or (identity is None and not re.fullmatch(
                r'sha256:[0-9a-f]{64}', str(metadata.get('image', ''))))):
        raise ValueError('原容器身份不完整，不能按名称终止。')
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    def query(args):
        return subprocess.run([docker_executable(), *args], capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=5, **options)

    def inspect(target):
        inspected = query(['container', 'inspect', target])
        if inspected.returncode:
            error = (inspected.stderr + inspected.stdout).lower()
            if 'no such container' in error or 'no such object' in error:
                return None
            raise OSError('无法核验原任务容器。')
        actual = json.loads(inspected.stdout)[0]
        actual_id = actual.get('Id')
        saved_labels = actual.get('Config', {}).get('Labels') or {}
        if (not isinstance(actual_id, str) or not re.fullmatch(r'[0-9a-f]{64}', actual_id)
                or (identity and actual_id != identity) or actual.get('Name') != '/' + name
                or any(saved_labels.get(key) != value for key, value in labels.items())
                or (metadata.get('image') and actual.get('Image') != metadata['image'])):
            raise ValueError('原容器归属已变化，未终止任何容器。')
        if type(actual.get('State', {}).get('Running')) is not bool:
            raise OSError('原任务容器运行状态不完整。')
        return actual

    # 创建命令可能未返回 ID；按预先登记的唯一名称只查询，标签和镜像均吻合后
    # 固定实际 ID。Docker 明确报告不存在，与引擎无响应是不同的状态。
    actual = inspect(identity or name)
    if actual is None:
        return {'stopped': True, 'resource_state': 'missing', 'container_id': identity}
    identity = actual['Id']
    if actual['State']['Running']:
        query(['kill', identity])
        # 即使 kill 报错，也可能是容器刚好自行退出；最终状态必须再次查询确认。
        actual = inspect(identity)
        if actual is not None and actual['State']['Running']:
            raise OSError('Docker 未确认原任务容器已停止。')
    return {'stopped': True, 'resource_state': 'missing' if actual is None else 'stopped',
            'container_id': identity}


def release_task_resources(owner, *, root=None, completed=False, docker=None):
    if not completed:
        return {'removed': [], 'preserved': True}
    catalog = resource_catalog(root)
    with catalog._db() as db:
        identities = [row[0] for row in db.execute(
            'SELECT resource FROM resource_refs WHERE owner=?', (owner,))]
        # 只有调用方已确认任务终结才释放它的遗留租约；未知任务不走此分支。
        db.execute('DELETE FROM resource_leases WHERE owner=?', (owner,))
    catalog.unlink(owner)
    with _active_lock:
        _cancelled.discard((str(current_root(root)), owner))
    return catalog.cleanup(identities, docker=docker)


def delete_owned_docker_resource(kind, identity, metadata, *, query=None, stop_running=False):
    """核验不可变 ID 或创建标签后才删除，不接受容器内部提供的 Shell。"""
    import subprocess
    from agent.execution.docker import docker_executable

    if kind not in {'container', 'image', 'volume', 'network'} or not identity or identity.startswith('-'):
        raise ValueError('无效 Docker 资源。')
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    def execute(args):
        return subprocess.run([docker_executable(), *args], capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=15, **options)
    query = query or execute
    inspected = query([kind, 'inspect', identity])
    if inspected.returncode:
        if any(text in (inspected.stderr + inspected.stdout).lower() for text in
               ('no such container', 'no such image', 'no such volume', 'no such network', 'no such object')):
            return
        raise OSError('Docker 资源检查失败。')
    item = json.loads(inspected.stdout)[0]
    if kind == 'container':
        if not metadata.get('container_id') or item.get('Id') != metadata['container_id']:
            raise ValueError('容器身份未确认，保留现场。')
        labels = metadata.get('labels') or {}
        if any((item.get('Config', {}).get('Labels') or {}).get(key) != value
               for key, value in labels.items()):
            raise ValueError('容器归属标签不一致。')
        if item.get('State', {}).get('Running'):
            if not stop_running:
                raise ValueError('运行中的容器不能被清理。')
            stopped = query(['kill', item['Id']])
            if stopped.returncode:
                raise OSError('容器停止未确认，保留现场。')
        if metadata.get('builder'):
            builder = metadata['builder']
            if (not builder.startswith('monoplanner-build-') or not labels
                    or identity != 'buildx_buildkit_' + metadata.get('builder_node', '')
                    or metadata.get('builder_volume') != identity + '_state'):
                raise ValueError('构建容器记录不完整，保留现场。')
            volume = metadata['builder_volume']
            found = query(['volume', 'inspect', volume])
            if found.returncode:
                raise OSError('构建缓存卷未确认，保留现场。')
            volume_labels = json.loads(found.stdout)[0].get('Labels') or {}
            if (volume_labels.get('com.monoplanner.builder') != builder
                    or any(volume_labels.get(key) != value for key, value in labels.items())):
                raise ValueError('构建缓存卷归属不一致。')
            removed = query(['buildx', 'rm', '--force', builder])
            if removed.returncode:
                raise OSError('Docker 未删除构建器。')
            # builder 元数据可能丢失；只补删刚核验过的原 ID 和自有卷。
            for resource_kind, target in (('container', metadata['container_id']), ('volume', volume)):
                result = query([resource_kind, 'rm', target])
                if result.returncode and not any(text in (result.stderr + result.stdout).lower() for text in (
                        'no such container', 'no such volume', 'no such object')):
                    raise OSError('构建器仍有未清理的资源。')
            return
    elif kind == 'image':
        labels = item.get('Config', {}).get('Labels') or {}
        managed = (metadata.get('fingerprint') and labels.get('org.monoplanner.project-fingerprint')
                   == metadata['fingerprint']) or (metadata.get('base_key')
                   and labels.get('org.monoplanner.managed-base') == metadata['base_key'])
        downloaded = metadata.get('downloaded') == 'moby/buildkit:v0.34.0'
        if item.get('Id') != identity or not (managed or downloaded):
            raise ValueError('镜像归属未确认。')
    elif kind == 'network':
        labels = metadata.get('labels') or {}
        if (not labels or item.get('Id') != metadata.get('network_id') or item.get('Containers')
                or any((item.get('Labels') or {}).get(key) != value for key, value in labels.items())):
            raise ValueError('准备网络身份未确认或仍有容器连接。')
    elif kind == 'volume':
        labels = item.get('Labels') or {}
        if metadata.get('builder'):
            expected = metadata.get('labels') or {}
            if (not expected or labels.get('com.monoplanner.builder') != metadata['builder']
                    or any(labels.get(key) != value for key, value in expected.items())):
                raise ValueError('构建缓存卷归属未确认。')
        elif not metadata.get('identity') or labels.get('com.monoplanner.dependency') != metadata['identity']:
            raise ValueError('依赖卷归属未确认。')
    removed = query([kind, 'rm', identity])
    if removed.returncode:
        raise OSError('Docker 未删除资源；可能仍被其他任务使用。')


def release_project(repo, *, root=None, retain=True):
    catalog = resource_catalog(root)
    owner = 'project:' + hashlib.sha256(str(Path(repo).resolve()).encode()).hexdigest()
    if retain:
        return {'retained': True, 'removed': []}
    with catalog._db() as db:
        identities = [row[0] for row in db.execute(
            'SELECT resource FROM resource_refs WHERE owner=?', (owner,))]
    catalog.unlink(owner)
    return catalog.cleanup(identities, docker=delete_owned_docker_resource)
