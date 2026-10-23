"""内容寻址的内核及运行时；任务绑定原件不随应用安装目录变化。"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import uuid

from agent.application.lease import ServiceLease
from agent.settings import PROJECT_ROOT
from agent.storage.records import digest, save
from agent.storage.resources import preflight_space, remove_private_tree
from agent.runtime.distribution import python_files, tree_files


PROTOCOL = 2
SUPPORTED_PROTOCOLS = {1, 2}
BUNDLE_FORMAT = 1


def identity(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Invalid immutable engine identity')
    return value


def safe_path(root, name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name
            or not relative.parts):
        raise ValueError('Invalid engine manifest path')
    root = Path(root).resolve()
    path = root.joinpath(*relative.parts)
    for part in [root, *path.parents, path]:
        if part.is_relative_to(root) and (part.is_symlink() or part.is_junction()):
            raise ValueError('Engine files cannot be links')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Engine file escaped its owned directory')
    return path


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def registry_lock(root):
    folder = Path(root) / 'engine-registry'
    folder.mkdir(parents=True, exist_ok=True)
    lease = ServiceLease(folder)
    lease.acquire()
    try:
        yield
    finally:
        lease.close()


def _publish(root, area, manifest, sources):
    identifier = digest(manifest)
    folder = Path(root) / area
    folder.mkdir(parents=True, exist_ok=True)
    target = safe_path(folder, identifier)
    if target.exists():
        _verify(target, identifier)
        return identifier
    preflight_space(folder, sum(path.stat().st_size for path in sources.values()))
    staging = folder / ('staging-' + uuid.uuid4().hex)
    staging.mkdir()
    try:
        for name, original in sources.items():
            destination = safe_path(staging, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
            if checksum(destination) != manifest['files'][name]:
                raise ValueError('Engine source changed while freezing: ' + name)
        save(staging / 'manifest.json', manifest)
        staging.rename(target)
    finally:
        if staging.exists():
            remove_private_tree(staging, folder)
    _verify(target, identifier)
    return identifier


def _verify(folder, identifier):
    manifest = json.loads(safe_path(folder, 'manifest.json').read_text('utf-8'))
    if digest(manifest) != identity(identifier) or manifest.get('format') != BUNDLE_FORMAT:
        raise ValueError('Engine manifest identity changed')
    # 冻结后多出来的配置或模块也必须拒绝，不能因分发过滤规则忽略被注入的 .env。
    actual = set()
    for path in folder.rglob('*'):
        if path.is_symlink() or path.is_junction():
            raise ValueError('Engine files cannot be links')
        if path.is_file():
            actual.add(path.relative_to(folder).as_posix())
    actual.discard('manifest.json')
    if actual != set(manifest['files']):
        raise ValueError('Engine file inventory changed')
    for name, expected in manifest['files'].items():
        if checksum(safe_path(folder, name)) != expected:
            raise ValueError('Engine file integrity failure: ' + name)
    return manifest


def _manifest(sources, **extra):
    return {'format': BUNDLE_FORMAT, **extra,
            'files': {name: checksum(path) for name, path in sorted(sources.items())}}


def _register(root, *, source=None):
    """只在新任务绑定时冻结；读取或恢复已有任务不能注册当前源码替代旧版本。"""
    root, source = Path(root).resolve(), Path(source or PROJECT_ROOT).resolve()
    runtime = source / 'launcher/runtime'
    python, _ = python_files(runtime / 'python', source / 'requirements.lock')
    runtime_sources = {'python/' + p.relative_to(runtime / 'python').as_posix(): p for p in python}
    for area in ('node', 'git'):
        for path in tree_files(runtime / area):
            runtime_sources[area + '/' + path.relative_to(runtime / area).as_posix()] = path
    for required in ('python/python.exe', 'node/node.exe', 'git/cmd/git.exe'):
        if required not in runtime_sources:
            raise ValueError('Frozen production runtime is missing: ' + required)
    sources = {}
    for area in ('agent', 'resources'):
        for path in tree_files(source / area):
            sources[path.relative_to(source).as_posix()] = path
    for name in ('requirements.txt', 'requirements-web.txt', 'requirements.lock',
                 'launcher/Applications.cs', 'launcher/FilePicker.cs', 'LICENSE'):
        sources[name] = source / name
    for name in ('MonoPlanner.Applications.exe', 'MonoPlanner.FilePicker.exe'):
        path = runtime / name
        if path.is_file():
            sources['launcher/runtime/' + name] = path
    from agent.storage.recovery import code_key

    runtime_id = _publish(root, 'engine-runtimes', _manifest(runtime_sources), runtime_sources)
    manifest = _manifest(sources, protocol=PROTOCOL, runtime_id=runtime_id,
                         code_key=code_key(source))
    return _publish(root, 'engines', manifest, sources)


def register(root, *, source=None):
    with registry_lock(root):
        return _register(root, source=source)


def resolve(root, identifier):
    root = Path(root).resolve()
    folder = safe_path(root / 'engines', identity(identifier))
    manifest = _verify(folder, identifier)
    if manifest.get('protocol') not in SUPPORTED_PROTOCOLS:
        raise ValueError('The task needs an unsupported worker protocol; no work was dispatched')
    runtime_id = identity(manifest['runtime_id'])
    runtime = safe_path(root / 'engine-runtimes', runtime_id)
    _verify(runtime, runtime_id)
    return folder, runtime, manifest


def binding(root, task, *, source=None, create=False):
    if not isinstance(task, str) or len(task) != 32 or any(c not in '0123456789abcdef' for c in task):
        raise ValueError('Invalid work identity')
    root = Path(root).resolve()
    directory = safe_path(root, 'tasks/' + task)
    path = safe_path(directory, 'engine-binding.json')
    if not path.exists():
        if not create:
            raise ValueError('Task engine binding is missing; no replacement was selected')
        if any((directory / name).exists() for name in ('state.json', 'history.sqlite3', 'existing/state.json')):
            raise ValueError('Pre-baseline task needs its original application; no migration was attempted')
        directory.mkdir(parents=True, exist_ok=True)
        with registry_lock(root):
            if path.exists():
                raise ValueError('Task acquired another engine binding; inspect before continuing')
            identifier = _register(root, source=source)
            value = {'format': 1, 'protocol': PROTOCOL, 'task_id': task, 'engine_id': identifier}
            save(path, value)
    value = json.loads(path.read_text('utf-8'))
    if (set(value) != {'format', 'protocol', 'task_id', 'engine_id'} or value['format'] != 1
            or value['protocol'] not in SUPPORTED_PROTOCOLS or value['task_id'] != task):
        raise ValueError('Task engine binding changed')
    _, _, manifest = resolve(root, value['engine_id'])
    if manifest['protocol'] != value['protocol']:
        raise ValueError('Task protocol differs from its pinned engine')
    return value


def collect(root):
    """仅回收无任务引用的内核；未知或损坏的绑定令回收停止。"""
    root = Path(root).resolve()
    with registry_lock(root):
        retained = set()
        for path in (root / 'tasks').glob('*/engine-binding.json'):
            value = binding(root, path.parent.name)
            retained.add(value['engine_id'])
        runtimes = set()
        removed = []
        for folder in (root / 'engines').glob('*'):
            if not folder.is_dir() or folder.name.startswith('staging-'):
                continue
            manifest = _verify(folder, folder.name)
            if folder.name in retained:
                runtimes.add(manifest['runtime_id'])
            else:
                remove_private_tree(folder, root / 'engines')
                removed.append(folder.name)
        for folder in (root / 'engine-runtimes').glob('*'):
            if folder.is_dir() and folder.name not in runtimes and not folder.name.startswith('staging-'):
                _verify(folder, folder.name)
                remove_private_tree(folder, root / 'engine-runtimes')
        return {'removed': removed, 'retained': sorted(retained)}
