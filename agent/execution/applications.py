"""发现 Windows 已安装应用，不解释或执行登记中的启动命令。"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import stat
from threading import RLock
import time
from agent.settings import PROJECT_ROOT

_cache, _updated, _lock = [], 0, RLock()


def executable_path(value):
    """只接受真实的本地可执行文件，不接受快捷方式或启动命令。"""
    path = Path(value)
    if (
        not path.is_absolute()
        or path.suffix.lower() != '.exe'
        or '..' in path.parts
        or str(path).startswith(('\\\\', '//'))
        or ':' in str(path)[2:]
        or not path.is_file()
    ):
        raise ValueError('应用入口必须是已有的本机 EXE 文件')
    for parent in (path, *path.parents):
        info = parent.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
            raise ValueError('应用入口不能经过链接或 Windows 重解析点')
    return path.resolve()


def discover(refresh=False):
    global _cache, _updated
    with _lock:
        if _cache and not refresh and time.monotonic() - _updated < 300:
            return [dict(a) for a in _cache if available(a)]
        found = []
        if os.name == 'nt':
            import winreg

            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                    try:
                        with winreg.OpenKey(
                            hive,
                            r'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths',
                            0,
                            winreg.KEY_READ | view,
                        ) as registry:
                            for index in range(winreg.QueryInfoKey(registry)[0]):
                                name = winreg.EnumKey(registry, index)
                                try:
                                    with winreg.OpenKey(registry, name) as key:
                                        path = Path(
                                            os.path.expandvars(
                                                winreg.QueryValue(key, None).strip('"')
                                            )
                                        )
                                    if (
                                        path.is_absolute()
                                        and path.is_file()
                                        and path.suffix.lower() == '.exe'
                                    ):
                                        found.append(
                                            {
                                                'name': path.stem,
                                                'path': str(path.resolve()),
                                                'kind': 'exe',
                                            }
                                        )
                                except OSError:
                                    continue
                    except OSError:
                        continue
            helper = PROJECT_ROOT / 'launcher/runtime/MonoPlanner.Applications.exe'
            if helper.is_file():
                response = subprocess.run(
                    [str(helper), 'list'],
                    capture_output=True,
                    encoding='utf-8',
                    timeout=15,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    env=_launch_environment(),
                )
                if response.returncode == 0:
                    found.extend(json.loads(response.stdout))
            notepad = Path(os.environ['SystemRoot']) / 'System32/notepad.exe'
            if notepad.is_file():
                found.append({'name': '记事本', 'path': str(notepad), 'kind': 'exe'})
        items = {}
        for item in found:
            if not available(item):
                continue
            target = item.get('aumid') if item['kind'] == 'packaged' else item['path']
            identity = hashlib.sha256(
                (item['kind'] + ':' + target.casefold()).encode()
            ).hexdigest()[:24]
            items[identity] = {**item, 'id': identity}
        _cache, _updated = (
            sorted(items.values(), key=lambda a: a['name'].casefold()),
            time.monotonic(),
        )
        return [dict(a) for a in _cache]


def available(app):
    if app.get('kind') == 'packaged':
        return bool(app.get('aumid') and '!' in app['aumid'])
    try:
        executable_path(app['path'])
        return app.get('kind', 'exe') == 'exe'
    except (OSError, KeyError, TypeError, ValueError):
        return False


def describe_executable(value):
    path = executable_path(value)
    target = str(path)
    identity = hashlib.sha256(('exe:' + target.casefold()).encode()).hexdigest()[:24]
    return {'name': path.stem, 'path': target, 'kind': 'exe', 'id': identity}


def identity(app):
    if app.get('kind', 'exe') == 'exe':
        with executable_path(app['path']).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashlib.sha256(json.dumps(app, sort_keys=True).encode()).hexdigest()


def _launch_environment():
    """外部应用不能继承助手的凭据。"""
    names = (
        'SystemRoot',
        'WINDIR',
        'COMSPEC',
        'SystemDrive',
        'USERPROFILE',
        'LOCALAPPDATA',
        'APPDATA',
        'HOMEDRIVE',
        'HOMEPATH',
    )
    values = {k: os.environ[k] for k in names if k in os.environ}
    values.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    return values


def start(app, path=None):
    # 用户打开的应用独立于助手服务；只有此受控入口允许脱离桌面作业对象。
    flags = subprocess.CREATE_BREAKAWAY_FROM_JOB if os.name == 'nt' else 0
    if app.get('kind', 'exe') == 'exe':
        entry = executable_path(app['path'])
        process = subprocess.Popen(
            [str(entry)] + ([str(path)] if path else []), close_fds=True,
            env=_launch_environment(), creationflags=flags,
        )
        return process.pid
    helper = PROJECT_ROOT / 'launcher/runtime/MonoPlanner.Applications.exe'
    result = subprocess.run(
        [str(helper), 'activate', app['aumid'], str(path or '')],
        capture_output=True,
        encoding='utf-8',
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW | flags,
        env=_launch_environment(),
    )
    if result.returncode:
        raise OSError('Windows 未能启动此应用：' + result.stderr.strip())
    return int(result.stdout.strip())
