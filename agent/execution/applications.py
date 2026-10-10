"""Discover installed Windows applications without interpreting launch commands."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from threading import RLock
import time
from agent.settings import PROJECT_ROOT

_cache, _updated, _lock = [], 0, RLock()


def discover(refresh=False):
    global _cache, _updated
    with _lock:
        if _cache and not refresh and time.monotonic() - _updated < 300:
            return [dict(a) for a in _cache]
        found = []
        if os.name == 'nt':
            import winreg
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
                    try:
                        with winreg.OpenKey(hive, r'SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths',
                                            0, winreg.KEY_READ | view) as registry:
                            for index in range(winreg.QueryInfoKey(registry)[0]):
                                name = winreg.EnumKey(registry, index)
                                try:
                                    with winreg.OpenKey(registry, name) as key:
                                        path = Path(os.path.expandvars(winreg.QueryValue(key, None).strip('"')))
                                    if path.is_absolute() and path.is_file() and path.suffix.lower() == '.exe':
                                        found.append({'name': path.stem, 'path': str(path.resolve()), 'kind': 'exe'})
                                except OSError:
                                    continue
                    except OSError:
                        continue
            helper = PROJECT_ROOT/'launcher/runtime/MonoPlanner.Applications.exe'
            if helper.is_file():
                response = subprocess.run([str(helper), 'list'], capture_output=True, encoding='utf-8',
                                          timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
                if response.returncode == 0:
                    found.extend(json.loads(response.stdout))
            notepad = Path(os.environ['SystemRoot'])/'System32/notepad.exe'
            if notepad.is_file():
                found.append({'name': '记事本', 'path': str(notepad), 'kind': 'exe'})
        items = {}
        for item in found:
            target = item.get('aumid') if item['kind'] == 'packaged' else item['path']
            identity = hashlib.sha256((item['kind']+':'+target.casefold()).encode()).hexdigest()[:24]
            items[identity] = {**item, 'id': identity}
        _cache, _updated = sorted(items.values(), key=lambda a: a['name'].casefold()), time.monotonic()
        return [dict(a) for a in _cache]


def identity(app):
    if app.get('kind', 'exe') == 'exe':
        with Path(app['path']).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashlib.sha256(json.dumps(app, sort_keys=True).encode()).hexdigest()


def start(app, path=None):
    from agent.execution.native import clean_environment
    if app.get('kind', 'exe') == 'exe':
        process = subprocess.Popen([app['path']] + ([str(path)] if path else []), close_fds=True,
                                   env=clean_environment())
        return process.pid
    helper = PROJECT_ROOT/'launcher/runtime/MonoPlanner.Applications.exe'
    result = subprocess.run([str(helper), 'activate', app['aumid'], str(path or '')],
                            capture_output=True, encoding='utf-8', timeout=15,
                            creationflags=subprocess.CREATE_NO_WINDOW, env=clean_environment())
    if result.returncode:
        raise OSError('Windows 未能启动此应用：'+result.stderr.strip())
    return int(result.stdout.strip())
