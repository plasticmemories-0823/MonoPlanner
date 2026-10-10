"""Windows LPAC execution, isolated source copies and checked publication.

The host owns records, approval and publication. No model program receives host
credentials or write access to the real checkout. Full access is an explicit mode.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from agent.execution.local import LocalEnvironment
from agent.execution.process import log_excerpt
from agent.settings import PROJECT_ROOT


def clean_environment(extra=None):
    names = ('SystemRoot', 'WINDIR', 'COMSPEC', 'SystemDrive', 'USERPROFILE',
             'LOCALAPPDATA', 'APPDATA', 'HOMEDRIVE', 'HOMEPATH')
    values = {k: os.environ[k] for k in names if k in os.environ}
    values.update(PYTHONUTF8='1', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    for key, value in (extra or {}).items():
        if key in {'PYTHONUTF8', 'PYTHONUNBUFFERED', 'PYTHONDONTWRITEBYTECODE', 'NODE_OPTIONS'}:
            values[key] = value
    return values


def safe_tree(root):
    from agent.source.snapshots import _link
    root = Path(root)
    for directory, folders, files in os.walk(root, followlinks=False):
        for name in [*folders, *files]:
            path = Path(directory)/name
            if _link(path) or (path.is_file() and path.stat().st_nlink != 1):
                raise ValueError('隔离副本含有链接或多重硬链接，拒绝发布：' + str(path))


class NativeEnvironment(LocalEnvironment):
    def __init__(self, repo_root, timeout=120, artifact_root=None, *, access_mode='sandbox',
                 read_roots=(), read_only=False):
        super().__init__(repo_root, timeout, artifact_root)
        if access_mode not in {'sandbox', 'full'}:
            raise ValueError('权限必须为 sandbox 或 full')
        if self.artifact_root.resolve().is_relative_to(self.repo_root):
            raise ValueError('隔离运行记录必须位于工作目录之外')
        self.access_mode, self.read_roots, self.read_only = access_mode, list(read_roots), read_only
        from agent.storage.preferences import state_root
        for root in self.read_roots:
            directory = Path(root).resolve()
            if any(protected.resolve().is_relative_to(directory) for protected in (state_root(), PROJECT_ROOT)):
                raise ValueError('只读授权包含助手私有目录，请选择更具体的资料目录')
        self.helper = PROJECT_ROOT/'launcher/runtime/MonoPlanner.Sandbox.exe'
        self.runtime = PROJECT_ROOT/'launcher/runtime/python'
        self.python_executable = str(self.runtime/'python.exe')
        self.runtime_files = ()
        self.current_copy = None

    def host_to_container(self, path):
        path = Path(path)
        if self.current_copy and path.is_relative_to(self.repo_root):
            return str(self.current_copy/path.relative_to(self.repo_root))
        return str(path)

    def _argv(self, command, shell, box, source, inputs, outputs):
        mappings = []
        if shell:
            # Keep the command body byte-exact. The wrapper only selects UTF-8 and
            # the Windows shell; it is not a textual rewrite of supplied code.
            body = inputs/'operation.cmd'
            body.write_bytes(command.encode('utf-8'))
            wrapper = inputs/'wrapper.cmd'
            wrapper.write_text('@echo off\ncall "..\\inputs\\operation.cmd"\nexit /b %errorlevel%\n', encoding='utf-8')
            return [os.environ.get('COMSPEC', r'C:\Windows\System32\cmd.exe'), '/d', '/c', r'..\inputs\wrapper.cmd'], mappings
        argv = list(command)
        if not argv or not Path(argv[0]).is_absolute():
            raise ValueError('隔离执行器需要明确的程序路径')
        if Path(argv[0]).name.lower() in {'python.exe', 'pythonw.exe'}:
            argv[0] = self.python_executable
        else:
            # Only a resolved executable is copied, never a DLL search path from
            # arbitrary environment variables. Project-specific deps may be absent.
            executable = Path(argv[0])
            target = inputs/executable.name
            shutil.copy2(executable, target)
            argv[0] = str(target)
        for index, value in enumerate(argv[1:], 1):
            prefix, _, location = value.partition('pycache_prefix=')
            is_cache = 'pycache_prefix=' in value
            candidate = Path(location if is_cache else value)
            if not candidate.is_absolute():
                continue
            if candidate.is_relative_to(self.repo_root):
                replacement = source/candidate.relative_to(self.repo_root)
            elif any(candidate.is_relative_to(Path(root)) for root in self.read_roots):
                continue  # Explicit read-only ACL, never a publication target.
            elif candidate.is_relative_to(Path(box).parent) or candidate.is_relative_to(self.artifact_root) or candidate.is_relative_to(outputs.parent.parent):
                replacement = outputs/('%02d-' % index + candidate.name)
                if candidate.is_file():
                    replacement = inputs/('%02d-' % index + candidate.name)
                    shutil.copy2(candidate, replacement)
                else:
                    if candidate.is_dir() or is_cache:
                        replacement.mkdir(exist_ok=True)
                    mappings.append((replacement, candidate))
            else:
                raise ValueError('未授权的执行参数路径：' + str(candidate))
            argv[index] = prefix+'pycache_prefix='+str(replacement) if is_cache else str(replacement)
        return argv, mappings

    def _publish(self, before, source, box):
        from agent.source.snapshots import SourceSnapshot, _safe_file, _link
        safe_tree(source)
        after = SourceSnapshot.capture(source, box/'after')
        current = SourceSnapshot.capture(self.repo_root, box/'check', excluded_roots=(self.artifact_root,),
                                         runtime_files=self.runtime_files)
        if current.files != before.files:
            raise ValueError('真实工作目录在隔离执行期间发生变化，修改未发布')
        changes = [name for name in before.files.keys() | after.files.keys()
                   if before.files.get(name) != after.files.get(name)]
        # Validate ALL destinations before publishing the first file.
        for name in changes:
            path = self.repo_root/name
            for ancestor in (path, *path.parents):
                if ancestor.exists() and _link(ancestor):
                    raise ValueError('发布路径含重解析点：'+name)
                if ancestor == self.repo_root:
                    break
            if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
                raise ValueError('发布目标不是独立普通文件：'+name)
        published = []
        try:
            for name in changes:
                path = self.repo_root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                if name in after.files:
                    temp = path.with_name(path.name+'.monoplanner-'+uuid.uuid4().hex)
                    shutil.copy2(_safe_file(source, Path(name)), temp)
                    os.replace(temp, path)
                else:
                    path.unlink()
                published.append(name)
        except BaseException:
            for name in reversed(published):
                path = self.repo_root/name
                if name in before.files:
                    shutil.copy2(before.root/name, path)
                elif path.exists():
                    path.unlink()
            raise
        return changes

    def run_process(self, command, *, stdout_path, stderr_path, timeout=None, shell=False,
                    env=None, on_started=None):
        timeout = self.timeout if timeout is None else timeout
        if timeout <= 0:
            raise ValueError('timeout 必须大于 0')
        if self.access_mode == 'full':
            from agent.execution.process import run_process
            environment = clean_environment(env)
            environment['PATH'] = os.environ.get('PATH', '')
            return run_process(command, cwd=self.repo_root, stdout_path=stdout_path,
                               stderr_path=stderr_path, timeout=timeout, shell=shell,
                               env=environment, on_started=on_started)
        if os.name != 'nt' or not self.helper.is_file() or not (self.runtime/'python.exe').is_file():
            raise OSError('原生沙盒或私有 Python 运行时不可用；没有退回普通权限')
        from agent.source.snapshots import SourceSnapshot
        box = self.artifact_root/'native'/uuid.uuid4().hex
        box.mkdir(parents=True)
        before = SourceSnapshot.capture(self.repo_root, box/'before', excluded_roots=(self.artifact_root,),
                                        runtime_files=self.runtime_files)
        source = box/'work'
        shutil.copytree(before.root, source)
        inputs, outputs, temporary = box/'inputs', box/'outputs', box/'tmp'
        for folder in (inputs, outputs, temporary):
            folder.mkdir()
        argv, mappings = self._argv(command, shell, box, source, inputs, outputs)
        environment = clean_environment(env)
        environment.update(TEMP=str(temporary), TMP=str(temporary), HOME=str(temporary),
                           PATH=str(self.runtime)+';'+str(Path(os.environ['SystemRoot'])/'System32'))
        readable = [str(self.runtime), str(inputs), *self.read_roots]
        writable = [str(outputs), str(temporary)]
        (readable if self.read_only else writable).append(str(source))
        request = {'argv': argv, 'cwd': str(source), 'readable': readable, 'writable': writable,
                   'env': environment, 'timeout_ms': max(1, int(timeout*1000)), 'result': str(box/'status.json')}
        if shell:
            request['shell_body'] = command
        (box/'request.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
        for path in (stdout_path, stderr_path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with Path(stdout_path).open('wb') as out, Path(stderr_path).open('wb') as err:
            process = subprocess.Popen([str(self.helper)], stdin=subprocess.PIPE, stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW, env=clean_environment())
            if on_started:
                on_started(process.pid)
            # Helper owns timeout and the Job. If the controller disappears,
            # closing the Job kills children; unfinished status is never replayed.
            process.communicate(json.dumps(request, ensure_ascii=False).encode('utf-8'))
        result = json.loads((box/'status.json').read_text(encoding='utf-8-sig')) if (box/'status.json').is_file() else {}
        if process.returncode != 0 or result.get('status') == 'initializing' or result.get('isolation') != 'lpac':
            return {'exit_code': None, 'timed_out': False, 'error': '原生隔离执行未完成，未发布任何修改',
                    'error_kind': 'sandbox_unavailable', 'record': str(box/'status.json')}
        for produced, destination in mappings:
            if produced.is_file():
                safe_tree(outputs)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(produced, destination)
        changes = [] if self.read_only or result['timed_out'] else self._publish(before, source, box)
        result.update(stdout_log=str(stdout_path), stderr_log=str(stderr_path),
                      changed_files=changes, sandbox_record=str(box/'status.json'), seconds=time.monotonic()-started)
        return result

    def execute(self, command):
        folder = self.artifact_root/('command-'+uuid.uuid4().hex)
        result = self.run_process(command, stdout_path=folder/'stdout.log', stderr_path=folder/'stderr.log', shell=True)
        result.update(command=command, stdout=log_excerpt(folder/'stdout.log'), stderr=log_excerpt(folder/'stderr.log'))
        return result
