"""原生 Windows 作业对象验证：助手工作随窗口退出，用户打开的应用独立存活。"""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch
import uuid

import pytest

from agent.execution import applications


pytestmark = pytest.mark.skipif(os.name != 'nt', reason='Windows desktop job ownership')


class BasicLimits(ctypes.Structure):
    _fields_ = [
        ('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
        ('flags', ctypes.c_uint32), ('minimum_working_set', ctypes.c_size_t),
        ('maximum_working_set', ctypes.c_size_t), ('active_process_limit', ctypes.c_uint32),
        ('affinity', ctypes.c_size_t), ('priority_class', ctypes.c_uint32),
        ('scheduling_class', ctypes.c_uint32),
    ]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ('basic', BasicLimits), ('io', ctypes.c_uint64 * 6),
        ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
        ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t),
    ]


def kernel():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateJobObjectW': ([ctypes.c_void_p, ctypes.c_wchar_p], ctypes.c_void_p),
        'OpenJobObjectW': ([ctypes.c_uint32, ctypes.c_bool, ctypes.c_wchar_p], ctypes.c_void_p),
        'SetInformationJobObject': (
            [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32], ctypes.c_bool),
        'AssignProcessToJobObject': ([ctypes.c_void_p, ctypes.c_void_p], ctypes.c_bool),
        'IsProcessInJob': (
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)], ctypes.c_bool),
        'OpenProcess': ([ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32], ctypes.c_void_p),
        'WaitForSingleObject': ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_uint32),
        'TerminateProcess': ([ctypes.c_void_p, ctypes.c_uint32], ctypes.c_bool),
        'CloseHandle': ([ctypes.c_void_p], ctypes.c_bool),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result
    return api


def require(success):
    if not success:
        raise ctypes.WinError(ctypes.get_last_error())
    return success


def wait_file(path, process=None):
    deadline = time.monotonic() + 15
    while not path.exists():
        if process is not None and process.poll() is not None:
            error = path.parent / 'error.txt'
            raise AssertionError(error.read_text() if error.exists() else 'Fixture exited early')
        if time.monotonic() >= deadline:
            raise AssertionError('Fixture did not become ready')
        time.sleep(0.02)


def write_children(folder, values):
    target = folder / 'children.json'
    temporary = folder / 'children.tmp'
    temporary.write_text(json.dumps(values), encoding='utf-8')
    temporary.replace(target)


def backend_fixture(folder):
    wait_file(folder / 'start')
    config = json.loads((folder / 'config.json').read_text())
    internal = subprocess.Popen([config['application']], creationflags=subprocess.CREATE_NO_WINDOW)
    children = {'internal': internal.pid}
    write_children(folder, children)
    app = applications.describe_executable(config['application'])
    children['external'] = applications.start(app)
    write_children(folder, children)

    api = kernel()
    job = require(api.OpenJobObjectW(0x4, False, config['job']))
    try:
        for name, identity in list(children.items()):
            process = require(api.OpenProcess(0x1000, False, identity))
            try:
                member = ctypes.c_int()
                require(api.IsProcessInJob(process, job, ctypes.byref(member)))
                children[name + '_in_job'] = bool(member.value)
            finally:
                api.CloseHandle(process)
    finally:
        api.CloseHandle(job)
    write_children(folder, children)
    (folder / 'ready').touch()
    time.sleep(120)


def desktop_fixture(folder):
    api = kernel()
    config = json.loads((folder / 'config.json').read_text())
    job = require(api.CreateJobObjectW(None, config['job']))
    backend = None
    try:
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000 | 0x800
        require(api.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)))
        backend = subprocess.Popen(
            [sys.executable, '-B', str(Path(__file__).resolve()), '--backend', str(folder)],
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        process = require(api.OpenProcess(0x0101, False, backend.pid))
        try:
            require(api.AssignProcessToJobObject(job, process))
        finally:
            api.CloseHandle(process)
        (folder / 'start').touch()
        wait_file(folder / 'stop', backend)
    finally:
        api.CloseHandle(job)
        if backend is not None:
            backend.wait(timeout=5)


@pytest.fixture(scope='module')
def harmless_application(tmp_path_factory):
    folder = tmp_path_factory.mktemp('desktop-ownership')
    source = folder / 'WaitApplication.cs'
    application = folder / 'WaitApplication.exe'
    source.write_text(
        'internal static class WaitApplication { '
        'public static void Main() { System.Threading.Thread.Sleep(120000); } }',
        encoding='utf-8',
    )
    compiler = Path(os.environ['WINDIR']) / 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    if not compiler.is_file():
        pytest.skip('Windows .NET Framework compiler is not installed')
    subprocess.run(
        [str(compiler), '/nologo', '/target:winexe', '/platform:x64',
         '/out:' + str(application), str(source)],
        check=True, capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return application


def test_external_launch_is_explicitly_detached_and_does_not_retry_inside_job(tmp_path):
    application = tmp_path / 'application.exe'
    application.write_bytes(b'owned fixture')
    app = applications.describe_executable(str(application))
    with patch.object(applications.subprocess, 'Popen', side_effect=OSError('breakaway refused')) as launch:
        with pytest.raises(OSError, match='breakaway refused'):
            applications.start(app)
    launch.assert_called_once()
    assert launch.call_args.kwargs['creationflags'] == subprocess.CREATE_BREAKAWAY_FROM_JOB
    assert 'MONOPLANER_SERVICE_TOKEN' not in launch.call_args.kwargs['env']
    assert 'MONOPLANER_DESKTOP_JOB' not in launch.call_args.kwargs['env']


def test_packaged_activation_uses_detached_helper_without_relaxing_arguments():
    with patch.object(applications.subprocess, 'run') as launch:
        launch.return_value = subprocess.CompletedProcess([], 0, '12345', '')
        assert applications.start({'kind': 'packaged', 'aumid': 'Fixture!Application'}) == 12345
    assert launch.call_args.args[0][1:] == ['activate', 'Fixture!Application', '']
    assert launch.call_args.kwargs['creationflags'] == (
        subprocess.CREATE_NO_WINDOW | subprocess.CREATE_BREAKAWAY_FROM_JOB)


@pytest.mark.parametrize('exit_mode', ['close', 'crash'])
def test_real_external_application_survives_desktop_exit(
    tmp_path, harmless_application, exit_mode,
):
    config = {
        'job': 'Local\\MonoPlanner-ownership-test-' + uuid.uuid4().hex,
        'application': str(harmless_application),
    }
    (tmp_path / 'config.json').write_text(json.dumps(config), encoding='utf-8')
    environment = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    desktop = subprocess.Popen(
        [sys.executable, '-B', str(Path(__file__).resolve()), '--desktop', str(tmp_path)],
        env=environment, creationflags=subprocess.CREATE_NO_WINDOW,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    api, handles = kernel(), {}
    try:
        wait_file(tmp_path / 'ready', desktop)
        children = json.loads((tmp_path / 'children.json').read_text())
        for name in ('internal', 'external'):
            handles[name] = require(api.OpenProcess(0x101001, False, children[name]))
        assert children['internal_in_job'] is True
        assert children['external_in_job'] is False
        started = time.monotonic()
        if exit_mode == 'close':
            (tmp_path / 'stop').touch()
        else:
            desktop.terminate()
        desktop.wait(timeout=5)
        assert api.WaitForSingleObject(handles['internal'], 5000) == 0
        assert api.WaitForSingleObject(handles['external'], 0) == 258
        assert time.monotonic() - started < 5
    finally:
        if desktop.poll() is None:
            desktop.terminate()
        desktop.wait(timeout=5)
        # 失败途中也回收私有无害程序，保留句柄时不再根据 PID 查找。
        children_file = tmp_path / 'children.json'
        if children_file.exists():
            children = json.loads(children_file.read_text())
            for name in ('internal', 'external'):
                if name not in handles and name in children:
                    handle = api.OpenProcess(0x101001, False, children[name])
                    if handle:
                        handles[name] = handle
        for handle in handles.values():
            try:
                if api.WaitForSingleObject(handle, 0) == 258:
                    api.TerminateProcess(handle, 1)
                    api.WaitForSingleObject(handle, 5000)
            finally:
                api.CloseHandle(handle)


if __name__ == '__main__':
    folder = Path(sys.argv[2])
    try:
        if sys.argv[1] == '--desktop':
            desktop_fixture(folder)
        elif sys.argv[1] == '--backend':
            backend_fixture(folder)
        else:
            raise ValueError('Unknown native ownership fixture mode')
    except BaseException as error:
        (folder / 'error.txt').write_text(str(error), encoding='utf-8')
        raise
