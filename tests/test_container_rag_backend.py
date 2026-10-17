"""固定容器资料协议的离线契约；不启动 Docker，不访问真实任务资料。"""

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agent.model.errors import ModelInterrupted
from agent.rag import container_probe as probe
from agent.rag.container_backend import ContainerBackend, ContainerReadError
from agent.task.control import Signal, using


def payload(root, path=None, **values):
    return {'path': str(path or root), 'allowed_root': str(root), 'denied_roots': [], **values}


def run_probe(operation, values):
    result = subprocess.run(
        [sys.executable, '-I', '-S', '-B', str(Path(probe.__file__)),
         json.dumps({'operation': operation, 'payload': values})],
        capture_output=True, text=True, encoding='utf-8', timeout=10, check=True,
    )
    assert not result.stderr
    return json.loads(result.stdout)


def test_subprocess_exact_original_and_stat(tmp_path):
    source = tmp_path / '中文.txt'
    raw = b'\xef\xbb\xbf' + '第一行\r\n\tsecond\n'.encode('utf-8')
    source.write_bytes(raw)
    metadata = run_probe('stat', payload(tmp_path, source))
    assert metadata['ok'] and metadata['result']['size'] == len(raw)
    answer = run_probe('read', payload(tmp_path, source))
    assert answer['ok'], answer
    result = answer['result']
    assert base64.b64decode(result['data_base64']) == raw
    assert result['sha256'] == hashlib.sha256(raw).hexdigest()
    assert result['mtime_ns'] == source.stat().st_mtime_ns
    assert source.read_bytes() == raw


def test_directory_pages_keep_stable_names_and_no_bodies(tmp_path):
    for name in ['z.txt', 'a.txt', 'm.txt', 'b.txt']:
        (tmp_path / name).write_text('do not return file bodies', encoding='utf-8')
    first = run_probe('list', payload(tmp_path, limit=2))['result']
    second = run_probe('list', payload(tmp_path, limit=2, after=first['next_after']))['result']
    assert [entry['name'] for entry in first['entries']] == ['a.txt', 'b.txt']
    assert [entry['name'] for entry in second['entries']] == ['m.txt', 'z.txt']
    assert not first['complete'] and second['complete'] and second['next_after'] == ''
    assert 'do not return file bodies' not in json.dumps(first)


@pytest.mark.parametrize('operation', ['stat', 'read', 'list'])
def test_denied_and_parent_traversal_are_rejected(tmp_path, operation):
    secret = tmp_path / 'secret'
    secret.mkdir()
    answer = run_probe(operation, payload(tmp_path, secret, denied_roots=[str(secret)]))
    assert not answer['ok'] and answer['error']['code'] == 'denied_path'
    answer = run_probe(operation, payload(tmp_path, str(tmp_path / 'secret' / '..')))
    assert not answer['ok'] and answer['error']['code'] == 'invalid_path'
    listing = run_probe('list', payload(tmp_path, denied_roots=[str(secret)]))['result']
    assert listing['entries'][0]['type'] == 'blocked'


def test_oversized_file_and_invalid_fields_are_not_read(tmp_path):
    source = tmp_path / 'big.txt'
    source.write_bytes(b'x' * 33)
    result = run_probe('read', payload(tmp_path, source, max_bytes=32))
    assert not result['ok'] and result['error']['code'] == 'source_too_large'
    assert 'data_base64' not in result
    result = run_probe('read', payload(tmp_path, source, command='erase data'))
    assert not result['ok'] and result['error']['code'] == 'invalid_payload'


def test_source_changed_mid_read_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / 'mutable.txt'
    source.write_bytes(b'abc')
    original = probe.os.read
    changed = False

    def mutate(fd, count):
        nonlocal changed
        result = original(fd, count)
        if result and not changed:
            changed = True
            source.write_bytes(b'different')
        return result

    monkeypatch.setattr(probe.os, 'read', mutate)
    with pytest.raises(probe.ProbeError, match='changed'):
        probe.dispatch({'operation': 'read', 'payload': payload(tmp_path, source)})


def test_symlink_escape_and_internal_link(tmp_path):
    root = tmp_path / 'allowed'
    root.mkdir()
    inside, outside = root / 'inside.txt', tmp_path / 'outside.txt'
    inside.write_bytes(b'inside')
    outside.write_bytes(b'outside')
    try:
        (root / 'safe.txt').symlink_to(inside)
        (root / 'escape.txt').symlink_to(outside)
    except OSError:
        pytest.skip('The host does not grant symbolic-link creation.')
    safe = run_probe('read', payload(root, root / 'safe.txt'))
    assert safe['ok'] and base64.b64decode(safe['result']['data_base64']) == b'inside'
    unsafe = run_probe('read', payload(root, root / 'escape.txt'))
    assert not unsafe['ok'] and unsafe['error']['code'] == 'outside_root'


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='FIFO files require a POSIX host.')
def test_special_file_never_blocks_read(tmp_path):
    fifo = tmp_path / 'pipe'
    os.mkfifo(fifo)
    answer = run_probe('read', payload(tmp_path, fifo))
    assert not answer['ok'] and answer['error']['code'] == 'special_file'


class FakeEnvironment:
    _container_id = 'a' * 64
    image = 'sha256:' + 'b' * 64
    python_executable = '/opt/python/bin/python'
    timeout = 99
    runtime_facts = {'python': '3.12.4', 'source': '/testbed', 'checker_errors': {}}

    def __init__(self):
        self.calls = []
        self.running = True
        self.cancelled = False
        self.mounts = [{'Destination': '/testbed', 'Source': '/host/project', 'RW': True,
                        'Type': 'bind'}]
        raw = b'original\r\n'
        self.result = {'path': '/usr/include/a.h', 'realpath': '/usr/include/a.h',
                       'type': 'file', 'size': len(raw), 'mtime_ns': 42,
                       'sha256': hashlib.sha256(raw).hexdigest(),
                       'data_base64': base64.b64encode(raw).decode('ascii')}

    def _check_cancelled(self):
        if self.cancelled:
            raise InterruptedError('paused')

    def _docker(self, arguments, **kwargs):
        self.calls.append((arguments, kwargs))
        if arguments[0] == 'inspect':
            value = {'container_id': self._container_id, 'image_id': self.image,
                     'running': self.running, 'mounts': self.mounts}
        elif arguments[:2] == ['image', 'inspect']:
            value = {'image_id': self.image, 'repo_digests': []}
        else:
            assert arguments[0] == 'exec'
            value = {'ok': True, 'result': self.result}
        return SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr='')


def request_payload(**changes):
    return {'path': '/usr/include/a.h', 'allowed_root': '/usr/include', **changes}


def test_binding_and_fixed_exec_never_start_or_use_shell():
    environment = FakeEnvironment()
    backend = ContainerBackend(environment)
    bound = backend.binding()
    assert bound['image_id'] == environment.image and bound['repo_digests'] == []
    assert bound['toolchain_versions']['values'] == {'python': '3.12.4'}
    result = backend.call('read', request_payload())
    assert result['sha256'] == environment.result['sha256']
    argv, options = environment.calls[-1]
    assert argv[:7] == ['exec', environment._container_id, environment.python_executable,
                       '-I', '-S', '-B', '-c']
    request = json.loads(argv[-1])
    assert request['operation'] == 'read' and request['payload']['denied_roots'] == []
    assert 0 < request['deadline_seconds'] < options['timeout']
    assert options['timeout'] == 30 and 'shell' not in options
    assert all(arguments[0] in {'inspect', 'image', 'exec'} for arguments, _ in environment.calls)


def test_unavailable_stopped_and_paused_do_not_exec():
    environment = FakeEnvironment()
    environment._container_id = None
    with pytest.raises(ContainerReadError, match='No existing'):
        ContainerBackend(environment).call('stat', request_payload())
    assert not environment.calls
    environment._container_id = 'a' * 64
    environment.running = False
    with pytest.raises(ContainerReadError, match='not running'):
        ContainerBackend(environment).call('stat', request_payload())
    assert all(arguments[0] != 'exec' for arguments, _ in environment.calls)
    environment.calls.clear()
    environment.cancelled = True
    with pytest.raises(ModelInterrupted) as caught:
        ContainerBackend(environment).binding()
    assert caught.value.kind == 'pause' and not caught.value.sent
    assert not environment.calls


def test_pause_signal_prevents_new_docker_dispatch():
    environment, signal = FakeEnvironment(), Signal()
    signal.interrupt('pause')
    with using(signal), pytest.raises(ModelInterrupted):
        ContainerBackend(environment).binding()
    assert not environment.calls


def test_pause_after_query_prevents_next_dispatch(monkeypatch):
    environment = FakeEnvironment()
    original = environment._docker

    def cancel_after(arguments, **options):
        result = original(arguments, **options)
        environment.cancelled = True
        return result

    monkeypatch.setattr(environment, '_docker', cancel_after)
    with pytest.raises(ModelInterrupted):
        ContainerBackend(environment).binding()
    assert len(environment.calls) == 1


def test_fixed_probe_deadline_uses_process_alarm(monkeypatch):
    registered = {}
    fake = SimpleNamespace(
        SIGALRM=14, ITIMER_REAL=0,
        signal=lambda number, callback: registered.update(number=number, callback=callback),
        setitimer=lambda timer, seconds: registered.update(timer=timer, seconds=seconds),
    )
    monkeypatch.setattr(probe, 'signal', fake)
    probe._arm_deadline(0.5)
    assert registered['number'] == 14 and registered['seconds'] == 0.5
    with pytest.raises(probe.ProbeError, match='deadline'):
        registered['callback'](14, None)
    with pytest.raises(probe.ProbeError):
        probe._arm_deadline(30)


def test_transport_timeout_is_not_retried(monkeypatch):
    environment = FakeEnvironment()

    def timeout(arguments, **options):
        environment.calls.append((arguments, options))
        raise subprocess.TimeoutExpired(arguments, options['timeout'])

    monkeypatch.setattr(environment, '_docker', timeout)
    with pytest.raises(ContainerReadError) as caught:
        ContainerBackend(environment).binding()
    assert caught.value.details['code'] == 'container_query_timeout'
    assert len(environment.calls) == 1


def test_binding_changes_and_bad_hash_are_rejected():
    environment = FakeEnvironment()
    backend = ContainerBackend(environment)
    backend.binding()
    environment.mounts = []
    with pytest.raises(ContainerReadError, match='mounts changed'):
        backend.call('read', request_payload())
    assert all(arguments[0] != 'exec' for arguments, _ in environment.calls)
    environment = FakeEnvironment()
    environment.result['sha256'] = '0' * 64
    with pytest.raises(ContainerReadError, match='bytes failed verification'):
        ContainerBackend(environment).call('read', request_payload())


@pytest.mark.parametrize('operation,values', [
    ('shell', request_payload()),
    ('read', request_payload(command='cat /etc/passwd')),
    ('read', request_payload(path='/usr/include/../../etc/passwd')),
    ('read', request_payload(max_bytes=probe.MAX_BYTES + 1)),
    ('read', request_payload(path='C:/host/file')),
])
def test_invalid_request_rejected_before_any_docker_call(operation, values):
    environment = FakeEnvironment()
    with pytest.raises(ContainerReadError):
        ContainerBackend(environment).call(operation, values)
    assert not environment.calls
