"""真实独立进程及原生检查点的升级回归；模型和项目都使用隔离离线替身。"""

import json
from pathlib import Path
import shutil
import time
import uuid

import pytest

from agent.runtime import bundles, bridge
from agent.settings import PROJECT_ROOT
from agent.storage.records import save
from agent.storage.recovery import code_key
from support import fixture_delivery


OFFLINE_APPEND = r'''

_native_run = run_work
def run_work(job, emit, controls, *, root=None, **unused):
    import sys, subprocess
    from langchain_core.messages import AIMessage
    from agent.execution.local import LocalEnvironment
    from unittest.mock import patch

    class Environment(LocalEnvironment):
        def __init__(self, repo, image, task_root, **kwargs):
            super().__init__(repo, artifact_root=Path(task_root).parent.parent / 'execution')
            self.image = image
            self.runtime_facts = {'offline': True}
        def start(self):
            return self
        def close(self):
            pass

    class Model:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            if job.get('probe_wait'):
                from threading import Event
                from agent.task.control import current
                from agent.model.errors import ModelInterrupted
                signal = current()
                stop = Event()
                (Path(root) / 'waiting.txt').write_text('waiting', encoding='utf-8')
                with signal.subscribe(stop.set):
                    stop.wait(15)
                raise ModelInterrupted(signal.kind or 'pause', sent=False)
            if ENGINE_RELEASE != 'A' and any(
                c['name'] == 'submit_plan' for m in messages for c in getattr(m, 'tool_calls', [])
            ):
                from agent.model.errors import ModelRequestNotSent
                raise ModelRequestNotSent('task_cancelled', 'Offline upgrade fixture stopped')
            tool, args = 'submit_plan', {'kind': 'blocked', 'note': ENGINE_RELEASE}
            if ENGINE_RELEASE == 'A':
                calls = [c for m in messages for c in getattr(m, 'tool_calls', [])]
                try:
                    last = json.loads(messages[-1].content)
                except (ValueError, TypeError):
                    last = {}
                if not any(c['name'] == 'get_boundary_hints' for c in calls):
                    tool, args = 'get_boundary_hints', {'keys': ['general']}
                elif last.get('outcome') == 'acceptance_preflight':
                    tool, args = 'freeze_acceptance', {
                        'trial_id': last['trial_id'], 'reason': 'Keep the original required behavior'}
                else:
                    program = "def value():\n    return 4\n"
                    code = ("from pathlib import Path; p=Path('edits.txt'); "
                            "p.write_text(p.read_text()+'x' if p.exists() else 'x'); "
                            "Path('sample.py').write_text(" + repr(program) + ")")
                    test_code = "\n".join([
                        "import unittest", "from sample import value",
                        "class Check(unittest.TestCase):", "    def test_value(self):",
                        "        self.assertEqual(value(), 4)", "",
                    ])
                    args = {'kind': 'candidate', 'note': ENGINE_RELEASE,
                        'steps': [{'id': 'edit', 'op': 'shell', 'command': subprocess.list2cmdline([sys.executable, '-B', '-X', 'utf8', '-c', code])}],
                        'test_code': test_code,
                        'execution_checks': [{'id': 'actual', 'requirement_id': 'fixture_delivery',
                            'requirement_quote': '', 'artifact_type': 'repository_entry',
                            'consumer': 'command', 'entry_file': 'sample.py', 'interpreter': 'python',
                            'imports': '', 'setup': '', 'expression': '', 'variables': [],
                            'result_variable': '', 'argv': [], 'cases': []}]}
            return AIMessage(content='', tool_calls=[{'name': tool, 'id': 'offline-call', 'args': args}],
                usage_metadata={'input_tokens': 10, 'output_tokens': 10, 'total_tokens': 20})

    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch(__name__ + '.require_binding', side_effect=lambda repo, value: value))
        if job.get('crash_after_effect'):
            original_persist = WorkRunner.persist
            def crash_once(owner, event, payload=None):
                marker = owner.root / 'offline-crash-injected'
                if event == 'step_finished' and not marker.exists():
                    marker.write_text('The file effect and receipt already exist', encoding='utf-8')
                    import os
                    os._exit(81)
                return original_persist(owner, event, payload)
            stack.enter_context(patch.object(WorkRunner, 'persist', crash_once))
        return _native_run(job, emit, controls, root=root, model=Model(), environment_factory=Environment)
'''


@pytest.fixture(scope='module')
def engines(tmp_path_factory):
    area = tmp_path_factory.mktemp('upgrade-engines')
    root = area / 'state'
    base = bundles.register(root)
    folder, _, manifest = bundles.resolve(root, base)
    versions = {}
    for release in ('A', 'B'):
        source = area / release
        source.mkdir()
        for name in manifest['files']:
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / name, destination)
        path = source / 'agent/application/work_bridge.py'
        path.write_text(path.read_text('utf-8') + '\nENGINE_RELEASE = ' + repr(release) + OFFLINE_APPEND,
                        encoding='utf-8')
        worker = source / 'agent/runtime/worker.py'
        worker.write_text(worker.read_text('utf-8') + '''
_native_query = query
def query(root, task, method, arguments):
    if method == 'offline_slow_query':
        time.sleep(10)
        return {'offline': True}
    return _native_query(root, task, method, arguments)
''', encoding='utf-8')
        sources = {p.relative_to(source).as_posix(): p for p in bundles.tree_files(source)}
        contents = bundles._manifest(sources, protocol=bundles.PROTOCOL, runtime_id=manifest['runtime_id'],
                                     code_key=code_key(source))
        versions[release] = bundles._publish(root, 'engines', contents, sources)
        assert source.parent == area
        shutil.rmtree(source)
    return root, versions


def job_for(root, engine, tmp_path, *, pause=False):
    task = uuid.uuid4().hex
    repo = tmp_path / task
    repo.mkdir()
    (repo / 'sample.py').write_text('def value():\n    return 0\n', encoding='utf-8')
    if engine:
        save(root / 'tasks' / task / 'engine-binding.json', {
            'format': 1, 'protocol': bundles.PROTOCOL, 'task_id': task, 'engine_id': engine,
        })
    return {'task_id': task, 'state_root': str(root), 'repo': str(repo),
            'request': 'value must return 4', 'settings': {},
            'confirmed_requirements': fixture_delivery('value must return 4'),
            'execution_binding': {'image': 'sha256:' + 'b' * 64, 'runtime': {}},
            'pause_after_steps': 1 if pause else None}


def run(job, monkeypatch, controls=()):
    monkeypatch.setattr('agent.settings.load_settings', lambda *_: {'api_key': 'offline-secret'})
    return bridge.run_work(job, lambda *_: None, lambda: list(controls))


def test_new_host_resumes_old_engine_real_checkpoint_without_replaying_edit(engines, tmp_path, monkeypatch):
    root, versions = engines
    job = job_for(root, versions['A'], tmp_path, pause=True)
    first = run(job, monkeypatch)
    assert first['status'] == 'paused', first
    directory = root / 'tasks' / job['task_id']
    frozen = (directory / 'frozen-acceptance.json').read_bytes()
    original_request = (directory / 'requests/call-0001/request.json').read_bytes()
    assert (Path(job['repo']) / 'edits.txt').read_text() == 'x'
    # 模拟应用已经换成 B：恢复不得调用当前版本注册入口。
    monkeypatch.setattr(bundles, '_register', lambda *_args, **_kwargs: pytest.fail('selected new engine'))
    # 冻结工作不得解释宿主以后版本的应用表。
    import sqlite3
    with sqlite3.connect(root / 'application.sqlite3') as database:
        database.execute('CREATE TABLE future_host_data (value TEXT)')
        database.execute('PRAGMA user_version=99')
    result = run({**job, 'pause_after_steps': None}, monkeypatch,
                 [{'id': 'continue-on-new-host', 'kind': 'resume'}])
    assert result['status'] == 'verified_success', result
    assert (Path(job['repo']) / 'edits.txt').read_text() == 'x'
    assert (directory / 'frozen-acceptance.json').read_bytes() == frozen
    assert (directory / 'requests/call-0001/request.json').read_bytes() == original_request
    assert result['usage']['calls'] >= first['usage']['calls']
    assert bridge.read(root, job['task_id'], 'status')['status'] == 'verified_success'
    monkeypatch.setattr(bundles, '_register', lambda *_args, **_kwargs: versions['B'])
    newer = job_for(root, None, tmp_path)
    assert run(newer, monkeypatch)['status'] == 'task_cancelled'
    assert bundles.binding(root, job['task_id'])['engine_id'] != bundles.binding(root, newer['task_id'])['engine_id']


def test_unknown_operation_and_duplicate_owner_are_not_resumed(engines, tmp_path, monkeypatch):
    from agent.application.lease import ServiceLease

    root, versions = engines
    job = job_for(root, versions['A'], tmp_path)
    directory = root / 'tasks' / job['task_id']
    save(directory / 'hard-interruption.json', {'status': 'execution_unknown'})
    result = run(job, monkeypatch)
    assert result['status'] == 'execution_unknown'
    assert not (directory / 'requests').exists()
    lease = ServiceLease(directory)
    lease.acquire()
    try:
        with pytest.raises(RuntimeError, match='MonoPlanner'):
            run(job, monkeypatch)
    finally:
        lease.close()


def test_engine_corruption_and_missing_binding_never_fall_back(engines, tmp_path, monkeypatch):
    root, versions = engines
    job = job_for(root, versions['B'], tmp_path)
    folder, _, _ = bundles.resolve(root, versions['B'])
    extra = folder / '.env'
    extra.write_text('UNEXPECTED=1', encoding='utf-8')
    try:
        with pytest.raises(ValueError, match='inventory'):
            run(job, monkeypatch)
    finally:
        extra.unlink()
    missing = uuid.uuid4().hex
    save(root / 'tasks' / missing / 'state.json', {'status': 'paused'})
    with pytest.raises(ValueError, match='Pre-baseline'):
        bundles.binding(root, missing, create=True)


def test_host_pipe_close_stops_worker_without_restarting(engines, tmp_path):
    root, versions = engines
    job = job_for(root, versions['A'], tmp_path)
    job['probe_wait'] = True
    marker = root / 'waiting.txt'
    marker.unlink(missing_ok=True)
    session = bridge.Session(root, job['task_id'])
    session.send('run', root=str(root), task_id=job['task_id'], engine_id=versions['A'], job=job,
                 credential='offline-secret')
    deadline = time.monotonic() + 25
    try:
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        assert marker.exists()
    finally:
        session.close()
    assert session.process.poll() is not None
    assert not (Path(job['repo']) / 'edits.txt').exists()
    assert b'offline-secret' not in (root / 'tasks' / job['task_id'] / 'engine-worker.log').read_bytes()


@pytest.mark.parametrize('kind,expected', [('pause', 'paused'), ('message', 'awaiting_user')])
def test_worker_controls_interrupt_pending_model_and_keep_receipts(engines, tmp_path, kind, expected):
    import queue

    root, versions = engines
    job = job_for(root, versions['A'], tmp_path)
    job['probe_wait'] = True
    marker = root / 'waiting.txt'
    marker.unlink(missing_ok=True)
    session = bridge.Session(root, job['task_id'])
    session.send('run', root=str(root), task_id=job['task_id'], engine_id=versions['A'], job=job,
                 credential='offline-secret')
    command = {'id': 'user-control', 'kind': kind,
               'text': 'Keep all original requirements' if kind == 'message' else ''}
    deadline = time.monotonic() + 25
    try:
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        if kind == 'message':
            session.send('interrupt', id=command['id'])
        receipt = session.control(command, wait=True)
        assert session.control(command, wait=True) == receipt
        while True:
            value = session.messages.get(timeout=25)
            assert value['kind'] != 'error', value
            if value['kind'] == 'result':
                assert value['result']['status'] == expected, value
                assert command['id'] in value['result'].get('applied_inputs', [])
                break
    finally:
        session.close()
    assert not (Path(job['repo']) / 'edits.txt').exists()


def test_referenced_engines_cannot_be_collected(engines):
    root, versions = engines
    for release, engine in versions.items():
        task = uuid.uuid4().hex
        save(root / 'tasks' / task / 'engine-binding.json', {
            'format': 1, 'protocol': bundles.PROTOCOL, 'task_id': task, 'engine_id': engine,
        })
    result = bundles.collect(root)
    assert set(versions.values()).issubset(result['retained'])
    assert not set(versions.values()) & set(result['removed'])


def test_real_worker_exit_before_sql_can_continue_without_repeating_file_effect(engines, tmp_path, monkeypatch):
    root, versions = engines
    job = {**job_for(root, versions['A'], tmp_path), 'crash_after_effect': True}
    with pytest.raises(bridge.WorkerError):
        run(job, monkeypatch)
    assert (Path(job['repo']) / 'edits.txt').read_text() == 'x'
    directory = root / 'tasks' / job['task_id']
    assert (directory / 'offline-crash-injected').exists()
    original = (directory / 'requests/call-0001/request.json').read_bytes()
    result = run(job, monkeypatch, [{'id': 'after-crash', 'kind': 'resume'}])
    assert result['status'] == 'verified_success', result
    assert (Path(job['repo']) / 'edits.txt').read_text() == 'x'
    assert (directory / 'requests/call-0001/request.json').read_bytes() == original
    assert result['usage']['unknown_calls'] == 0


def test_lost_worker_result_requests_owned_cleanup_without_restarting(tmp_path, monkeypatch):
    import queue
    from threading import Event
    from types import SimpleNamespace

    task = 'f' * 32
    stopped, sent = [], []
    session = SimpleNamespace(
        binding={'engine_id': 'fixture'}, finished=Event(), messages=queue.Queue(),
        send=lambda *args, **kwargs: sent.append(args[0]), close=lambda: None,
        process=SimpleNamespace(poll=lambda: 81),
    )
    session.messages.put({'kind': 'error', 'error': 'UnexpectedExit'})
    monkeypatch.setattr(bridge, 'Session', lambda *_args, **_kwargs: session)
    monkeypatch.setattr('agent.settings.load_settings', lambda *_: {'api_key': 'offline'})
    monkeypatch.setattr(bridge, 'pause_work', lambda task, command, **kwargs:
                        stopped.append((task, kwargs['root'])))
    with pytest.raises(bridge.WorkerError):
        bridge._run_work({'task_id': task}, lambda *_: None, lambda: [], tmp_path, task)
    assert sent == ['run'] and stopped == [(task, tmp_path)]


def test_open_but_silent_host_pipe_stops_worker_without_waiting_for_restart(engines, tmp_path):
    root, versions = engines
    job = {**job_for(root, versions['A'], tmp_path), 'probe_wait': True}
    marker = root / 'waiting.txt'
    marker.unlink(missing_ok=True)
    session = bridge.Session(root, job['task_id'])
    session.send('run', root=str(root), task_id=job['task_id'], engine_id=versions['A'],
                 job=job, credential='offline-secret')
    deadline = time.monotonic() + 60
    try:
        while not marker.exists() and time.monotonic() < deadline:
            session.send('heartbeat')
            time.sleep(0.25)
        assert marker.exists()
        # 故意保持宿主进程和管道存活，只停止协议心跳。
        session.process.wait(timeout=14)
        assert not session.process.stdin.closed
        assert not (Path(job['repo']) / 'edits.txt').exists()
    finally:
        session.close()


def test_slow_read_query_does_not_block_heartbeat_or_user_pause(engines, tmp_path):
    from threading import Event, Thread

    root, versions = engines
    job = {**job_for(root, versions['A'], tmp_path), 'probe_wait': True}
    marker = root / 'waiting.txt'
    marker.unlink(missing_ok=True)
    session = bridge.Session(root, job['task_id'])
    session.send('run', root=str(root), task_id=job['task_id'], engine_id=versions['A'],
                 job=job, credential='offline-secret')
    stop = Event()

    def heartbeat():
        while not stop.wait(0.5):
            if session.process.poll() is not None:
                return
            try:
                session.send('heartbeat')
            except bridge.WorkerError:
                return

    thread = Thread(target=heartbeat)
    thread.start()
    try:
        deadline = time.monotonic() + 60
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists()
        session.send('query', id='slow', method='offline_slow_query', arguments={})
        assert session.reply('slow', timeout=13) == {'offline': True}
        assert session.process.poll() is None
        assert not (root / 'control' / job['task_id'] / 'host-lease-expired.json').exists()
        receipt = session.control({'id': 'explicit-after-query', 'kind': 'pause', 'text': ''}, wait=True)
        assert receipt['command_id'] == 'explicit-after-query'
    finally:
        stop.set()
        thread.join(timeout=2)
        session.close()
