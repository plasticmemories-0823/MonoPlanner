"""宿主与冻结内核的唯一生产桥接；管道失败不会重发工作或模型请求。"""

import json
import os
from pathlib import Path
import queue
import subprocess
from threading import Event, Lock, Thread
import time
import uuid

from agent.runtime.bundles import PROTOCOL, binding, resolve, safe_path
from agent.storage.preferences import state_root


_sessions = {}
_sessions_lock = Lock()
BOOTSTRAP = (
    "import os,sys; p=os.path.dirname(sys.executable); "
    "sys.path[:]=[sys.argv.pop(1),*[s for s in sys.path if s==p or s.startswith(p+os.sep)]]; "
    "from agent.runtime.worker import main; main()"
)


class WorkerError(RuntimeError):
    """进程边界未得到完整回执；原任务保留，不自动重试。"""


class WorkerUnavailable(ValueError):
    """尚未派发工作的内核或运行设置错误，不是假装执行结果未知。"""


class Session:
    def __init__(self, root, task, *, create=False):
        self.root, self.task = Path(root).resolve(), task
        self.binding = binding(self.root, task, create=create)
        self.source, runtime, _ = resolve(self.root, self.binding['engine_id'])
        environment = dict(os.environ)
        for name in ('DEEPSEEK_API_KEY', 'OPENAI_API_KEY', 'MONOPLANER_WORKER_CREDENTIAL',
                     'PYTHONPATH', 'PYTHONHOME'):
            environment.pop(name, None)
        environment['PATH'] = os.pathsep.join([
            str(runtime / 'git/cmd'), str(runtime / 'git/usr/bin'), str(runtime / 'node'),
            environment.get('PATH', ''),
        ])
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        self.log = (self.root / 'tasks' / task / 'engine-worker.log').open('ab')
        self.process = subprocess.Popen(
            [str(runtime / 'python/python.exe'), '-B', '-X', 'utf8', '-u', '-c',
             BOOTSTRAP, str(self.source)], cwd=self.source, env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
        )
        self.messages = queue.Queue()
        self.replies = {}
        self.sent = {}
        self.lock = Lock()
        self.reply_lock = Lock()
        self.finished = Event()
        Thread(target=self._read, name='engine-events', daemon=True).start()

    def _read(self):
        try:
            for line in self.process.stdout:
                value = json.loads(line)
                if value.get('protocol') != PROTOCOL:
                    raise WorkerError('Worker returned an unsupported protocol')
                if value['kind'] == 'reply':
                    with self.reply_lock:
                        self.replies[value['id']] = value
                else:
                    self.messages.put(value)
        except Exception:
            self.messages.put({'kind': 'error', 'error': 'InvalidWorkerResponse'})
        finally:
            self.finished.set()

    def send(self, kind, **values):
        with self.lock:
            if self.process.poll() is not None:
                raise WorkerError('Worker exited before command receipt')
            try:
                self.process.stdin.write(json.dumps(
                    {'protocol': PROTOCOL, 'kind': kind, **values}, ensure_ascii=False
                ) + '\n')
                self.process.stdin.flush()
            except (OSError, ValueError) as exc:
                raise WorkerError('Worker command receipt is unknown; no retry') from exc

    def control(self, value, *, wait=False):
        identity = value['id']
        with self.reply_lock:
            previous = self.sent.get(identity)
            if previous is not None and previous != value:
                raise WorkerError('Control identity was reused for different content')
            fresh = identity not in self.sent
            self.sent[identity] = value
        if fresh:
            self.send('control', id=identity, value=value)
        if wait:
            return self.reply(identity)

    def reply(self, identity, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.reply_lock:
                value = self.replies.get(identity)
            if value:
                if 'error' in value:
                    raise WorkerError('Frozen worker query failed: ' + value['error'])
                return value['result']
            if self.finished.wait(0.05):
                break
        raise WorkerError('Worker receipt unavailable; inspect the original task before continuing')

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=15)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            # 原生暂停先有机会保存现场；最终只终止本桥接拥有的准确进程对象。
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.process.stdout.close()
            self.log.close()


def run_work(job, emit, controls):
    from agent.application.lease import ServiceLease

    root = Path(job.get('state_root') or state_root()).resolve()
    task = job['task_id']
    if not isinstance(task, str) or len(task) != 32 or any(c not in '0123456789abcdef' for c in task):
        raise ValueError('Invalid work identity')
    directory = safe_path(root, 'tasks/' + task)
    directory.mkdir(parents=True, exist_ok=True)
    lease = ServiceLease(directory)
    lease.acquire()
    try:
        return _run_work(job, emit, controls, root, task)
    finally:
        lease.close()


def _run_work(job, emit, controls, root, task):
    from agent.settings import load_settings

    key = (str(root), task)
    try:
        settings = load_settings(job.get('settings', job.get('runtime_settings', {})))
        session = Session(root, task, create=True)
    except (ValueError, OSError) as exc:
        raise WorkerUnavailable(str(exc)) from exc
    received_result = False
    try:
        session.send('run', root=str(root), task_id=task,
                     engine_id=session.binding['engine_id'], job=job,
                     credential=settings['api_key'], initial_controls=controls())
        with _sessions_lock:
            if key in _sessions:
                raise WorkerError('Task already has an active worker')
            _sessions[key] = session
        while True:
            for command in controls():
                if session.finished.is_set() or session.process.poll() is not None:
                    break
                session.control(command)
            try:
                value = session.messages.get(timeout=0.1)
            except queue.Empty:
                if session.finished.is_set():
                    raise WorkerError('Worker ended without a final saved result')
                continue
            if value['kind'] == 'event':
                emit(value['event'], value['payload'])
            elif value['kind'] == 'result':
                received_result = True
                return value['result']
            elif value['kind'] == 'error':
                raise WorkerError('Frozen worker stopped: ' + value['error'])
    finally:
        session.close()
        with _sessions_lock:
            _sessions.pop(key, None)
        if not received_result:
            # 工作进程异常退出仍须停止原容器，不能遗留无人管理的命令。
            # 沿用与桌面退出相同的所属资源协议，不派发新工作或请求模型。
            try:
                pause_work(task, 'worker-exited-' + uuid.uuid4().hex, root=root)
            except Exception:
                import logging

                logging.getLogger(__name__).exception('Worker exit cleanup is unconfirmed')


def _session(root, task):
    with _sessions_lock:
        return _sessions.get((str(Path(root).resolve()), task))


def pause_work(task, command, *, root=None):
    root = Path(root or state_root()).resolve()
    session = _session(root, task)
    if session:
        return session.control({'id': command, 'kind': 'pause', 'text': ''}, wait=True)
    # 未派发或已退出的任务没有可取消连接；沿既有归属台账停止容器，不启动内核。
    from agent.application.work_bridge import pause_work as stop_owned

    return stop_owned(task, command, root=root)


def supplement_work(task, command, *, root=None):
    # 原文由调度器的持久控制队列交付，避免先发一个空补充占用相同命令编号。
    session = _session(root or state_root(), task)
    if session:
        session.send('interrupt', id=command)


def read(root, task, method, arguments=None):
    arguments = arguments or {}
    session = _session(root, task)
    if session:
        identity = 'query-' + uuid.uuid4().hex
        session.send('query', id=identity, method=method, arguments=arguments)
        return session.reply(identity)
    session = Session(root, task)
    try:
        session.send('query', root=str(Path(root).resolve()), task_id=task,
                     engine_id=session.binding['engine_id'], method=method, arguments=arguments)
        value = session.messages.get(timeout=30)
        if value['kind'] != 'result':
            raise WorkerError('Frozen task query failed: ' + value.get('error', 'UnknownResponse'))
        return value['result']
    finally:
        session.close()


def activity(root, tasks, cursors, page_size=100):
    """各任务的原内核解释活动原件，宿主只合并稳定的公开投影。"""
    from agent.storage.activity import read_activity

    combined = {'items': [], 'live': [], 'cursors': dict(cursors), 'more': False,
                'usage': {}, 'display_warning': ''}
    warnings = []
    for task in tasks:
        if (Path(root) / 'tasks' / task['id'] / 'engine-binding.json').exists():
            page = read(root, task['id'], 'activity', {
                'tasks': [task], 'cursors': cursors, 'page_size': page_size,
            })
        else:
            page = read_activity(root, [task], cursors, page_size)
        for key in ('items', 'live'):
            combined[key].extend(page.get(key, []))
        for key in ('cursors', 'usage'):
            combined[key].update(page.get(key, {}))
        combined['more'] |= page['more']
        if page.get('display_warning'):
            warnings.append(page['display_warning'])
    combined['display_warning'] = '; '.join(dict.fromkeys(warnings))
    return combined
