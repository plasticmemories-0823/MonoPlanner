"""固定内核的私有管道入口；不启动网页服务，不读写宿主应用数据库。"""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from threading import Event, Lock, Thread
import traceback
import time

from agent.runtime.bundles import PROTOCOL


def query(root, task, method, arguments):
    from agent.storage.history import LocalHistory

    directory = Path(root) / 'tasks' / task
    if method == 'record':
        from agent.application.records import read_task_record

        return read_task_record(directory, **arguments)
    if method == 'activity':
        from agent.storage.activity import read_activity

        if any(row.get('id') != task for row in arguments.get('tasks', [])):
            raise ValueError('Worker query cannot read another task')
        return read_activity(Path(root), **arguments)
    if method not in {'status', 'find'} or (arguments and method == 'status'):
        raise ValueError('Unsupported worker query')
    import sqlite3
    from agent.storage.usage import summarize
    from agent.task.acceptance_limit import public

    history = LocalHistory(directory)
    try:
        try:
            history.connection = sqlite3.connect(
                (directory / 'history.sqlite3').as_uri() + '?mode=ro', uri=True
            )
            state = history.load()
        except sqlite3.DatabaseError:
            history.close()
            with history.lease():
                state = history.load()
        if state is None:
            return {'status': 'interrupted', 'usage': summarize({}),
                    'acceptance_limit': None, 'applied_inputs': []}
        if method == 'find':
            return history.find(arguments['query'])
        return {'status': state['status'], 'usage': summarize(state),
                'acceptance_limit': public(state), 'continuation': state.get('continuation'),
                'reason': state.get('reason', ''), 'applied_inputs': state.get('applied_inputs', [])}
    finally:
        history.close()


def main():
    # 私有控制管道不能被 Git、搜索或用户命令继承。复制为不可继承的描述符，
    # 然后把进程的普通 stdin 指向空输入、stdout 指向日志。
    input_stream = os.fdopen(os.dup(sys.stdin.fileno()), 'r', encoding='utf-8')
    output = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    with open(os.devnull, 'rb') as empty:
        os.dup2(empty.fileno(), 0)
    os.dup2(sys.stderr.fileno(), 1)
    if os.name == 'nt':
        import ctypes
        import msvcrt

        set_handle = ctypes.windll.kernel32.SetStdHandle
        set_handle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
        set_handle.restype = ctypes.c_int
        if not set_handle(-10 & 0xffffffff, msvcrt.get_osfhandle(0)) or not set_handle(
            -11 & 0xffffffff, msvcrt.get_osfhandle(1)
        ):
            raise OSError('Could not isolate worker standard handles')
    sys.stdout = sys.stderr
    lock = Lock()

    def send(kind, **values):
        raw = json.dumps({'protocol': PROTOCOL, 'kind': kind, **values}, ensure_ascii=False)
        with lock:
            output.write(raw + '\n')
            output.flush()

    first = json.loads(input_stream.readline())
    if first.get('protocol') != PROTOCOL:
        raise ValueError('Worker protocol is unsupported')
    root, task = Path(first['root']).resolve(), first['task_id']
    from agent.runtime.bundles import binding

    bound = binding(root, task)
    if bound['engine_id'] != first['engine_id']:
        raise ValueError('Worker engine differs from the task binding')
    from agent.storage.recovery import code_key
    from agent.settings import PROJECT_ROOT
    manifest = json.loads((PROJECT_ROOT / 'manifest.json').read_text('utf-8'))
    if manifest['code_key'] != code_key():
        raise ValueError('Frozen worker source identity differs')
    if first['kind'] == 'query':
        try:
            result = query(root, task, first['method'], first.get('arguments', {}))
            send('result', result=result)
        except Exception as exc:
            send('error', error=type(exc).__name__, message=str(exc)[:1000])
        return
    if first['kind'] != 'run':
        raise ValueError('Unsupported worker command')
    if (first['job'].get('task_id') != task
            or Path(first['job'].get('state_root', root)).resolve() != root):
        raise ValueError('Worker job does not belong to the bound task directory')
    # 凭据只来自宿主私有管道，不归档到内核包、任务参数或子进程命令行。
    key = first.pop('credential', '')
    if key:
        os.environ['MONOPLANER_WORKER_CREDENTIAL'] = key
    os.environ['MONOPLANER_STATE_DIR'] = str(root)
    controls = {}
    for command in first.get('initial_controls', []):
        if (not isinstance(command.get('id'), str) or not command['id']
                or command.get('kind') not in {'pause', 'resume', 'message'}
                or command['id'] in controls):
            raise ValueError('Invalid initial worker controls')
        controls[command['id']] = command
    control_results = {}
    controls_lock = Lock()
    completed = Event()
    host_lost = Event()
    last_heartbeat = [time.monotonic()]
    interrupted_messages = set()
    from agent.application.work_bridge import run_work, pause_work, supplement_work

    def current_controls():
        with controls_lock:
            return list(controls.values())

    def receive():
        try:
            for line in input_stream:
                packet = json.loads(line)
                if packet.get('protocol') != PROTOCOL:
                    raise ValueError('Worker protocol changed')
                if packet['kind'] == 'heartbeat':
                    last_heartbeat[0] = time.monotonic()
                    send('heartbeat')
                    continue
                request = packet['id']
                if packet['kind'] == 'interrupt':
                    if request not in interrupted_messages:
                        supplement_work(task, request, root=root)
                        interrupted_messages.add(request)
                    send('reply', id=request, result={'received': True})
                    continue
                if packet['kind'] == 'query':
                    try:
                        value = query(root, task, packet['method'], packet.get('arguments', {}))
                        send('reply', id=request, result=value)
                    except Exception as exc:
                        send('reply', id=request, error=type(exc).__name__)
                    continue
                if packet['kind'] != 'control':
                    raise ValueError('Unsupported worker control')
                value = packet['value']
                if value.get('id') != request or value.get('kind') not in {'pause', 'resume', 'message'}:
                    raise ValueError('Invalid worker control identity')
                with controls_lock:
                    previous = controls.get(request)
                    if previous is not None and previous != value:
                        raise ValueError('Control identity was reused for different content')
                    controls[request] = value
                if request not in control_results:
                    if value['kind'] == 'pause':
                        receipt = pause_work(task, request, root=root)
                    elif value['kind'] == 'message':
                        if request not in interrupted_messages:
                            supplement_work(task, request, root=root)
                            interrupted_messages.add(request)
                        receipt = {'received': True}
                    else:
                        receipt = {'received': True}
                    control_results[request] = receipt
                send('reply', id=request, result=control_results[request])
        except Exception:
            traceback.print_exc(file=sys.stderr)
        # 宿主退出或管道损坏必须中止实际工作，不能成为无人管理的后台执行器。
        if completed.is_set():
            return
        host_lost.set()
        with controls_lock:
            controls['host-disconnected'] = {'id': 'host-disconnected', 'kind': 'pause', 'text': ''}
        pause_work(task, 'host-disconnected', root=root)

    def host_alive():
        return not host_lost.is_set() and time.monotonic() - last_heartbeat[0] <= 8

    def watch_host():
        while not completed.wait(0.25):
            if host_alive():
                continue
            host_lost.set()
            from agent.task.control import interrupt

            interrupt(root, task, 'pause', 'host-lease-expired')
            Thread(target=lambda: pause_work(task, 'host-lease-expired', root=root),
                   name='lost-host-stop', daemon=True).start()
            # 即使停止回调卡在 Docker API，也不能让失联内核再次派发。
            # 容器自己的租约会独立到期；原生请求与未知预留仍由下次继续核对。
            if not completed.wait(3):
                os._exit(78)
            return

    with redirect_stdout(sys.stderr):
        Thread(target=receive, name='engine-control', daemon=True).start()
        Thread(target=watch_host, name='host-lease-watch', daemon=True).start()
        send('ready', engine_id=bound['engine_id'], task_id=task)
        try:
            from agent.runtime.liveness import host_scope

            with host_scope(host_alive):
                result = run_work(
                    first['job'], lambda kind, payload: send('event', event=kind, payload=payload),
                    current_controls, root=root,
                )
            completed.set()
            send('result', result=result)
        except Exception as exc:
            completed.set()
            traceback.print_exc(file=sys.stderr)
            send('error', error=type(exc).__name__, message=str(exc)[:1000])
        finally:
            os.environ.pop('MONOPLANER_WORKER_CREDENTIAL', None)


if __name__ == '__main__':
    main()
