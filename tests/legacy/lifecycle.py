"""核验并收尾服务重启后遗留的运行状态，不恢复或重放任务。"""

from contextlib import contextmanager, asynccontextmanager, closing
from datetime import datetime
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import uuid
from starlette.exceptions import HTTPException

from agent.storage.activity import checked
from tests.legacy.conversations import ConversationStore
from agent.task import control as interrupts

STOPPED = {
    'paused',
    'awaiting_user',
    'chat_complete',
    'verified_success',
    'task_blocked',
    'settings_required',
    'output_limit_reached',
    'runtime_error',
}
ACTIVE_RECORDS = (
    'request_pending',
    'running_step',
    'verification_running',
    'acceptance_running',
    'assistant_running',
    'web_running',
    'web_request_pending',
)


def runtime_api():
    from langgraph_runtime.database import connect
    from langgraph_runtime.ops import Runs, Threads
    from langgraph_runtime.metrics import get_metrics

    return connect, Runs, Threads, get_metrics


@contextmanager
def stopped_archive(root, session):
    """持有任务租约并核验最后原件，避免把正在执行或结果不明的操作判成空闲。"""
    task = session.get('task_id', '')
    if len(task) != 32 or any(c not in '0123456789abcdef' for c in task):
        raise ValueError('Invalid task identity')
    directory = Path(root) / 'tasks' / task
    lock_path = directory / '.history.lock'
    if lock_path.is_symlink() or directory.is_symlink() or directory.is_junction():
        raise ValueError('Task archive cannot be a link')
    with lock_path.open('r+b') as lock:
        lock.seek(0)
        if os.name == 'nt':
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            unlock = lambda: msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            unlock = lambda: fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        try:
            db_path = directory / 'history.sqlite3'
            if db_path.is_symlink():
                raise ValueError('Task index cannot be a link')
            with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)) as db:
                row = db.execute(
                    'SELECT path,sha256 FROM commits ORDER BY seq DESC LIMIT 1'
                ).fetchone()
            if not row:
                raise ValueError('Task has no committed stop record')
            record = checked(directory, *row)
            state = record['state']
            if (
                session.get('status') not in STOPPED
                or state.get('status') != session['status']
                or any(state.get(key) for key in ACTIVE_RECORDS)
                or (state.get('compaction_pending') or {}).get('stage') == 'committing'
            ):
                raise ValueError('Task stop is not confirmed')
            yield {
                'task': task,
                'checkpoint': row[0],
                'sha256': row[1],
                'mtime': (directory / row[0]).stat().st_mtime,
            }
        finally:
            lock.seek(0)
            unlock()


@asynccontextmanager
async def archive_lease(root, session):
    owner = stopped_archive(root, session)
    acquisition = asyncio.create_task(asyncio.to_thread(owner.__enter__))
    try:
        evidence = await asyncio.shield(acquisition)
    except asyncio.CancelledError:
        await acquisition
        await asyncio.to_thread(owner.__exit__, None, None, None)
        raise
    try:
        yield evidence
    finally:
        await asyncio.to_thread(owner.__exit__, None, None, None)


def saved_sessions(root):
    store = ConversationStore(root)
    with store.connection() as db:
        return [
            (row['thread'], json.loads(row['data']))
            for row in db.execute('SELECT thread,data FROM sessions')
        ]


def save_recovery(root, run, evidence):
    folder = Path(root) / 'service-recovery'
    folder.mkdir(exist_ok=True)
    record_path = folder / (uuid.uuid4().hex + '.json')
    record_path.write_text(
        json.dumps({'run': run, 'evidence': evidence}, default=str, indent=2), 'utf-8'
    )


async def reconcile(root, *, api=None):
    """仅收尾无执行者、无排队请求且有可信停止原件的遗留框架记录。"""
    connect, Runs, Threads, metrics = api or runtime_api()
    if metrics()['workers']['active'] != 0 or interrupts.active_count(root):
        return {'reconciled': 0}
    sessions = await asyncio.to_thread(saved_sessions, root)
    restored = 0
    async with connect() as connection:
        if (await Runs.stats(connection))['n_pending']:
            return {'reconciled': 0}
        for thread, session in sessions:
            if session.get('status') not in STOPPED:
                continue
            try:
                async with archive_lease(root, session) as evidence:
                    records = await Runs.search(
                        connection, uuid.UUID(thread), status='running', limit=100
                    )
                    candidates = [record async for record in records]
                    if not candidates:
                        latest = await Runs.search(connection, uuid.UUID(thread), limit=1)
                        last = await anext(latest, None)
                        if last is None:
                            continue  # 本地历史可能保留了已删除的框架会话。
                        saved = await Threads.get(connection, uuid.UUID(thread))
                        framework_thread = await anext(saved)
                        if (
                            framework_thread.get('status') == 'busy'
                            and not framework_thread.get('interrupts')
                            and last
                            and last['status'] in {'success', 'interrupted'}
                        ):
                            created = last['created_at']
                            created = (
                                datetime.fromisoformat(created)
                                if isinstance(created, str)
                                else created
                            )
                            if evidence['mtime'] >= created.timestamp():
                                if metrics()['workers']['active'] != 0 or interrupts.active_count(
                                    root
                                ):
                                    return {'reconciled': restored}
                                if (await Runs.stats(connection))['n_pending']:
                                    return {'reconciled': restored}
                                await asyncio.to_thread(save_recovery, root, last, evidence)
                                await Threads.set_status(connection, uuid.UUID(thread), None, None)
                                restored += 1
                    for run in candidates:
                        created = run['created_at']
                        created = (
                            datetime.fromisoformat(created) if isinstance(created, str) else created
                        )
                        if evidence['mtime'] < created.timestamp():
                            continue  # 新请求不能被上一轮暂停记录提前结束。
                        if metrics()['workers']['active'] != 0 or interrupts.active_count(root):
                            return {'reconciled': restored}
                        if (await Runs.stats(connection))['n_pending']:
                            return {'reconciled': restored}
                        # 先保留旧框架记录；失败时不消除阻止退出的状态。
                        await asyncio.to_thread(save_recovery, root, run, evidence)
                        await Runs.cancel(
                            connection,
                            [run['run_id']],
                            thread_id=uuid.UUID(thread),
                            action='interrupt',
                        )
                        saved = await Threads.get(connection, uuid.UUID(thread))
                        framework_thread = await anext(saved)
                        if not framework_thread.get('interrupts'):
                            # 不写入 values 或检查点，原有消息与恢复游标保持不变。
                            await Threads.set_status(connection, uuid.UUID(thread), None, None)
                        restored += 1
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
            except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
                continue  # 无法证明空闲的任务仍由原保护规则阻止关闭。
    return {'reconciled': restored}
