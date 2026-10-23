"""独立编码任务入口；历史查询不调用模型，旧记录只由冻结版本恢复。"""

import argparse
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid

from agent.runtime.bridge import run_work
from agent.execution.projects import workspace_binding
from agent.storage.history import LocalHistory
from agent.storage.preferences import Preferences
from agent.storage.records import digest, save


def _history(folder, query=None):
    """只读旧索引和原件，不取得执行租约或创建检查点。"""
    for name in ('preparation-failed.json', 'preparation-interrupted.json'):
        if query is None and (folder / name).exists():
            from agent.source.snapshots import _safe_file

            return json.loads(_safe_file(folder, Path(name)).read_bytes())
    if not (folder / 'history.sqlite3').is_file():
        return None
    if (folder / 'engine-binding.json').exists():
        from agent.runtime.bridge import read

        return read(folder.parent.parent, folder.name, 'find' if query is not None else 'status',
                    {'query': query} if query is not None else {})
    store = LocalHistory(folder)
    connection = None
    try:
        connection = sqlite3.connect(
            (folder / 'history.sqlite3').as_uri() + '?mode=ro', uri=True
        )
        store.connection = connection
        return store.find(query) if query is not None else store.load()
    finally:
        if connection is not None:
            connection.close()
            store.connection = None
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir', required=True, help='独立记录目录，不能位于工作源码内')
    parser.add_argument('--repo')
    parser.add_argument('--task')
    parser.add_argument('--verify-command')
    parser.add_argument('--acceptance')
    parser.add_argument('--requirements', help='人类确认的交付目标 JSON；缺失时只允许调查')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--pause-after-steps', type=int)
    parser.add_argument('--find', help='Search existing history metadata; no model or shell')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--continue', dest='continue_task', action='store_true',
                        help='明确继续原任务；成功任务复查，耗尽维度追加一批原始额度')
    parser.add_argument('--continue-acceptance', action='store_true',
                        help='明确批准增加一批验收轮次，从原任务断点继续')
    args = parser.parse_args(argv)
    for name in ('max_steps', 'pause_after_steps'):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(name + ' must be positive')
    for name in ('verify_command', 'acceptance'):
        value = getattr(args, name)
        if value is not None and not value.strip():
            parser.error(name + ' cannot be empty')

    root = Path(args.state_dir).expanduser().resolve()
    job_path = root / 'bridge-job.json'
    job = None
    if job_path.exists():
        if job_path.is_symlink():
            parser.error('任务入口记录不能是链接。')
        job = json.loads(job_path.read_text(encoding='utf-8'))
        if (job.get('format') != 1
                or not re.fullmatch(r'[0-9a-f]{32}', job.get('task_id', ''))
                or job.get('state_root') != str(root)):
            parser.error('任务入口原件无效或记录目录已改变；不会迁移旧任务。')
    folder = root / 'tasks' / job['task_id'] if job else root
    old = _history(folder)
    if args.continue_task and job is None:
        parser.error('Continue requires an existing task')
    if (args.continue_task or args.continue_acceptance) and (args.find is not None or args.status):
        parser.error('Continuation cannot be combined with read-only queries')
    from agent.task import acceptance_limit

    limit = acceptance_limit.public(old or {})
    if args.continue_acceptance and not (limit or {}).get('active'):
        parser.error('当前任务不处于可批准增加验收额度的暂停状态。')
    if args.find is not None or args.status:
        if old is None:
            parser.error('No existing indexed task at this path')
        value = (_history(folder, args.find) if args.find is not None else
                 {key: old.get(key) for key in ('status', 'reason', 'source', 'plan_count',
                                                               'acceptance_limit', 'continuation')})
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return 0
    if old is not None and job is None:
        parser.error('旧任务须使用冻结源码和原执行后端恢复，不迁移检查点')

    if job is None:
        if not args.repo or not args.task:
            parser.error('A new task requires --repo and --task')
        repo = Path(args.repo).expanduser().resolve()
        if not repo.is_dir() or root == repo or root.is_relative_to(repo):
            parser.error('请选择已有工作目录，并把 --state-dir 放在工作目录之外。')
        job = {
            'format': 1, 'task_id': uuid.uuid4().hex, 'kind': 'coding',
            'state_root': str(root), 'repo': str(repo), 'request': args.task,
            'settings': Preferences().defaults(), 'public_conversation': [],
            'execution_binding': workspace_binding(str(repo)),
            'verify_command': args.verify_command, 'acceptance': args.acceptance,
            'budget': {'max_steps': args.max_steps or 1000},
            'pause_after_steps': args.pause_after_steps,
        }
        if args.requirements:
            from agent.task.requirements import confirmed

            specification = json.loads(Path(args.requirements).read_text(encoding='utf-8-sig'))
            job['confirmed_requirements'] = confirmed(args.task, specification['targets'],
                confirmation={'id': 'cli:' + str(Path(args.requirements).resolve()),
                              'text': 'User explicitly supplied --requirements'})
        root.mkdir(parents=True, exist_ok=True)
        # 原件只创建一次；同时启动的另一个 CLI 不能覆盖任务身份。
        with job_path.open('x', encoding='utf-8') as stream:
            json.dump(job, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        if args.requirements:
            parser.error('Existing requirement contracts cannot be replaced by a resume flag')
        for name, stored in (('repo', job['repo']), ('task', job['request']),
                             ('verify_command', job.get('verify_command')),
                             ('acceptance', job.get('acceptance'))):
            requested = getattr(args, name)
            if name == 'repo' and requested is not None:
                requested = str(Path(requested).expanduser().resolve())
            if requested is not None and requested != stored:
                parser.error('已有任务的需求、项目和验收原件不能替换，请使用新的 --state-dir。')

    invocation = {**job, 'budget': dict(job['budget'])}
    if args.max_steps is not None and args.max_steps != job['budget']['max_steps']:
        parser.error('已有任务额度只能用 --continue 按批追加，不能替换原始限额。')
    # 步数暂停是本次调用的调试边界，不在下次显式续跑时重复触发。
    invocation['pause_after_steps'] = args.pause_after_steps
    if args.continue_acceptance or (args.continue_task and (limit or {}).get("active")):
        controls = [{'id': uuid.uuid5(uuid.UUID(job['task_id']), limit['pause_id']).hex, 'kind': 'resume',
                     'acceptance_pause_id': limit['pause_id']}]
    elif (limit or {}).get('active'):
        controls = []
    else:
        from agent.task.resumption import stopped

        token = ((old or {}).get('continuation') or {}).get('stop_id')
        # 崩溃可能发生在首次检查点之前，或留下 executing 状态。人工继续仍须
        # 交付明确控制；是否还在运行由真实租约判断，不靠旧状态猜测。
        controls = ([{
            'id': uuid.uuid5(uuid.UUID(job['task_id']), token or digest(old or job)).hex,
            'kind': 'resume', 'native_stop_id': token,
        }] if args.continue_task else [])
        if old and stopped(old.get('status')) and not controls:
            print(json.dumps(old, ensure_ascii=False, indent=2))
            return 0
    save(root / 'cli-invocations' / (uuid.uuid4().hex + '.json'),
         {'task_id': job['task_id'], 'budget': invocation['budget'],
          'pause_after_steps': invocation.get('pause_after_steps'), 'controls': controls})
    from agent.application.lease import LeaseBusy

    try:
        result = run_work(invocation, lambda *_: None, lambda: controls)
    except LeaseBusy:
        # 仅租约占用可返回运行状态；内部故障不能被伪装成“正在运行”。
        print(json.dumps({'status': 'running', 'task_id': job['task_id'],
                          'dispatch_started': False}, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {'messages', 'rounds', 'verification'}},
                     ensure_ascii=False, indent=2))
    return 0 if result['status'] in {'verified_success', 'paused'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
