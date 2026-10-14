"""从已提交的本地任务原件派生只读聊天活动。"""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from agent.source.snapshots import _safe_file
from agent.acceptance.contracts import tests_ready_for_candidate


def checked(root, path, digest):
    raw = _safe_file(root, Path(path)).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('Activity record checksum mismatch: ' + path)
    return json.loads(raw)


def projection(root, task, seq, record, db):
    state, event, payload = record['state'], record['event'], record['payload']
    public = state.get('public_events', [])
    anchor = public[-1]['id'] if public else task['input_id']
    if state.get('conversation_turn'):
        anchor = state['conversation_turn']['id']
    call = payload.get('call')
    row = {'after_id': anchor, 'seq': seq, 'task_id': task['id']}
    if event == 'environment_preparation':
        status = payload.get('status', 'preparing')
        row.update(
            id=f"{task['id']}-environment",
            kind='operation',
            label=payload.get('message', '正在准备项目环境'),
            detail='',
            status=(
                'completed'
                if status == 'ready'
                else (
                    'failed'
                    if status == 'failed'
                    else 'interrupted' if status == 'cancelled' else 'running'
                )
            ),
        )
        return row
    if event.startswith('model_request_') or event == 'model_response_received':
        call = call or state['budget']['used_steps']
        row.update(
            id=f"{task['id']}-thinking-{call}",
            kind='thinking',
            label='正在思考',
            status='running',
            detail='',
        )
        if event == 'model_response_received':
            path = 'requests/call-%04d/response.json' % call
            ref = db.execute('SELECT sha256 FROM artifacts WHERE path=?', (path,)).fetchone()
            if not ref:
                raise ValueError('Model response has no committed original')
            response = checked(root, path, ref[0])
            thought = response['data'].get('additional_kwargs', {}).get('reasoning_content', '')
            finish = response['data'].get('response_metadata', {}).get('finish_reason')
            row.update(
                label='回复达到输出上限' if finish == 'length' else '思考',
                status='interrupted' if finish == 'length' else 'completed',
                detail=thought if isinstance(thought, str) else '',
            )
        elif event == 'model_request_interrupted':
            path = 'requests/call-%04d/interruption.json' % call
            ref = db.execute('SELECT sha256 FROM artifacts WHERE path=?', (path,)).fetchone()
            if not ref:
                raise ValueError('Interrupted request has no committed original')
            interrupted = checked(root, path, ref[0])
            row.update(
                label='回复已中断',
                status='interrupted',
                detail=interrupted.get('reasoning_content', ''),
            )
        elif event != 'model_request_started':
            row.update(label='模型请求未完成', status='unknown')
        return row
    if event in {'step_started', 'step_finished'}:
        pending = state['pending']
        step = next(s for s in pending['plan']['steps'] if s['id'] == payload['id'])
        op = step['op']
        if op == 'shell':
            label, detail = '正在运行命令', step['command']
        elif op == 'search':
            label, detail = '正在搜索', f"{step['path']}\n{step['query']}"
        else:
            label = '正在读取'
            detail = f"{step['path']} · 从第 {step['offset'] + 1} 行起，最多 {step['limit']} 行"
            if step.get('from_step'):
                detail += '\n来源搜索：' + step['from_step']
        row.update(
            id=f"{task['id']}-operation-{pending['id']}-{step['id']}",
            kind='operation',
            label=label,
            detail=detail,
            status='running' if event == 'step_started' else payload['status'],
        )
        return row
    phases = {
        'acceptance_trial_started': ('trial', '正在原始源码上试跑验收'),
        'acceptance_trial_finished': ('trial', '原始源码验收结果'),
        'final_acceptance_started': ('acceptance', '正在运行实际验收'),
        'final_acceptance_finished': ('acceptance', '实际验收结果'),
    }
    if event in {'web_started', 'web_finished', 'web_unknown'}:
        name = payload.get('name', '')
        args = payload.get('args', {})
        row.update(
            id=f"{task['id']}-web-{payload['id']}",
            kind='operation',
            label='正在搜索网页' if name == 'web_search' else '正在读取网页',
            detail=args.get('query') or args.get('url', ''),
            status=payload.get('status', 'unknown' if event == 'web_unknown' else 'running'),
        )
        return row
    if event in {'assistant_started', 'assistant_finished', 'assistant_unknown'}:
        name, args = payload['name'], payload.get('args', {})
        args = args if isinstance(args, dict) else {}
        labels = {
            'list_files': '正在列出目录',
            'read_file': '正在读取文件',
            'search_files': '正在搜索文件',
            'run_existing': '正在运行已有任务',
            'web_search': '正在搜索网页',
            'web_fetch': '正在读取网页',
            'list_applications': '正在查找应用',
            'open_file': '正在准备打开文件',
            'launch_application': '正在准备启动应用',
        }
        labels.update(
            list_directory='正在列出目录',
            search_file_contents='正在搜索文件内容',
            run_project_task='正在运行项目任务',
        )
        detail = args.get('path') or args.get('entry') or args.get('query') or args.get('url', '')
        if name in {'search_files', 'search_file_contents'}:
            detail += '\n' + args.get('query', '')
        if name in {'run_existing', 'run_project_task'}:
            detail += '\n参数：' + json.dumps(args.get('args', []), ensure_ascii=False)
        row.update(
            id=f"{task['id']}-assistant-{payload['request_call']}-{payload['id']}",
            kind='operation',
            label=labels.get(name, '工具调用'),
            detail=detail,
            started_at=payload.get('started_at'),
            seconds=payload.get('seconds'),
            child_usage=payload.get('child_usage'),
            exit_code=payload.get('exit_code'),
            status=payload.get('status', 'unknown' if event == 'assistant_unknown' else 'running'),
        )
        return row
    if event in phases:
        key, label = phases[event]
        identity = payload.get('trial_id') or payload.get('plan')
        row.update(
            id=f"{task['id']}-{key}-{identity}",
            kind='operation',
            label=label,
            detail='',
            stage=key,
            status=payload.get('status', 'running'),
        )
        if event == 'acceptance_trial_finished':
            ref = (state.get('acceptance_trial') or {}).get('baseline_result')
            report = checked(root, ref['record'], ref['sha256']) if ref else None
            if not report:
                row['summary'] = '原始源码试跑结束，结果待核对。'
            elif not tests_ready_for_candidate(report):
                row['summary'] = '原始源码试跑未形成可用验收，需要检查配置或测试。'
            elif payload.get('status') == 'passed':
                row['summary'] = '原始源码试跑通过，接下来审查验收是否覆盖需求。'
            else:
                row['summary'] = '原始源码试跑发现未通过项，接下来审查验收与修复方案。'
            if report:
                row['detail'] = json.dumps(report, ensure_ascii=False, indent=2)
        elif event == 'final_acceptance_finished':
            status = payload.get('status')
            row['summary'] = (
                '修改后实际验收通过。'
                if status == 'passed'
                else (
                    '修改后仍有未通过项，需要根据诊断继续修正。'
                    if status == 'failed'
                    else '修改后验收未完整完成，需要核对运行诊断。'
                )
            )
            if state.get('verification'):
                row['detail'] = json.dumps(state['verification'], ensure_ascii=False, indent=2)
        return row
    return None


def read_activity(root, tasks, cursors, page_size=100):
    """不获取任务租约、不写检查点、不调用模型，也不执行任务。"""
    rows, positions, more, usage, live, warnings = [], dict(cursors), False, {}, [], []
    for task in tasks:
        identity = task['id']
        if not re.fullmatch(r'[a-f0-9]{32}', identity):
            raise ValueError('Invalid activity task ID')
        cursor = cursors.get(identity, 0)
        if type(cursor) is not int or cursor < 0:
            raise ValueError('Invalid activity cursor')
        folder = Path(root) / 'tasks' / identity
        database = folder / 'history.sqlite3'
        if not database.exists():
            continue
        _safe_file(folder, Path('history.sqlite3'))
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=5)) as db:
            commits = db.execute(
                'SELECT seq,path,sha256 FROM commits WHERE seq>? ORDER BY seq LIMIT ?',
                (cursor, page_size + 1),
            ).fetchall()
            more |= len(commits) > page_size
            for seq, path, digest in commits[:page_size]:
                record = checked(folder, path, digest)
                from agent.storage.usage import summarize

                usage[identity] = summarize(record['state'])
                row = projection(folder, task, seq, record, db)
                if row:
                    rows.append(row)
                positions[identity] = seq
        from agent.storage.live_reasoning import read_live

        live_key, version_key = identity + '-live', identity + '-live-version'
        previous, version = cursors.get(live_key, 0), cursors.get(version_key, 0)
        if any(type(v) is not int or v < 0 for v in (previous, version)):
            raise ValueError('Invalid live activity cursor')
        fragments, position, version, pending, display_error = read_live(folder, previous, version)
        for row in fragments:
            if not row['after_id']:
                row['after_id'] = task['input_id']
        live.extend(fragments)
        positions.update({live_key: position, version_key: version})
        more |= pending
        if display_error:
            warnings.append(display_error)
    return {
        'items': rows,
        'live': live,
        'cursors': positions,
        'more': more,
        'usage': usage,
        'display_warning': '; '.join(dict.fromkeys(warnings)),
    }
