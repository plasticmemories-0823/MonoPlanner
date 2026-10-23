"""聊天只读的工作诊断白名单，不提供工作推理和冻结测试正文。"""

from contextlib import closing
import hashlib
import json
import re
from pathlib import Path
import sqlite3

from agent.source.snapshots import _safe_file


def diagnostic_facts(state):
    # 只投影诊断字段，不把模型私有推理、凭据、完整请求或冻结测试正文交给聊天。
    result = {key: state.get(key) for key in ('status', 'reason', 'plan_error', 'interruption',
                                             'continuation', 'acceptance_limit')}
    result['evidence_scope'] = (
        'Committed checkpoint diagnostics and recent operation references only. '
        'Missing operation records do not prove that nothing executed or that no files exist. '
        'A complete side-effect audit requires the referenced operation receipts and environment.'
    )
    result['interrupted_requests'] = [
        {key: row[key] for key in ('call', 'purpose', 'kind', 'request_sent', 'usage',
                                   'partial_output_executable', 'causes') if key in row}
        for row in state.get('interrupted_requests', [])[-20:]
    ]
    result['verification'] = {key: value for key, value in (state.get('verification') or {}).items()
                              if key in {'status', 'tests_ready', 'execution_exception', 'record'}}
    pending = state.get('pending') or {}
    result['current_plan'] = {key: pending[key] for key in ('id', 'cursor') if key in pending}
    result['verification_running_at_checkpoint'] = bool(state.get('verification_running'))
    result['operation_records'] = [
        {key: row[key] for key in ('id', 'op', 'status', 'record') if key in row}
        for row in (state.get('last_result') or {}).get('steps', [])[-50:]
    ]
    return result


def read_work_record(store, conversation, record, offset, limit, *, task_id=None):
    if (not isinstance(record, str) or type(offset) is not int or offset < 0
            or type(limit) is not int or not 1 <= limit <= 16000):
        raise ValueError('记录路径、offset 和 limit 无效')
    path = Path(record)
    parts = path.parts
    if (path.is_absolute() or '..' in parts or ':' in record or '\\' in record
            or not (record == 'diagnostics' or (len(parts) == 4 and parts[0] == 'plans')
                    or (len(parts) == 3 and parts[0] == 'assistant'))
            or (record != 'diagnostics' and path.name not in {'result.json', 'stdout.log', 'stderr.log'})):
        raise ValueError('只允许已提交操作结果及 stdout/stderr，不读取私有验收或模型记录')
    value = store.get(conversation)
    task = task_id if task_id is not None else value['work_id']
    selected = store.work(task)
    if task_id is not None and (selected is None or selected['conversation_id'] != conversation):
        raise ValueError('The selected task does not belong to this conversation')
    if not task:
        raise ValueError('当前对话没有工作记录')
    root = store.root / 'tasks' / task
    if (root / 'engine-binding.json').exists():
        from agent.runtime.bridge import read

        return read(store.root, task, 'record', {'record': record, 'offset': offset, 'limit': limit})
    return read_task_record(root, record, offset, limit)


def read_task_record(root, record, offset, limit):
    """由对应内核解析原生原件；查询仅允许已提交的操作结果与输出。"""
    root = Path(root)
    if record == 'diagnostics':
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16000:
            raise ValueError('Invalid diagnostic page')
        from agent.storage.history import LocalHistory
        if not root.is_dir():
            raise FileNotFoundError('The task has no saved native directory')
        value = diagnostic_facts({})
        value['checkpoint_available'] = False
        if (root / 'history.sqlite3').exists():
            database = _safe_file(root, Path('history.sqlite3'))
            archive = LocalHistory(root)
            try:
                archive.connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
                state = archive.load()
                value = diagnostic_facts(state or {})
                value['checkpoint_available'] = state is not None
                value.update(current_execution_facts(root, state or {}, archive.connection))
                value['acceptance_draft'] = acceptance_draft_facts(root, state or {})
            finally:
                archive.close()
        # 准备失败可能早于原生检查点；保留原件哈希和读取时的来源性质。
        value['saved_receipts'] = []
        for name in ('preparation-failed.json', 'preparation-interrupted.json',
                     'resource-cleanup-error.json', 'hard-interruption.json',
                     'pause-container.json', 'execution-runtime-facts.json'):
            if not (root / name).exists():
                continue
            target = _safe_file(root, Path(name))
            if target.stat().st_size > 2_000_000:
                raise ValueError('Saved diagnostic receipt exceeds 2 MB')
            raw = target.read_bytes()
            saved = json.loads(raw)
            facts = {key: saved[key] for key in ('status', 'stop_reason', 'reason', 'error_type',
                                                 'container_id', 'state') if key in saved}
            if name == 'execution-runtime-facts.json':
                facts['runtime'] = (saved.get('result') or {}).get('facts', {})
            value['saved_receipts'].append({'record': name,
                                            'sha256': hashlib.sha256(raw).hexdigest(),
                                            'provenance': 'saved receipt; not a new live environment probe',
                                            'facts': facts})
        text = json.dumps(value, ensure_ascii=False)
        end = min(len(text), offset + limit)
        return {'record': record, 'text': text[offset:end], 'offset': offset,
                'next_offset': end if end < len(text) else None, 'truncated': end < len(text)}
    path = Path(record)
    parts = path.parts
    if (type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 16000
            or path.is_absolute() or '..' in parts or ':' in record or '\\' in record
            or not ((len(parts) == 4 and parts[0] == 'plans')
                    or (len(parts) == 3 and parts[0] == 'assistant'))
            or path.name not in {'result.json', 'stdout.log', 'stderr.log'}):
        raise ValueError('Only committed task results and bounded output may be read')
    database = _safe_file(root, Path('history.sqlite3'))
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        row = db.execute('SELECT sha256 FROM artifacts WHERE path=?', (path.as_posix(),)).fetchone()
    if row is None:
        raise ValueError('操作结果尚未提交，不能作为确定事实读取')
    target = _safe_file(root, path)
    if target.stat().st_size > 2_000_000:
        raise ValueError('该诊断超过2MB，请在本地记录中检查')
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row[0]:
        raise ValueError('工作原件已变化，停止读取')
    if path.name == 'result.json':
        value = json.loads(raw)
        data = value.get('data') or {}
        allowed = {'status', 'exit_code', 'timed_out', 'error', 'seconds', 'op', 'path',
                   'outcome', 'record', 'child_usage'}
        facts = {key: item for key, item in value.items() if key in allowed}
        if isinstance(data, dict):
            facts['data'] = {key: item for key, item in data.items() if key in allowed}
        text = json.dumps(facts, ensure_ascii=False)
    else:
        text = raw.decode('utf-8', errors='replace')
    end = min(len(text), offset + limit)
    return {'record': record, 'text': text[offset:end], 'offset': offset,
            'next_offset': end if end < len(text) else None, 'truncated': end < len(text)}


def acceptance_draft_facts(root, state):
    """只读取已绑定的试跑原件，给聊天具体冻结障碍而不暴露测试正文。"""
    ref = (state.get('acceptance_trial') or {}).get('baseline_result')
    if not ref:
        return None
    name = ref.get('record', '')
    if not re.fullmatch(r'acceptance/[0-9a-f]{64}/[A-Za-z0-9_./-]+', name):
        raise ValueError('Invalid acceptance diagnostic reference')
    target = _safe_file(root, Path(name))
    if target.stat().st_size > 2_000_000:
        raise ValueError('Acceptance diagnostic exceeds 2 MB')
    raw = target.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref.get('sha256'):
        raise ValueError('Acceptance diagnostic original changed')
    report = json.loads(raw)
    from agent.acceptance.contracts import tests_ready_for_candidate
    from agent.acceptance.revisions import freeze_feedback

    ready = tests_ready_for_candidate(report)
    return {'record': name, 'sha256': ref['sha256'], 'ready_for_freeze': ready,
            'status': report.get('status'), 'tests_run': report.get('tests_run'),
            'reason': '' if ready else freeze_feedback(report)}


def current_execution_facts(root, state, database):
    """补充当前计划的已提交操作与未完成验收证据，不暴露冻结断言或模型原文。"""
    result = {'current_operation_records': [], 'acceptance_receipts': []}
    plan = (state.get('pending') or {}).get('id', '')
    if not re.fullmatch(r'p[0-9]{4,12}', plan):
        return result
    for row in database.execute(
        'SELECT path,sha256 FROM artifacts WHERE path GLOB ? ORDER BY path LIMIT 51',
        ('plans/' + plan + '/*/result.json',),
    ):
        name, expected = row
        if len(Path(name).parts) != 4:
            continue
        target = _safe_file(root, Path(name))
        if target.stat().st_size > 2_000_000:
            raise ValueError('Current operation diagnostic exceeds 2 MB')
        raw = target.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Current operation original changed')
        saved = json.loads(raw)
        result['current_operation_records'].append({
            'record': name, 'sha256': expected,
            **{key: saved[key] for key in ('id', 'op', 'status') if key in saved},
        })
    attempt = state.get('verification_attempt', 0)
    if type(attempt) is not int or not 0 <= attempt <= 100000:
        raise ValueError('Invalid saved verification attempt')
    base = root / 'feedback' / plan
    if attempt:
        base = base / ('recovery-%04d' % attempt)
    # 使用检查点指定的当前计划；只读固定报告名，最多十六个验收目录。
    folders = sorted((base / 'verification').glob('acceptance-*')) if base.exists() else []
    if len(folders) > 16:
        raise ValueError('Too many current verification directories for bounded diagnostics')
    for folder in folders:
        for filename in ('candidate.tests.json', 'candidate.execution-error.json',
                         'candidate.result.json'):
            relative = (folder / filename).relative_to(root)
            if not (root / relative).exists():
                continue
            target = _safe_file(root, relative)
            if target.stat().st_size > 2_000_000:
                raise ValueError('Saved acceptance diagnostic exceeds 2 MB')
            raw = target.read_bytes()
            saved = json.loads(raw)
            facts = {key: saved[key] for key in ('phase', 'status', 'reason', 'error_kind',
                                                 'tests_run', 'failures', 'errors', 'skipped')
                     if key in saved}
            process = saved.get('execution_process') or {}
            facts['execution_process'] = {key: process[key] for key in (
                'exit_code', 'timed_out', 'process_cleanup_confirmed', 'container_id', 'error')
                if key in process}
            result['acceptance_receipts'].append({
                'record': relative.as_posix(), 'sha256': hashlib.sha256(raw).hexdigest(),
                'provenance': 'Saved diagnostic original; may precede checkpoint commit. '
                              'This is not permission to replay or proof of overall success.',
                'facts': facts,
            })
    return result
