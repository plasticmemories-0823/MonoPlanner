"""聊天只读的工作诊断白名单，不提供工作推理和冻结测试正文。"""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from agent.source.snapshots import _safe_file


def read_work_record(store, conversation, record, offset, limit):
    if (not isinstance(record, str) or type(offset) is not int or offset < 0
            or type(limit) is not int or not 1 <= limit <= 16000):
        raise ValueError('记录路径、offset 和 limit 无效')
    path = Path(record)
    parts = path.parts
    if (path.is_absolute() or '..' in parts or ':' in record or '\\' in record
            or not ((len(parts) == 4 and parts[0] == 'plans')
                    or (len(parts) == 3 and parts[0] == 'assistant'))
            or path.name not in {'result.json', 'stdout.log', 'stderr.log'}):
        raise ValueError('只允许已提交操作结果及 stdout/stderr，不读取私有验收或模型记录')
    value = store.get(conversation)
    task = value['work_id']
    if not task:
        raise ValueError('当前对话没有工作记录')
    root = store.root / 'tasks' / task
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
