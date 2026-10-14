"""产品会话的唯一写入口；工作原件仍由内核自己的任务记录管理。"""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
import uuid

from agent.storage.preferences import Preferences
from agent.storage.usage import summarize


ACTIVE = {'queued', 'running', 'needs_plan', 'executing', 'pausing'}
TERMINAL = {'verified_success', 'task_blocked', 'task_cancelled', 'chat_complete',
            'existing_complete', 'stalled', 'budget_exhausted', 'preparation_failed'}
UNKNOWN = {'model_request_unknown', 'execution_unknown', 'state_conflict'}
RESUMABLE = {'paused', 'awaiting_user', 'output_limit_reached'}
SUPPLEMENTABLE = (ACTIVE - {'pausing'}) | {'paused', 'awaiting_user'}
WORK_STATE_FIELDS = {'status', 'reason', 'updated_at', 'revision', 'usage', 'control_unconfirmed'}


class Conflict(ValueError):
    """请求与已保存事实冲突，不能通过重复发送消除。"""


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def identity(value):
    return str(uuid.UUID(str(value)))


class ApplicationStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'application.sqlite3'
        if self.path.is_symlink():
            raise ValueError('Application database cannot be a link')
        with self.connection() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS app_conversations (
                    id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS app_messages (
                    seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                    conversation TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS app_commands (
                    seq INTEGER PRIMARY KEY, id TEXT NOT NULL,
                    conversation TEXT NOT NULL, kind TEXT NOT NULL,
                    body TEXT NOT NULL, state TEXT NOT NULL, task_id TEXT,
                    UNIQUE(conversation,id));
                CREATE TABLE IF NOT EXISTS app_events (
                    seq INTEGER PRIMARY KEY, conversation TEXT NOT NULL,
                    kind TEXT NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS app_events_conversation
                    ON app_events(conversation,seq);
                CREATE TABLE IF NOT EXISTS app_work (
                    id TEXT PRIMARY KEY, conversation TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS app_work_state (
                    id TEXT PRIMARY KEY, conversation TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS app_model_messages (
                    seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL,
                    conversation TEXT NOT NULL, lane TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS app_calls (
                    id TEXT PRIMARY KEY, conversation TEXT NOT NULL, lane TEXT NOT NULL,
                    state TEXT NOT NULL, request TEXT NOT NULL, response_id TEXT,
                    usage TEXT, seconds REAL, error TEXT);
                CREATE TABLE IF NOT EXISTS app_progress (
                    task_id TEXT PRIMARY KEY, conversation TEXT NOT NULL,
                    pending_seq INTEGER NOT NULL DEFAULT 0,
                    reported_seq INTEGER NOT NULL DEFAULT 0,
                    last_started REAL NOT NULL DEFAULT 0, inflight INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS app_context (
                    conversation TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS app_titles (
                    conversation TEXT PRIMARY KEY, data TEXT NOT NULL);
            ''')

    @contextmanager
    def connection(self):
        with closing(sqlite3.connect(self.path, timeout=15)) as db:
            db.row_factory = sqlite3.Row
            db.execute('PRAGMA synchronous=FULL')
            db.execute('PRAGMA busy_timeout=15000')
            with db:
                yield db

    @staticmethod
    def _get(db, conversation, *, writable=False):
        row = db.execute('SELECT data FROM app_conversations WHERE id=?', (conversation,)).fetchone()
        if row is None:
            raise KeyError('Conversation does not exist')
        value = json.loads(row['data'])
        if writable and value['visibility'] in {'purging', 'purged'}:
            raise Conflict('Conversation is being deleted')
        return value

    @staticmethod
    def _put(db, value):
        value['updated_at'] = timestamp()
        db.execute('UPDATE app_conversations SET data=? WHERE id=?',
                   (encode(value), value['thread_id']))

    @staticmethod
    def _event(db, conversation, kind, payload):
        cursor = db.execute(
            'INSERT INTO app_events(conversation,kind,payload,created) VALUES(?,?,?,?)',
            (conversation, kind, encode(payload), timestamp()),
        )
        return cursor.lastrowid

    def create(self, conversation=None, repo='', runtime_settings=None):
        conversation = identity(conversation or uuid.uuid4())
        settings = Preferences.validate(runtime_settings or Preferences(self.root).defaults())
        if repo:
            path = Path(repo).expanduser().resolve()
            if not path.is_dir():
                raise ValueError('Workspace must be an existing directory')
            repo = str(path)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT data FROM app_conversations WHERE id=?', (conversation,)).fetchone()
            if old:
                return json.loads(old['data'])
            legacy_path = self.root / 'conversations.sqlite3'
            if legacy_path.is_file():
                with closing(sqlite3.connect(legacy_path.as_uri() + '?mode=ro', uri=True)) as legacy:
                    if legacy.execute("SELECT 1 FROM sqlite_master WHERE name='sessions'").fetchone():
                        if legacy.execute('SELECT 1 FROM sessions WHERE thread=?', (conversation,)).fetchone():
                            raise Conflict('旧版本会话仅可查看；请使用新的会话编号')
            value = {
                'thread_id': conversation, 'created_at': timestamp(), 'updated_at': timestamp(),
                'repo': repo, 'runtime_settings': settings, 'status': 'idle',
                'chat_status': 'idle', 'visibility': 'active', 'work_id': None,
                'title': {'title': '新对话', 'status': 'temporary', 'revision': 0},
                'read_only': False,
                'configuration_revision': 0,
            }
            db.execute('INSERT INTO app_conversations VALUES(?,?)', (conversation, encode(value)))
            self._event(db, conversation, 'conversation_created', {})
        return value

    def get(self, conversation):
        with self.connection() as db:
            return self._get(db, identity(conversation))

    def list(self):
        with self.connection() as db:
            values = [json.loads(row[0]) for row in db.execute('SELECT data FROM app_conversations')]
        known = {v['thread_id'] for v in values}
        values.extend(v for v in self.legacy() if v['thread_id'] not in known)
        return sorted((v for v in values if v['visibility'] != 'purged'),
                      key=lambda v: v['updated_at'], reverse=True)

    def legacy(self, conversation=None):
        """旧会话只读展示，既不迁移检查点，也不接管原运行任务。"""
        path = self.root / 'conversations.sqlite3'
        if not path.is_file() or path.is_symlink():
            return []
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if 'sessions' not in tables:
                return []
            query, args = 'SELECT thread,data FROM sessions', ()
            if conversation:
                query += ' WHERE thread=?'
                args = (conversation,)
            results = []
            for row in db.execute(query, args):
                saved = json.loads(row['data'])
                if saved.get('application_managed'):
                    continue
                title = None
                if 'conversation_titles' in tables:
                    record = db.execute('SELECT data FROM conversation_titles WHERE thread=?',
                                        (row['thread'],)).fetchone()
                    title = json.loads(record[0]) if record else None
                visible = db.execute('SELECT visibility FROM presentation WHERE thread=?',
                                     (row['thread'],)).fetchone() if 'presentation' in tables else None
                messages = [{'id': m['data'].get('id', ''), 'type': m['type'],
                             'content': m['data']['content'], 'source': 'user' if m['type'] == 'human' else 'chat'}
                            for m in saved.get('public_messages', [])]
                results.append({
                    'thread_id': row['thread'], 'created_at': '', 'updated_at': '',
                    'repo': saved.get('repo', ''), 'runtime_settings': saved.get('runtime_settings', {}),
                    'status': saved.get('status', 'paused'), 'chat_status': 'idle',
                    'work_id': saved.get('task_id'), 'read_only': True,
                    'title': {key: (title or {}).get(key, default) for key, default in
                              (('title', '旧版对话'), ('status', 'complete'), ('revision', 0))},
                    'visibility': visible[0] if visible else 'active', 'messages': messages,
                })
        return results

    @staticmethod
    def _message(db, conversation, message_id, role, content, source):
        old = db.execute('SELECT data FROM app_messages WHERE id=?', (message_id,)).fetchone()
        value = {'id': message_id, 'type': role, 'content': content,
                 'source': source, 'created_at': timestamp()}
        if old:
            original = json.loads(old['data'])
            if any(original[k] != value[k] for k in ('id', 'type', 'content', 'source')):
                raise Conflict('Message identity was reused for different content')
            return
        db.execute('INSERT INTO app_messages(id,conversation,data) VALUES(?,?,?)',
                   (message_id, conversation, encode(value)))
        ApplicationStore._event(db, conversation, 'message', value)

    def publish(self, conversation, message_id, text, source='chat'):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            value = self._get(db, conversation, writable=True)
            self._message(db, conversation, message_id, 'ai', text, source)
            self._put(db, value)

    def submit(self, conversation, command):
        conversation = identity(conversation)
        allowed = {'id', 'kind', 'text', 'repo', 'runtime_settings'}
        if not isinstance(command, dict) or set(command) - allowed:
            raise ValueError('Invalid command fields')
        command = dict(command)
        command['id'] = identity(command.get('id'))
        kind = command.get('kind')
        if kind not in {'message', 'pause', 'resume', 'supplement', 'workspace'}:
            raise ValueError('Invalid command kind')
        if kind in {'message', 'supplement'} and (
            not isinstance(command.get('text'), str) or not command['text'].strip()
        ):
            raise ValueError('Message text is required')
        if len(command.get('text', '')) > 1_000_000:
            raise ValueError('Message is too large')
        if 'runtime_settings' in command:
            command['runtime_settings'] = Preferences.validate(command['runtime_settings'])
        if 'repo' in command:
            repo = Path(command['repo']).expanduser().resolve() if command['repo'] else None
            if repo and not repo.is_dir():
                raise ValueError('Workspace must be an existing directory')
            command['repo'] = str(repo) if repo else ''
        body = encode(command)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            value = self._get(db, conversation, writable=True)
            old = db.execute('SELECT body,state FROM app_commands WHERE conversation=? AND id=?',
                             (conversation, command['id'])).fetchone()
            if old:
                if old['body'] != body:
                    raise Conflict('Command identity was reused for different content')
                return {'received': True, 'id': command['id'], 'applied': old['state'] == 'applied'}
            task = self._work(db, value.get('work_id'))
            if kind == 'workspace' and task and task['status'] not in TERMINAL:
                raise Conflict('Pause the original task and select the workspace in a new conversation')
            if kind in {'pause', 'resume', 'supplement'} and not task:
                if kind != 'pause':
                    raise Conflict('There is no work task to control')
            if kind == 'resume' and task and task['status'] in UNKNOWN:
                raise Conflict('Unknown work results require inspection, not replay')
            if kind == 'resume' and task and task['status'] not in RESUMABLE:
                raise Conflict('当前任务不处于可恢复状态')
            if kind == 'supplement' and task and task['status'] not in SUPPLEMENTABLE:
                raise Conflict('当前任务不能接收补充；输出上限请先明确恢复，终态请新建工作')
            if kind in {'workspace', 'message'}:
                before_configuration = (value['repo'], value['runtime_settings'])
                if 'repo' in command and command['repo'] != value['repo']:
                    if task and task['status'] not in TERMINAL:
                        raise Conflict('An active task cannot change workspace')
                    value['repo'] = command['repo']
                if 'runtime_settings' in command:
                    value['runtime_settings'] = command['runtime_settings']
                if before_configuration != (value['repo'], value['runtime_settings']):
                    value['configuration_revision'] = value.get('configuration_revision', 0) + 1
            db.execute('INSERT INTO app_commands(id,conversation,kind,body,state,task_id) VALUES(?,?,?,?,?,?)',
                       (command['id'], conversation, kind, body, 'pending',
                        task['task_id'] if task and kind != 'message' else None))
            if kind in {'message', 'supplement'}:
                self._message(db, conversation, command['id'], 'human', command['text'], 'user')
                if not value['title']['revision']:
                    value['title'] = {'title': ' '.join(command['text'].split())[:80],
                                      'status': 'temporary', 'revision': 1}
            if kind == 'message':
                value['chat_status'] = 'queued'
            elif kind == 'pause' and task and task['status'] in ACTIVE:
                task['status'] = 'pausing'
                self._put_work_state(db, task)
            self._put(db, value)
            self._event(db, conversation, 'command_received', {'id': command['id'], 'kind': kind})
        return {'received': True, 'id': command['id'], 'applied': False}

    def commands(self, *, conversation=None, task=None, states=('pending',)):
        query = 'SELECT * FROM app_commands WHERE state IN (%s)' % ','.join('?' for _ in states)
        params = list(states)
        if conversation:
            query += ' AND conversation=?'
            params.append(conversation)
        if task:
            query += ' AND task_id=?'
            params.append(task)
        with self.connection() as db:
            return [dict(row, value=json.loads(row['body']))
                    for row in db.execute(query + ' ORDER BY seq', params)]

    def queued_work_count(self):
        with self.connection() as db:
            return db.execute("SELECT COUNT(*) FROM app_work_state "
                              "WHERE json_extract(data,'$.status')='queued'").fetchone()[0]

    def command_state(self, conversation, command_id, state, *, error=None):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('UPDATE app_commands SET state=? WHERE conversation=? AND id=?',
                       (state, conversation, command_id))
            self._event(db, conversation, 'command_state',
                        {'id': command_id, 'state': state, 'error': error})

    def chat_state(self, conversation, status):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            value = self._get(db, conversation)
            value['chat_status'] = status
            if not value['work_id']:
                value['status'] = status
            self._put(db, value)
            self._event(db, conversation, 'chat_state', {'status': status})

    @staticmethod
    def _work(db, task):
        row = db.execute('SELECT w.data AS original,s.data AS state FROM app_work w '
                         'JOIN app_work_state s ON s.id=w.id WHERE w.id=?', (task,)).fetchone() if task else None
        return {**json.loads(row['original']), **json.loads(row['state'])} if row else None

    @staticmethod
    def _put_work_state(db, work):
        mutable = {key: work[key] for key in WORK_STATE_FIELDS if key in work}
        db.execute('UPDATE app_work_state SET data=? WHERE id=?', (encode(mutable), work['task_id']))

    def work(self, task):
        with self.connection() as db:
            return self._work(db, task)

    def work_for_conversation(self, conversation):
        with self.connection() as db:
            return [self._work(db, row[0]) for row in db.execute(
                'SELECT id FROM app_work WHERE conversation=?', (conversation,))]

    def create_work(self, conversation, command_id, *, kind='coding', arguments=None,
                    work_options=None):
        from agent.execution.projects import workspace_binding
        from agent.storage.read_sources import ReadSources

        sources = ReadSources(self.root).get(conversation)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            value = self._get(db, conversation, writable=True)
            current = self._work(db, value['work_id'])
            if current and current.get('input_id') == command_id:
                return current
            if current and current['status'] not in TERMINAL:
                raise Conflict('A work task already exists; send an explicit supplement')
            if not value['repo']:
                raise Conflict('请先选择工作目录，再开始工作')
            command = db.execute('SELECT body FROM app_commands WHERE conversation=? AND id=?',
                                 (conversation, command_id)).fetchone()
            if not command:
                raise ValueError('Work requires a saved user input')
            request = json.loads(command['body'])['text']
            history = [json.loads(row['data']) for row in db.execute(
                'SELECT data FROM app_messages WHERE conversation=? ORDER BY seq', (conversation,))]
            public = [{'type': row['type'], 'data': {'content': row['content'], 'id': row['id'],
                       'additional_kwargs': {}, 'response_metadata': {}, 'type': row['type']}}
                      for row in history if row['source'] in {'user', 'chat'} and row['id'] != command_id]
            task = uuid.uuid4().hex
            work = {'task_id': task, 'conversation_id': conversation, 'input_id': command_id,
                    'repo': value['repo'], 'request': request, 'public_conversation': public,
                    'settings': value['runtime_settings'], 'kind': kind, 'arguments': arguments,
                    'state_root': str(self.root), 'read_sources': sources,
                    'execution_binding': workspace_binding(value['repo'], root=self.root),
                    'status': 'queued', 'reason': '', 'created_at': timestamp(),
                    'updated_at': timestamp(), 'revision': 1, 'usage': {}}
            options = work_options or {}
            if set(options) - {'verify_command', 'acceptance', 'budget', 'pause_after_steps'}:
                raise ValueError('Unexpected frozen work option')
            work.update(options)
            # 交接原文只插入一次。后续推进只更新下面的小状态行，不重写完整讨论。
            original = {key: item for key, item in work.items() if key not in WORK_STATE_FIELDS}
            mutable = {key: item for key, item in work.items() if key in WORK_STATE_FIELDS}
            db.execute('INSERT INTO app_work VALUES(?,?,?)', (task, conversation, encode(original)))
            db.execute('INSERT INTO app_work_state VALUES(?,?,?)', (task, conversation, encode(mutable)))
            value.update(work_id=task, status='queued')
            self._put(db, value)
            self._event(db, conversation, 'work_created', {'task_id': task})
        return work

    def work_state(self, task, status=None, *, reason=None, usage=None, applied_inputs=(), **facts):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT conversation,data FROM app_work_state WHERE id=?', (task,)).fetchone()
            if row is None:
                raise KeyError('Work task does not exist')
            work = {**json.loads(row['data']), 'task_id': task, 'conversation_id': row['conversation']}
            if status is not None:
                work['status'] = status
            if reason is not None:
                work['reason'] = reason
            if usage is not None:
                work['usage'] = usage
            # 终止未确认属于持久保护，后来的正常收尾或再次暂停不能将它变为可恢复。
            if work.get('control_unconfirmed'):
                work['status'] = 'execution_unknown'
                work['reason'] = '任务终止未确认，已保留现场；不能直接恢复或重放操作。'
            work['updated_at'] = timestamp()
            work['revision'] += 1
            self._put_work_state(db, work)
            value = self._get(db, work['conversation_id'])
            value['status'] = work['status']
            self._put(db, value)
            for command_id in applied_inputs:
                db.execute("UPDATE app_commands SET state='applied' WHERE task_id=? AND id=?",
                           (task, command_id))
            seq = self._event(db, work['conversation_id'], 'work_state',
                              {'task_id': task, 'status': work['status'], 'reason': work['reason'],
                               'revision': work['revision'], **facts})
        return seq

    def control_unconfirmed(self, task, command_id):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            work = self._work(db, task)
            if work is None:
                raise KeyError('Work task does not exist')
            commands = work.setdefault('control_unconfirmed', [])
            if command_id not in commands:
                commands.append(command_id)
            work.update(status='execution_unknown',
                        reason='任务终止未确认，已保留现场；不能直接恢复或重放操作。',
                        updated_at=timestamp(), revision=work['revision'] + 1)
            self._put_work_state(db, work)
            value = self._get(db, work['conversation_id'])
            value['status'] = work['status']
            self._put(db, value)
            self._event(db, work['conversation_id'], 'work_state', {
                'task_id': task, 'status': work['status'], 'reason': work['reason'],
                'revision': work['revision'],
            })

    def reserve_control(self, conversation, task, command):
        """同一控制命令只派发一次；已保存但结果不明时也不能自动重发。"""
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute(
                "SELECT 1 FROM app_events WHERE conversation=? AND kind='control_dispatched' "
                "AND json_extract(payload,'$.id')=? LIMIT 1",
                (conversation, command['id']),
            ).fetchone()
            if old:
                return False
            self._event(db, conversation, 'control_dispatched', {
                'id': command['id'], 'kind': command['kind'], 'task_id': task,
            })
            return True

    def record_event(self, conversation, kind, payload):
        with self.connection() as db:
            self._get(db, conversation, writable=True)
            return self._event(db, conversation, kind, payload)

    def pause_after(self, conversation, command_id):
        with self.connection() as db:
            return bool(db.execute(
                "SELECT 1 FROM app_commands WHERE conversation=? AND kind='pause' AND seq>"
                '(SELECT seq FROM app_commands WHERE conversation=? AND id=?) LIMIT 1',
                (conversation, conversation, command_id)).fetchone())

    def events(self, conversation, after=0, limit=200):
        if type(after) is not int or after < 0:
            raise ValueError('Event cursor must be a nonnegative integer')
        with self.connection() as db:
            self._get(db, conversation)
            rows = db.execute('SELECT * FROM app_events WHERE conversation=? AND seq>? ORDER BY seq LIMIT ?',
                              (conversation, after, limit + 1)).fetchall()
        events = [{'seq': row['seq'], 'kind': row['kind'], 'conversation_id': conversation,
                   'payload': json.loads(row['payload']), 'created_at': row['created']}
                  for row in rows[:limit]]
        return {'events': events, 'cursor': events[-1]['seq'] if events else after,
                'more': len(rows) > limit}

    def work_events(self, conversation, task, after=0, limit=50, *, latest=True, through=None):
        upper = ' AND seq<=?' if through is not None else ''
        params = [conversation, after, task]
        if through is not None:
            params.append(through)
        params.append(limit + 1)
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM app_events WHERE conversation=? AND seq>? AND kind IN "
                "('work_state','work_fact') AND json_extract(payload,'$.task_id')=? "
                + upper + (' ORDER BY seq DESC' if latest else ' ORDER BY seq ASC')
                + ' LIMIT ?', params).fetchall()
        selected = list(reversed(rows[:limit])) if latest else rows[:limit]
        return {'events': [{'seq': row['seq'], 'kind': row['kind'],
                            'payload': json.loads(row['payload'])} for row in selected],
                'cursor': selected[-1]['seq'] if selected else after, 'truncated': len(rows) > limit}

    def model_message(self, conversation, lane, message_id, raw):
        data = encode(raw)
        with self.connection() as db:
            old = db.execute('SELECT data FROM app_model_messages WHERE id=?', (message_id,)).fetchone()
            if old and old['data'] != data:
                raise Conflict('Model message original changed')
            db.execute('INSERT OR IGNORE INTO app_model_messages(id,conversation,lane,data) VALUES(?,?,?,?)',
                       (message_id, conversation, lane, data))
        return message_id

    def model_history(self, conversation, lane='chat'):
        with self.connection() as db:
            return [(row['id'], json.loads(row['data'])) for row in db.execute(
                'SELECT id,data FROM app_model_messages WHERE conversation=? AND lane=? ORDER BY seq',
                (conversation, lane))]

    def context(self, conversation):
        with self.connection() as db:
            row = db.execute('SELECT data FROM app_context WHERE conversation=?', (conversation,)).fetchone()
            return json.loads(row[0]) if row else {}

    def save_context(self, conversation, state, event, payload=None):
        # 历史在 app_model_messages 中只保存一次；这里只保存视图及不可变引用。
        saved = {key: value for key, value in state.items() if key != 'history'}
        with self.connection() as db:
            db.execute('INSERT INTO app_context VALUES(?,?) ON CONFLICT(conversation) '
                       'DO UPDATE SET data=excluded.data', (conversation, encode(saved)))
            if event != 'context_admitted':
                self._event(db, conversation, 'chat_context', {'event': event, **(payload or {})})

    def begin_call(self, conversation, lane, request):
        call = uuid.uuid4().hex
        with self.connection() as db:
            db.execute('INSERT INTO app_calls VALUES(?,?,?,?,?,?,?,?,?)',
                       (call, conversation, lane, 'reserved', encode(request), None, None, None, None))
        return call

    def mark_sent(self, call):
        with self.connection() as db:
            db.execute("UPDATE app_calls SET state='sent' WHERE id=? AND state='reserved'", (call,))

    def finish_call(self, call, *, response_id=None, usage=None, seconds=0, error=None):
        with self.connection() as db:
            old = db.execute('SELECT state FROM app_calls WHERE id=?', (call,)).fetchone()
            state = 'complete' if response_id else ('unknown' if old and old[0] == 'sent' else 'not_sent')
            db.execute('UPDATE app_calls SET state=?,response_id=?,usage=?,seconds=?,error=? WHERE id=?',
                       (state, response_id,
                        encode(usage) if usage else None, seconds, error, call))

    def usage(self, conversation):
        result = {}
        with self.connection() as db:
            rows = db.execute("SELECT lane,usage,seconds FROM app_calls WHERE conversation=? "
                              "AND state IN ('sent','complete','unknown')",
                              (conversation,)).fetchall()
        for lane in ('chat', 'progress', 'title'):
            result[lane] = summarize({'model_usage': [
                {'usage': json.loads(row['usage']) if row['usage'] else None,
                 'seconds': row['seconds'] or 0} for row in rows if row['lane'] == lane]})
            result[lane]['purpose'] = lane
        result['chat']['context'] = self.context(conversation).get('context_usage')
        work_usage = summarize({'model_usage': []})
        with self.connection() as db:
            saved_states = [json.loads(row[0]) for row in db.execute(
                'SELECT data FROM app_work_state WHERE conversation=? ORDER BY rowid', (conversation,))]
        for work in saved_states:
            saved = work.get('usage') or {}
            for key in ('input_tokens', 'output_tokens', 'total_tokens', 'cached_input_tokens',
                        'cache_known_input_tokens', 'cache_unknown_calls', 'uncached_input_tokens',
                        'calls', 'unknown_calls', 'model_seconds', 'elapsed_seconds'):
                work_usage[key] += saved.get(key, 0)
            if saved.get('context'):
                work_usage['context'] = saved['context']
        work_usage['cache_rate'] = (
            work_usage['cached_input_tokens'] / work_usage['cache_known_input_tokens']
            if work_usage['cache_known_input_tokens'] else None)
        result['work'] = work_usage
        return result

    def progress_pending(self, task, seq):
        work = self.work(task)
        with self.connection() as db:
            db.execute('INSERT INTO app_progress(task_id,conversation,pending_seq) VALUES(?,?,?) '
                       'ON CONFLICT(task_id) DO UPDATE SET pending_seq=MAX(pending_seq,excluded.pending_seq)',
                       (task, work['conversation_id'], seq))

    def reserve_progress(self, now=None):
        now = time.time() if now is None else now
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM app_progress WHERE inflight=0 AND pending_seq>reported_seq '
                             'AND last_started<=? ORDER BY pending_seq LIMIT 1', (now - 30,)).fetchone()
            if row is None:
                return None
            db.execute('UPDATE app_progress SET inflight=1,last_started=? WHERE task_id=?',
                       (now, row['task_id']))
            return dict(row)

    def finish_progress(self, task, through):
        with self.connection() as db:
            db.execute('UPDATE app_progress SET inflight=0,reported_seq=MAX(reported_seq,?) WHERE task_id=?',
                       (through, task))

    def recover_interrupted(self):
        """只分类已消失的执行者；不会重发请求、启动工作或推进原操作。"""
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute("UPDATE app_calls SET state='unknown',error='ServiceInterrupted' WHERE state='sent'")
            db.execute("UPDATE app_calls SET state='not_sent',error='ServiceInterrupted' WHERE state='reserved'")
            db.execute("UPDATE app_commands SET state='unknown' WHERE state='running'")
            db.execute('UPDATE app_progress SET inflight=0,reported_seq=pending_seq WHERE inflight=1')
            for row in db.execute('SELECT conversation,data FROM app_titles').fetchall():
                title = json.loads(row['data'])
                if title['status'] not in {'pending', 'running'}:
                    continue
                title.update(status='unknown' if title.get('request_sent') else 'failed',
                             error_type='ServiceInterrupted', revision=title['revision'] + 1)
                db.execute('UPDATE app_titles SET data=? WHERE conversation=?',
                           (encode(title), row['conversation']))
                value = self._get(db, row['conversation'])
                value['title'] = {key: title[key] for key in ('title', 'status', 'revision')}
                self._put(db, value)
            conversations = [json.loads(r[0]) for r in db.execute('SELECT data FROM app_conversations')]
            for value in conversations:
                work = self._work(db, value['work_id'])
                if work and work['status'] in ACTIVE:
                    work.update(status='execution_unknown', reason='服务中断，需核对原任务记录后继续')
                    self._put_work_state(db, work)
                    value['status'] = work['status']
                if value['chat_status'] == 'running':
                    value['chat_status'] = 'model_request_unknown'
                self._put(db, value)
            # 旧的补充或恢复若已失去执行对象，明确结束命令，不能让退出检查永远繁忙。
            pending = db.execute("SELECT conversation,id,kind,task_id FROM app_commands "
                                 "WHERE state='pending' AND kind IN ('resume','supplement')").fetchall()
            for row in pending:
                work = self._work(db, row['task_id'])
                allowed = RESUMABLE if row['kind'] == 'resume' else SUPPLEMENTABLE
                if work is None or work['status'] not in allowed:
                    db.execute("UPDATE app_commands SET state='failed' WHERE conversation=? AND id=?",
                               (row['conversation'], row['id']))
                    self._event(db, row['conversation'], 'command_state', {
                        'id': row['id'], 'state': 'failed', 'error': 'WorkNoLongerAcceptsControl',
                    })

    def snapshot(self, conversation):
        conversation = identity(conversation)
        with self.connection() as db:
            try:
                value = self._get(db, conversation)
            except KeyError:
                legacy = self.legacy(conversation)
                if not legacy:
                    raise
                return {**legacy[0], 'work': None, 'usage': {}, 'activities': [],
                        'workspace_bound': True, 'event_cursor': 0, 'pending_messages': []}
            work = self._work(db, value['work_id'])
            messages = [json.loads(row['data']) for row in db.execute(
                'SELECT data FROM app_messages WHERE conversation=? ORDER BY seq', (conversation,))]
            cursor = db.execute('SELECT MAX(seq) FROM app_events WHERE conversation=?',
                                (conversation,)).fetchone()[0] or 0
            activities = {}
            for row in db.execute("SELECT payload FROM app_events WHERE conversation=? AND kind='activity' ORDER BY seq",
                                  (conversation,)):
                item = json.loads(row[0])
                old = activities.get(item['id'], {})
                if item.get('append'):
                    item['detail'] = old.get('detail', '') + item.get('detail', '')
                activities[item['id']] = {**old, **item}
        usage = self.usage(conversation)
        public_work = ({key: work.get(key) for key in
                       ('task_id', 'status', 'reason', 'revision', 'created_at', 'updated_at')}
                       if work else None)
        return {**value, 'work': public_work, 'messages': messages, 'pending_messages': [],
                'workspace_bound': bool(work), 'usage': usage, 'event_cursor': cursor,
                'activities': list(activities.values()), 'read_only': False}

    def visibility(self, conversation, visibility):
        if visibility not in {'active', 'archived', 'deleted'}:
            raise ValueError('Invalid visibility')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            value = self._get(db, conversation, writable=True)
            work = self._work(db, value['work_id'])
            if visibility != 'active' and (value['chat_status'] in ACTIVE or
                                          (work and work['status'] in ACTIVE)):
                raise Conflict('请先暂停并等待当前操作结束')
            value['visibility'] = visibility
            self._put(db, value)
            self._event(db, conversation, 'visibility', {'visibility': visibility})
        return {'thread_id': conversation, 'visibility': visibility}

    def title(self, conversation, title):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            value = self._get(db, conversation, writable=True)
            value['title'] = title
            self._put(db, value)
            self._event(db, conversation, 'title', title)

    def purge(self, conversation, cleanup):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            value = self._get(db, conversation)
            if value['visibility'] not in {'deleted', 'purging', 'purged'}:
                raise Conflict('请先将对话移入回收站')
            if value['visibility'] == 'purged':
                return {'thread_id': conversation, 'deleted': True}
            if value['chat_status'] in ACTIVE:
                raise Conflict('请先暂停当前回复')
            tasks = [self._work(db, row[0]) for row in db.execute(
                'SELECT id FROM app_work WHERE conversation=?', (conversation,))]
            if any(task['status'] in ACTIVE or task['status'] in UNKNOWN for task in tasks):
                raise Conflict('任务仍在运行或结果未确认，不能删除')
            from agent.application.purge import validate_deletion

            validate_deletion(self, db, value, tasks)
            value['visibility'] = 'purging'
            self._put(db, value)
        cleanup(conversation, [task['task_id'] for task in tasks])
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            for table in ('app_messages', 'app_commands', 'app_events', 'app_work', 'app_work_state',
                          'app_model_messages', 'app_calls', 'app_progress', 'app_context', 'app_titles'):
                db.execute('DELETE FROM ' + table + ' WHERE conversation=?', (conversation,))
            value = self._get(db, conversation)
            value.update(visibility='purged', repo='', work_id=None,
                         title={'title': '', 'status': 'deleted', 'revision': 0})
            self._put(db, value)
        return {'thread_id': conversation, 'deleted': True}
