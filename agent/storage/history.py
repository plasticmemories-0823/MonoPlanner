"""SQLite indexes the existing archive; files hold originals, one loop owns execution."""
from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from agent.storage.records import save
from agent.source.snapshots import _safe_file


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class LocalHistory:
    MESSAGE_CACHE_BYTES = 32 * 1024 * 1024
    MESSAGE_CACHE_ITEMS = 4096
    AUDIT_EVENTS = {'model_request_started', 'step_started', 'plan_accepted',
                    'acceptance_frozen', 'acceptance_trial_started', 'task_returned',
                    'assistant_started', 'web_started', 'final_acceptance_started'}

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.stats = {'read_cache_hits': 0, 'resumed': 0, 'direct_returns': 0}
        self.last_commit = 0
        self.message_cache = OrderedDict()
        self.message_cache_bytes = 0
        self.stats.update(message_cache_hits=0, message_serializations=0, message_audits=0)

    def close(self):
        self.message_cache.clear()
        self.message_cache_bytes = 0
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @contextmanager
    def lease(self):
        lock_path = self.root/'.history.lock'
        if lock_path.is_symlink():
            raise ValueError('History lock cannot be a link')
        with lock_path.open('a+b') as lock:
            if lock.tell() == 0:
                lock.write(b'0'); lock.flush()
            lock.seek(0)
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError('Task already owned by another process') from exc
            try:
                self._open()
                yield self
            finally:
                if os.name == 'nt':
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _open(self):
        self.message_cache.clear()
        self.message_cache_bytes = 0
        db = self.root/'history.sqlite3'
        if db.is_symlink():
            raise ValueError('History database cannot be a link')
        if not db.exists() and (self.root/'state.json').exists():
            raise ValueError('Existing archive has no valid index; explicit migration is required')
        self.connection = sqlite3.connect(db, timeout=5)
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS commits (seq INTEGER PRIMARY KEY, path TEXT, sha256 TEXT);
            CREATE TABLE IF NOT EXISTS artifacts (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, kind TEXT NOT NULL, query TEXT, scope TEXT, source TEXT);
            CREATE TABLE IF NOT EXISTS read_cache (key TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS artifact_kind ON artifacts(kind);
        ''')
        row = self.connection.execute('SELECT COALESCE(MAX(seq),0) FROM commits').fetchone()
        self.last_commit = row[0]
        if not (self.root/'state.json').exists():
            save(self.root/'state.json', {'format':'task-history-reference','database':'history.sqlite3',
                'meaning':'Progress references only; original history is in this archive.'})

    def _immutable(self, relative, value, raw=None):
        path = self.root/relative
        raw = encode(value).encode('utf-8') if raw is None else raw
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != raw:
                raise ValueError('Immutable history changed: '+relative)
        else:
            with path.open('xb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        return {'path':relative, 'sha256':sha(raw)}

    def _checked(self, reference):
        path = _safe_file(self.root, Path(reference['path']))
        raw = path.read_bytes()
        if sha(raw) != reference['sha256']:
            raise ValueError('Recorded history checksum mismatch: '+reference['path'])
        return json.loads(raw)

    def _reference(self, path):
        path = Path(path)
        safe = _safe_file(self.root, path.relative_to(self.root))
        return {'path':safe.relative_to(self.root).as_posix(), 'sha256':sha(safe.read_bytes())}

    def _message(self, message):
        key = id(message)
        cached = self.message_cache.get(key)
        if cached and cached['original'] is message and cached['value'] == message:
            self.message_cache.move_to_end(key)
            self.stats['message_cache_hits'] += 1
            return dict(cached['reference'])
        if cached:
            self.message_cache_bytes -= self.message_cache.pop(key)['size']
        raw = encode(message).encode('utf-8')
        self.stats['message_serializations'] += 1
        reference = self._immutable('history/messages/'+sha(raw)+'.json', message, raw)
        # Keep a separate value to detect in-place edits, and bound both copies.
        size = len(raw) * 2
        if size <= self.MESSAGE_CACHE_BYTES:
            while self.message_cache and (self.message_cache_bytes + size > self.MESSAGE_CACHE_BYTES
                    or len(self.message_cache) >= self.MESSAGE_CACHE_ITEMS):
                self.message_cache_bytes -= self.message_cache.popitem(last=False)[1]['size']
            self.message_cache[key] = {'original': message, 'value': deepcopy(message),
                                      'reference': reference, 'size': size}
            self.message_cache_bytes += size
        return dict(reference)

    def _audit_messages(self, packed):
        references = packed['history_refs'] + packed.get('chat_history_refs', [])
        for name in ('pending', 'compaction_pending'):
            reference = (packed.get(name) or {}).get('response_ref')
            if reference:
                references.append(reference)
        for reference in {r['path']: r for r in references}.values():
            raw = _safe_file(self.root, Path(reference['path'])).read_bytes()
            if sha(raw) != reference['sha256']:
                raise ValueError('Recorded history checksum mismatch: '+reference['path'])
            self.stats['message_audits'] += 1

    def _register(self, reference, kind, query='', scope='', source=''):
        old = self.connection.execute('SELECT sha256 FROM artifacts WHERE path=?', (reference['path'],)).fetchone()
        if old and old[0] != reference['sha256']:
            raise ValueError('Original file changed: '+reference['path'])
        self.connection.execute('INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?,?)',
            (reference['path'],reference['sha256'],kind,query,scope,source))

    def record(self, state, event, payload):
        packed = dict(state)
        packed['history_refs'] = [self._message(m) for m in packed.pop('history')]
        if 'chat_history' in packed:
            packed['chat_history_refs'] = [self._message(m) for m in packed.pop('chat_history')]
        compaction = packed.get('compaction_pending')
        if compaction and compaction.get('response'):
            compaction = dict(compaction)
            compaction['response_ref'] = self._message(compaction.pop('response'))
            packed['compaction_pending'] = compaction
        pending = packed.get('pending')
        if pending:
            pending = dict(pending)
            pending['response_ref'] = self._message(pending.pop('response'))
            pending['rows_refs'] = [self._reference(self.root/r['record']) for r in pending.pop('rows')]
            packed['pending'] = pending
        if event in self.AUDIT_EVENTS:
            self._audit_messages(packed)
        number = self.last_commit + 1
        import uuid
        name = 'history/checkpoints/%06d-%s.json' % (number, uuid.uuid4().hex)
        reference = self._immutable(name, {'state':packed,'event':event,'payload':payload})
        with self.connection:
            if event in {'step_started','step_finished'}:
                pid = state['pending']['id']
                folder = self.root/'plans'/pid/(pid+'-'+payload['id'])
                req = json.loads((folder/'request.json').read_text(encoding='utf-8'))
                for filename in ('request.json','result.json','stdout.log','stderr.log'):
                    file = folder/filename
                    if file.is_file():
                        self._register(self._reference(file), req['op'], req.get('query',''),
                            req.get('path',''), payload.get('before',''))
                if event == 'step_finished' and payload.get('cache_key') and payload['status']=='succeeded':
                    item = self._reference(self.root/payload['record'])
                    self.connection.execute('INSERT OR REPLACE INTO read_cache VALUES(?,?,?)',
                        (payload['cache_key'],item['path'],item['sha256']))
            if event in {'model_request_started','model_response_received','model_request_interrupted'}:
                call = payload['call']
                folder = self.root/'requests'/('call-%04d' % call)
                for name in ('request.json','response.json','interruption.json'):
                    if (folder/name).is_file():
                        self._register(self._reference(folder/name), 'model_record')
            if event == 'plan_accepted':
                self._register(self._reference(self.root/'plans'/state['pending']['id']/'plan.json'),'plan')
            if event == 'plan_finished':
                self._register(self._reference(self.root/'plans'/payload['plan']/'result.json'),'plan_result')
            self.connection.execute('INSERT INTO commits VALUES(?,?,?)',
                (number,reference['path'],reference['sha256']))
        self.last_commit = number
        with (self.root/'events.jsonl').open('a',encoding='utf-8') as out:
            out.write(encode({'kind':event,'payload':payload,'checkpoint':reference['path']})+'\n')

    def load(self):
        row = self.connection.execute('SELECT seq,path,sha256 FROM commits ORDER BY seq DESC LIMIT 1').fetchone()
        if row is None:
            return None
        value = self._checked({'path':row[1],'sha256':row[2]})['state']
        for path, digest in self.connection.execute('SELECT path,sha256 FROM artifacts'):
            if sha(_safe_file(self.root,Path(path)).read_bytes()) != digest:
                raise ValueError('Original record changed: '+path)
        value['history'] = [self._checked(r) for r in value.pop('history_refs')]
        if 'chat_history_refs' in value:
            value['chat_history'] = [self._checked(r) for r in value.pop('chat_history_refs')]
        compaction = value.get('compaction_pending')
        if compaction and compaction.get('response_ref'):
            compaction['response'] = self._checked(compaction.pop('response_ref'))
        pending = value.get('pending')
        if pending:
            pending['response'] = self._checked(pending.pop('response_ref'))
            pending['rows'] = [self._checked(r) for r in pending.pop('rows_refs')]
        return value

    def cached(self, key):
        row = self.connection.execute('SELECT path,sha256 FROM read_cache WHERE key=?',(key,)).fetchone()
        if not row:
            return None
        value = self._checked({'path':row[0],'sha256':row[1]})
        if value['status'] != 'succeeded':
            return None
        self.stats['read_cache_hits'] += 1
        return value

    def register_derived(self, path, kind, source):
        """Called only by the owning main thread, after an immutable file is ready."""
        with self.connection:
            self._register(self._reference(path), kind, source=source)

    def find(self, query, limit=20, prefix=""):
        if Path(prefix).is_absolute() or ".." in Path(prefix).parts or ":" in prefix:
            raise ValueError("History scope must be relative")
        if not isinstance(query,str) or not 1 <= limit <= 50:
            raise ValueError('Use a literal query and limit 1..50')
        rows = self.connection.execute('''SELECT path,sha256,kind,query,scope,source FROM artifacts
            WHERE instr(lower(path || ' ' || COALESCE(query,'') || ' ' || COALESCE(scope,'')),lower(?))>0
            AND kind != 'source_navigation' AND (?='' OR path=? OR substr(path,1,?)=?)
            ORDER BY path LIMIT ?''',(query,prefix,prefix,len(prefix.rstrip('/')+'/'),prefix.rstrip('/')+'/',limit)).fetchall()
        return [dict(zip(('path','sha256','kind','query','scope','source'),r)) for r in rows]
