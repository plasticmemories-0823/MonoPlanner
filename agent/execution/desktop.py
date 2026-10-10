"""Local previews and durable, target-bound external-open requests."""
import hashlib
import json
import mimetypes
from pathlib import Path
import subprocess
import time
import uuid

from agent.execution.assistant_files import AssistantFiles
from agent.storage.conversations import ConversationStore
from agent.storage.read_sources import ReadSources
from agent.storage.permissions import Permissions
from agent.execution import applications as installed
from agent.execution import file_actions

DOCUMENTS = {'.txt', '.md', '.html', '.htm', '.css', '.json', '.yaml', '.yml', '.xml',
             '.csv', '.log', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.pdf',
             '.docx', '.xlsx', '.pptx', '.rtf'}
PREVIEW_TEXT = {'.py', '.js', '.ts', '.tsx', '.jsx', '.c', '.cpp', '.h', '.ps1', '.sh',
                '.bat', '.cmd', '.toml', '.ini', '.sql', '.svg', *DOCUMENTS} - {
                    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.pdf', '.docx', '.xlsx', '.pptx'}
COMMAND_ENGINES = {'cmd', 'powershell', 'pwsh', 'wscript', 'cscript', 'mshta', 'rundll32',
                   'regsvr32', 'bash', 'sh', 'wsl', 'node', 'ruby', 'perl', 'java', 'dotnet'}


def document_application(app):
    name = Path(app.get('path', '')).stem.casefold()
    if app.get('kind', 'exe') == 'exe' and (name in COMMAND_ENGINES or name.startswith(('python', 'pypy'))):
        raise ValueError('文档打开不能使用 Shell、脚本解释器或代码运行器')


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def applications(refresh=False):
    return installed.discover(refresh)


class DesktopStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.store = ConversationStore(self.root)
        self.permissions = Permissions(self.root)
        with self.store.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS desktop_requests '
                       '(id TEXT PRIMARY KEY, thread TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS desktop_files '
                       '(thread TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY(thread,path))')
            db.execute('CREATE TABLE IF NOT EXISTS desktop_applications (id TEXT PRIMARY KEY, data TEXT NOT NULL)')

    def applications(self, refresh=False):
        found = {item['id']: item for item in applications(refresh)}
        with self.store.connection() as db:
            for row in db.execute('SELECT data FROM desktop_applications'):
                item = json.loads(row['data'])
                if Path(item['path']).is_file():
                    found[item['id']] = item
        return list(found.values())

    def authorize_file(self, thread, path):
        ReadSources(self.root).add(thread, path, 'file')
        return str(Path(path).resolve())

    def path(self, thread, path, repo, read_roots):
        sources = ReadSources(self.root).get(thread, read_roots)
        return AssistantFiles(repo, read_sources=sources).path(path)

    def put(self, thread, data, status):
        identity = uuid.uuid4().hex
        with self.store.connection() as db:
            self.store.require_available(db, self.store.thread(thread))
            db.execute('INSERT INTO desktop_requests VALUES (?,?,?,?)',
                       (identity, self.store.thread(thread), status, json.dumps(data, ensure_ascii=False)))
        return {**data, 'id': identity, 'status': status}

    def list(self, thread):
        with self.store.connection() as db:
            rows = db.execute('SELECT id,status,data FROM desktop_requests WHERE thread=? ORDER BY rowid',
                              (self.store.thread(thread),))
            return [{**json.loads(r['data']), 'id': r['id'], 'status': r['status']} for r in rows]

    def get(self, thread, identity):
        return next((r for r in self.list(thread) if r['id'] == identity), None)

    def preview(self, thread, path, repo, roots):
        target = self.path(thread, path, repo, roots)
        if target.suffix.lower() not in PREVIEW_TEXT | {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.pdf'}:
            raise ValueError('此类型不支持应用内预览，请选择外部软件')
        if target.stat().st_size > 10*1024*1024:
            raise ValueError('预览文件最大 10 MiB')
        raw = target.read_bytes()
        mime = mimetypes.guess_type(target.name)[0] or 'text/plain'
        if target.suffix.lower() in PREVIEW_TEXT and target.suffix.lower() not in {'.html', '.htm'}:
            raw.decode('utf-8-sig')
            mime = 'text/plain; charset=utf-8'
        sha = hashlib.sha256(raw).hexdigest()
        folder = self.root/'previews'; folder.mkdir(exist_ok=True)
        saved = folder/(uuid.uuid4().hex+'.bin'); saved.write_bytes(raw)
        return self.put(thread, {'kind': 'preview', 'path': str(target), 'sha256': sha,
                                'mime': mime, 'record': str(saved), 'bytes': len(raw)}, 'ready')

    def request_open(self, thread, application_id, file_path=None):
        if Permissions(self.root).get(thread)['effective'] == 'sandbox':
            raise ValueError('沙盒权限禁止外部启动；可使用应用内预览，或由用户切换为风险询问')
        if file_path:
            file_path = file_actions.target_path(str(file_path), readonly=True)
        app = next((a for a in self.applications() if a['id'] == application_id), None)
        if not app:
            raise ValueError('未找到此应用编号，请重新查询已安装应用')
        if file_path and Path(file_path).suffix.lower() not in DOCUMENTS:
            raise ValueError('外部文档打开不允许运行脚本、快捷方式或可执行文件')
        if file_path:
            document_application(app)
        data = {'kind': 'external', 'application': app, 'path': str(file_path) if file_path else '',
                'sha256': fingerprint(file_path) if file_path else '', 'application_sha256': installed.identity(app)}
        data['target'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return self.put(thread, data, 'awaiting_approval')

    def request_file_operation(self, thread, args, repo):
        if Permissions(self.root).get(thread)['effective'] == 'sandbox':
            raise ValueError('沙盒权限禁止目录外文件操作；须由用户选择风险询问或完全权限')
        data = file_actions.prepare(args, repo)
        data['target'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return self.put(thread, data, 'awaiting_approval')

    def reject(self, thread, identity, target):
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self.store.require_available(db, self.store.thread(thread))
            row = db.execute('SELECT status,data FROM desktop_requests WHERE id=? AND thread=?',
                             (identity, self.store.thread(thread))).fetchone()
            if not row or json.loads(row['data']).get('target') != target:
                raise ValueError('批准目标不匹配')
            if row['status'] == 'awaiting_approval':
                db.execute('UPDATE desktop_requests SET status=? WHERE id=?', ('rejected', identity))
        return self.get(thread, identity)

    def approve(self, thread, identity, target):
        thread = self.store.thread(thread)
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self.store.require_available(db, thread)
            mode = db.execute('SELECT effective FROM permissions WHERE thread=?', (thread,)).fetchone()
            if not mode or mode['effective'] == 'sandbox':
                raise ValueError('当前沙盒权限不允许外部操作')
            row = db.execute('SELECT status,data FROM desktop_requests WHERE id=? AND thread=?', (identity, thread)).fetchone()
            if not row:
                raise ValueError('找不到打开请求')
            data = json.loads(row['data'])
            if data.get('kind') not in {'external', 'file_operation'} or data.get('target') != target:
                raise ValueError('打开授权与目标不匹配')
            if row['status'] != 'awaiting_approval':
                return {**data, 'id': identity, 'status': row['status']}
            app = None
            if data['kind'] == 'external':
                app = next((a for a in self.applications(refresh=True) if a['id'] == data['application']['id']), None)
                if not app or installed.identity(app) != data['application_sha256']:
                    raise ValueError('软件已变化，请重新请求打开')
                if data['path']:
                    document_application(app)
                    path = file_actions.target_path(data['path'], readonly=True)
                    if fingerprint(path) != data['sha256']:
                        raise ValueError('文件已变化，请重新请求打开')
            db.execute('UPDATE desktop_requests SET status=? WHERE id=?',
                       ('executing' if data['kind'] == 'file_operation' else 'launching', identity))
        try:
            if data['kind'] == 'file_operation':
                outcome = file_actions.execute(data, self.root/'file-operation-backups'/thread/(identity+'.bin'))
                data.update({k: v for k, v in outcome.items() if k != 'status'})
                status = 'completed'
            else:
                data.update(pid=installed.start(app, data['path'] or None), opened_at=time.time())
                status = 'opened'
        except file_actions.FileActionError as error:
            data['error'], status = str(error), 'unknown' if error.changed else 'failed'
            backup = self.root/'file-operation-backups'/thread/(identity+'.bin')
            if backup.is_file():
                data['backup'] = str(backup)
        except subprocess.TimeoutExpired:
            data['error'], status = '启动结果不明，不自动重试', 'unknown'
        except OSError as error:
            data['error'], status = str(error), 'failed'
        with self.store.connection() as db:
            db.execute('UPDATE desktop_requests SET status=?,data=? WHERE id=?',
                       (status, json.dumps(data, ensure_ascii=False), identity))
        return {**data, 'id': identity, 'status': status}


def invoke(loop, name, args):
    from agent.storage.preferences import state_root
    store, thread = DesktopStore(state_root()), loop.conversation_id
    if not thread:
        raise ValueError('应用打开需要已保存的对话')
    if name == 'list_applications':
        return {'applications': [a for a in store.applications() if args['query'].casefold() in a['name'].casefold()]}
    if name == 'request_file_operation':
        result = store.request_file_operation(thread, args, loop.repo)
    elif name == 'open_file':
        if not args['application_id']:
            return store.preview(thread, args['path'], loop.repo, loop.state.get('read_roots', []))
        result = store.request_open(thread, args['application_id'], args['path'])
    else:
        result = store.request_open(thread, args['application_id'])
    if loop.access_mode == 'full':
        result = store.approve(thread, result['id'], result['target'])
    return {k: v for k, v in result.items() if k != 'content'}
