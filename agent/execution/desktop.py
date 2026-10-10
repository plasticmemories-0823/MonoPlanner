"""Local previews and durable, target-bound external-open requests."""
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import sqlite3
import subprocess
import time
import uuid

from agent.execution.assistant_files import AssistantFiles
from agent.storage.conversations import ConversationStore

DOCUMENTS = {'.txt', '.md', '.html', '.htm', '.css', '.json', '.yaml', '.yml', '.xml',
             '.csv', '.log', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.pdf',
             '.docx', '.xlsx', '.pptx', '.rtf'}
PREVIEW_TEXT = {'.py', '.js', '.ts', '.tsx', '.jsx', '.c', '.cpp', '.h', '.ps1', '.sh',
                '.bat', '.cmd', '.toml', '.ini', '.sql', '.svg', *DOCUMENTS} - {
                    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.pdf', '.docx', '.xlsx', '.pptx'}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def applications():
    """Use registered paths, never names or command lines invented by a model."""
    found = {'记事本': Path(os.environ.get('SystemRoot', r'C:\Windows'))/'System32/notepad.exe'}
    if os.name == 'nt':
        import winreg
        for label, name in [('Microsoft Edge', 'msedge.exe'), ('Google Chrome', 'chrome.exe'),
                            ('Visual Studio Code', 'Code.exe'), ('Adobe Acrobat', 'Acrobat.exe')]:
            for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
                try:
                    with winreg.OpenKey(hive, 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\'+name) as key:
                        value = winreg.QueryValue(key, None).strip('"')
                    path = Path(value)
                    if path.is_absolute() and path.is_file() and path.suffix.lower() == '.exe':
                        found[label] = path.resolve()
                        break
                except OSError:
                    pass
    return [{'id': hashlib.sha256(str(path).casefold().encode()).hexdigest()[:24],
             'name': name, 'path': str(path)} for name, path in found.items() if path.is_file()]


class DesktopStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.store = ConversationStore(self.root)
        with self.store.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS desktop_requests '
                       '(id TEXT PRIMARY KEY, thread TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS desktop_files '
                       '(thread TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY(thread,path))')
            db.execute('CREATE TABLE IF NOT EXISTS desktop_applications (id TEXT PRIMARY KEY, data TEXT NOT NULL)')

    def applications(self):
        found = {item['id']: item for item in applications()}
        with self.store.connection() as db:
            for row in db.execute('SELECT data FROM desktop_applications'):
                item = json.loads(row['data'])
                if Path(item['path']).is_file():
                    found[item['id']] = item
        return list(found.values())

    def register_application(self, path):
        path = Path(path)
        if not path.is_absolute() or path.suffix.lower() != '.exe':
            raise ValueError('请选择 Windows 应用程序（.exe）')
        target = AssistantFiles(str(path.parent)).path(path.name)
        item = {'id': hashlib.sha256(str(target).casefold().encode()).hexdigest()[:24],
                'name': target.stem, 'path': str(target)}
        with self.store.connection() as db:
            db.execute('INSERT OR REPLACE INTO desktop_applications VALUES (?,?)',
                       (item['id'], json.dumps(item, ensure_ascii=False)))
        return item

    def authorize_file(self, thread, path):
        path = Path(path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError('请选择一个已有文件')
        # Single-file authorization still excludes credentials and links.
        target = AssistantFiles(str(path.parent)).path(path.name)
        with self.store.connection() as db:
            self.store.require_available(db, self.store.thread(thread))
            db.execute('INSERT OR IGNORE INTO desktop_files VALUES (?,?)',
                       (self.store.thread(thread), str(target)))
        return str(target)

    def path(self, thread, path, repo, read_roots):
        files = AssistantFiles(repo, read_roots)
        try:
            return files.path(path)
        except ValueError:
            target = Path(path)
            if not target.is_absolute():
                raise
            with self.store.connection() as db:
                row = db.execute('SELECT path FROM desktop_files WHERE thread=? AND path=?',
                                 (self.store.thread(thread), str(target.resolve()))).fetchone()
            if not row:
                raise ValueError('文件不在授权范围，请通过选择窗口授权')
            return AssistantFiles(str(target.parent)).path(target.name)

    def put(self, thread, data, status):
        identity = uuid.uuid4().hex
        with self.store.connection() as db:
            self.store.require_available(db, self.store.thread(thread))
            db.execute('INSERT INTO desktop_requests VALUES (?,?,?,?)',
                       (identity, self.store.thread(thread), status, json.dumps(data, ensure_ascii=False)))
        return {'id': identity, 'status': status, **data}

    def list(self, thread):
        with self.store.connection() as db:
            rows = db.execute('SELECT id,status,data FROM desktop_requests WHERE thread=? ORDER BY rowid',
                              (self.store.thread(thread),))
            return [{'id': r['id'], 'status': r['status'], **json.loads(r['data'])} for r in rows]

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
        app = next((a for a in self.applications() if a['id'] == application_id), None)
        if not app:
            raise ValueError('application_id 不在已注册应用清单中')
        if file_path and Path(file_path).suffix.lower() not in DOCUMENTS:
            raise ValueError('外部文档打开不允许运行脚本、快捷方式或可执行文件')
        data = {'kind': 'external', 'application': app, 'path': str(file_path) if file_path else '',
                'sha256': fingerprint(file_path) if file_path else '', 'application_sha256': fingerprint(app['path'])}
        data['target'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
        return self.put(thread, data, 'awaiting_approval')

    def approve(self, thread, identity, target):
        thread = self.store.thread(thread)
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self.store.require_available(db, thread)
            row = db.execute('SELECT status,data FROM desktop_requests WHERE id=? AND thread=?', (identity, thread)).fetchone()
            if not row:
                raise ValueError('找不到打开请求')
            data = json.loads(row['data'])
            if data.get('kind') != 'external' or data.get('target') != target:
                raise ValueError('打开授权与目标不匹配')
            if row['status'] != 'awaiting_approval':
                return {'id': identity, 'status': row['status'], **data}
            app = next((a for a in self.applications() if a['id'] == data['application']['id']), None)
            if not app or fingerprint(app['path']) != data['application_sha256']:
                raise ValueError('软件已变化，请重新请求打开')
            if data['path']:
                path = Path(data['path'])
                AssistantFiles(str(path.parent)).path(path.name)
                if fingerprint(path) != data['sha256']:
                    raise ValueError('文件已变化，请重新请求打开')
            # Commit consumption BEFORE starting the process. A crash cannot replay it.
            db.execute('UPDATE desktop_requests SET status=? WHERE id=?', ('launching', identity))
        command = [app['path']] + ([data['path']] if data['path'] else [])
        try:
            process = subprocess.Popen(command, close_fds=True, env={k: v for k, v in os.environ.items()
                if not any(s in k.upper() for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD'))})
            data.update(pid=process.pid, opened_at=time.time())
            status = 'opened'
        except OSError as error:
            data['error'], status = str(error), 'failed'
        with self.store.connection() as db:
            db.execute('UPDATE desktop_requests SET status=?,data=? WHERE id=?',
                       (status, json.dumps(data, ensure_ascii=False), identity))
        return {'id': identity, 'status': status, **data}


def invoke(loop, name, args):
    from agent.storage.preferences import state_root
    store, thread = DesktopStore(state_root()), loop.conversation_id
    if not thread:
        raise ValueError('应用打开需要已保存的对话')
    if name == 'list_applications':
        return {'applications': [a for a in store.applications() if args['query'].casefold() in a['name'].casefold()]}
    if name == 'open_file':
        path = store.path(thread, args['path'], loop.repo, loop.state.get('read_roots', []))
        if not args['application_id']:
            return store.preview(thread, str(path), loop.repo, loop.state.get('read_roots', []))
        result = store.request_open(thread, args['application_id'], path)
    else:
        result = store.request_open(thread, args['application_id'])
    if loop.access_mode == 'full':
        return store.approve(thread, result['id'], result['target'])
    return {**result, 'notice': '请点击聊天中的打开按钮，仅授权这个目标一次；任务仍在沙盒中。'}
