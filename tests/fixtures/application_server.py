"""独立前厅的离线浏览器服务；只能由 tests 显式启动，生产不加载。

先设置独立 MONOPLANER_STATE_DIR、MONOPLANER_SERVICE_TOKEN 和两端口，再执行：
python -m uvicorn tests.fixtures.application_server:app --host 127.0.0.1 --port 3059
"""

import json
import os
import sqlite3
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
import time
import uuid

from langchain_core.messages import AIMessage, ToolMessage

from agent.application import api
from agent.application.scheduler import Scheduler
from agent.application.store import ApplicationStore
from agent.model.errors import ModelInterrupted
from agent.storage.preferences import Preferences, state_root
from agent.task import control


def forbidden(*args, **kwargs):
    raise AssertionError('Offline application fixture must not contact a model or install software')


# 即使今后新增路径误用默认模型工厂，也不能触达真实密钥和网络。
import agent.model.client as model_client
import agent.model.network as model_network
import agent.application.titles as title_module
import agent.execution.preparation as preparation
import agent.execution.desktop as desktop_module
import agent.server.desktop as desktop_api

model_client.create_model = forbidden
model_network.model_http_clients = forbidden
title_module.create_model = forbidden
Preferences.key = lambda self: 'offline-fixture-not-a-real-key'


class OfflinePreparation:
    def status(self, repo):
        return {'status': 'ready', 'stage': 'ready', 'repo': repo,
                'message': '离线环境样本，无下载和安装', 'mode': 'automatic'}

    start = status
    retry = status
    cancel = status

    def cancel_owner(self, owner):
        return []


preparation.jobs = lambda root=None: OfflinePreparation()
preparation.active_count = lambda root=None: 0


def native_action(value):
    with (state_root() / 'fixture-native-actions.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def choose_file(command, **kwargs):
    if Path(command[0]).name != 'MonoPlanner.FilePicker.exe' or command[1:] != ['file']:
        raise AssertionError('Only the explicit offline file selection is permitted')
    selected = Path(os.environ['MONOPLANER_FIXTURE_READ_FILE']).resolve()
    if not selected.is_file():
        raise AssertionError('Offline selected file must exist')
    native_action({'kind': 'choose_file', 'path': str(selected)})
    return SimpleNamespace(returncode=0, stdout=str(selected), stderr='')


def start_application(application, path=None):
    if application['id'] != 'offline-viewer':
        raise AssertionError('Only the offline application identity is permitted')
    native_action({'kind': 'launch', 'application': application['id'], 'path': str(path or '')})
    return 424242


# 仅替代不可在无头浏览器操作的原生边界；授权、目标校验、消费与数据持久化保持真实。
desktop_api.subprocess = SimpleNamespace(run=choose_file, CREATE_NO_WINDOW=0)
desktop_module.applications = lambda refresh=False: [{
    'id': 'offline-viewer', 'name': '离线查看器', 'kind': 'packaged', 'aumid': 'OfflineFixture!Viewer',
}]
desktop_module.installed.start = start_application


def response(text, tool=None, args=None):
    return AIMessage(
        content=text,
        tool_calls=[{'name': tool, 'args': args or {}, 'id': uuid.uuid4().hex}] if tool else [],
        additional_kwargs={'reasoning_content': '离线思考片段：确认请求与已保存记录。'},
        response_metadata={'finish_reason': 'stop'},
        usage_metadata={'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120,
                        'input_token_details': {'cache_read': 50}},
    )


class OfflineChat:
    def bind_tools(self, tools):
        return self

    def invoke_with_progress(self, messages, callback):
        for fragment in ('正在确认请求。', '读取已有事实。', '准备公开回复。'):
            signal = control.current()
            if signal and signal.kind:
                raise ModelInterrupted(signal.kind)
            callback(fragment)
            time.sleep(0.35)
        return self.invoke(messages)

    def invoke(self, messages):
        last = messages[-1]
        if messages[0].content.startswith('Write a concise professional progress'):
            facts = json.loads(last.content) if not isinstance(last, ToolMessage) else {}
            task = facts.get('task') or {}
            return response('任务记录已更新，当前状态：' + task.get('status', '已保存'))
        if isinstance(last, ToolMessage):
            return response('已完成查询：' + last.content[:400])
        text = str(last.content)
        if '开始工作' in text or '准备失败工作' in text:
            return response('已保存任务，将在独立工作流程中执行。', 'start_work')
        if '查询进度' in text:
            return response('', 'get_work_status')
        if '联网' in text:
            return response('读取离线网页样本。', 'web_search', {'query': 'offline'})
        if '查询文件' in text:
            return response('读取文件。', 'read_file',
                            {'path': 'demo.py', 'offset': 0, 'limit': 80, 'reread_reason': ''})
        if '读取授权资料' in text:
            return response('读取所选资料。', 'read_file', {
                'path': os.environ['MONOPLANER_FIXTURE_READ_FILE'],
                'offset': 0, 'limit': 80, 'reread_reason': '',
            })
        if '读取未授权相邻文件' in text:
            return response('检查资料边界。', 'read_file', {
                'path': str(Path(os.environ['MONOPLANER_FIXTURE_READ_FILE']).with_name('unselected-note.txt')),
                'offset': 0, 'limit': 80, 'reread_reason': '',
            })
        if '外部打开文件' in text:
            return response('申请一次打开。', 'open_file', {
                'path': str(Path(os.environ['MONOPLANER_FIXTURE_READ_FILE']).resolve()),
                'application_id': 'offline-viewer',
            })
        if '预览文件' in text:
            return response('准备预览。', 'open_file',
                            {'path': 'demo.html', 'application_id': ''})
        return response('已收到：' + text + '。这次聊天不改变后台工作。')


class OfflineTitle:
    def invoke(self, messages):
        return response('离线功能验证')


_signals = {}
_guard = Lock()


def pause(task, command):
    native_action({'kind': 'pause_signalled', 'task': task})
    with _guard:
        signal = _signals.setdefault(task, Event())
    signal.set()


def work(job, emit, controls):
    native_action({'kind': 'work_started', 'task': job['task_id']})
    if '准备失败工作' in job['request']:
        return {'status': 'preparation_failed', 'applied_inputs': [],
                'stop_reason': '开工准备失败：离线模拟磁盘空间不足'}
    with _guard:
        signal = Event()
        _signals[job['task_id']] = signal
    applied = []
    for index in range(160):
        if (state_root() / 'fixture-storage-failure').is_file():
            _failed_writes.set()
            emit('work_state', {'status': 'executing', 'reason': '触发离线记录失败'})
        pending = controls()
        for item in pending:
            if item['id'] in applied:
                continue
            applied.append(item['id'])
            if item['kind'] == 'pause':
                signal.set()
            elif item['kind'] == 'message':
                emit('work_state', {'event': 'user_input_applied', 'status': 'executing',
                                    'reason': '已收到补充原文：' + item['text'],
                                    'applied_inputs': applied})
        if signal.wait(0.25):
            return {'status': 'paused', 'applied_inputs': applied, 'stop_reason': '离线任务已暂停'}
        if index % 8 == 0:
            emit('work_state', {'event': 'plan_finished', 'status': 'executing',
                                'reason': '离线检查阶段 ' + str(index // 8),
                                'applied_inputs': applied,
                                'usage': {'calls': 1, 'total_tokens': 120, 'input_tokens': 100,
                                          'output_tokens': 20, 'cached_input_tokens': 50,
                                          'cache_known_input_tokens': 100,
                                          'uncached_input_tokens': 50, 'unknown_calls': 0}})
    return {'status': 'verified_success', 'applied_inputs': applied, 'stop_reason': '离线验收通过'}


store = ApplicationStore(state_root())
_failed_writes = Event()


def fail_when_requested(method):
    def call(*args, **kwargs):
        if _failed_writes.is_set():
            raise sqlite3.OperationalError('database or disk is full')
        return method(*args, **kwargs)
    return call


# 故障由独立测试目录中的标记触发，只注入记录写入失败；读取及受管取消仍走真实代码。
for _name in ('work_state', 'submit', 'record_event', 'command_state', 'chat_state', 'publish'):
    setattr(store, _name, fail_when_requested(getattr(store, _name)))
service = Scheduler(store, model_factory=lambda settings: OfflineChat(), title_factory=OfflineTitle,
                    work_factory=work, pause_callback=pause, supplement_callback=lambda *args: None,
                    prepare_callback=lambda repo: {'status': 'ready', 'repo': repo})
real_tool = service.chatbot._invoke_tool


def tool(conversation, call, operation):
    if call['name'] in {'web_search', 'web_fetch'}:
        return {'text': '离线网页样本，没有联网', 'sources': [{'url': 'https://example.com/offline'}]}
    if call['name'] in {'list_files', 'read_file', 'search_files', 'open_file'}:
        return real_tool(conversation, call, operation)
    return {'error': '该外部操作未在此离线样本执行', 'actions_executed': False}


service.chatbot.invoke_tool = tool
api._service = service

# 默认不缩短30秒生产限频；测试可通过已保存关键事件验证即刻首报与后续合并。
from agent.server.app import app
