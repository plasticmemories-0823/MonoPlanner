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

from langchain_core.messages import AIMessage, ToolMessage, message_to_dict

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
    def status(self, repo, *, verify=False):
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
        if messages[0].content.startswith('Give a brief first-person progress'):
            facts = json.loads(last.content) if not isinstance(last, ToolMessage) else {}
            task = facts.get('task') or {}
            return response('我刚完成一轮本地检查，结果已记录。' if task else '我正在核对刚收到的记录。')
        if isinstance(last, ToolMessage):
            return response('已完成查询：' + last.content[:400])
        text = str(last.content)
        if text == '帮我画鹈鹕骑自行车':
            return response('我会在所选目录下新建 pelican-bike，制作能直接打开的动画网页。是否按这个计划开始？')
        if text == '按这个计划开始' or '开始工作' in text or '准备失败工作' in text:
            original = next((row for row in messages if getattr(row, 'type', None) == 'human'
                             and row.content == '帮我画鹈鹕骑自行车'), last)
            return response('请确认保存的交付方案。', 'prepare_work', {
                'project_path': 'pelican-bike' if text == '按这个计划开始' else '.',
                'create': text == '按这个计划开始', 'request_message_ids': [original.id],
                'targets': [{'id': 'fixture', 'artifact_type': 'repository_entry',
                             'target_runtime': 'python', 'entry': 'demo.py', 'behavior': str(original.content),
                             'input_scope': 'All original requirements', 'imports': '', 'setup': '',
                             'expression': '', 'variables': [], 'required_values': [], 'argv': [],
                             'quotes': [{'source_id': original.id, 'text': str(original.content)}]}],
            })
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
    from agent.storage.history import LocalHistory
    from agent.storage.live_reasoning import LiveReasoning

    from agent.application.project_location import ensure_location

    task_root = state_root() / 'tasks' / job['task_id']
    task_root.mkdir(parents=True, exist_ok=True)
    ensure_location(job['project_location'], task_root)
    native_action({'kind': 'work_started', 'task': job['task_id']})
    native_action({'kind': 'prepare', 'repo': job['repo']})
    if '准备失败工作' in job['request']:
        return {'status': 'preparation_failed', 'applied_inputs': [],
                'stop_reason': '开工准备失败：离线模拟磁盘空间不足'}
    with _guard:
        signal = Event()
        _signals[job['task_id']] = signal

    # 使用真实只读活动接口的本地原件；命令只作为样本保存，绝不执行。
    history = LocalHistory(state_root() / 'tasks' / job['task_id'])
    with history.lease():
        previous = history.load() or {}
        call = previous.get('budget', {}).get('used_steps', 0) + 1
        state = {'history': [], 'budget': {'used_steps': call}}
        history.record(state, 'model_request_started', {'call': call})
    live = LiveReasoning(history.root, call, job['input_id'])
    live.append('工作思考第一段')
    applied = []
    try:
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
                if live.row['status'] == 'running':
                    live.finish('interrupted')
                return {'status': 'paused', 'applied_inputs': applied,
                        'stop_reason': '离线任务已暂停'}
            if index == 4:
                live.append('，已确认要运行检查')
                live.flush()
            if index == 8:
                live.finish('completed')
                plan_id = 'fixture-' + str(call)
                step = {'id': 'probe', 'op': 'shell',
                        'command': "printf 'offline-operation-marker'"}
                folder = history.root / 'plans' / plan_id / (plan_id + '-probe')
                folder.mkdir(parents=True)
                (folder / 'request.json').write_text(json.dumps(step), encoding='utf-8')
                state['pending'] = {'id': plan_id, 'plan': {'steps': [step]},
                                    'response': message_to_dict(response('')), 'rows': []}
                with history.lease():
                    history.record(state, 'step_started', {'id': 'probe'})
            if index == 12:
                with history.lease():
                    history.record(state, 'step_finished', {'id': 'probe', 'status': 'succeeded'})
            if index % 8 == 0:
                emit('work_state', {'event': 'plan_finished', 'status': 'executing',
                                    'reason': '离线检查阶段 ' + str(index // 8),
                                    'applied_inputs': applied,
                                    'usage': {'calls': 1, 'total_tokens': 120, 'input_tokens': 100,
                                              'output_tokens': 20, 'cached_input_tokens': 50,
                                              'cache_known_input_tokens': 100,
                                              'uncached_input_tokens': 50, 'unknown_calls': 0}})
        return {'status': 'verified_success', 'applied_inputs': applied, 'stop_reason': '离线验收通过'}
    finally:
        history.close()


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
                    work_factory=work, pause_callback=pause, supplement_callback=lambda *args: None)
real_tool = service.chatbot._invoke_tool


def tool(conversation, call, operation):
    if call['name'] in {'web_search', 'web_fetch'}:
        return {'text': '离线网页样本，没有联网', 'sources': [{'url': 'https://example.com/offline'}]}
    if call['name'] in {'list_files', 'read_file', 'retrieve_evidence', 'open_file'}:
        return real_tool(conversation, call, operation)
    return {'error': '该外部操作未在此离线样本执行', 'actions_executed': False}


service.chatbot.invoke_tool = tool
api._service = service

# 默认不缩短30秒生产限频；测试可通过已保存关键事件验证即刻首报与后续合并。
from agent.server.app import app
