"""公开 LangGraph 对话引用私有、由单一执行者管理的任务检查点。"""

from pathlib import Path
import logging
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.errors import GraphBubbleUp
from langgraph.types import Overwrite

from agent.model.client import create_model
from agent.model.errors import ModelRequestNotSent
from agent.settings import load_settings
from tests.legacy.conversations import ConversationStore
from agent.task.budget import StepBudget
from tests.legacy.runtime import ConversationRunner
from agent.storage.preferences import Preferences, state_root
from agent.execution.assistant_files import authorized_roots
from agent.storage.permissions import Permissions
from agent.storage.read_sources import ReadSources
from agent.storage.records import save
from agent.storage.usage import summarize
from agent.task import control as interrupts
from agent.execution.projects import workspace_binding

ACTIVE = {'needs_plan', 'executing'}
FINISHED = {'chat_complete', 'verified_success', 'task_blocked', 'settings_required'}


def finished(session, store):
    return session['status'] in FINISHED or (
        session['status'] == 'runtime_error'
        and (
            (
                session.get('initialization_failed')
                and not (store.path.parent / 'tasks' / session['task_id']).exists()
            )
            or store.error_is_settled(session['task_id'])
        )
    )


class ConversationState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    repo: str
    task_id: str
    status: str
    result: dict
    dispatch: str
    runtime_settings: dict
    turn_status: str
    workspace_bound: bool
    read_roots: list[str]
    read_sources: list[dict]


# ========================================================================
# 延迟模型初始化与私有流控制
# ========================================================================
def model_factory(settings=None):
    model = create_model(load_settings(settings))
    # 界面同时订阅 messages-tuple 和公开的图状态值。
    # 禁止 LangChain 自动将私有工具参数通过流公开。
    model.client.tags = list(
        dict.fromkeys([*(model.client.tags or []), 'nostream', 'langsmith:do-not-render'])
    )
    return model


class LazyModel:
    def __init__(self, settings=None):
        self.model = None
        self.settings = settings

    def bind_tools(self, tools):
        if self.model is None:
            try:
                self.model = model_factory(self.settings)
            except ValueError as exc:
                raise ModelRequestNotSent('settings_required', str(exc)) from exc
        return self.model.bind_tools(tools)


# ========================================================================
# 公开结果与会话标识
# ========================================================================
def public_result(result):
    return {
        key: result.get(key)
        for key in (
            'status',
            'patch_path',
            'artifact_dir',
            'planner_calls',
            'fresh_verification',
            'usage',
            'existing_runs',
            'output_limit',
        )
    }


def thread_id(config):
    return ConversationStore.thread(config.get('configurable', {}).get('thread_id'))


# ========================================================================
# 输入接收与会话归档
# ========================================================================
def intake(state: ConversationState, config: RunnableConfig):
    store, thread = ConversationStore(state_root()), thread_id(config)
    session = store.session(thread)
    new_conversation = session is None

    # ========================================================================
    # 恢复公开历史并校验本次用户输入
    # ========================================================================
    saved = messages_from_dict((session or {}).get('public_messages', []))
    incoming = state.get('messages') or []
    public = add_messages(saved, incoming)
    latest = next((m for m in reversed(incoming or public) if isinstance(m, HumanMessage)), None)
    if latest is None:
        raise ValueError('No saved user input; an explicit message is required')
    if not latest.id:
        raise ValueError('User messages require a stable ID')
    if isinstance(latest.content, list) and any(
        block.get('type') != 'text' for block in latest.content
    ):
        raise ValueError('Only text messages are supported; attachments were not submitted')

    # ========================================================================
    # 解析工作目录与用户授权的只读资料
    # ========================================================================
    selected = state.get('repo') or (session or {}).get('repo', '')
    repo = str(Path(selected).expanduser().resolve()) if selected else ''
    roots = authorized_roots(state.get('read_roots', (session or {}).get('read_roots', [])))
    sources = ReadSources(state_root()).get(thread, roots)
    roots = [s['path'] for s in sources if s['kind'] == 'directory']
    # 新图可能只包含本次输入；若直接追加恢复列表，
    # 归并器会把新输入排在旧消息之前。
    presentation = {
        'messages': Overwrite(public),
        'repo': repo,
        'read_roots': roots,
        'read_sources': sources,
    }

    # ========================================================================
    # 按消息 ID 入队，复用活动任务或创建新任务
    # ========================================================================
    record = store.enqueue(thread, latest.id, 'message', latest.text)
    if session and record['applied']:
        can_resume = session['status'] in {
            'paused',
            'awaiting_user',
            'output_limit_reached',
        } and bool(store.pending(thread))
        return {
            **presentation,
            'task_id': session['task_id'],
            'status': 'needs_plan' if can_resume else session['status'],
        }
    if session and not finished(session, store):
        if session['repo'] != repo:
            raise ValueError('Open a new conversation to change the active workspace')
        session.update(last_input=latest.id, read_roots=roots, read_sources=sources)
    else:
        store.acknowledge(thread, [record['id']])
        previous = (session or {}).get('activity_tasks', [])
        if session and not previous:
            previous = [{'id': session['task_id'], 'input_id': session['last_input']}]
        session = {
            'repo': repo,
            'task_id': uuid.uuid4().hex,
            'request': latest.text,
            'history': messages_to_dict(public[:-1]),
            'status': 'needs_plan',
            'last_input': latest.id,
            'public_messages': messages_to_dict(public),
        }
        session['runtime_settings'] = Preferences().defaults()
        session['execution_binding'] = workspace_binding(repo)
        session['read_roots'] = roots
        session['read_sources'] = sources
        session['activity_tasks'] = previous + [{'id': session['task_id'], 'input_id': latest.id}]

    # ========================================================================
    # 保存会话，并为新会话预留后台标题任务
    # ========================================================================
    from tests.legacy.server_titles import reservation, schedule

    first_message = next((m for m in public if isinstance(m, HumanMessage)), latest)
    title_created = store.save_session(
        thread,
        session,
        title_request=(
            reservation(first_message.id, first_message.text) if new_conversation else None
        ),
    )
    if title_created:
        try:
            schedule(store.path.parent, thread)
        except Exception:
            logging.getLogger(__name__).exception(
                'Title scheduling failed; main conversation continues'
            )
    return {
        **presentation,
        'task_id': session['task_id'],
        'status': 'needs_plan',
        'runtime_settings': session['runtime_settings'],
        'dispatch': None,
        'turn_status': None,
    }


# ========================================================================
# 单步任务执行与公开状态回写
# ========================================================================
def step(state: ConversationState, config: RunnableConfig, phase='decide'):
    store, thread = ConversationStore(state_root()), thread_id(config)
    session = store.session(thread)
    if not session or session['task_id'] != state['task_id']:
        raise ValueError('Conversation/task association changed')

    # ========================================================================
    # 在当前权限边界内创建执行器并运行一个阶段
    # ========================================================================
    writer = get_stream_writer()
    task_dir = state_root() / 'tasks' / session['task_id']
    runner = None
    try:
        permission = Permissions(state_root()).boundary(thread)
        session['read_sources'] = ReadSources(state_root()).get(
            thread, session.get('read_roots', [])
        )
        session['read_roots'] = [
            s['path'] for s in session['read_sources'] if s['kind'] == 'directory'
        ]
        runner = ConversationRunner(
            LazyModel(session['runtime_settings']),
            session['repo'],
            phase=phase,
            execution_binding=session.get('execution_binding', {'backend': 'docker'}),
            read_roots=session.get('read_roots', []),
            read_sources=session['read_sources'],
            access_mode=permission['effective'],
            conversation_id=thread,
            state_dir=task_dir,
            interactive=True,
            yield_boundary=phase != 'work',
            event_sink=writer,
            runtime_settings=session['runtime_settings'],
            control_source=lambda: store.pending(thread),
        )
        with interrupts.active(state_root(), thread):
            result = runner.run(
                messages_from_dict(session['history']),
                session['request'],
                StepBudget(max_steps=1000),
            )
    except GraphBubbleUp:
        raise  # LangGraph 中断必须保留自身的暂停与检查点语义。
    except Exception as exc:
        logging.getLogger(__name__).exception('Conversation task failed before returning a result')
        runner_state = runner.state if runner else {}
        # 只有尚未创建任何任务归档的失败才能确定已结束。
        # 现有检查点和结果不明的操作仍需保护。
        session['initialization_failed'] = not runner_state and not task_dir.exists()
        save(
            state_root() / 'errors' / f"{session['task_id']}.json",
            {
                'task_id': session['task_id'],
                'phase': phase,
                'error_type': type(exc).__name__,
                'initialization_failed': session['initialization_failed'],
            },
        )
        continuation_failed = bool(runner_state.get('output_limit', {}).get('active'))
        result = {
            'status': 'state_conflict' if continuation_failed else 'runtime_error',
            'artifact_dir': str(task_dir),
            'fresh_verification': False,
            'usage': summarize(runner_state),
        }
    runner_state = runner.state if runner else {}

    # 确认已应用输入，并收集任务产生的公开消息
    status = result['status']
    store.acknowledge(thread, runner_state.get('applied_inputs', []))
    updates = [
        (AIMessage if r['type'] == 'ai' else HumanMessage)(content=r['content'], id=r['id'])
        for r in runner_state.get('public_events', [])
    ]

    # ========================================================================
    # 合并执行结束时到达的输入，并决定下一轮状态
    # ========================================================================
    queued = store.pending(thread)
    # 与最终检查点同时到达的消息进入下一轮，不能创建第二个执行者，
    # 也不能自动重试刚刚完成的任务。
    if finished({**session, 'status': status}, store) and any(
        r['kind'] == 'message' for r in queued
    ):
        updates.extend(
            HumanMessage(content=r['text'], id=r['id']) for r in queued if r['kind'] == 'message'
        )
        status = 'new_input'
    elif status in {'paused', 'awaiting_user'} and queued and not result.get('pause_requested'):
        status = 'needs_plan'

    # ========================================================================
    # 根据停止原因补充公开结果提示
    # ========================================================================
    if (
        status not in ACTIVE
        and status != 'chat_complete'
        and result.get('turn_status') != 'chat_complete'
    ):
        text = (
            '实际验收通过。'
            if status == 'verified_success'
            else (
                '操作等待你的批准，可以在操作卡中批准或拒绝。'
                if status == 'awaiting_user'
                and 'awaiting_approval' in runner_state.get('desktop_approvals', {}).values()
                else (
                    '任务未能启动，已保存错误记录。请检查运行环境后重新提交。'
                    if session.get('initialization_failed')
                    else (
                        '请先在设置中配置可用的 DeepSeek API Key，然后重新发送消息。'
                        if status == 'settings_required'
                        else (
                            runner_state['reason']
                            if status == 'awaiting_user'
                            and (
                                'Docker' in runner_state.get('reason', '')
                                or runner_state.get('environment_preparation', {}).get('status')
                                == 'failed'
                            )
                            else (
                                '已暂停，可以补充要求或继续。'
                                if status in {'paused', 'awaiting_user'}
                                else '任务尚未完成，已保留运行记录，供后续检查。'
                            )
                        )
                    )
                )
            )
        )
        if status == 'state_conflict' and runner_state.get('output_limit', {}).get('active'):
            text = '无法继续：原任务的源码、验收记录或容器状态未通过核验。没有重跑已完成的操作；请先核对本地记录。'
        if status == 'output_limit_reached':
            limit = runner_state.get('output_limit', {})
            used = limit.get('output_tokens')
            usage_text = f'本次已输出 {used:,} Token（含思考）。' if isinstance(used, int) else ''
            text = (
                '本次回复达到输出上限，尚未提交完整操作。'
                + usage_text
                + '已保留进度和记录；可点击“继续任务”，或先询问停止原因。'
            )
        identity = session['task_id'] + '-' + status
        if status == 'output_limit_reached':
            identity += '-' + str(runner_state.get('output_limit', {}).get('call', 'unknown'))
        updates.append(AIMessage(content=text, id=identity))
    # 展示恢复不恢复执行游标；执行游标及未知操作保护
    # 仅保存在任务检查点中。
    session.update(
        status=result['status'],
        result=public_result(result),
        execution_binding=runner_state.get('execution_binding', session.get('execution_binding')),
        workspace_bound=bool(runner_state.get('baseline') or runner_state.get('environment_key')),
        public_messages=messages_to_dict(add_messages(state.get('messages', []), updates)),
    )
    store.save_session(thread, session)
    return {
        'messages': updates,
        'status': status,
        'result': public_result(result),
        'dispatch': result.get('dispatch'),
        'turn_status': result.get('turn_status'),
        'runtime_settings': session['runtime_settings'],
        'workspace_bound': session['workspace_bound'],
        'read_roots': session.get('read_roots', []),
        'read_sources': session.get('read_sources', []),
    }


# ========================================================================
# 后续节点分派
# ========================================================================
def route(state):
    if state['status'] == 'new_input':
        return 'intake'
    if state['status'] not in ACTIVE:
        return END
    if state.get('dispatch'):
        return {
            'chat': 'chat_reply',
            'web': 'read_web',
            'read': 'read_local',
            'run': 'run_existing',
            'work': 'work',
        }[state['dispatch']]
    return 'work' if state['status'] == 'executing' else 'decide'


# ========================================================================
# 图节点与条件边连接
# ========================================================================
builder = StateGraph(ConversationState)
builder.add_node('intake', intake).add_node('decide', step)
builder.add_node('chat_reply', lambda state, config: step(state, config, 'chat'))
builder.add_node('read_web', lambda state, config: step(state, config, 'web'))
builder.add_node('read_local', lambda state, config: step(state, config, 'assistant'))
builder.add_node('run_existing', lambda state, config: step(state, config, 'assistant'))
builder.add_node('work', lambda state, config: step(state, config, 'work'))
builder.add_edge(START, 'intake').add_conditional_edges('intake', route)
for name in ('decide', 'chat_reply', 'read_web', 'read_local', 'run_existing', 'work'):
    builder.add_conditional_edges(name, route)
graph = builder.compile()
