"""Public LangGraph conversation references private, single-owner task checkpoints."""
from pathlib import Path
import uuid
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, messages_from_dict, messages_to_dict
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.model.client import create_model
from agent.model.errors import ModelRequestNotSent
from agent.settings import load_settings
from agent.storage.conversations import ConversationStore
from agent.task.budget import StepBudget
from agent.server.runtime import ConversationRunner
from agent.storage.preferences import Preferences, state_root
from agent.execution.assistant_files import authorized_roots

ACTIVE = {'needs_plan', 'executing'}
FINISHED = {'chat_complete', 'verified_success', 'task_blocked', 'settings_required'}


def finished(session, store):
    return session['status'] in FINISHED or (session['status'] == 'runtime_error'
                                            and store.error_is_settled(session['task_id']))


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


def model_factory(settings=None):
    model = create_model(load_settings(settings))
    # The UI subscribes to messages-tuple as well as public graph values.
    # Suppress automatic LangChain streaming of private tool arguments.
    model.client.tags = ['langsmith:nostream', 'langsmith:do-not-render']
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


def public_result(result):
    return {key: result.get(key) for key in
            ('status', 'patch_path', 'artifact_dir', 'planner_calls', 'fresh_verification', 'usage', 'existing_runs')}


def thread_id(config):
    return ConversationStore.thread(config.get('configurable', {}).get('thread_id'))


def intake(state: ConversationState, config: RunnableConfig):
    store, thread = ConversationStore(state_root()), thread_id(config)
    session = store.session(thread)
    public = state.get('messages') or messages_from_dict((session or {}).get('public_messages', []))
    latest = next((m for m in reversed(public) if isinstance(m, HumanMessage)), None)
    if latest is None:
        raise ValueError('No saved user input; an explicit message is required')
    if not latest.id:
        raise ValueError('User messages require a stable ID')
    if isinstance(latest.content, list) and any(block.get('type') != 'text' for block in latest.content):
        raise ValueError('Only text messages are supported; attachments were not submitted')
    selected = state.get('repo') or (session or {}).get('repo', '')
    repo = str(Path(selected).expanduser().resolve()) if selected else ''
    roots = authorized_roots(state.get('read_roots', (session or {}).get('read_roots', [])))
    presentation = {'messages': public, 'repo': repo, 'read_roots': roots}
    record = store.enqueue(thread, latest.id, 'message', latest.text)
    if session and record['applied']:
        can_resume = session['status'] in {'paused', 'awaiting_user'} and bool(store.pending(thread))
        return {**presentation, 'task_id': session['task_id'], 'status': 'needs_plan' if can_resume else session['status']}
    if session and not finished(session, store):
        if session['repo'] != repo:
            raise ValueError('Open a new conversation to change the active workspace')
        session.update(last_input=latest.id, read_roots=roots)
    else:
        store.acknowledge(thread, [record['id']])
        previous = (session or {}).get('activity_tasks', [])
        if session and not previous:
            previous = [{'id': session['task_id'], 'input_id': session['last_input']}]
        session = {'repo': repo, 'task_id': uuid.uuid4().hex, 'request': latest.text,
                   'history': messages_to_dict(public[:-1]), 'status': 'needs_plan',
                   'last_input': latest.id, 'public_messages': messages_to_dict(public)}
        session['runtime_settings'] = Preferences().defaults()
        session['read_roots'] = roots
        session['activity_tasks'] = previous + [{'id': session['task_id'], 'input_id': latest.id}]
    store.save_session(thread, session)
    return {**presentation, 'task_id': session['task_id'], 'status': 'needs_plan',
            'runtime_settings': session['runtime_settings'], 'dispatch': None, 'turn_status': None}


def step(state: ConversationState, config: RunnableConfig, phase='decide'):
    store, thread = ConversationStore(state_root()), thread_id(config)
    session = store.session(thread)
    if not session or session['task_id'] != state['task_id']:
        raise ValueError('Conversation/task association changed')
    writer = get_stream_writer()
    runner = ConversationRunner(LazyModel(session['runtime_settings']), session['repo'], phase=phase,
                        read_roots=session.get('read_roots', []),
                        state_dir=state_root()/'tasks'/session['task_id'], interactive=True,
                        event_sink=writer, runtime_settings=session['runtime_settings'],
                        control_source=lambda: store.pending(thread))
    result = runner.run(messages_from_dict(session['history']), session['request'], StepBudget(max_steps=1000))
    status = result['status']
    store.acknowledge(thread, runner.state.get('applied_inputs', []))
    updates = [(AIMessage if r['type'] == 'ai' else HumanMessage)(content=r['content'], id=r['id'])
               for r in runner.state.get('public_events', [])]
    queued = store.pending(thread)
    # A message racing the final checkpoint starts the next turn, not a
    # second owner or an automatic retry of the just-completed task.
    if finished({'status': status, 'task_id': session['task_id']}, store) and any(r['kind'] == 'message' for r in queued):
        updates.extend(HumanMessage(content=r['text'], id=r['id']) for r in queued if r['kind'] == 'message')
        status = 'new_input'
    elif status in {'paused', 'awaiting_user'} and queued:
        status = 'needs_plan'
    if status not in ACTIVE and status != 'chat_complete' and result.get('turn_status') != 'chat_complete':
        text = ('实际验收通过。' if status == 'verified_success' else
                '请先在设置中配置可用的 DeepSeek API Key，然后重新发送消息。' if status == 'settings_required' else
                '已在安全位置暂停，可以补充要求或继续。' if status in {'paused', 'awaiting_user'} else
                f'任务状态：{status}；尚未确认完成，已保留现场。')
        updates.append(AIMessage(content=text, id=session['task_id']+'-'+status))
    # Presentation recovery never restores execution cursors. Those remain
    # solely in the task checkpoint, including unknown-operation protection.
    session.update(status=result['status'], result=public_result(result),
                   workspace_bound=bool(runner.state.get('baseline')),
                   public_messages=messages_to_dict(add_messages(state.get('messages', []), updates)))
    store.save_session(thread, session)
    return {'messages': updates, 'status': status, 'result': public_result(result),
            'dispatch': result.get('dispatch'), 'turn_status': result.get('turn_status'),
            'runtime_settings': session['runtime_settings'], 'workspace_bound': session['workspace_bound'],
            'read_roots': session.get('read_roots', [])}


def route(state):
    if state['status'] == 'new_input':
        return 'intake'
    if state['status'] not in ACTIVE:
        return END
    if state.get('dispatch'):
        return {'chat': 'chat_reply', 'web': 'read_web', 'read': 'read_local', 'run': 'run_existing', 'work': 'work'}[state['dispatch']]
    return 'work' if state['status'] == 'executing' else 'decide'


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
