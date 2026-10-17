"""在请求或操作完成后的边界处理公开回复与用户引导。"""

from dataclasses import dataclass
import json
from langchain_core.messages import HumanMessage, ToolMessage, message_to_dict, messages_from_dict


@dataclass(frozen=True)
class TextReply:
    text: str


@dataclass(frozen=True)
class WebReply:
    calls: list


PROGRESS = {
    'model_request_started': ('model', '正在等待模型回复'),
    'model_response_received': ('model_finished', '模型回复已保存'),
    'plan_accepted': ('planning', '操作已校验，准备执行'),
    'step_started': ('executing', '正在执行操作'),
    'step_finished': ('operation_finished', '操作结果已保存'),
    'assistant_started': ('executing', '正在查询资料或运行已有任务'),
    'assistant_finished': ('operation_finished', '操作结果已保存'),
    'rag_started': ('executing', '正在检索项目证据'),
    'rag_completed': ('operation_finished', '检索证据已保存'),
    'acceptance_trial_started': ('acceptance', '正在原始源码上试跑验收'),
    'final_acceptance_started': ('acceptance', '正在运行实际验收'),
    'final_acceptance_finished': ('acceptance_finished', '实际验收结果已保存'),
    'plan_rejected': ('repair', '提交信息需要修正，正在处理'),
    'user_input_applied': ('updated', '新要求已接入，重新安排剩余工作'),
    'paused': ('paused', '已暂停'),
}


def publish(loop, response):
    content = messages_from_dict([response])[0].content
    if not isinstance(content, str) or not content.strip():
        return
    identity = f'{loop.root.name}-reply-{loop.budget.used_steps}'
    replies = loop.state.setdefault('public_replies', [])
    if not any(row['id'] == identity for row in replies):
        replies.append({'id': identity, 'content': content})
        loop.state.setdefault('public_events', []).append(
            {'id': identity, 'type': 'ai', 'content': content}
        )
        loop.persist('public_reply', {'id': identity})


def accept_text(loop, response):
    loop.state['history'].append(response)
    publish(loop, response)
    active = bool(loop.state['plan_count'] or loop.state.get('acceptance_draft'))
    loop.state.update(
        response_pending=None,
        status='awaiting_user' if active else 'chat_complete',
        reason='Text reply does not establish actual acceptance',
    )
    loop.persist('text_reply_saved')


def discard_unexecuted_response(loop):
    raw = loop.state.get('response_pending')
    if raw:
        response = messages_from_dict([raw])[0]
        loop.state['history'].append(raw)
        for call in [*response.tool_calls, *response.invalid_tool_calls]:
            loop.state['history'].append(
                message_to_dict(
                    ToolMessage(
                        content=json.dumps(
                            {'outcome': 'superseded_by_user', 'actions_executed': False}
                        ),
                        name=call['name'],
                        tool_call_id=call['id'],
                    )
                )
            )
        loop.state['response_pending'] = None


def apply_controls(loop):
    if loop.control_source is None:
        return False
    changed = False
    for control in loop.control_source():
        applied = loop.state.setdefault('applied_inputs', [])
        if control['id'] in applied:
            continue
        if control['kind'] == 'message':
            if loop.state.get('pending'):
                loop.finish_plan('superseded_by_user')
            discard_unexecuted_response(loop)
            loop.state.pop('acceptance_release', None)
            loop.state.pop('repair_draft', None)
            loop.state['plan_error'] = ''
            loop.state['history'].append(
                message_to_dict(HumanMessage(content=control['text'], id=control['id']))
            )
            loop.state.setdefault('user_updates', []).append(
                {'id': control['id'], 'text': control['text']}
            )
            loop.state['rag_request_id'] = control['id']
            loop.state.setdefault('public_events', []).append(
                {'id': control['id'], 'type': 'human', 'content': control['text']}
            )
            loop.state.update(status='needs_plan', reason='User supplemented the task')
            changed = True
        elif control['kind'] == 'pause':
            loop.state.update(status='paused', reason='User requested pause', pause_requested=True)
            changed = True
        elif control['kind'] == 'resume':
            loop.state.update(
                status='executing' if loop.state.get('pending') else 'needs_plan',
                reason='User resumed task',
                pause_requested=False,
            )
            changed = True
        applied.append(control['id'])
        loop.persist(
            'paused' if control['kind'] == 'pause' else 'user_input_applied', {'id': control['id']}
        )
    return changed
