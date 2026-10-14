"""输出触顶是一次已知的停止；用户批准后才能发起下一次请求。"""

import json
from copy import deepcopy

from langchain_core.messages import AIMessage, HumanMessage, message_to_dict

from agent.model.errors import ModelRequestNotSent
from agent.storage.records import digest

STATUS = 'output_limit_reached'
UNKNOWN = (
    'request_pending',
    'running_step',
    'verification_running',
    'acceptance_running',
    'assistant_running',
    'web_running',
    'web_request_pending',
)


def active(state):
    return bool(state.get('output_limit', {}).get('active'))


def public(state):
    value = state.get('output_limit', {})
    if not value:
        return None
    return {
        key: value.get(key)
        for key in ('call', 'max_tokens', 'output_tokens', 'reasoning_tokens', 'active')
    }


def guard(loop, response):
    data = response['data']
    finish = data.get('response_metadata', {}).get('finish_reason')
    if finish != 'length':
        if finish in {'content_filter', 'aborted', 'insufficient_system_resource'}:
            raise ModelRequestNotSent(
                'incomplete_model_response',
                'Incomplete reply retained; no operations executed or automatic retry.',
            )
        return
    usage = data.get('usage_metadata') or {}
    last = (loop.state.get('model_usage') or [{}])[-1]
    if not active(loop.state) or loop.state['output_limit'].get('response_sha256') != digest(
        response
    ):
        previous = deepcopy(loop.state.get('output_limit')) if active(loop.state) else None
        turn = loop.state.get('conversation_turn')
        value = {
            'active': True,
            'call': last.get('call', loop.budget.used_steps),
            'max_tokens': last.get('max_output_tokens'),
            'output_tokens': usage.get('output_tokens'),
            'reasoning_tokens': usage.get('output_token_details', {}).get('reasoning'),
            'response_sha256': digest(response),
            'response': response,
            'conversation': deepcopy(turn),
            'previous_limit': previous,
        }
        if turn:
            loop.complete_turn()
            loop.state['conversation_turn'] = None
        loop.just_reached_output_limit = True
        loop.state['output_limit'] = value
        loop.state.update(
            status=STATUS, reason='本次回复达到输出上限，尚未提交完整操作。', dispatch=None
        )
        loop.persist('output_limit_reached', public(loop.state))
    raise ModelRequestNotSent(STATUS, '本次回复达到输出上限，尚未提交完整操作。')


def resume_input(loop):
    source = getattr(loop, 'mailbox', None) or loop.control_source
    if source is None:
        return None
    applied = set(loop.state.get('applied_inputs', []))
    return next(
        (row for row in source() if row['kind'] == 'resume' and row['id'] not in applied), None
    )


def resume_validated(loop):
    """只在历史、源码、验收和环境身份核验之后调用。"""
    if not active(loop.state):
        return False
    control = resume_input(loop)
    if not control:
        return False
    value = loop.state['output_limit']
    if (
        any(loop.state.get(key) for key in UNKNOWN)
        or (loop.state.get('compaction_pending') or {}).get('stage') == 'committing'
    ):
        raise ValueError('仍有请求或操作结果未确认，不能继续。')
    raw = value['response']
    saved = json.loads(
        (loop.root / 'requests' / f"call-{value['call']:04d}" / 'response.json').read_text('utf-8')
    )
    if digest(raw) != value['response_sha256'] or digest(saved) != value['response_sha256']:
        raise ValueError('输出触顶记录已变化，不能继续。')
    if loop.environment is not None and hasattr(loop.environment, 'restart_after_output_limit'):
        receipt = value.get('container_stop')
        if not receipt:
            raise ValueError('缺少容器正常停止记录，不能自动启动。')
        retrying = bool(value.get('restart_requested'))
        value['restart_requested'] = control['id']
        loop.persist('output_limit_restart_requested', {'id': control['id']})
        loop.environment.restart_after_output_limit(receipt, allow_running=retrying)
        loop.environment_started = True

    # 原始工具片段留在响应原件中；模型历史只追加可用文字和明确的未执行说明。
    data = raw['data']
    content = data.get('content')
    reasoning = data.get('additional_kwargs', {}).get('reasoning_content')
    projection = message_to_dict(
        AIMessage(
            content=content if isinstance(content, str) else '',
            additional_kwargs=(
                {'reasoning_content': reasoning} if isinstance(reasoning, str) else {}
            ),
        )
    )
    notice = message_to_dict(
        HumanMessage(
            content='The previous reply reached its output-token limit. Its text and reasoning are retained as incomplete context. '
            'No tool calls from that reply were executed; discarded tool fragments remain in the original response record. '
            'The user explicitly requested continuation. Continue after the last completed operation; do not replay completed work.',
            id='output-limit-continue-' + control['id'],
        )
    )
    turn = value.get('conversation')
    if turn:
        turn = deepcopy(turn)
        turn['messages'].extend([projection, notice])
        turn.pop('response', None)
        turn.pop('stage', None)
        loop.state['conversation_turn'] = turn
        loop.state['response_pending'] = turn.get('work_response')
        loop.just_reached_output_limit = True
        loop.state['output_limit'] = value.get('previous_limit') or {}
    else:
        loop.state['history'].extend([projection, notice])
        loop.state['response_pending'] = None
        value['active'] = False
    value['continued_by'] = control['id']
    loop.state.setdefault('output_limit_history', []).append(value)
    source = getattr(loop, 'mailbox', None) or loop.control_source
    processed = []
    for row in source():
        if row['kind'] in {'pause', 'resume'}:
            processed.append(row['id'])
        if row['id'] == control['id']:
            break
    applied = loop.state.setdefault('applied_inputs', [])
    applied.extend(identity for identity in processed if identity not in applied)
    from agent.task.control import current

    signal = current()
    if signal:
        for identity in processed:
            signal.consume_input(identity)
    loop.state.update(status='needs_plan', reason='用户已请求继续', pause_requested=False)
    loop.state.pop('dispatch', None)
    loop.state.pop('turn_status', None)
    loop.persist('output_limit_continued', {'id': control['id'], 'call': value['call']})
    return True
