"""人工续跑的共同契约：停止不是删除，授权追加而不是重置。"""

from copy import deepcopy

from langchain_core.messages import HumanMessage, message_to_dict

from agent.storage.records import digest


PROTOCOL = 2
ACTIVE = {'queued', 'running', 'needs_plan', 'executing', 'pausing'}
SUCCESS = {'verified_success', 'existing_complete'}
LIMITS = ('max_steps', 'max_tokens', 'timeout_seconds', 'max_service_cost_micros')


def explicit_continue(text):
    # 只解析完整的人类命令，不从引文、否定句或新需求中提取关键词。
    return isinstance(text, str) and text.strip().rstrip('。.!！').strip() in {
        '继续', '继续任务', '继续复查', 'continue', 'resume',
    }


def stopped(status):
    return bool(status) and status not in ACTIVE


def describe(status, stop_id, *, reason='', additions=None):
    return {
        'protocol': PROTOCOL, 'available': stopped(status), 'stop_id': stop_id,
        'mode': 'review' if status in SUCCESS else 'continue',
        'reason': reason, 'additions': additions or {},
        'requirements': [
            '核验任务原件、权限、环境和操作回执；缺失或损坏时先修复。',
            '保留未知用量；部分执行先调查，不重放已完成操作。',
        ],
    }


def additions(state, budget):
    original = state.get('budget_initial') or {key: getattr(budget, key) for key in LIMITS}
    limits = {}
    if budget.working_steps == 0:
        limits['max_steps'] = original['max_steps']
    occupied = budget.reported_tokens + budget.unknown_reserved_tokens + budget.inflight_reserved_tokens
    if budget.max_tokens is not None and (
        occupied >= budget.max_tokens or state.get('status') == 'token_budget_exhausted'
    ):
        limits['max_tokens'] = original['max_tokens']
    if budget.remaining_time() is not None and budget.remaining_time() <= 0:
        limits['timeout_seconds'] = original['timeout_seconds']
    occupied_cost = (budget.service_cost_micros + budget.unknown_service_cost_micros
                     + budget.inflight_service_cost_micros)
    if budget.max_service_cost_micros is not None and (
        occupied_cost >= budget.max_service_cost_micros
        or state.get('status') == 'service_budget_exhausted'
    ):
        # 零额度是明确禁止该服务，继续不能把它变成付费权限。
        if original['max_service_cost_micros']:
            limits['max_service_cost_micros'] = original['max_service_cost_micros']
    return limits


def grant_budget(state, budget, command_id):
    ledger = state.setdefault('budget_grants', [])
    if any(row['command_id'] == command_id for row in ledger):
        return
    state.setdefault('budget_initial', {key: getattr(budget, key) for key in LIMITS})
    changes = additions(state, budget)
    before = {key: getattr(budget, key) for key in changes}
    for key, amount in changes.items():
        setattr(budget, key, before[key] + amount)
    ledger.append({
        'command_id': command_id, 'stop_id': (state.get('continuation') or {}).get('stop_id'),
        'before': before, 'after': {key: getattr(budget, key) for key in changes},
    })


def save_stop(loop):
    state = loop.state
    if not stopped(state.get('status')):
        return
    old = state.get('continuation') or {}
    if old.get('available') and old.get('status') == state['status']:
        return
    token = digest([state['identity'], len(state.get('continuation_history', [])),
                    state['status'], loop.budget.request_sequence,
                    state.get('last_plan'), len(state.get('rounds', []))])
    value = describe(state['status'], token, reason=state.get('reason', ''),
                     additions=additions(state, loop.budget))
    value['status'] = state['status']
    state['continuation'] = value


def input_command(loop):
    source = getattr(loop, 'mailbox', None) or loop.control_source
    applied = set(loop.state.get('applied_inputs', []))
    return next((row for row in source() if row['kind'] == 'resume' and row['id'] not in applied),
                None) if source else None


def resume_validated(loop, command=None):
    """调用者已核验原件和环境；只改变调度入口，不跳过未决操作检查。"""
    state = loop.state
    if getattr(loop, 'current_cancel', False):
        return False
    command = command or input_command(loop)
    if command is None:
        recovery_id = getattr(loop, 'recovery_id', None)
        if recovery_id:
            command = {'id': recovery_id, 'kind': 'resume'}
    if (not command or command['id'] in state.get('rejected_recovery_ids', []) or any(row['command_id'] == command['id']
                          for row in state.get('continuation_history', []))):
        return False
    if any(state.get(name) for name in (
        'request_pending', 'running_step', 'verification_running', 'acceptance_running',
        'assistant_running', 'web_running', 'web_request_pending', 'rag_running', 'rag_request_pending',
    )) or (state.get('compaction_pending') or {}).get('stage') == 'committing':
        return False
    old = deepcopy(state.get('continuation') or {})
    if command.get('native_stop_id') and command['native_stop_id'] != old.get('stop_id'):
        raise ValueError('The task stop identity changed; refresh before continuing')
    previous = old.get('status', state['status'])
    record = {
        'command_id': command['id'], 'stop_id': old.get('stop_id'), 'status': previous,
        'reason': old.get('reason', state.get('reason', '')),
        'mode': 'review' if previous in SUCCESS else 'continue',
        'verification': deepcopy(state.get('verification', {})),
        'rounds': len(state.get('rounds', [])), 'elapsed_seconds': state.get('elapsed_seconds', 0),
        'cancelled': loop.budget.cancelled, 'cancel_reason': loop.budget.cancel_reason,
    }
    grant_budget(state, loop.budget, command['id'])
    loop.budget.cancelled, loop.budget.cancel_reason = False, ''
    if previous in SUCCESS:
        # 已结束计划不会重放。重复计划指纹继续保留，模型必须调查并重新规划。
        state.pop('dispatch', None)
        state.pop('turn_status', None)
        state.pop('repair_draft', None)
        state['plan_error'] = ''
        state.update(status='needs_plan', pending=None)
    else:
        state['status'] = 'executing' if state.get('pending') else 'needs_plan'
    state.setdefault('continuation_history', []).append(record)
    state['continuation'] = {**old, 'available': False, 'continued_by': command['id']}
    state.update(pause_requested=False, reason='User explicitly continued the original task')
    applied = state.setdefault('applied_inputs', [])
    if command['id'] not in applied:
        applied.append(command['id'])
    state['history'].append(message_to_dict(HumanMessage(
        content=(
            'The user explicitly requested a new review of this previously successful task. '
            if previous in SUCCESS else 'The user explicitly requested continuation of this task. '
        ) + 'Keep the original requirements, evidence, frozen acceptance and accumulated usage. '
            'Use the preserved investigation to plan further work. Do not replay completed '
            'modifications or side-effect commands. A model-reported obstacle is not proof of '
            'an external impossibility. Prior success does not establish this review result.',
        id='continue-' + command['id'],
    )))
    loop.persist('task_resumed', record)
    return True
