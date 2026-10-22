"""验收额度用尽只暂停；人工批准按批增加额度，不重置历史或其他预算。"""

from copy import deepcopy

from langchain_core.messages import HumanMessage, message_to_dict

from agent.storage.records import digest


STATUS = 'acceptance_limit_reached'
UNKNOWN = (
    'hard_interruption', 'request_pending', 'running_step', 'verification_running',
    'acceptance_running', 'assistant_running', 'web_running', 'web_request_pending',
    'rag_running', 'rag_request_pending',
)


def active(state):
    return bool((state.get('acceptance_limit') or {}).get('active'))


def public(state):
    value = state.get('acceptance_limit')
    if not value:
        return None
    return {key: value.get(key) for key in (
        'active', 'completed_rounds', 'round_limit', 'round_increment',
        'pause_id', 'last_verification_status',
    )}


def unsafe(state):
    return state.get('status') in {
        'model_request_unknown', 'execution_unknown', 'web_request_unknown',
        'state_conflict', 'compaction_commit_unknown',
    } or any(state.get(key) for key in UNKNOWN) or (
        state.get('compaction_pending') or {}
    ).get('stage') == 'committing'




def _value(state, budget):
    return {
        'active': True,
        'completed_rounds': len(state['rounds']),
        'round_limit': budget['round_limit'],
        'round_increment': budget['round_increment'],
        'pause_id': digest([
            state['identity'], budget['round_limit'], len(state['rounds']),
            (state.get('pending') or {}).get('id'), state.get('last_plan'),
        ]),
        'last_verification_status': (state.get('verification') or {}).get('status', 'not_run'),
    }




def budget(loop):
    value = loop.state.setdefault('acceptance_budget', {
        'round_increment': loop.max_rounds, 'round_limit': loop.max_rounds,
    })
    if any(type(value.get(key)) is not int or value[key] <= 0
           for key in ('round_increment', 'round_limit')):
        raise ValueError('Invalid saved acceptance round budget')
    return value


def exhausted(loop):
    return len(loop.state['rounds']) >= budget(loop)['round_limit']


def pause(loop, *, persist=True):
    """在已知操作边界保存待判断状态，未执行计划及执行游标原样保留。"""
    if unsafe(loop.state):
        raise ValueError('Unconfirmed work cannot become an acceptance-budget pause')
    value = loop.state.get('acceptance_limit')
    if not active(loop.state):
        value = _value(loop.state, budget(loop))
        loop.state['acceptance_limit'] = value
    loop.state.update(
        status=STATUS,
        reason=(f"本批验收额度已用完，累计完成 {value['completed_rounds']} 轮，"
                f"尚未通过验收。已保存断点；是否增加 {value['round_increment']} 轮并继续？"),
    )
    if persist:
        loop.persist(STATUS, public(loop.state))


def resume_input(loop):
    source = getattr(loop, 'mailbox', None) or loop.control_source
    if source is None:
        return None
    applied = set(loop.state.get('applied_inputs', []))
    return next((row for row in source()
                 if row['kind'] == 'resume' and row['id'] not in applied), None)


def resume_validated(loop):
    """只在原件、源码、冻结验收和环境核验后，以一个人工命令批准一批额度。"""
    if not active(loop.state):
        return False
    if loop.state.get('status') not in {STATUS, 'paused', 'awaiting_user'}:
        return False
    control = resume_input(loop)
    if not control:
        return False
    if unsafe(loop.state) or loop.budget.cancelled:
        raise ValueError('Unconfirmed or cancelled work cannot continue')
    value = loop.state['acceptance_limit']
    token = control.get('acceptance_pause_id')
    if token != value['pause_id']:
        # 旧页面或延迟到达的命令只能被消费，不能授权后来的一次暂停。
        loop.state.setdefault('applied_inputs', []).append(control['id'])
        loop.persist('acceptance_continue_stale', {'id': control['id']})
        return False
    previous = deepcopy(value)
    ledger = budget(loop)
    ledger['round_limit'] = max(ledger['round_limit'], len(loop.state['rounds'])) + ledger[
        'round_increment'
    ]
    value['active'] = False
    previous.update(continued_by=control['id'], new_round_limit=ledger['round_limit'])
    loop.state.setdefault('acceptance_limit_history', []).append(previous)
    loop.state['history'].append(message_to_dict(HumanMessage(
        content=(f"The user explicitly approved {ledger['round_increment']} additional final "
                 f"acceptance rounds (cumulative limit {ledger['round_limit']}). Continue from "
                 "the saved checkpoint. Do not repeat completed operations or replace frozen "
                 "acceptance. Other task budgets and safety constraints are unchanged."),
        id='acceptance-continue-' + control['id'],
    )))
    # 同一暂停上已排队的重复点击合并；后来的暂停必须有新的明确授权。
    source = getattr(loop, 'mailbox', None) or loop.control_source
    applied = loop.state.setdefault('applied_inputs', [])
    for row in source():
        if (row['kind'] == 'resume' and row['id'] not in applied
                and row.get('acceptance_pause_id') in {None, previous['pause_id']}):
            applied.append(row['id'])
    loop.state.update(
        status='executing' if loop.state.get('pending') else 'needs_plan',
        reason='用户已批准增加验收额度，继续原任务。', pause_requested=False,
    )
    loop.persist('acceptance_limit_continued', {
        'id': control['id'], 'pause_id': previous['pause_id'],
        'round_limit': ledger['round_limit'], 'round_increment': ledger['round_increment'],
    })
    return True
