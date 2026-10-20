"""验收额度用尽只暂停；人工批准按批增加额度，不重置历史或其他预算。"""

from copy import deepcopy
import sqlite3

from langchain_core.messages import HumanMessage, message_to_dict

from agent.storage.records import digest


STATUS = 'acceptance_limit_reached'
LEGACY_REASON = 'Final candidate attempt limit reached'
# 只兼容已审查的 main a6c3ca215d72b6335e5301a04616e9b627acf81c。
# 不把任意旧源码或结果不明的检查点变为可执行状态。
LEGACY_CODE_KEY = 'a4b590cf1f08116c36c959d2ce102ed9a9cd621307f62f6c6642f806ffd42e27'
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


def is_legacy_exhausted(state):
    result = state.get('last_result') or {}
    return bool(
        state.get('code_key') == LEGACY_CODE_KEY
        and state.get('status') == 'stalled'
        and state.get('reason') == LEGACY_REASON
        and result.get('outcome') == 'candidate_limit_reached'
        and result.get('final_acceptance') == {'status': 'not_run', 'reason': LEGACY_REASON}
        and state.get('rounds') and not state.get('pending') and not unsafe(state)
        and not (state.get('budget') or {}).get('cancelled')
    )


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


def legacy_projection(directory):
    """应用只读核对旧原件并展示继续入口；实际恢复仍须完整内核校验。"""
    from pathlib import Path
    from agent.source.snapshots import _safe_file
    from agent.storage.history import LocalHistory

    directory = Path(directory)
    if not directory.is_dir() or (directory / 'hard-interruption.json').exists():
        return None
    archive = None
    try:
        database = _safe_file(directory, Path('history.sqlite3'))
        archive = LocalHistory(directory)
        archive.connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
        state = archive.load()
        if not state or not is_legacy_exhausted(state):
            return None
        rounds = len(state['rounds'])
        return _value(state, {'round_limit': rounds, 'round_increment': rounds})
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
        return None
    finally:
        if archive is not None:
            archive.close()


def release_unexecuted_legacy_plan(loop):
    """旧版已拒绝且一步未执行的候选可重新提交，不误触相同计划防空转保护。"""
    import json
    from pathlib import Path
    from agent.source.snapshots import _safe_file

    result = loop.state['last_result']
    if result.get('steps'):
        return
    path = Path('plans') / result['plan'] / 'plan.json'
    original = json.loads(_safe_file(loop.root, path).read_text('utf-8'))
    if original['id'] != result['plan'] or original['plan']['kind'] != 'candidate':
        raise ValueError('Legacy exhausted candidate identity changed')
    fingerprint = digest([original['plan'], original['base_source']])
    loop.state['seen_plans'] = [
        item for item in loop.state['seen_plans'] if item != fingerprint
    ]


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
