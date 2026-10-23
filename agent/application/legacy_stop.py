"""只读投影已绑定旧内核的验收额度字段；新内核不再产生或执行该暂停。"""

STATUS = 'acceptance_limit_reached'


def public(state):
    value = state.get('acceptance_limit')
    if not value:
        return None
    return {key: value.get(key) for key in (
        'active', 'completed_rounds', 'round_limit', 'round_increment',
        'pause_id', 'last_verification_status',
    )}
