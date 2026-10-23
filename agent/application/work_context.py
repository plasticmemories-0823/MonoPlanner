"""为聊天投影当前任务与按钮事实，不启动工作或改写原生状态。"""

from pathlib import Path

from agent.task.resumption import stopped


COMMAND_STATES = ('pending', 'running', 'applied', 'deferred', 'failed', 'unknown')
COMPLETED = {'verified_success', 'existing_complete'}


def activity_facts(store, task):
    """把派发状态与最近已提交事件翻译成用户能理解的阶段，不推测文件写入。"""
    page = store.work_events(task['conversation_id'], task['task_id'], 0, 6, latest=True)
    payloads = [row['payload'] for row in page['events']]
    markers = {
        'plan_rejected', 'plan_accepted', 'plan_finished', 'step_started', 'step_finished',
        'acceptance_trial_started', 'acceptance_trial_finished',
        'final_acceptance_preparing', 'final_acceptance_started', 'final_acceptance_finished',
    }
    event = next((row.get('event') for row in reversed(payloads)
                  if row.get('event') in markers), None)
    status = task['status']
    phase, message = {
        'queued': ('queued', '任务已排队，正在等待开始。'),
        'running': ('starting', '任务已启动，正在做启动准备。暂时不需要你操作。'),
        'needs_plan': ('planning', '正在分析项目并准备下一步，任务会继续推进。'),
        'executing': ('working', '正在处理项目中的实现和检查步骤。'),
        'pausing': ('stopping', '正在结束本次执行并确认运行资源已经停止。'),
        'verified_success': ('completed', '任务已完成，并通过本次验收。'),
        'existing_complete': ('completed', '这次已有程序的运行已结束。'),
    }.get(status, ('stopped', '任务已停止，具体原因需要结合已保存的记录核对。'))
    if status == 'needs_plan':
        if event == 'plan_rejected':
            phase, message = 'reworking', '当前处理方式需要调整，正在继续修正。'
        elif event in {'acceptance_trial_started', 'acceptance_trial_finished'}:
            phase, message = 'preparing_checks', '正在准备和试跑验收测试，任务会自行继续。'
    elif status == 'executing' and event in {
        'final_acceptance_preparing', 'final_acceptance_started', 'final_acceptance_finished',
    }:
        phase, message = 'verifying', '正在运行验收检查，确认实现是否符合要求。'
    writes = any(step.get('op') in {'write', 'patch'} and step.get('status') == 'succeeded'
                 for row in payloads for step in (row.get('details') or {}).get('steps', []))
    return {
        'phase': phase, 'message': message, 'file_changes_confirmed': writes,
        'evidence_cursor': page['cursor'],
        'meaning': 'This is the current recorded activity. Starting or planning is not file editing. '
                   'False file_changes_confirmed means these recent facts do not establish file changes; '
                   'it is not proof that the project has no files. Do not infer activity from the goal.',
    }


def task_facts(store, task):
    if task is None:
        return None
    from agent.application.legacy_stop import public

    value = {key: task.get(key) for key in (
        'task_id', 'kind', 'repo', 'project_path', 'status', 'reason', 'revision',
        'request', 'updated_at',
    )}
    status = task['status']
    value['activity'] = activity_facts(store, task)
    if status in COMPLETED:
        lifecycle = 'completed'
    elif status == 'pausing':
        lifecycle = 'stopping'
    elif stopped(status):
        lifecycle = 'stopped'
    else:
        lifecycle = 'working'
    value.update(lifecycle=lifecycle, needs_user_action=lifecycle == 'stopped',
                 acceptance_limit=public(task), diagnostic_record='diagnostics')
    value['meaning'] = {
        'needs_plan': 'Work is preparing its next internal plan. It is not waiting for human confirmation.',
        'verified_success': 'The coding task completed and passed its acceptance. Review is optional.',
        'existing_complete': 'This separate existing-command run ended. It does not replace a coding acceptance result.',
        'pausing': 'Owned execution is being stopped and checked; shutdown is not yet confirmed.',
    }.get(status, 'Use lifecycle and the saved events; do not infer a human action from an internal reason.')
    controls = store.commands(conversation=task['conversation_id'], task=task['task_id'],
                              states=COMMAND_STATES)
    value['pause_commands'] = [
        {'id': row['id'], 'state': row['state']} for row in controls if row['kind'] == 'pause'
    ][-20:]
    packet = task.get('confirmed_requirements')
    application_state = 'initial'
    for row in controls:
        updated = row.get('value', {}).get('confirmed_requirements')
        if updated and row['state'] in {'pending', 'running', 'applied', 'deferred'}:
            packet, application_state = updated, row['state']
    if packet:
        confirmation = packet['confirmation']
        actions = store.commands(conversation=task['conversation_id'], states=COMMAND_STATES)
        action = next((row for row in actions if row['id'] == confirmation['id']), None)
        value['confirmation'] = {
            'id': confirmation['id'], 'state': action['state'] if action else 'confirmed',
            'meaning': 'Human confirmation is already saved. Do not ask for the same confirmation again.',
        }
        confirmed = {
            'revision': packet['revision'], 'digest': packet['digest'],
            'confirmation_id': confirmation['id'], 'application_state': application_state,
            'source_ids': [row['id'] for row in packet['sources']],
        }
        if lifecycle in {'stopped', 'completed'}:
            confirmed['targets'] = [{key: target[key] for key in (
                'id', 'artifact_type', 'target_runtime', 'entry', 'behavior', 'input_scope',
            )} for target in packet['targets']]
        value['confirmed_requirements'] = confirmed
    return value


def context(store, conversation, *, task_id=None):
    selected = store.get(conversation)
    task = store.work(task_id if task_id is not None else selected['work_id'])
    if task_id is not None and (task is None or task['conversation_id'] != conversation):
        raise ValueError('The selected task does not belong to this conversation')
    coding = next((item for item in reversed(store.work_for_conversation(conversation))
                   if item['kind'] == 'coding'), None)
    actions = store.commands(conversation=conversation, states=COMMAND_STATES)
    buttons = [{key: row.get(key) for key in ('id', 'kind', 'state', 'task_id')}
               for row in actions if row['kind'] in {'confirm_work', 'pause', 'resume'}][-20:]
    return {
        'source': 'saved_work_context',
        'task': task_facts(store, task),
        'coding_task': task_facts(store, coding) if coding and (not task or coding['task_id'] != task['task_id']) else None,
        'button_actions': buttons,
    }


def completion_notice(task):
    """已提交完成状态使用固定简短通知，不等待额外模型解释或重跑验证。"""
    if task['status'] not in COMPLETED:
        return None
    project = Path(task['repo']).name
    path = task['repo'].replace('`', '\\`')
    if task['status'] == 'verified_success':
        return f'{project} 已完成，并通过本次验收。\n\n文件保存在：`{path}`。'
    return f'这次已有程序的运行已结束。项目目录：`{path}`。'
