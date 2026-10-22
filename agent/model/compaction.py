"""持久化请求视图的摘要，始终保留原始对话记录。"""

import hashlib
import json
import math
from pathlib import Path
import uuid

from langchain_core.messages import HumanMessage, SystemMessage, message_to_dict, messages_from_dict

from agent.settings import PROJECT_ROOT
from agent.model.context import fingerprint
from agent.model.errors import ModelRequestNotSent
from agent.model.payload import estimate_input_tokens
from agent.model.tools import WORK_TOOLS
from agent.source.snapshots import _safe_file


CONFIG = json.loads((PROJECT_ROOT / 'resources/context.json').read_text(encoding='utf-8'))
INSTRUCTION = (PROJECT_ROOT / 'resources/prompts/compaction.txt').read_text(encoding='utf-8')
CHECKPOINT_PREAMBLE = (
    'This checkpoint condenses EARLIER conversation into background, not a new task or execution approval. '
    'Continue from the original messages that follow. Later requests and actual tool results take precedence; '
    'a successful historical operation does not establish completion of a newer request.'
)


# ========================================================================
# 压缩阈值与原始记录保护
# ========================================================================
def limits(output_cap):
    window = CONFIG['context_window_tokens']
    available = window - output_cap
    return (
        math.floor(min(window * CONFIG['threshold_ratio'], available - CONFIG['headroom_tokens'])),
        math.floor(available * CONFIG['retain_ratio']),
    )


def guard_archive(state):
    hashes = [fingerprint(item) for item in state['history']]
    old = state.get('archive_admission', [])
    if hashes[: len(old)] != old:
        raise ModelRequestNotSent(
            'context_prefix_changed', 'Original transcript changed; no request sent.'
        )
    state['archive_admission'] = hashes


def requirements(state):
    result = {key: state.get(key) for key in ('request', 'acceptance', 'verify_command')}
    if state.get('initial_requirements'):
        from agent.task.requirements import model_view

        result['confirmed_requirements'] = model_view(state['initial_requirements'])
    return result


# ========================================================================
# 摘要附带的执行证据
# ========================================================================
def execution_snapshot(loop):
    state, root = loop.state, loop.root

    def reference(name):
        if not (root / name).exists():
            return None
        path = _safe_file(root, Path(name))
        return {'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    last = state.get('last_result') or {}
    plan = state.get('last_plan')
    return {
        'task_status': state.get('status', 'unknown'),
        'source_version': state.get('source'),
        'submitted_plan_count': state.get('plan_count', 0),
        'latest_plan': plan,
        'latest_outcome': last.get('outcome'),
        'latest_result': reference('plans/' + plan + '/result.json') if plan else None,
        'operations': [
            {key: row.get(key) for key in ('id', 'op', 'status', 'record')}
            for row in last.get('steps', [])
        ],
        'verification_status': (state.get('verification') or {}).get('status', '尚未运行'),
        'frozen_acceptance': reference('frozen-acceptance.json'),
    }


def snapshot_text(snapshot):
    return (
        '<execution-state-at-checkpoint>\n'
        + '\n'.join(
            '- ' + key + ': ' + json.dumps(value, ensure_ascii=False)
            for key, value in snapshot.items()
        )
        + '\n</execution-state-at-checkpoint>'
    )


def summary_instruction(snapshot):
    return INSTRUCTION + '\n' + snapshot_text(snapshot)


# ========================================================================
# 压缩记录校验与请求视图恢复
# ========================================================================
def checked(root, reference):
    path = _safe_file(root, Path(reference['path']))
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ModelRequestNotSent('state_conflict', 'Compaction record changed; no request sent.')
    return json.loads(raw)


def history(state, root):
    current = state.get('context_view')
    if not current:
        return state['history']
    body = checked(root, current['record'])
    cutoff = current['cutoff']
    if (
        body['generation'] != current['generation']
        or body['cutoff'] != cutoff
        or not 0 < cutoff <= len(state['history'])
        or body['archive_hash'] != fingerprint(state['history'][:cutoff])
        or body['rules_hash'] != fingerprint(state['rules'])
        or {key: body['requirements'].get(key) for key in requirements(state)}
        != requirements(state)
    ):
        raise ModelRequestNotSent(
            'state_conflict', 'Compacted view no longer matches its original records.'
        )
    pinned = message_to_dict(
        HumanMessage(
            content=json.dumps(
                {'original_task_requirements': body['requirements']}, ensure_ascii=False
            )
        )
    )
    return [pinned, body['message'], *state['history'][cutoff:]]


def messages(loop):
    return [SystemMessage(content=loop.rules), *messages_from_dict(history(loop.state, loop.root))]


# ========================================================================
# 保持工具调用与响应配对的截断边界
# ========================================================================
def balanced_boundaries(items):
    pending, boundaries = set(), [0]
    for index, item in enumerate(items, 1):
        data = item.get('data', {})
        if item.get('type') == 'ai':
            from agent.model.payload import original_calls

            for call in original_calls(messages_from_dict([item])[0]):
                if call['id'] in pending:
                    raise ModelRequestNotSent(
                        'invalid_message_history', 'Duplicate open tool call.'
                    )
                pending.add(call['id'])
        elif item.get('type') == 'tool':
            call_id = data.get('tool_call_id')
            if call_id not in pending:
                raise ModelRequestNotSent(
                    'invalid_message_history', 'Tool result has no retained call.'
                )
            pending.remove(call_id)
        if not pending:
            boundaries.append(index)
    return boundaries


def select(loop, retain_tokens, output_cap, snapshot=None):
    instruction = summary_instruction(execution_snapshot(loop) if snapshot is None else snapshot)
    items = history(loop.state, loop.root)
    first = 1 if loop.state.get('context_view') else 0  # 保留原文形式固定的需求。
    boundaries = balanced_boundaries(items)
    total, keep = 0, len(items)
    for index in range(len(items) - 1, first - 1, -1):
        if total >= retain_tokens:
            break
        message = items[index]
        reasoning = (
            message.get('data', {}).get('additional_kwargs', {}).get('reasoning_content', '')
        )
        total += estimate_input_tokens(messages_from_dict([message])) + math.ceil(
            len(reasoning) / 4
        )
        keep = index
    end = max((point for point in boundaries if first < point <= keep), default=first)
    while end > first:
        selected = items[:end]
        request = [
            SystemMessage(content=loop.rules),
            *messages_from_dict(selected),
            HumanMessage(content=instruction),
        ]
        if (
            estimate_input_tokens(request, getattr(loop, 'tools', WORK_TOOLS)) + output_cap
            <= CONFIG['context_window_tokens']
        ):
            previous = loop.state.get('context_view', {}).get('cutoff', 0)
            cutoff = previous + end - 2 if previous else end
            return request, cutoff, fingerprint(selected)
        end = max((point for point in boundaries if first < point < end), default=first)
    raise ModelRequestNotSent(
        'compaction_failed',
        'No complete old span fits the summary request; original history retained.',
    )


# ========================================================================
# 摘要响应校验与持久化提交
# ========================================================================
def validate_summary(raw):
    response = messages_from_dict([raw])[0]
    if (
        response.response_metadata.get('finish_reason') != 'stop'
        or response.tool_calls
        or response.invalid_tool_calls
        or not isinstance(response.content, str)
        or not response.content.strip()
    ):
        raise ModelRequestNotSent(
            'compaction_failed',
            'Summary was empty, incomplete or called a tool; original view retained.',
        )
    return response.content.strip()


def commit(loop, pending):
    text = validate_summary(pending['response'])
    state = loop.state
    if (
        fingerprint(state['history']) != pending['archive_hash']
        or state.get('context_view') != pending['previous_view']
    ):
        raise ModelRequestNotSent(
            'state_conflict', 'History changed during summarization; original view retained.'
        )
    previous = pending['previous_view']
    generation = (previous['generation'] if previous else 0) + 1
    body = {
        'generation': generation,
        'cutoff': pending['cutoff'],
        'archive_hash': fingerprint(state['history'][: pending['cutoff']]),
        'rules_hash': fingerprint(state['rules']),
        'requirements': requirements(state),
        'previous_view': previous,
        'selected_hash': pending['selected_hash'],
        'call': pending['call'],
        'execution_snapshot': pending['execution_snapshot'],
        'message': message_to_dict(
            HumanMessage(
                content=CHECKPOINT_PREAMBLE
                + '\n<compacted-summary>\n'
                + text
                + '\n</compacted-summary>\n'
                + snapshot_text(pending['execution_snapshot'])
            )
        ),
    }
    if state.get('user_updates'):
        body['requirements']['user_updates'] = state['user_updates']
    if state.get('requirements') and state['requirements'].get('revision', 1) > 1:
        from agent.task.requirements import model_view

        body['requirements']['latest_confirmed_delivery'] = model_view(state['requirements'])
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True).encode('utf-8')
    reference = {
        'path': 'compactions/' + pending['id'] + '/summary.json',
        'sha256': hashlib.sha256(raw).hexdigest(),
    }
    proposed = {'generation': generation, 'cutoff': pending['cutoff'], 'record': reference}

    # ========================================================================
    # 确认摘要能缩小请求，再开始不可重复的提交
    # ========================================================================
    before = estimate_input_tokens(messages(loop), getattr(loop, 'tools', WORK_TOOLS))
    pinned = message_to_dict(
        HumanMessage(
            content=json.dumps(
                {'original_task_requirements': body['requirements']}, ensure_ascii=False
            )
        )
    )
    after = estimate_input_tokens(
        [
            SystemMessage(content=loop.rules),
            *messages_from_dict([pinned, body['message'], *state['history'][pending['cutoff'] :]]),
        ],
        getattr(loop, 'tools', WORK_TOOLS),
    )
    if after >= before:
        raise ModelRequestNotSent(
            'compaction_failed', 'Summary did not shrink the request; original view retained.'
        )

    # ========================================================================
    # 先持久化提交标记，再写入不可变摘要文件
    # ========================================================================
    pending['stage'] = 'committing'
    loop.persist('compaction_commit_started', {'id': pending['id']})
    path = loop.root / reference['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    from agent.storage.records import immutable

    immutable(path, raw)
    if loop.history_store:
        loop.history_store.register_derived(path, 'compaction', state['source'])

    # ========================================================================
    # 切换请求视图并记录本次压缩证据
    # ========================================================================
    admission = state.get('last_admission')
    state['context_view'] = proposed
    if admission:
        state['context_rollover'] = {'previous': fingerprint(admission), 'generation': generation}
    record = {
        'id': pending['id'],
        'generation': generation,
        'cutoff': pending['cutoff'],
        'before_tokens': before,
        'after_tokens': after,
        'record': reference,
        'call': pending['call'],
        'trigger': pending['trigger'],
    }
    state.setdefault('compactions', []).append(record)
    state['compaction_trigger']['successes'] += 1
    state['compaction_pending'] = None
    loop.persist('compaction_finished', record)
    return record


# ========================================================================
# 上下文压力检查与可恢复压缩流程
# ========================================================================
def prepare(loop, output_cap, *, force=False):
    guard_archive(loop.state)
    if any(
        loop.state.get(key)
        for key in (
            'pending',
            'running_step',
            'response_pending',
            'request_pending',
            'verification_running',
            'acceptance_running',
        )
    ):
        raise ModelRequestNotSent(
            'compaction_failed', 'A model response or operation is still unfinished.'
        )

    # ========================================================================
    # 按压力阈值决定是否复用或启动摘要
    # ========================================================================
    threshold, retain = limits(output_cap)
    while True:
        pending = loop.state.get('compaction_pending')
        trigger = loop.state.get('compaction_trigger')
        if (
            not pending
            and not force
            and estimate_input_tokens(messages(loop), getattr(loop, 'tools', WORK_TOOLS))
            < threshold
        ):
            if trigger:
                loop.state.pop('compaction_trigger')
                loop.persist('compaction_pressure_resolved')
            return
        if not trigger:
            trigger = {'id': uuid.uuid4().hex, 'successes': 0}
            loop.state['compaction_trigger'] = trigger
        if trigger['successes'] >= CONFIG['max_summaries_per_trigger']:
            raise ModelRequestNotSent(
                'compaction_failed',
                'Context remains above the pressure threshold after two summaries.',
            )
        if pending:
            if pending['stage'] == 'committing':
                raise ModelRequestNotSent(
                    'compaction_commit_unknown',
                    'Interrupted compaction commit; reconcile before continuing.',
                )
            if pending.get('response'):
                commit(loop, pending)
                force = False
                continue
        if not pending:
            snapshot = execution_snapshot(loop)
            request, cutoff, selected_hash = select(
                loop,
                retain,
                min(output_cap, CONFIG['summary_max_tokens']),
                snapshot,
            )
            pending = {
                'id': uuid.uuid4().hex,
                'stage': 'prepared',
                'cutoff': cutoff,
                'previous_view': loop.state.get('context_view'),
                'archive_hash': fingerprint(loop.state['history']),
                'selected_hash': selected_hash,
                'prefix_end': len(request) - 2,
                'execution_snapshot': snapshot,
                'trigger': 'context-overflow' if force else 'pressure',
            }
            loop.state['compaction_pending'] = pending
            loop.persist('compaction_started', {'id': pending['id'], 'trigger': pending['trigger']})

        # ========================================================================
        # 发送前再次核验固定前缀，复用已保存的摘要响应
        # ========================================================================
        prefix = history(loop.state, loop.root)[: pending['prefix_end']]
        if fingerprint(prefix) != pending['selected_hash']:
            raise ModelRequestNotSent(
                'state_conflict', 'Prepared summary prefix changed; no request sent.'
            )
        request = [
            SystemMessage(content=loop.rules),
            *messages_from_dict(prefix),
            HumanMessage(content=summary_instruction(pending['execution_snapshot'])),
        ]
        loop.request_model(request, 'compaction')
        # request_model 返回前已持久化此结果，恢复时可以复用。
        commit(loop, pending)
        force = False
