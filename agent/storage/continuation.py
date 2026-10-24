"""核对持久化原件再推进游标；未知操作不原样重放，未知费用不清零。"""

import json
from pathlib import Path

from langchain_core.messages import messages_from_dict, message_to_dict, ToolMessage

from agent.source.snapshots import SourceSnapshot, _safe_file
from agent.storage.records import digest, save


def read(root, relative):
    return json.loads(_safe_file(root, Path(relative)).read_bytes())


def authorized(loop):
    identity = getattr(loop, 'recovery_id', None)
    return bool(identity and identity not in loop.state.get('used_recovery_ids', []))


def recovered_time(loop, seconds):
    if isinstance(seconds, (int, float)) and seconds >= 0:
        loop.prior_elapsed += seconds
        loop.budget.started_at -= seconds


def _known_response(loop):
    state = loop.state
    pending = state.get('request_pending')
    if not pending:
        return
    settled = next((row for row in state.get('model_usage', [])
                    if row['call'] == pending['call']), None)
    if (type(pending.get('reservation')) is not int or pending['reservation'] < 0
            or (not settled and pending['reservation'] > loop.budget.inflight_reserved_tokens)):
        raise ValueError('Pending model reservation is inconsistent')
    folder = loop.root / 'requests' / ('call-%04d' % pending['call'])
    response_path = folder / 'response.json'
    if response_path.exists():
        raw = read(loop.root, response_path.relative_to(loop.root))
        metadata = read(loop.root, folder.relative_to(loop.root) / 'completion.json')
        if (metadata['response_digest'] != digest(raw)
                or metadata['request'] != pending):
            raise ValueError('Saved response does not match the pending request')
        response = messages_from_dict([raw])[0]
        if settled:
            if settled.get('usage') != response.usage_metadata:
                raise ValueError('Saved model usage differs from the response receipt')
        else:
            recovered_time(loop, metadata['seconds'])
            loop.budget.inflight_reserved_tokens -= pending['reservation']
            loop.budget.record_usage(response, pending['reservation'])
            state.setdefault('model_usage', []).append({
                'call': pending['call'], 'purpose': pending['purpose'],
                'seconds': metadata['seconds'], 'usage': response.usage_metadata,
                'max_output_tokens': metadata['max_output_tokens'],
            })
        if pending['purpose'] == 'compaction':
            state['compaction_pending'].update(
                stage='response_saved', response=raw, call=pending['call'],
            )
        else:
            state['response_pending'] = raw
        state['request_pending'] = None
        loop.persist('model_response_received', {'call': pending['call'], 'recovered': True})
        return
    # 请求已发出但没有完整响应：结算为未知一次，不释放预留，也不自动补发。
    if not settled:
        loop.budget.inflight_reserved_tokens -= pending['reservation']
        loop.budget.record_usage(None, pending['reservation'])
        state.setdefault('model_usage', []).append({
            'call': pending['call'], 'purpose': pending['purpose'], 'usage': None,
            'seconds': 0, 'seconds_unknown': True, 'interrupted': True,
        })
    state.setdefault('unknown_requests', []).append(dict(pending))
    state.update(request_pending=None, status='model_request_unknown',
                 reason='Interrupted request has no complete response; explicit Continue is required.')
    if pending['purpose'] == 'compaction':
        state['compaction_pending'] = None
    loop.persist('model_request_reconciled', {'call': pending['call'], 'usage': 'unknown'})


def _step(loop):
    state, running = loop.state, loop.state.get('running_step')
    if not running:
        return False
    pending = state['pending']
    if not isinstance(running, dict) or not pending:
        return False
    cursor, steps = pending.get('cursor'), (pending.get('plan') or {}).get('steps')
    if (type(cursor) is not int or not isinstance(steps, list)
            or not 0 <= cursor < len(steps)):
        raise ValueError('Interrupted operation cursor is inconsistent with the saved plan')
    request = steps[cursor]
    if (running['plan'] != pending['id'] or running['id'] != request['id']
            or running['before'] != state['source']):
        raise ValueError('Interrupted operation identity changed')
    folder = loop.root / 'plans' / pending['id'] / (pending['id'] + '-' + request['id'])
    if read(loop.root, folder.relative_to(loop.root) / 'request.json') != request:
        raise ValueError('Interrupted operation request changed')
    path = folder / 'result.json'
    if path.exists():
        row = read(loop.root, path.relative_to(loop.root))
        complete = read(loop.root, folder.relative_to(loop.root) / 'completion.json')
        if complete['request'] != request or complete['result_digest'] != digest(row):
            raise ValueError('Saved operation result checksum mismatch')
        if (row['id'] != running['id'] or row['op'] != running['op']
                or row['before'] != running['before'] or row['after'] != loop.evidence.current
                or row['record'] != path.relative_to(loop.root).as_posix()):
            raise ValueError('Saved operation result or source changed')
    else:
        receipt = folder / 'execution-result.json'
        if receipt.exists():
            saved = read(loop.root, receipt.relative_to(loop.root))
            if saved['request'] != request or saved['before'] != running['before']:
                raise ValueError('Execution receipt differs from pending operation')
            row = saved['result']
        else:
            from agent.execution.executor import recover_process_result

            row = recover_process_result(loop.environment, loop.root, folder, request)
        if row is None:
            if not authorized(loop) or not getattr(loop.environment, 'recover_partial_effects', False):
                return False
            # 本步骤可能留下部分副作用。结束旧计划，要求先检查现状，禁止盲目重放。
            row = {'status': 'failed', 'seconds': 0, 'seconds_unknown': True, 'data': {
                'interrupted': True, 'effects': 'unconfirmed',
                'error': 'Execution was interrupted. Effects may be partial. Inspect the current '
                         'source and retained logs before planning a repair; do not replay this command.',
            }}
            state['recovery_inspection_required'] = True
            state.setdefault('uncertain_commands', []).append(request.get('command', ''))
        row.update(id=running['id'], op=running['op'], before=running['before'],
                   after=loop.evidence.current, cache_key=None,
                   record=path.relative_to(loop.root).as_posix())
        save(folder / 'completion.json', {'request': request, 'result_digest': digest(row)})
        save(path, row)
    pending['rows'].append(row)
    pending['cursor'] += 1
    state.update(running_step=None, source=row['after'])
    recovered_time(loop, row.get('seconds', 0))
    loop.persist('step_finished', row)
    return True


def _verification(loop):
    state = loop.state
    if not state.get('verification_running'):
        return
    folder = loop.root / 'feedback' / state['pending']['id']
    attempt = state.get('verification_attempt', 0)
    if attempt:
        folder = folder / ('recovery-%04d' % attempt)
    receipt = folder / 'result.json'
    if not receipt.exists():
        if authorized(loop):
            # 验收只在隔离副本里执行。保留未完成尝试，下一次使用新目录。
            state.setdefault('interrupted_verifications', []).append({
                'plan': state['pending']['id'], 'source': state['source'],
                'attempt': state.get('verification_attempt', 0),
            })
            state['verification_attempt'] = state.get('verification_attempt', 0) + 1
            state['verification_running'] = False
            loop.persist('verification_interrupted')
        return
    report = read(loop.root, receipt.relative_to(loop.root))
    complete = read(loop.root, folder.relative_to(loop.root) / 'completion.json')
    if (complete['report_digest'] != digest(report)
            or complete['acceptance_digest'] != digest(state['verification_spec'])):
        raise ValueError('Saved acceptance result checksum mismatch')
    from agent.acceptance.final import require_source_integrity

    require_source_integrity(report)
    manifest = folder / 'candidate/manifest.json'
    body = read(loop.root, manifest.relative_to(loop.root))
    snapshot = SourceSnapshot(Path(body['root']), body['files'], body['digest'], manifest,
                              body.get('excluded_files', {}))
    if (complete['candidate_digest'] != snapshot.digest
            or not snapshot.verify_integrity()['valid']
            or snapshot.files != dict(loop.evidence.view().files)):
        raise ValueError('Saved acceptance candidate changed')
    snapshot.write_patch(loop.baseline, loop.root / 'candidate.patch')
    state['rounds'].append({'plan': state['pending']['id'], 'verification': report,
                            'candidate_digest': snapshot.digest})
    state.update(verification=report, verification_running=False)
    loop.persist('final_acceptance_finished', {
        'plan': state['pending']['id'], 'record': str(receipt),
        'status': report['status'], 'recovered': True,
    })


def _web(loop):
    """只读联网工具可补登记或明确结束未知尝试；不重发原服务请求。"""
    state = loop.state
    pending = state.get('web_pending')
    if not pending or not state.get('web_running'):
        return
    call = pending['response']['data']['tool_calls'][pending['cursor']]
    folder = loop.root / 'web' / ('call-%04d-%02d' % (loop.budget.request_sequence, pending['cursor']))
    if call['id'] != state['web_running'] or read(loop.root, folder.relative_to(loop.root) / 'request.json') != call:
        raise ValueError('Interrupted web tool identity changed')
    request = state.get('web_request_pending')
    if request and not any(row['call'] == request['call'] for row in state.get('web_usage', [])):
        receipt_path = folder / 'provider-completion.json'
        receipt = read(loop.root, receipt_path.relative_to(loop.root)) if receipt_path.exists() else None
        if receipt and receipt['pending'] != request:
            raise ValueError('Web service receipt differs from reservation')
        if receipt and receipt.get('response_sha256'):
            import hashlib

            response = _safe_file(loop.root, folder.relative_to(loop.root) / 'provider-response.json').read_bytes()
            if hashlib.sha256(response).hexdigest() != receipt['response_sha256']:
                raise ValueError('Saved web service response changed')
        outcome = receipt['outcome'] if receipt else 'unknown'
        if outcome not in {'succeeded', 'empty', 'failed', 'not_sent', 'unknown'}:
            raise ValueError('Unsupported web settlement receipt')
        amount = request['reserved_cost_micros']
        if type(amount) is not int or not 0 <= amount <= loop.budget.inflight_service_cost_micros:
            raise ValueError('Interrupted web reservation changed')
        loop.budget.inflight_service_cost_micros -= amount
        loop.budget.service_cost_micros += amount if outcome == 'succeeded' else 0
        loop.budget.unknown_service_cost_micros += amount if outcome == 'unknown' else 0
        if outcome == 'not_sent':
            loop.budget.service_calls -= 1
        state.setdefault('web_usage', []).append({
            **request, 'status': outcome, 'cost_micros': None if outcome == 'unknown' else amount if outcome == 'succeeded' else 0,
            'reserved_cost_micros': amount if outcome == 'unknown' else 0,
            'seconds': receipt['seconds'] if receipt else 0, 'seconds_unknown': not bool(receipt),
        })
        if receipt:
            recovered_time(loop, receipt['seconds'])
        loop.persist('web_usage_reconciled', {'call': request['call'], 'outcome': outcome})
    result_path, completion_path = folder / 'result.json', folder / 'completion.json'
    if result_path.exists() and completion_path.exists():
        result = read(loop.root, result_path.relative_to(loop.root))
        receipt = read(loop.root, completion_path.relative_to(loop.root))
        if receipt['request'] != call or receipt['result_digest'] != digest(result):
            raise ValueError('Saved web tool result changed')
        data = result['data']
    elif authorized(loop):
        data = {'status': 'unknown', 'reason': 'The previous read-only web attempt has no complete tool receipt. '
                'Previously recorded fees and any unknown reservation are retained. '
                'Do not automatically resend that request.'}
        save(folder / ('continued-' + digest(loop.recovery_id) + '.json'), data)
    else:
        state.update(status='web_request_unknown', reason='Web receipt requires explicit continuation')
        return
    from agent.model.tool_results import normalize_result
    from agent.model.tool_payloads import project

    visible = project(loop, normalize_result(call['name'], data, source_version=state.get('source'),
                                             project=loop.environment.repo_root))
    pending['results'].append(message_to_dict(ToolMessage(
        content=json.dumps(visible, ensure_ascii=False), name=call['name'], tool_call_id=call['id'])))
    pending['cursor'] += 1
    state.update(web_running=None, web_request_pending=None)
    loop.persist('web_finished', {'id': call['id'], 'recovered': True})


def reconcile(loop):
    """完整结果可无损补登记；进一步请求或不完整执行只能由继续命令放行。"""
    state = loop.state
    previous_unknowns = len(state.get('unknown_requests', []))
    _known_response(loop)
    if authorized(loop) and len(state.get('unknown_requests', [])) > previous_unknowns:
        # 继续命令不能顺带批准审计中新发现的未知请求；保留费用并再次确认。
        state.setdefault('rejected_recovery_ids', []).append(loop.recovery_id)
        state.setdefault('used_recovery_ids', []).append(loop.recovery_id)
        state.setdefault('applied_inputs', []).append(loop.recovery_id)
        loop.persist('continue_requires_unknown_confirmation', {'id': loop.recovery_id})
        return False
    recovered_step = _step(loop)
    if state['source'] != loop.evidence.current:
        raise ValueError('Workspace source changed outside the interrupted operation')
    _verification(loop)
    _web(loop)
    if state.get('acceptance_running'):
        from agent.acceptance.revisions import resume_trial

        resume_trial(loop, allow_execute=authorized(loop))
    if authorized(loop) and (state.get('compaction_pending') or {}).get('stage') == 'committing':
        from agent.model.compaction import commit

        commit(loop, state['compaction_pending'])
    if authorized(loop):
        state.setdefault('used_recovery_ids', []).append(loop.recovery_id)
        state.pop('hard_interruption', None)
        if state['status'] in {
            'model_request_unknown', 'execution_unknown', 'runtime_error', 'interrupted',
            'compaction_commit_unknown', 'state_conflict',
        }:
            state.update(status='executing' if state.get('pending') else 'needs_plan', reason='')
        applied = state.setdefault('applied_inputs', [])
        if loop.recovery_id not in applied:
            applied.append(loop.recovery_id)
        if state['status'] in {'paused', 'awaiting_user'}:
            state.update(status='executing' if state.get('pending') else 'needs_plan',
                         pause_requested=False)
        loop.persist('explicit_recovery_authorized', {'id': loop.recovery_id})
        marker = loop.root / 'hard-interruption.json'
        active = ('running_step', 'verification_running', 'acceptance_running', 'assistant_running')
        if marker.exists() and not any(state.get(name) for name in active):
            from agent.storage.records import immutable
            import hashlib

            raw = _safe_file(loop.root, Path(marker.name)).read_bytes()
            filename = 'interruption-' + hashlib.sha256(raw).hexdigest() + '.json'
            immutable(loop.root / 'recovery' / filename, raw)
            marker.unlink()
    return recovered_step
