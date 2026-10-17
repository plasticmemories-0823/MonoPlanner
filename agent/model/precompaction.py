"""工作上下文预压缩；后台只写独立原件，主线程独占检查点和预算结算。"""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from threading import Event, Thread
import time
from urllib.parse import urlsplit
import uuid

from langchain_core.messages import AIMessage, message_to_dict, messages_from_dict, messages_to_dict

from agent.model import compaction
from agent.model.client import ConfiguredModel, configure_call, request_timeout
from agent.model.context import fingerprint
from agent.model.errors import ModelInterrupted, ModelRequestNotSent
from agent.model.payload import estimate_input_tokens, request_reservation
from agent.source.snapshots import _safe_file
from agent.storage.records import save
from agent.storage.requests import save_request
from agent.task import control
from agent.task.budget import RequestReservation


def enabled(model):
    """只对已确认使用 DeepSeek API 的工作模型启用固定输入水位。"""
    if not isinstance(model, ConfiguredModel):
        return False
    client = model.client
    address = getattr(client, 'api_base', None) or getattr(client, 'openai_api_base', None)
    return isinstance(address, str) and urlsplit(address).hostname == 'api.deepseek.com'


def limits():
    config = compaction.CONFIG.get('deepseek_work', {})
    return config.get('soft_input_tokens', 192_000), config.get('hard_input_tokens', 256_000)


def _independent(model):
    """复制请求配置而不是连接；凭据只保留在内存中。"""
    from agent.model.deepseek import DeepSeekClient
    from agent.model.network import model_http_clients
    from agent.model.streaming import run

    source = model.client
    clients = model_http_clients(model.policy.api_timeout)
    try:
        client = DeepSeekClient(
            model=source.model_name,
            api_key=source.api_key,
            base_url=source.api_base,
            temperature=source.temperature,
            max_tokens=model.policy.max_output_tokens,
            timeout=model.policy.api_timeout,
            max_retries=0,
            tags=['nostream'],
            **clients,
        )
    except BaseException:
        clients['http_client'].close()
        run(clients['http_async_client'].aclose(), control.Signal())
        raise

    def close():
        clients['http_client'].close()
        run(clients['http_async_client'].aclose(), control.Signal())

    return replace(model, client=client), close


class Job:
    """持有一个可取消请求；不引用 TaskRunner 的可变状态或历史数据库。"""

    def __init__(self, root, pending, model, request, parent_signal):
        self.root, self.pending = root, deepcopy(pending)
        self.model, self.request = model, deepcopy(request)
        self.signal, self.done = control.Signal(), Event()
        self.parent_signal = parent_signal
        self.error = None
        self.worker = Thread(target=self._run, name='work-precompaction', daemon=True)
        self.subscription = ExitStack()
        if parent_signal:
            self.subscription.enter_context(parent_signal.subscribe(self._parent_interrupt))

    def _parent_interrupt(self):
        self.signal.interrupt(self.parent_signal.kind or 'pause')

    def start(self):
        self.worker.start()

    def _run(self):
        started = time.monotonic()
        folder = self.root / 'requests' / ('call-%04d' % self.pending['call'])
        result = {'status': 'unknown', 'call': self.pending['call']}
        cleanup = None
        dispatch_started = False
        try:
            selected, cleanup = _independent(self.model)
            selected = configure_call(selected, 'compaction', timeout=self.pending['timeout'])
            with control.using(self.signal):
                if self.signal.kind:
                    raise ModelInterrupted(self.signal.kind, sent=False)
                dispatch_started = True
                response = selected.invoke_with_progress(self.request, lambda text: None)
            raw = message_to_dict(response)
            save(folder / 'response.json', raw)
            result.update(status='complete', response_hash=fingerprint(raw))
        except ModelInterrupted as exc:
            result.update(status='interrupted' if exc.sent else 'not_sent', kind=exc.kind)
        except ModelRequestNotSent:
            result['status'] = 'not_sent'
        except Exception as exc:
            # 详细网络异常可能包含地址或凭据，独立结果只记录异常类型。
            result['error_type'] = type(exc).__name__
            if not dispatch_started:
                result['status'] = 'not_sent'
        finally:
            result['seconds'] = time.monotonic() - started
            try:
                save(self.root / self.pending['directory'] / 'outcome.json', result)
            except BaseException as exc:
                self.error = exc
            try:
                if cleanup:
                    cleanup()
            except Exception:
                pass  # 已归档回复有效；连接释放异常不能伪造第二次调用。
            self.subscription.close()
            self.done.set()


def _pending_path(loop, pending, name):
    directory = 'compactions/' + pending['id']
    if pending['directory'] != directory or not pending['id'].isalnum():
        raise ModelRequestNotSent('state_conflict', 'Invalid precompaction record location.')
    return _safe_file(loop.root, Path(directory) / name)


def _check_snapshot(loop, pending):
    cutoff = pending['cutoff']
    if (
        type(cutoff) is not int
        or not 0 < cutoff <= len(loop.state['history'])
        or fingerprint(loop.state['history'][:cutoff]) != pending['archive_hash']
        or fingerprint(loop.state['rules']) != pending['rules_hash']
        or compaction.requirements(loop.state) != pending['requirements']
        or loop.state.get('context_view') != pending['previous_view']
    ):
        raise ModelRequestNotSent(
            'state_conflict', 'Prepared summary prefix or task rules changed; no request sent.'
        )


def _ticket(loop, pending):
    return RequestReservation(
        owner=id(loop.budget), call_id=pending['call'], used_steps=loop.budget.used_steps,
        purpose='compaction', remaining_steps=loop.budget.remaining_steps,
        working_steps=loop.budget.working_steps, report_steps=0,
        reported_tokens=loop.budget.reported_tokens,
        unknown_usage_steps=loop.budget.unknown_usage_steps,
        reserved_tokens=pending['reservation'],
    )


def _guard_unconfirmed_stop(loop, pending):
    path = loop.root / pending['directory'] / 'stop-unconfirmed.json'
    if path.exists():
        marker = json.loads(_pending_path(loop, pending, path.name).read_text('utf-8'))
        if marker.get('call') != pending['call']:
            raise ModelRequestNotSent('state_conflict', 'Summary stop record identity changed.')
        raise ModelRequestNotSent(
            'model_request_unknown',
            'Summary cancellation was not confirmed; inspect retained records before continuing.',
        )


def reconcile(loop):
    """只在工作主线程调用；可恢复完整回复，未知请求绝不重发。"""
    pending = loop.state.get('precompaction_pending')
    if not pending:
        return False
    # 元数据与原件一致即可结算；当前历史是否仍兼容，留到采纳时单独检查。
    # 即使用户补充或规则冲突使摘要不可采用，已发生的调用也不能漏记费用。
    original = json.loads(_pending_path(loop, pending, 'job.json').read_text('utf-8'))
    if any(pending.get(key) != value for key, value in original.items()
           if key != 'stage'):
        raise ModelRequestNotSent('state_conflict', 'Prepared summary metadata changed.')
    job = getattr(loop, '_precompaction_job', None)
    if job and not job.done.is_set():
        _guard_unconfirmed_stop(loop, pending)
        return False
    if pending.get('accounted'):
        _guard_unconfirmed_stop(loop, pending)
        return pending.get('status') == 'complete'
    path = loop.root / pending['directory'] / 'outcome.json'
    if path.exists():
        outcome = json.loads(_pending_path(loop, pending, 'outcome.json').read_text('utf-8'))
    else:
        outcome = {'status': 'unknown', 'call': pending['call'], 'seconds': 0}
        response_path = Path('requests') / ('call-%04d' % pending['call']) / 'response.json'
        if (loop.root / response_path).exists():
            # 完整响应已落盘但后台尚未来得及写结束标记时，复用原件，不补发请求。
            raw = json.loads(_safe_file(loop.root, response_path).read_text('utf-8'))
            outcome.update(status='complete', response_hash=fingerprint(raw))
    if outcome.get('call') != pending['call']:
        raise ModelRequestNotSent('state_conflict', 'Summary outcome belongs to another call.')
    response = None
    if outcome['status'] == 'complete':
        path = Path('requests') / ('call-%04d' % pending['call']) / 'response.json'
        raw = json.loads(_safe_file(loop.root, path).read_text('utf-8'))
        if fingerprint(raw) != outcome['response_hash']:
            raise ModelRequestNotSent('state_conflict', 'Saved summary response changed.')
        try:
            response = messages_from_dict([raw])[0]
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelRequestNotSent(
                'state_conflict', 'Saved summary is not a parseable assistant response.'
            ) from exc
        if not isinstance(response, AIMessage):
            raise ModelRequestNotSent('state_conflict', 'Saved summary is not an assistant response.')
        # 可解析的实际回复先结算；长度截断、空摘要或工具调用由采纳阶段统一拒绝。
        pending.update(response=raw, stage='response_saved')
    ticket = _ticket(loop, pending)
    if outcome['status'] == 'not_sent':
        loop.budget.cancel_request(ticket)
    else:
        loop.budget.settle_request(ticket, response)
        loop.state.setdefault('model_usage', []).append({
            'call': pending['call'], 'purpose': 'compaction', 'seconds': outcome['seconds'],
            'usage': response.usage_metadata if response else None,
            'max_output_tokens': pending['output_cap'], 'background': True,
        })
    pending.update(accounted=True, status=outcome['status'], kind=outcome.get('kind'))
    if pending['status'] == 'not_sent' and not pending['kind']:
        pending.update(
            status='failed', failure='Summary request could not be prepared; no request sent.'
        )
    loop.persist(
        'precompaction_response_saved', {'call': pending['call'], 'status': pending['status']}
    )
    _guard_unconfirmed_stop(loop, pending)
    return pending['status'] == 'complete'


def _start(loop, output_cap, force):
    _, retain = compaction.limits(output_cap)
    snapshot = compaction.execution_snapshot(loop)
    cap = min(output_cap, compaction.CONFIG['summary_max_tokens'])
    request, cutoff, selected_hash = compaction.select(loop, retain, cap, snapshot)
    timeout = request_timeout(loop.model, loop.budget.remaining_time())
    ticket = loop.budget.reserve_request(
        request_reservation(request, loop.tools, cap), purpose='compaction'
    )
    identity = uuid.uuid4().hex
    pending = {
        'id': identity, 'directory': 'compactions/' + identity, 'stage': 'prepared',
        'cutoff': cutoff, 'previous_view': deepcopy(loop.state.get('context_view')),
        'archive_hash': fingerprint(loop.state['history'][:cutoff]),
        'rules_hash': fingerprint(loop.state['rules']),
        'requirements': deepcopy(compaction.requirements(loop.state)),
        'selected_hash': selected_hash, 'execution_snapshot': snapshot,
        'trigger': 'context-overflow' if force else 'precompaction',
        'call': ticket.call_id, 'reservation': ticket.reserved_tokens, 'output_cap': cap,
        'timeout': timeout, 'prefix_guard': True,
    }
    try:
        folder = loop.root / 'requests' / ('call-%04d' % ticket.call_id)
        save_request(loop.root, folder, messages_to_dict(request), 'compaction', loop.history_store)
        save(loop.root / pending['directory'] / 'job.json', pending)
    except BaseException:
        loop.budget.cancel_request(ticket)
        raise
    loop.state['precompaction_pending'] = pending
    loop.persist('precompaction_started', {'id': identity, 'call': ticket.call_id})
    job = Job(loop.root, pending, loop.model.bind_tools(loop.tools), request, control.current())
    loop._precompaction_job = job
    try:
        job.start()
    except BaseException:
        save(loop.root / pending['directory'] / 'outcome.json', {
            'status': 'not_sent', 'call': pending['call'], 'seconds': 0,
        })
        job.subscription.close()
        job.done.set()
        reconcile(loop)
        raise


def _adopt(loop):
    pending = loop.state['precompaction_pending']
    if pending['stage'] == 'committing':
        raise ModelRequestNotSent('compaction_commit_unknown', 'Interrupted summary commit.')
    _check_snapshot(loop, pending)
    loop.state['compaction_pending'] = pending
    loop.state.setdefault('compaction_trigger', {'id': pending['id'], 'successes': 0})
    compaction.commit(loop, pending)
    loop.state.pop('precompaction_pending', None)
    loop._precompaction_job = None
    loop.persist('precompaction_adopted', {'id': pending['id'], 'call': pending['call']})


def prepare(loop, output_cap, *, force=False):
    """请求前的安全边界：软水位并发准备，硬水位等待完整摘要。"""
    if not enabled(loop.model):
        return compaction.prepare(loop, output_cap, force=force, preserve_recent=True)
    compaction.guard_archive(loop.state)
    if any(loop.state.get(key) for key in (
        'pending', 'running_step', 'response_pending', 'request_pending',
        'verification_running', 'acceptance_running',
    )):
        raise ModelRequestNotSent('compaction_failed', 'A response or operation is unfinished.')
    soft, hard = limits()
    adopted = 0
    while True:
        pending = loop.state.get('precompaction_pending')
        if pending and reconcile(loop):
            try:
                _adopt(loop)
            except ModelRequestNotSent as exc:
                if exc.status != 'compaction_failed':
                    raise
                pending.update(status='failed', failure=exc.reason)
                loop.state['compaction_pending'] = None
                loop.persist('precompaction_failed', {'call': pending['call']})
            else:
                adopted += 1
                force = False
                pending = None
        used = estimate_input_tokens(compaction.messages(loop), loop.tools)
        if not force and used < soft:
            loop.state.pop('compaction_trigger', None)
            return
        if pending and pending.get('accounted'):
            if pending['status'] in {'interrupted', 'not_sent'}:
                raise ModelInterrupted(
                    pending.get('kind') or 'pause', sent=pending['status'] != 'not_sent'
                )
            if not force and used < hard:
                return  # 摘要失败不打断仍低于硬水位的工作，也不自动补发后台请求。
            raise ModelRequestNotSent(
                'compaction_failed' if pending['status'] == 'failed' else 'model_request_unknown',
                pending.get('failure') or 'Summary result is unknown; no automatic resend.',
            )
        successes = (loop.state.get('compaction_trigger') or {}).get('successes', 0)
        if max(adopted, successes) >= compaction.CONFIG['max_summaries_per_trigger']:
            if not force and used < hard:
                return
            raise ModelRequestNotSent(
                'compaction_failed', 'Context remains above its input watermark.'
            )
        if not pending:
            try:
                _start(loop, output_cap, force)
            except ModelRequestNotSent as exc:
                if exc.status != 'compaction_failed' or force or used >= hard:
                    raise
                # 只有近期原文时暂不压缩；硬水位前仍可继续，不删除受保护消息。
                return
        if not force and used < hard:
            return
        job = getattr(loop, '_precompaction_job', None)
        if job:
            started = time.monotonic()
            if not job.done.wait(timeout=job.pending['timeout'] + 5):
                job.signal.interrupt('pause')
                raise ModelRequestNotSent(
                    'model_request_unknown', 'Summary did not finish within its request timeout.'
                )
            loop.state.setdefault('precompaction_waits', []).append({
                'call': job.pending['call'], 'seconds': time.monotonic() - started,
            })


def close(loop, *, settle=True):
    """离开任务所有权前取消后台连接并归档；不采纳摘要或重发请求。"""
    job = getattr(loop, '_precompaction_job', None)
    if job:
        if not job.done.is_set():
            job.signal.interrupt('pause')
        if job.worker.ident is not None:
            job.worker.join(timeout=5)
        if not job.done.is_set():
            pending = loop.state['precompaction_pending']
            marker = _pending_path(loop, pending, 'job.json').parent / 'stop-unconfirmed.json'
            save(marker, {'call': pending['call'], 'status': 'stop_unconfirmed'})
            raise ModelRequestNotSent(
                'model_request_unknown', 'Summary cancellation is not confirmed; records retained.'
            )
    # 异常退出时只取消和等待，不让清理覆盖故障检查点；恢复流程另行结算。
    if settle and loop.state.get('precompaction_pending'):
        reconcile(loop)
        pending = loop.state['precompaction_pending']
        if pending.get('status') in {'interrupted', 'not_sent'}:
            loop.state.setdefault('precompaction_cancelled', []).append({
                key: pending.get(key) for key in ('id', 'call', 'status', 'kind')
            })
            loop.state.pop('precompaction_pending')
            loop.persist('precompaction_cancelled', {'call': pending['call']})
    loop._precompaction_job = None
