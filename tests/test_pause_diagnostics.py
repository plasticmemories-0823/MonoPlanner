"""暂停原因、只读证据与瞬时续租冲突的行为回归。"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.runtime.liveness import Lease
from agent.task.control import active


def test_stop_report_sees_current_confirmed_scope_without_private_tests(tmp_path):
    from agent.application.store import ApplicationStore
    from agent.application.chat import Chatbot
    from unittest.mock import patch
    import uuid

    repo = tmp_path / 'repo'
    repo.mkdir()
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(repo))['thread_id']
    command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Inspect the original task'}
    store.submit(conversation, command)
    work = store.create_work(conversation, command['id'])
    store.work_state(work['task_id'], 'execution_unknown', reason='Execution check timed out')
    packet = {'revision': 2, 'digest': 'confirmed-digest',
              'confirmation': {'id': 'confirmed-review'}, 'sources': [{'id': 'original'}],
              'targets': [{'id': 'entry', 'artifact_type': 'repository_entry',
                           'target_runtime': 'wine-cmd', 'entry': 'clean.cmd',
                           'behavior': 'Refuse normalized protected paths',
                           'input_scope': 'Both confirmed dot-segment forms',
                           'private_test_code': 'MUST_NOT_PUBLISH'}]}
    controls = [{'id': 'applied-review', 'kind': 'supplement', 'state': 'applied',
                 'value': {'confirmed_requirements': packet}}]
    with patch.object(store, 'commands', return_value=controls):
        facts = Chatbot(store, lambda *_: None).work_facts(conversation)
    confirmed = facts['task']['confirmed_requirements']
    assert confirmed['digest'] == packet['digest']
    assert confirmed['confirmation_id'] == 'confirmed-review'
    assert confirmed['targets'][0]['input_scope'] == 'Both confirmed dot-segment forms'
    assert 'MUST_NOT_PUBLISH' not in str(confirmed)


def sharing_error():
    error = PermissionError('sharing conflict')
    error.winerror = 5
    return error


def test_transient_heartbeat_replace_conflict_does_not_pause(tmp_path):
    with active(tmp_path, 'owner') as signal:
        lease = Lease(tmp_path, 'reader-contention')
        with patch('agent.runtime.liveness.atomic_write', side_effect=[sharing_error(), None]) as write:
            lease._write(lease.token + ' 1')
        assert write.call_count == 2
        assert signal.kind is None


def test_persistent_heartbeat_failure_retains_system_cause(tmp_path):
    with active(tmp_path, 'owner') as signal:
        lease = Lease(tmp_path, 'disk-failure')
        with patch.object(lease, '_write', side_effect=[None, OSError(28, 'disk full')]):
            lease.arm()
            lease.thread.join(timeout=4)
        facts = signal.diagnostics()
        assert signal.kind == 'pause'
        assert facts[-1]['identity'] == 'lifetime-renewal-failed'
        assert facts[-1]['error']['errno'] == 28
        assert facts[-1]['source'] == 'system'


def test_internal_interruption_is_not_labeled_as_user_action(tmp_path):
    from agent.task.runner import TaskRunner
    with active(tmp_path, 'owner') as signal:
        signal.interrupt('pause', 'host-lease-expired')
        runner = TaskRunner.__new__(TaskRunner)
        runner.state = {'history': []}
        runner.control_source = None
        runner.persist = lambda *_: None
        runner.on_model_interrupted('pause')
        assert runner.state['interruption']['causes'][0]['identity'] == 'host-lease-expired'
        assert 'user' not in runner.state['reason'].lower()
        assert 'The user interrupted' not in json.dumps(runner.state['history'])


def test_rejection_exposes_diagnostic_without_exposing_response():
    from agent.application.work_bridge import public_fact
    state = {'response_pending': {'reasoning_content': 'PRIVATE'}, 'history': []}
    result = public_fact(state, 'plan_rejected', {'reason': 'execution_checks[0].interpreter: unsupported',
                                               'call_ids': ['call-a']})
    assert result['reason'] == 'execution_checks[0].interpreter: unsupported'
    assert result['call_ids'] == ['call-a']
    assert 'PRIVATE' not in json.dumps(result)


def test_readonly_diagnostics_preserve_unknown_usage_and_hide_reasoning(tmp_path):
    from agent.application.records import read_task_record
    from agent.storage.history import LocalHistory
    history = LocalHistory(tmp_path)
    with history.lease():
        history.record({'history': [], 'status': 'paused', 'reason': 'Runtime heartbeat failed',
                        'plan_error': 'Invalid field /execution_checks/0',
                        'interrupted_requests': [{'call': 6, 'kind': 'pause', 'usage': 'unknown',
                                                  'reasoning_content': 'PRIVATE REASONING'}]},
                       'fixture', {})
    try:
        result = read_task_record(tmp_path, 'diagnostics', 0, 16000)
        facts = json.loads(result['text'])
        assert facts['interrupted_requests'][0]['usage'] == 'unknown'
        assert facts['plan_error'] == 'Invalid field /execution_checks/0'
        assert 'PRIVATE REASONING' not in result['text']
        with pytest.raises(ValueError):
            read_task_record(tmp_path, 'requests/call-0006/response.json', 0, 16000)
    finally:
        history.close()


def test_unsupported_runtime_is_rejected_before_work_proposal(tmp_path):
    from agent.acceptance.capabilities import require_targets
    with pytest.raises(ValueError, match='Work has not been started'):
        require_targets([{'artifact_type': 'repository_entry', 'target_runtime': 'windows-cmd'}])
    require_targets([{'artifact_type': 'repository_entry', 'target_runtime': 'python'}])


def make_work(tmp_path):
    import uuid
    from agent.application.store import ApplicationStore
    store = ApplicationStore(tmp_path / 'state')
    repo = tmp_path / 'repo'
    repo.mkdir()
    conversation = store.create(repo=str(repo))['thread_id']
    message = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Build a safe cleaner'}
    store.submit(conversation, message)
    task = store.create_work(conversation, message['id'])['task_id']
    return store, conversation, task


def test_stop_is_explained_once_even_if_older_event_pages_or_duplicate_stop_arrive(tmp_path):
    from agent.application.scheduler import Scheduler
    store, conversation, task = make_work(tmp_path)
    scheduler = Scheduler(store)
    batches = []
    scheduler.chatbot.progress = lambda _conversation, facts: (
        batches.append(facts) or {'output': '心跳续租失败，任务已暂停。'})
    try:
        for _ in range(65):
            scheduler._emit(task, 'work_state', {'status': 'executing'})
        scheduler._emit(task, 'work_state', {'status': 'paused', 'reason': 'Heartbeat failed'})
        scheduler._report(store.reserve_progress(now=100))
        scheduler._emit(task, 'work_state', {'status': 'paused', 'reason': 'Heartbeat failed'})
        scheduler._report(store.reserve_progress(now=131))
        assert len(batches) == 1
        assert 'diagnostics' in batches[0]
        notices = [m for m in store.snapshot(conversation)['messages'] if m['source'] == 'progress']
        assert len(notices) == 1
        scheduler._emit(task, 'work_state', {'status': 'executing'})
        scheduler._emit(task, 'work_state', {'status': 'paused', 'reason': 'Another stop'})
        scheduler._report(store.reserve_progress(now=162))
        assert len(batches) == 2
    finally:
        scheduler.close()


def test_failed_stop_explanation_has_saved_fallback_without_retry(tmp_path):
    from agent.application.scheduler import Scheduler
    store, conversation, task = make_work(tmp_path)
    scheduler = Scheduler(store)
    attempts = []

    def unavailable(*_):
        attempts.append(True)
        raise TimeoutError('unknown model response')

    scheduler.chatbot.progress = unavailable
    try:
        scheduler._emit(task, 'work_state', {'status': 'paused', 'reason': 'Host heartbeat lost'})
        scheduler._report(store.reserve_progress(now=100))
        scheduler._emit(task, 'work_state', {'status': 'paused', 'reason': 'Host heartbeat lost'})
        scheduler._report(store.reserve_progress(now=131))
        assert len(attempts) == 1
        notices = [m for m in store.snapshot(conversation)['messages'] if m['source'] == 'progress']
        assert len(notices) == 1
        assert 'Host heartbeat lost' in notices[0]['content']
        assert store.work(task)['status'] == 'paused'
        assert not scheduler.storage_error()
    finally:
        scheduler.close()


def test_preparation_failure_is_readable_without_creating_a_checkpoint(tmp_path):
    from agent.application.records import read_task_record
    root = tmp_path / 'task'
    root.mkdir()
    original = {'status': 'preparation_failed', 'error_type': 'PermissionError',
                'stop_reason': 'Dependency cache is not writable', 'credential': 'DO_NOT_EXPOSE'}
    (root / 'preparation-failed.json').write_text(json.dumps(original), encoding='utf-8')
    value = json.loads(read_task_record(root, 'diagnostics', 0, 16000)['text'])
    assert value['checkpoint_available'] is False
    assert value['saved_receipts'][0]['facts']['error_type'] == 'PermissionError'
    assert 'DO_NOT_EXPOSE' not in json.dumps(value)
    assert sorted(p.name for p in root.iterdir()) == ['preparation-failed.json']
    missing = tmp_path / 'missing'
    with pytest.raises(FileNotFoundError):
        read_task_record(missing, 'diagnostics', 0, 16000)
    assert not missing.exists()


def test_chat_interface_upgrade_preserves_original_history_and_requires_explicit_revision(tmp_path, monkeypatch):
    import uuid
    from agent.application import context
    from agent.application.chat import Chatbot
    from test_application_runtime import FakeModel, answer

    store, conversation, _ = make_work(tmp_path)
    model = FakeModel([answer('旧回复'), answer('新回复')])
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model)
    monkeypatch.setattr(context, 'CHAT_INTERFACE_REVISION', 1)
    monkeypatch.setattr(bot, '_rules', lambda *_: 'Original application rules')

    def ask(text):
        command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': text}
        store.submit(conversation, command)
        return bot.run(conversation, command)

    ask('旧问题')
    original = store.model_history(conversation)
    monkeypatch.setattr(context, 'CHAT_INTERFACE_REVISION', 2)
    monkeypatch.setattr(bot, '_rules', lambda *_: 'New application rules with diagnostic tools')
    ask('为什么暂停')
    assert store.model_history(conversation)[:len(original)] == original
    assert store.context(conversation)['prior_views'][-1]['interface_revision'] == 1
    monkeypatch.setattr(bot, '_rules', lambda *_: 'Unversioned tampering')
    with pytest.raises(Exception, match='settings changed'):
        ask('再次查询')
    assert len(model.requests) == 2


def test_unsupported_proposal_never_publishes_confirmation(tmp_path):
    from agent.application.requirements import prepare, source_messages
    from support import fixture_delivery

    store, conversation, _ = make_work(tmp_path)
    source = source_messages(store, conversation, [])['sources'][0]
    original = source_messages(store, conversation, [source['id']])['sources'][0]
    target = fixture_delivery(original['text'])['targets'][0]
    target.update(target_runtime='windows-cmd', entry='cleaner.bat',
                  quotes=[{'source_id': source['id'], 'text': original['text']}])
    with pytest.raises(ValueError, match='no supported execution adapter'):
        prepare(store, conversation, {'id': source['id']}, project_path='.', create=False,
                request_message_ids=[source['id']], targets=[target])
    assert not (store.root / 'work-proposals').exists()
    assert not [m for m in store.snapshot(conversation)['messages'] if m['source'] == 'work_proposal']


def test_stopped_verification_exposes_timeout_original_without_private_assertions(tmp_path):
    from langchain_core.messages import AIMessage, message_to_dict
    from agent.application.records import read_task_record
    from agent.storage.history import LocalHistory
    root = tmp_path / 'task'
    history = LocalHistory(root)
    with history.lease():
        history.record({'history': [], 'status': 'execution_unknown',
                        'reason': 'Incomplete execution receipt',
                        'pending': {'id': 'p0013', 'cursor': 2, 'rows': [],
                                    'response': message_to_dict(AIMessage(content='Saved candidate'))},
                        'verification_running': True}, 'fixture', {})
    history.close()
    folder = root / 'feedback/p0013/verification/acceptance-case'
    folder.mkdir(parents=True)
    (folder / 'candidate.execution-error.json').write_text(json.dumps({
        'status': 'error', 'tests_run': 7, 'test_code': 'PRIVATE_ASSERTIONS',
        'execution_process': {'exit_code': 124, 'timed_out': True,
                              'request': 'PRIVATE_RESPONSE', 'container_id': 'owned'},
    }), encoding='utf-8')
    value = read_task_record(root, 'diagnostics', 0, 16000)
    facts = json.loads(value['text'])
    receipt = facts['acceptance_receipts'][0]
    assert receipt['facts']['execution_process']['timed_out'] is True
    assert receipt['facts']['tests_run'] == 7
    assert 'not permission to replay' in receipt['provenance']
    assert 'PRIVATE' not in value['text']
    assert facts['current_plan'] == {'id': 'p0013', 'cursor': 2}


def test_last_report_request_is_reserved_for_explanation_not_another_tool_read(tmp_path):
    from agent.application.chat import Chatbot
    from agent.model.client import ConfiguredModel
    from test_application_runtime import FakeModel, answer

    class ReportingModel(FakeModel):
        def __init__(self):
            super().__init__([
                answer(tool_calls=[{'id': 'a', 'name': 'get_work_status', 'args': {}}]),
                answer(tool_calls=[{'id': 'b', 'name': 'get_work_status', 'args': {}}]),
                answer('独立入口验收超时，任务已停止；原件保留。'),
            ])
            self.purposes = []

        def for_purpose(self, purpose, timeout=None):
            self.purposes.append(purpose)
            return self

    store, conversation, task = make_work(tmp_path)
    store.work_state(task, 'paused', reason='Execution timeout')
    model = ReportingModel()
    bot = Chatbot(store, lambda *_: None, factory=lambda _: model)
    result = bot.progress(conversation, bot.work_facts(conversation))
    assert result['output'].startswith('独立入口验收超时')
    assert model.purposes == ['progress', 'progress', 'progress_final']
    ordinary = ConfiguredModel(object(), purpose='progress')
    final = ordinary.for_purpose('progress_final')
    assert 'tool_choice' not in ordinary.request_options
    assert final.request_options['tool_choice'] == 'none'
    assert final.max_output_tokens == ordinary.max_output_tokens
