"""所有人工续跑入口共享停止身份；恢复不重放、不清零，不通过聊天模型判定授权。"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import unittest
import uuid

import pytest

from agent.application.store import ApplicationStore, Conflict
from agent.task.budget import StepBudget
import test_task_runner as fixtures


@pytest.fixture
def stopped_work(tmp_path, monkeypatch):
    root, repo = tmp_path / 'state', tmp_path / 'repo'
    repo.mkdir()
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(root))
    store = ApplicationStore(root)
    conversation = store.create(repo=str(repo))['thread_id']
    message = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Implement the original task.'}
    store.submit(conversation, message)
    job = store.create_work(conversation, message['id'])
    store.command_state(conversation, message['id'], 'applied')
    return store, conversation, job


@pytest.mark.parametrize('status', [
    'task_blocked', 'task_cancelled', 'stalled', 'preparation_failed', 'budget_exhausted',
    'work_budget_exhausted', 'token_budget_exhausted', 'task_timeout', 'runtime_error',
    'paused', 'awaiting_user', 'model_request_unknown', 'execution_unknown',
    'web_request_unknown', 'compaction_commit_unknown', 'verified_success', 'existing_complete',
])
def test_stopped_work_exposes_bound_continuation(stopped_work, status):
    store, conversation, job = stopped_work
    store.work_state(job['task_id'], status, reason='Saved stop reason')
    snapshot = store.snapshot(conversation)
    continuation = snapshot['work']['continuation']
    assert continuation['available'] and continuation['stop_id']
    assert continuation['mode'] == ('review' if status in {'verified_success', 'existing_complete'} else 'continue')
    result = store.submit(conversation, {
        'id': str(uuid.uuid4()), 'kind': 'resume', 'resume_token': continuation['stop_id'],
    })
    assert result['received']
    assert store.get(conversation)['work_id'] == job['task_id']


def test_same_stop_coalesces_clicks_and_rejects_old_stop(stopped_work):
    store, conversation, job = stopped_work
    store.work_state(job['task_id'], 'task_blocked')
    token = store.snapshot(conversation)['work']['continuation']['stop_id']
    commands = [{'id': str(uuid.uuid4()), 'kind': 'resume', 'resume_token': token} for _ in range(8)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda value: store.submit(conversation, value), commands))
    assert sum(not row['applied'] for row in results) == 1
    store.work_state(job['task_id'], 'running')
    store.work_state(job['task_id'], 'task_blocked')
    assert store.snapshot(conversation)['work']['continuation']['stop_id'] != token
    with pytest.raises(Conflict):
        store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'resume', 'resume_token': token})


@pytest.mark.parametrize('text,expected', [
    ('继续', 'resume'), ('继续任务。', 'resume'), ('继续复查', 'resume'),
    ('为什么停了', 'message'), ('不要继续', 'message'), ('文档里写着“继续”', 'message'),
    ('继续，并把原要求改掉', 'message'),
])
def test_explicit_chat_continue_is_a_program_command(stopped_work, text, expected):
    store, conversation, job = stopped_work
    store.work_state(job['task_id'], 'task_blocked')
    command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': text}
    store.submit(conversation, command)
    saved = next(row for row in store.commands(conversation=conversation) if row['id'] == command['id'])
    assert saved['kind'] == expected
    assert store.snapshot(conversation)['messages'][-1]['content'] == text


class NativeContinuationTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    loop = fixtures.TaskRunnerTests.loop
    candidate = fixtures.TaskRunnerTests.candidate
    shell = fixtures.TaskRunnerTests.shell

    def test_unknown_web_attempt_can_continue_without_replaying_or_releasing_fee(self):
        from unittest.mock import patch
        from agent.task.runner import TaskRunner
        from langchain_core.messages import AIMessage
        from tests.fixtures.models import ScriptedModel
        class Crash(BaseException):
            pass
        directory = self.root / 'web-state'
        reply = AIMessage(content='', tool_calls=[{'id': 'lookup', 'name': 'web_search',
            'args': {'query': 'public documentation'}}],
            usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2})
        def crash(query, loop, folder):
            ticket = loop.budget.reserve_service('web_search')
            loop.state['web_request_pending'] = {
                'kind': 'service', 'call': ticket.call_id, 'purpose': 'web_search',
                'reserved_cost_micros': ticket.reserved_cost_micros,
            }
            loop.persist('web_request_started')
            raise Crash()
        first = TaskRunner(ScriptedModel([reply]), self.env, state_dir=directory,
                           runtime_settings={'online': True})
        with patch('agent.execution.web.WebProvider.search', side_effect=crash), self.assertRaises(Crash):
            first.run([], 'value returns 4', StepBudget(max_steps=10))
        resumed = TaskRunner(ScriptedModel([fixtures.reply({'kind': 'blocked', 'note': 'Continue inspected'})]),
                             self.env, state_dir=directory, runtime_settings={'online': True},
                             recovery_id='explicit-web-continue')
        with patch('agent.execution.web.WebProvider.search', side_effect=AssertionError('replayed')):
            result = resumed.run([], 'value returns 4', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'task_cancelled', result)
        self.assertEqual(resumed.budget.unknown_service_cost_micros, 5000)
        self.assertEqual(resumed.budget.inflight_service_cost_micros, 0)
        self.assertEqual(resumed.budget.used_steps, 2)
        self.assertIsNone(resumed.state['web_request_pending'])
        pairs = [m for m in resumed.state['history'] if m['type'] == 'tool'
                 and m['data'].get('tool_call_id') == 'lookup']
        self.assertEqual(len(pairs), 1)

    def test_cancelled_budget_requires_explicit_continue_and_keeps_history(self):
        directory = self.root / 'cancelled'
        budget = StepBudget(max_steps=10)
        budget.request_cancel('User stopped this task')
        first = self.loop([], state_dir=directory)
        self.assertEqual(first.run([], 'value returns 4', budget)['status'], 'task_cancelled')
        passive = self.loop([], state_dir=directory)
        self.assertEqual(passive.run([], 'value returns 4', StepBudget(max_steps=10))['status'],
                         'task_cancelled')
        resumed = self.loop([{'kind': 'blocked', 'note': 'Further investigation'}],
                            state_dir=directory, recovery_id='human-continue')
        result = resumed.run([], 'value returns 4', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'task_cancelled')
        self.assertEqual(resumed.state['continuation_history'][0]['cancel_reason'],
                         'User stopped this task')
        self.assertEqual(resumed.budget.used_steps, 1)

    def test_exhausted_budget_adds_original_batch_and_duplicate_does_not_dispatch(self):
        directory = self.root / 'quota'
        plan = {'kind': 'inspect', 'note': 'Read the entry', 'steps': [
            {'id': 'read', 'op': 'read', 'path': 'sample.py'}]}
        first = self.loop([plan], state_dir=directory)
        self.assertEqual(first.run([], 'value returns 4', StepBudget(max_steps=1))['status'],
                         'budget_exhausted')
        for count in (1, 2):
            next_loop = self.loop([{'kind': 'blocked', 'note': 'Another investigation'}],
                                 state_dir=directory, recovery_id=f'continue-{count}')
            next_loop.run([], 'value returns 4', StepBudget(max_steps=1))
            self.assertEqual(next_loop.budget.max_steps, 1 + count)
            self.assertEqual(next_loop.budget.used_steps, 1 + count)
        duplicate = self.loop([], state_dir=directory, recovery_id='continue-2')
        duplicate.run([], 'value returns 4', StepBudget(max_steps=1))
        self.assertEqual(self.model.calls, [])
        self.assertEqual(duplicate.budget.max_steps, 3)

    def test_stale_native_stop_does_not_call_model(self):
        directory = self.root / 'stale'
        first = self.loop([{'kind': 'blocked', 'note': 'Investigate'}], state_dir=directory)
        first.run([], 'value returns 4', StepBudget(max_steps=10))
        next_loop = self.loop([], state_dir=directory, control_source=lambda: [
            {'id': 'stale', 'kind': 'resume', 'native_stop_id': 'another-stop'}])
        with self.assertRaisesRegex(ValueError, 'stop identity'):
            next_loop.run([], 'value returns 4', StepBudget(max_steps=10))
        self.assertEqual(self.model.calls, [])

    def test_blocked_continues_same_history_and_cancellation_is_cleared(self):
        directory = self.root / 'state'
        first = self.loop([{'kind': 'blocked', 'note': 'Proof not found yet'}], state_dir=directory)
        result = first.run([], 'value returns 4', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'task_cancelled')
        original = deepcopy(first.state['history'])
        second = self.loop([{'kind': 'blocked', 'note': 'A new investigation stopped'}],
                           state_dir=directory, control_source=lambda: [{'id': 'continue-1', 'kind': 'resume'}])
        result = second.run([], 'value returns 4', StepBudget(max_steps=10))
        self.assertEqual(len(self.model.calls), 1)
        self.assertEqual(second.state['history'][:len(original)], original)
        self.assertEqual(second.budget.used_steps, 2)
        self.assertFalse(second.budget.cancelled)
        self.assertEqual(len(second.state['continuation_history']), 1)

    def test_success_continue_reviews_without_replaying_previous_edit(self):
        directory = self.root / 'state'
        first = self.loop([self.candidate(4)], state_dir=directory)
        result = first.run([], 'value returns 4', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'verified_success')
        original_rounds = deepcopy(first.state['rounds'])
        receipts = {str(p): p.read_bytes() for p in directory.glob('plans/*/*/completion.json')}
        second = self.loop([{'kind': 'blocked', 'note': 'Review needs more investigation'}],
                           state_dir=directory, control_source=lambda: [{'id': 'review-1', 'kind': 'resume'}])
        result = second.run([], 'value returns 4', StepBudget(max_steps=10))
        self.assertEqual(len(self.model.calls), 1)
        self.assertFalse(result['result_reused'])
        self.assertEqual(second.state['rounds'], original_rounds)
        self.assertTrue(all(__import__('pathlib').Path(p).read_bytes() == raw for p, raw in receipts.items()))
        self.assertEqual(second.state['continuation_history'][-1]['mode'], 'review')


@pytest.mark.parametrize('field,used_field,initial,used,expected', [
    ('max_steps', 'used_steps', 10, 10, 20),
    ('max_tokens', 'reported_tokens', 100, 100, 200),
    ('max_steps', 'used_steps', 10, 3, 10),
])
def test_budget_grant_only_extends_exhausted_dimension(field, used_field, initial, used, expected):
    from agent.task.resumption import grant_budget
    budget = StepBudget(**{field: initial})
    setattr(budget, used_field, used)
    state = {'budget': dict(vars(budget)), 'elapsed_seconds': 0}
    grant_budget(state, budget, 'explicit-resume')
    assert getattr(budget, field) == expected and getattr(budget, used_field) == used
    grant_budget(state, budget, 'explicit-resume')
    assert getattr(budget, field) == expected


def test_confirmed_storage_limits_and_unknown_reservations_are_not_reset():
    from agent.task.resumption import grant_budget
    budget = StepBudget(max_steps=20, max_tokens=100, max_service_cost_micros=0)
    budget.unknown_reserved_tokens = 80
    budget.unknown_usage_steps = 1
    budget.reported_tokens = 10
    state = {'status': 'token_budget_exhausted', 'budget': dict(vars(budget))}
    grant_budget(state, budget, 'one')
    assert budget.max_tokens == 200
    assert budget.max_steps == 20 and budget.max_service_cost_micros == 0
    assert budget.unknown_reserved_tokens == 80 and budget.unknown_usage_steps == 1


def test_v1_application_data_is_backed_up_before_resume_command_migration(tmp_path):
    import sqlite3
    from agent.storage.schema import initialize, DATA_VERSION
    path = tmp_path / 'application.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE app_commands (id TEXT)')
        db.execute("INSERT INTO app_commands VALUES ('original')")
        db.execute('PRAGMA user_version=1')
    initialize(path)
    with sqlite3.connect(path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == DATA_VERSION
        assert db.execute('SELECT id,resume_token FROM app_commands').fetchall() == [('original', None)]
    assert len(list((tmp_path / 'data-backups').glob('*.sqlite3'))) == 1


def test_model_free_chat_continue_does_not_reserve_title_or_chat_model(stopped_work):
    from agent.application.scheduler import Scheduler
    from unittest.mock import patch
    store, conversation, job = stopped_work
    store.work_state(job['task_id'], 'task_blocked')
    scheduler = Scheduler(store)
    with patch.object(scheduler, '_reserve_title') as title:
        scheduler.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '继续'})
    title.assert_not_called()
    assert not store.commands(conversation=conversation, states=('running',))
    assert store.commands(task=job['task_id'])[-1]['kind'] == 'resume'


def test_preparation_failure_can_be_repaired_without_deleting_original(tmp_path):
    from unittest.mock import patch
    from agent.application import work_bridge as bridge
    from agent.execution.projects import EnvironmentRequired
    from test_work_bridge import OfflineEnvironment, RecordedModel
    repo, root = tmp_path / 'repo', tmp_path / 'state'
    repo.mkdir()
    job = {'task_id': '7' * 32, 'repo': str(repo), 'request': 'Inspect this repository'}
    binding = {'image': 'offline', 'runtime': {}}
    with patch.object(bridge, 'prepare_binding', side_effect=EnvironmentRequired('Missing environment')):
        result = bridge.run_work(job, lambda *_: None, lambda: [], model=RecordedModel(),
                                 root=root, environment_factory=OfflineEnvironment)
    assert result['status'] == 'preparation_failed'
    failure = root / 'tasks' / job['task_id'] / 'preparation-failed.json'
    original = failure.read_bytes()
    model = RecordedModel()
    with patch.object(bridge, 'prepare_binding', return_value=binding) as prepare, patch.object(
        bridge, 'require_binding', side_effect=lambda repo, value: value,
    ):
        result = bridge.run_work(job, lambda *_: None,
                                 lambda: [{'id': 'retry-prepare', 'kind': 'resume'}],
                                 root=root, model=model, environment_factory=OfflineEnvironment)
    assert result['status'] == 'task_cancelled'
    assert len(model.calls) == 1 and prepare.call_args.kwargs['retry'] is True
    assert any(p.read_bytes() == original for p in failure.parent.glob('preparation-failures/*.json'))


def test_completed_existing_command_review_only_reads_current_source(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch
    from agent.application.work_bridge import _existing
    from agent.execution.local import LocalEnvironment
    from agent.storage.records import save
    repo, directory = tmp_path / 'repo', tmp_path / 'task'
    repo.mkdir()
    (repo / 'effect.txt').write_text('one side effect')
    old = {'status': 'completed', 'arguments': {},
           'result': {'status': 'existing_complete', 'existing_result': {'exit_code': 0}}}
    save(directory / 'existing/state.json', old)
    environment = LocalEnvironment(repo, artifact_root=tmp_path / 'execution')
    with patch('agent.server.assistant.validate'), patch('agent.execution.existing.execute') as execute:
        result = _existing({'arguments': {}}, environment, directory, {}, lambda *_: None, 'review')
    execute.assert_not_called()
    assert result['review']['commands_executed'] == 0
    assert result['previous_result'] == old['result'] and result['status'] == 'awaiting_user'
    assert (repo / 'effect.txt').read_text() == 'one side effect'
    assert json.loads((directory / 'existing/state.json').read_bytes()) == old


def test_storage_fault_can_only_be_released_after_explicit_idle_write_probe(stopped_work):
    from agent.application.scheduler import Scheduler
    store, conversation, job = stopped_work
    store.work_state(job['task_id'], 'task_blocked')
    scheduler = Scheduler(store)
    scheduler.storage_failed('fixture', OSError('full'))
    assert scheduler.storage_error()
    scheduler.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'message', 'text': '继续'})
    assert scheduler.storage_error() is None
    assert store.commands(task=job['task_id'])[-1]['kind'] == 'resume'


def test_scheduler_rejects_continue_for_a_stop_that_changed_before_dispatch(stopped_work):
    from agent.application.scheduler import Scheduler
    from unittest.mock import patch
    store, conversation, job = stopped_work
    store.work_state(job['task_id'], 'task_blocked')
    command = {'id': str(uuid.uuid4()), 'kind': 'resume'}
    store.submit(conversation, command)
    store.work_state(job['task_id'], 'running')
    store.work_state(job['task_id'], 'task_blocked')
    scheduler = Scheduler(store)
    with patch.object(scheduler, '_schedule_work') as dispatch:
        scheduler._tick()
    dispatch.assert_not_called()
    assert store.commands(task=job['task_id'], states=('failed',))[-1]['id'] == command['id']
