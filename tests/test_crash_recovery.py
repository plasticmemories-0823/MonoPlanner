"""在原件落盘与索引提交之间断电，恢复不能重复已发生的操作。"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.storage.history import LocalHistory
from agent.storage.records import save
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
import test_task_runner as f


class PowerLoss(BaseException):
    pass


class CrashRecoveryTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp
    candidate = f.TaskRunnerTests.candidate
    shell = f.TaskRunnerTests.shell

    def run_plans(self, plans, **options):
        self.model = f.Model(plans)
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / 'state', **options)
        return self.engine.run([], 'return 4', StepBudget(max_steps=8))

    def crash_after_save(self, module, predicate):
        def interrupted(path, data):
            save(path, data)
            if predicate(Path(path), data):
                raise PowerLoss()
        return patch(module + '.save', side_effect=interrupted)

    def test_completed_effect_before_sql_commit_is_not_executed_again(self):
        plan = self.candidate()
        plan['steps'].insert(0, self.shell(
            "from pathlib import Path; p=Path('counter'); "
            "p.write_text(str(int(p.read_text())+1 if p.exists() else 1))", 'counter',
        ))
        with self.crash_after_save('agent.task.operations', lambda path, data:
                                  path.name == 'result.json' and data.get('id') == 'counter'):
            with self.assertRaises(PowerLoss):
                self.run_plans([plan])
        self.assertEqual((self.repo / 'counter').read_text(), '1')
        result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual((self.repo / 'counter').read_text(), '1')
        self.assertFalse(self.model.calls)

    def test_saved_response_before_checkpoint_is_consumed_and_settled_once(self):
        with self.crash_after_save('agent.task.runner', lambda path, data:
                                  path.name == 'response.json' and any(
                                      call['name'] == 'submit_plan'
                                      for call in data.get('data', {}).get('tool_calls', []))):
            with self.assertRaises(PowerLoss):
                self.run_plans([self.candidate()])
        result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(self.engine.budget.unknown_usage_steps, 0)
        self.assertEqual(self.engine.budget.inflight_reserved_tokens, 0)
        calls = self.engine.state['model_usage']
        self.assertEqual(len({row['call'] for row in calls}), len(calls))

    def test_final_result_before_checkpoint_does_not_repeat_tests(self):
        with self.crash_after_save('agent.task.operations', lambda path, data:
                                  path.name == 'result.json' and path.parent.parent.name == 'feedback'):
            with self.assertRaises(PowerLoss):
                self.run_plans([self.candidate()])
        with patch('agent.task.operations.verify_final', side_effect=AssertionError('repeated')):
            result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(result['rounds']), 1)
        self.assertFalse(self.model.calls)

    def test_index_lost_after_durable_checkpoint_is_rebuilt(self):
        self.run_plans([self.candidate()])
        (self.root / 'state/history.sqlite3').unlink()
        result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success')
        self.assertTrue(result['result_reused'])
        self.assertFalse(self.model.calls)

    def test_corrupt_sql_index_is_archived_and_rebuilt_from_verified_journal(self):
        self.run_plans([self.candidate()])
        database = self.root / 'state/history.sqlite3'
        database.write_bytes(b'corrupted SQLite index')
        result = self.run_plans([])
        self.assertTrue(result['result_reused'])
        backups = list(database.parent.glob('history-corrupt-*.sqlite3'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), b'corrupted SQLite index')

    def test_durable_checkpoint_before_index_commit_is_reconciled(self):
        original = LocalHistory._index_checkpoint

        def crash(archive, number, reference, value):
            if value['event'] == 'step_finished':
                raise PowerLoss()
            return original(archive, number, reference, value)

        with patch.object(LocalHistory, '_index_checkpoint', crash), self.assertRaises(PowerLoss):
            self.run_plans([self.candidate()])
        with patch('agent.execution.executor.OperationExecutor.execute',
                   side_effect=AssertionError('completed command repeated')):
            result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertGreater(result['local_state_stats']['reconciled_checkpoints'], 0)

    def test_process_receipt_survives_parent_death_before_step_result(self):
        from agent.execution.local import LocalEnvironment

        original = LocalEnvironment.run_process

        def crash(environment, *args, **kwargs):
            result = original(environment, *args, **kwargs)
            if 'plan-steps' in str(kwargs['stdout_path']):
                raise PowerLoss()
            return result

        with patch.object(LocalEnvironment, 'run_process', crash), self.assertRaises(PowerLoss):
            self.run_plans([self.candidate()])
        with patch('agent.execution.executor.OperationExecutor.execute',
                   side_effect=AssertionError('completed command repeated')):
            result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success', result)

    def test_tampered_result_cannot_advance_the_cursor(self):
        with self.crash_after_save('agent.task.operations', lambda path, data:
                                  path.name == 'result.json' and 'op' in data):
            with self.assertRaises(PowerLoss):
                self.run_plans([self.candidate()])
        path = next((self.root / 'state/plans').glob('*/*/result.json'))
        value = json.loads(path.read_text('utf-8'))
        value['after'] = 'invented-source'
        save(path, value)
        result = self.run_plans([])
        self.assertEqual(result['status'], 'state_conflict')
        self.assertFalse(self.model.calls)

    def test_baseline_trial_result_is_reused_without_running_tests_again(self):
        with self.crash_after_save('agent.acceptance.revisions', lambda path, data:
                                  path.name == 'baseline-result.json'):
            with self.assertRaises(PowerLoss):
                self.run_plans([self.candidate()])
        with patch('agent.acceptance.revisions.run_report', side_effect=AssertionError('trial repeated')):
            result = self.run_plans([])
        self.assertEqual(result['status'], 'verified_success', result)

    def test_partial_managed_edit_requires_inspection_and_does_not_repeat_command(self):
        from agent.execution.local import LocalEnvironment

        self.env.recover_partial_effects = True  # 离线替身模拟无网络、任务独占的受管环境。
        original = LocalEnvironment.run_process
        plan = self.candidate()

        def crash(environment, *args, **kwargs):
            result = original(environment, *args, **kwargs)
            if 'plan-steps' in str(kwargs['stdout_path']):
                Path(kwargs['stdout_path']).with_suffix('.completion.json').unlink()
                raise PowerLoss()
            return result

        with patch.object(LocalEnvironment, 'run_process', crash), self.assertRaises(PowerLoss):
            self.run_plans([plan])
        inspect = {'kind': 'inspect', 'note': 'inspect actual partial effects',
                   'steps': [{'id': 'read', 'op': 'read', 'path': 'sample.py'}]}
        remaining = {**plan, 'steps': []}
        result = self.run_plans([inspect, remaining], recovery_id='inspect-after-crash')
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(self.engine.state['uncertain_commands'], [plan['steps'][0]['command']])
        self.assertNotIn('recovery_inspection_required', self.engine.state)
        first_tool = self.model.calls[0][-1].content
        self.assertIn('unconfirmed', first_tool)

    def test_interrupted_isolated_acceptance_keeps_attempt_and_runs_original_contract(self):
        from support import interrupt_after_persist

        interrupt, Interrupted = interrupt_after_persist('final_acceptance_started')
        with interrupt, self.assertRaises(Interrupted):
            self.run_plans([self.candidate()])
        frozen = (self.root / 'state/frozen-acceptance.json').read_bytes()
        result = self.run_plans([], recovery_id='continue-original-acceptance')
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual((self.root / 'state/frozen-acceptance.json').read_bytes(), frozen)
        self.assertEqual(len(self.engine.state['interrupted_verifications']), 1)
        self.assertFalse(self.model.calls)

    def test_unknown_request_requires_explicit_continue_and_keeps_reservation(self):
        from support import interrupt_after_persist

        interrupt, Interrupted = interrupt_after_persist('model_request_started')
        with interrupt, self.assertRaises(Interrupted):
            self.run_plans([self.candidate()])
        result = self.run_plans([])
        self.assertEqual(result['status'], 'model_request_unknown')
        reserved = self.engine.budget.unknown_reserved_tokens
        result = self.run_plans([self.candidate()], recovery_id='user-continue-1')
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(self.engine.budget.unknown_usage_steps, 1)
        self.assertEqual(self.engine.budget.unknown_reserved_tokens, reserved)

    def test_crash_while_capturing_acceptance_does_not_poison_the_next_attempt(self):
        from agent.source.snapshots import SourceSnapshot

        original = SourceSnapshot.capture

        def capture(repo, destination, **options):
            snapshot = original(repo, destination, **options)
            if 'feedback' in str(destination):
                raise PowerLoss()
            return snapshot

        with patch.object(SourceSnapshot, 'capture', capture), self.assertRaises(PowerLoss):
            self.run_plans([self.candidate()])
        result = self.run_plans([], recovery_id='continue-after-snapshot')
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertFalse(self.model.calls)

    def test_orphan_acceptance_proposal_is_not_mistaken_for_an_already_run_trial(self):
        with self.crash_after_save('agent.acceptance.revisions', lambda path, data:
                                  path.name == 'proposal.json'):
            with self.assertRaises(PowerLoss):
                self.run_plans([self.candidate()])
        result = self.run_plans([], recovery_id='continue-proposal')
        self.assertEqual(result['status'], 'verified_success', result)


def test_confirmed_supplement_survives_restart_but_waits_for_explicit_continue(tmp_path):
    import uuid
    from agent.application.store import ApplicationStore
    from support import fixture_delivery

    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(tmp_path))['thread_id']
    initial = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'initial request'}
    store.submit(conversation, initial)
    work = store.create_work(conversation, initial['id'])
    store.work_state(work['task_id'], 'running')
    addition = {'id': str(uuid.uuid4()), 'kind': 'supplement', 'text': 'keep this requirement',
                'confirmed_requirements': fixture_delivery('keep this requirement')}
    store.submit(conversation, addition)
    store.recover_interrupted()
    assert not store.commands(task=work['task_id'])
    saved = store.commands(task=work['task_id'], states=('deferred',))
    assert [row['value'] for row in saved] == [addition]
    store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'resume'})
    assert store.commands(task=work['task_id'])[0]['value'] == addition
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM app_calls').fetchone()[0] == 0


def test_unconfirmed_supplement_is_retained_until_a_confirmed_proposal_includes_it(tmp_path):
    import uuid
    import pytest
    from agent.application.store import ApplicationStore, Conflict
    from agent.storage.records import digest
    from support import fixture_delivery

    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(tmp_path))['thread_id']
    initial = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'initial request'}
    store.submit(conversation, initial)
    work = store.create_work(conversation, initial['id'])
    store.work_state(work['task_id'], 'running')
    addition = {'id': str(uuid.uuid4()), 'kind': 'supplement', 'text': 'new required behavior'}
    store.submit(conversation, addition)
    store.recover_interrupted()
    with pytest.raises(Conflict, match='尚未确认'):
        store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'resume'})
    packet = fixture_delivery(addition['text'])
    packet['sources'][0]['id'] = addition['id']
    packet['targets'][0]['quotes'][0]['source_id'] = addition['id']
    packet['digest'] = digest({key: value for key, value in packet.items() if key != 'digest'})
    confirmation = {**addition, 'id': str(uuid.uuid4()), 'confirmed_requirements': packet}
    store.submit(conversation, confirmation)
    store.submit(conversation, {'id': str(uuid.uuid4()), 'kind': 'resume'})
    pending = store.commands(task=work['task_id'])
    assert pending[0]['value'] == confirmation
    assert store.commands(task=work['task_id'], states=('coalesced',))[0]['id'] == addition['id']

