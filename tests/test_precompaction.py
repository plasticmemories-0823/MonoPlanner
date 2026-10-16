"""工作预压缩的水位、并发尾部、暂停与持久恢复；只使用本地假模型。"""

from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from threading import Event, Thread
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict

from agent.model import compaction, precompaction
from agent.model.client import ConfiguredModel
from agent.model.context import admit
from agent.model.errors import ModelInterrupted, ModelRequestNotSent
from agent.model.tools import CODING_TOOLS
from agent.storage.records import save
from agent.task import control
from agent.task.budget import StepBudget


def pair(identity, reasoning=''):
    return [
        message_to_dict(AIMessage(
            content='Inspect source.', additional_kwargs={'reasoning_content': reasoning},
            tool_calls=[{'id': identity, 'name': 'submit_plan', 'args': {'kind': 'inspect'}}],
        )),
        message_to_dict(ToolMessage(content='Read complete.', tool_call_id=identity)),
    ]


class Transport:
    def __init__(self):
        self.started, self.release, self.closed = Event(), Event(), Event()
        self.requests = []
        self.error = None

    def invoke_with_progress(self, messages, on_reasoning):
        self.requests.append(deepcopy(messages))
        self.started.set()
        signal = control.current()
        while not self.release.wait(0.005):
            if signal.kind:
                raise ModelInterrupted(signal.kind)
        if self.error:
            raise self.error
        return AIMessage(
            content='A recorded observation remains relevant; acceptance is still pending.',
            response_metadata={'finish_reason': 'stop'},
            usage_metadata={'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30},
        )


class Loop:
    def __init__(self, root):
        self.root, self.rules, self.history_store = root, 'fixed rules', None
        self.root.mkdir(exist_ok=True)
        self.model = ConfiguredModel(SimpleNamespace(api_base='https://api.deepseek.com/beta'))
        self.tools, self.budget = CODING_TOOLS, StepBudget(max_steps=20)
        self.state = {
            'history': [message_to_dict(HumanMessage(content='原始要求不能改写')),
                        *pair('old', 'Evidence ' * 16000), *pair('recent', 'Recent facts.')],
            'rules': self.rules, 'request': '原始要求不能改写', 'acceptance': '固定验收',
            'verify_command': None, 'source': 'source-1',
        }
        self.events, self.saved = [], None

    def persist(self, event, payload=None):
        self.state['budget'] = asdict(self.budget)
        self.saved = deepcopy(self.state)
        self.events.append(event)
        if self.history_store:
            self.history_store.record(self.state, event, payload or {})


class PrecompactionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.loop = Loop(self.root)
        self.transport = Transport()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(
            precompaction, '_independent', return_value=(self.transport, self.transport.closed.set)
        ))
        self.stack.enter_context(patch.dict(compaction.CONFIG, {
            'retain_ratio': 0.00005,
            'deepseek_work': {'soft_input_tokens': 15000, 'hard_input_tokens': 60000},
        }))
        self.addCleanup(self.stop)

    def stop(self):
        self.transport.release.set()
        job = getattr(self.loop, '_precompaction_job', None)
        if job:
            job.worker.join(3)

    def start(self):
        precompaction.prepare(self.loop, 256000)
        self.assertTrue(self.transport.started.wait(2))

    def finish(self):
        self.transport.release.set()
        self.assertTrue(self.loop._precompaction_job.done.wait(3))

    def test_only_deepseek_endpoint_uses_fixed_watermarks(self):
        self.assertTrue(precompaction.enabled(self.loop.model))
        self.assertFalse(precompaction.enabled(ConfiguredModel(
            SimpleNamespace(api_base='https://example.org/deepseek')
        )))
        self.assertFalse(precompaction.enabled(SimpleNamespace(model_name='deepseek-flash')))
        with patch.object(compaction, 'prepare') as fallback:
            self.loop.model = ConfiguredModel(object())
            precompaction.prepare(self.loop, 256000, force=True)
        fallback.assert_called_once_with(self.loop, 256000, force=True, preserve_recent=True)

    def test_watermarks_are_decimal_input_limits_and_do_not_change_output_cap(self):
        with patch.dict(compaction.CONFIG, {'deepseek_work': {}}):
            self.assertEqual(precompaction.limits(), (192000, 256000))
        self.assertEqual(self.loop.model.max_output_tokens, 256000)
        with patch.object(precompaction, 'estimate_input_tokens', return_value=14999):
            precompaction.prepare(self.loop, 256000)
        self.assertFalse(self.transport.requests)
        with patch.object(precompaction, 'estimate_input_tokens', return_value=15000):
            self.start()
        self.assertEqual(self.loop.state['precompaction_pending']['output_cap'], 65536)

    def test_default_soft_boundary_keeps_original_summary_instruction(self):
        with patch.dict(compaction.CONFIG, {'deepseek_work': {}}):
            with patch.object(precompaction, 'estimate_input_tokens', return_value=191999):
                precompaction.prepare(self.loop, 256000)
            self.assertFalse(self.transport.requests)
            with patch.object(precompaction, 'estimate_input_tokens', return_value=192000):
                self.start()
        snapshot = self.loop.state['precompaction_pending']['execution_snapshot']
        self.assertEqual(
            self.transport.requests[0][-1].content, compaction.summary_instruction(snapshot)
        )
        self.assertNotIn('reasoning_lineage', snapshot)

    def test_default_hard_boundary_waits_and_keeps_new_tail(self):
        self.start()
        tail = pair('after-background-start', 'New facts remain exact.')
        self.loop.state['history'].extend(tail)
        finished, errors = Event(), []

        def boundary():
            try:
                with (
                    patch.dict(compaction.CONFIG, {'deepseek_work': {}}),
                    patch.object(precompaction, 'estimate_input_tokens',
                                 side_effect=[256000, 1000]),
                ):
                    precompaction.prepare(self.loop, 256000)
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = Thread(target=boundary)
        worker.start()
        self.assertFalse(finished.wait(0.05))
        self.transport.release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors, errors)
        self.assertEqual(compaction.history(self.loop.state, self.root)[-2:], tail)
        self.assertEqual(len(self.transport.requests), 1)

    def test_user_supplement_cancels_background_without_changing_work_history(self):
        with control.active(self.root, 'task') as signal:
            self.start()
            signal.interrupt('message', 'new-input')
            self.assertTrue(self.loop._precompaction_job.done.wait(3))
            with self.assertRaises(ModelInterrupted) as stopped:
                precompaction.prepare(self.loop, 256000)
            self.assertEqual(stopped.exception.kind, 'message')
            precompaction.close(self.loop)
        self.assertNotIn('context_view', self.loop.state)

    def test_soft_watermark_starts_once_without_waiting_or_overwriting_main_context(self):
        self.loop.state['context_usage'] = {'used_tokens': 42}
        original = deepcopy(self.loop.state['history'])
        self.start()
        precompaction.prepare(self.loop, 256000)
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(self.loop.state['history'], original)
        self.assertEqual(self.loop.state['context_usage'], {'used_tokens': 42})
        self.assertNotIn('request_pending', self.loop.state)
        self.assertEqual(self.loop.budget.used_steps, 1)
        self.assertGreater(self.loop.budget.inflight_reserved_tokens, 0)

    def test_appended_tail_is_unchanged_and_admission_rollover_is_authorized(self):
        options = {'max_tokens': 256000}
        admit(self.loop.state, compaction.messages(self.loop), self.loop.tools, options)
        self.start()
        tail = pair('new', 'This new reasoning must remain exact.')
        self.loop.state['history'].extend(tail)
        self.finish()
        precompaction.prepare(self.loop, 256000)
        self.assertEqual(compaction.history(self.loop.state, self.root)[-2:], tail)
        admit(self.loop.state, compaction.messages(self.loop), self.loop.tools, options)
        self.assertEqual(self.loop.state['last_admission']['generation'], 1)
        self.assertEqual(self.loop.budget.reported_tokens, 30)
        self.assertEqual(self.loop.budget.inflight_reserved_tokens, 0)
        self.assertEqual(self.loop.state['model_usage'][0]['purpose'], 'compaction')

    def test_original_prefix_or_rules_changes_block_adoption(self):
        self.start()
        self.finish()
        for key, value in [('rules', 'changed rules'), ('request', 'changed requirement')]:
            with self.subTest(key=key):
                previous = self.loop.state[key]
                self.loop.state[key] = value
                with self.assertRaises(ModelRequestNotSent):
                    precompaction.prepare(self.loop, 256000)
                self.loop.state[key] = previous
        self.loop.state['history'][0]['data']['content'] = 'edited prefix'
        with self.assertRaises(ModelRequestNotSent):
            precompaction.prepare(self.loop, 256000)
        self.assertNotIn('context_view', self.loop.state)

    def test_current_source_can_advance_while_snapshot_remains_historical(self):
        self.start()
        self.loop.state['source'] = 'source-2'
        self.finish()
        precompaction.prepare(self.loop, 256000)
        record = compaction.checked(self.root, self.loop.state['context_view']['record'])
        self.assertEqual(record['execution_snapshot']['source_version'], 'source-1')
        self.assertEqual(self.loop.state['source'], 'source-2')

    def test_hard_watermark_waits_before_next_work_request(self):
        self.start()
        self.loop.state['history'].extend(pair('large-tail', 'New tail. ' * 15000))
        finished, errors = Event(), []

        def boundary():
            try:
                precompaction.prepare(self.loop, 256000)
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = Thread(target=boundary)
        worker.start()
        self.assertFalse(finished.wait(0.05))
        self.transport.release.set()
        worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(errors, errors)
        self.assertTrue(self.loop.state['precompaction_waits'])
        self.assertLess(estimate(self.loop), 60000)

    def test_pausing_parent_cancels_only_child_and_keeps_raw_history(self):
        original = deepcopy(self.loop.state['history'])
        with control.active(self.root, 'task') as signal:
            self.start()
            signal.interrupt('pause')
            precompaction.close(self.loop)
        self.assertTrue(self.transport.closed.is_set())
        self.assertEqual(self.loop.state['history'], original)
        self.assertNotIn('precompaction_pending', self.loop.state)
        self.assertEqual(self.loop.state['precompaction_cancelled'][0]['status'], 'interrupted')
        self.assertEqual(self.loop.budget.unknown_usage_steps, 1)
        self.assertNotIn('context_view', self.loop.state)

    def test_terminal_cleanup_cancels_background_without_interrupting_main_signal(self):
        with control.active(self.root, 'task') as signal:
            self.start()
            precompaction.close(self.loop)
            self.assertIsNone(signal.kind)
        self.assertTrue(self.transport.closed.is_set())

    def test_unconfirmed_stop_stays_blocked_after_late_response_and_restart(self):
        self.start()
        reserved = deepcopy(self.loop.saved)
        self.finish()
        pending = self.loop.state['precompaction_pending']
        job = self.loop._precompaction_job
        job.done = Event()
        job.worker = SimpleNamespace(ident=1, join=lambda timeout: None)
        with self.assertRaises(ModelRequestNotSent) as interrupted:
            precompaction.close(self.loop)
        self.assertEqual(interrupted.exception.status, 'model_request_unknown')
        marker = self.loop.root / pending['directory'] / 'stop-unconfirmed.json'
        self.assertTrue(marker.exists())
        self.loop._precompaction_job = None

        restored = Loop(self.loop.root)
        restored.state = reserved
        restored.budget = StepBudget(**reserved['budget'])
        with self.assertRaises(ModelRequestNotSent) as stopped:
            precompaction.reconcile(restored)
        self.assertEqual(stopped.exception.status, 'model_request_unknown')
        self.assertEqual(restored.budget.reported_tokens, 30)
        self.assertEqual(restored.budget.inflight_reserved_tokens, 0)
        with self.assertRaises(ModelRequestNotSent):
            precompaction.prepare(restored, 256000)
        self.assertEqual(restored.budget.reported_tokens, 30)
        self.assertEqual(len(self.transport.requests), 1)
        self.assertTrue(marker.exists())
        self.assertNotIn('context_view', restored.state)

    def test_saved_response_is_reused_once_after_process_restart(self):
        self.start()
        reserved_state = deepcopy(self.loop.saved)
        self.finish()
        restored = Loop(self.root)
        restored.state = reserved_state
        restored.budget = StepBudget(**reserved_state['budget'])
        precompaction.prepare(restored, 256000)
        precompaction.prepare(restored, 256000)
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(restored.budget.reported_tokens, 30)
        self.assertEqual(len(restored.state['model_usage']), 1)

    def test_response_saved_before_outcome_journal_is_reused(self):
        self.start()
        reserved_state = deepcopy(self.loop.saved)
        self.finish()
        pending = self.loop.state['precompaction_pending']
        (self.root / pending['directory'] / 'outcome.json').unlink()
        restored = Loop(self.root)
        restored.state = reserved_state
        restored.budget = StepBudget(**reserved_state['budget'])
        precompaction.prepare(restored, 256000)
        self.assertEqual(restored.budget.reported_tokens, 30)
        self.assertEqual(len(self.transport.requests), 1)

    def test_unusable_orphan_ai_responses_settle_once_without_resending(self):
        cases = {
            'length': {'content': 'Incomplete summary.',
                       'response_metadata': {'finish_reason': 'length'}},
            'empty': {'content': ''},
            'tool': {'content': '', 'tool_calls': [
                {'id': 'unexpected', 'name': 'submit_plan', 'args': {'kind': 'inspect'}},
            ]},
        }
        for name, changes in cases.items():
            with self.subTest(summary=name):
                self.loop = Loop(self.root / name)
                self.start()
                reserved = deepcopy(self.loop.saved)
                self.finish()
                pending = self.loop.state['precompaction_pending']
                response_path = self.loop.root / 'requests/call-0001/response.json'
                raw = json.loads(response_path.read_text('utf-8'))
                raw['data'].update(changes)
                save(response_path, raw)
                (self.loop.root / pending['directory'] / 'outcome.json').unlink()
                calls = len(self.transport.requests)

                restored = Loop(self.loop.root)
                restored.state = reserved
                restored.budget = StepBudget(**reserved['budget'])
                precompaction.prepare(restored, 256000)
                self.assertEqual(restored.budget.reported_tokens, 30)
                self.assertEqual(restored.budget.inflight_reserved_tokens, 0)
                self.assertEqual(len(restored.state['model_usage']), 1)
                self.assertEqual(restored.state['precompaction_pending']['status'], 'failed')
                self.assertNotIn('context_view', restored.state)

                resumed = Loop(self.loop.root)
                resumed.state = deepcopy(restored.saved)
                resumed.budget = StepBudget(**resumed.state['budget'])
                precompaction.prepare(resumed, 256000)
                with self.assertRaises(ModelRequestNotSent) as stopped:
                    precompaction.prepare(resumed, 256000, force=True)
                self.assertEqual(stopped.exception.status, 'compaction_failed')
                self.assertEqual(resumed.budget.reported_tokens, 30)
                self.assertEqual(resumed.budget.inflight_reserved_tokens, 0)
                self.assertEqual(len(resumed.state['model_usage']), 1)
                self.assertEqual(len(self.transport.requests), calls)

    def test_unknown_request_is_not_resent_or_double_counted(self):
        self.start()
        pending = deepcopy(self.loop.state['precompaction_pending'])
        self.transport.error = TimeoutError('lost response')
        self.finish()
        precompaction.prepare(self.loop, 256000)
        self.assertEqual(len(self.transport.requests), 1)
        for _ in range(2):
            with self.assertRaisesRegex(ModelRequestNotSent, 'no automatic resend'):
                precompaction.prepare(self.loop, 256000, force=True)
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(self.loop.budget.unknown_usage_steps, 1)
        self.assertEqual(self.loop.budget.unknown_reserved_tokens, pending['reservation'])

    def test_force_does_not_remove_recent_raw_tool_pair(self):
        self.transport.release.set()
        recent = deepcopy(self.loop.state['history'][-2:])
        precompaction.prepare(self.loop, 256000, force=True)
        self.assertEqual(compaction.history(self.loop.state, self.root)[-2:], recent)

    def test_protected_tail_alone_cannot_be_forced_out(self):
        self.loop.state['history'] = pair('single', 'protected ' * 30000)
        with patch.dict(compaction.CONFIG, {'retain_ratio': 0.9}):
            with self.assertRaisesRegex(ModelRequestNotSent, 'No complete old span'):
                precompaction.prepare(self.loop, 256000, force=True)
        self.assertFalse(self.transport.requests)

    def test_protected_tail_at_soft_watermark_can_continue_without_compaction(self):
        self.loop.state['history'] = pair('single', 'protected ' * 8000)
        with patch.dict(compaction.CONFIG, {'retain_ratio': 0.9}):
            precompaction.prepare(self.loop, 256000)
        self.assertFalse(self.transport.requests)
        self.assertNotIn('context_view', self.loop.state)

    def test_saved_response_tampering_does_not_replace_history(self):
        self.start()
        self.finish()
        path = self.root / 'requests/call-0001/response.json'
        response = json.loads(path.read_text('utf-8'))
        response['data']['content'] = 'forged completion'
        save(path, response)
        with self.assertRaisesRegex(ModelRequestNotSent, 'response changed'):
            precompaction.prepare(self.loop, 256000)
        self.assertNotIn('context_view', self.loop.state)

    def test_interrupted_commit_does_not_replay_summary_or_replace_view(self):
        self.start()
        self.finish()
        original = self.loop.persist

        def crash(event, payload=None):
            original(event, payload)
            if event == 'compaction_commit_started':
                raise InterruptedError('checkpoint interrupted')

        with patch.object(self.loop, 'persist', crash):
            with self.assertRaises(InterruptedError):
                precompaction.prepare(self.loop, 256000)
        restored = Loop(self.root)
        restored.state = deepcopy(self.loop.saved)
        restored.budget = StepBudget(**restored.state['budget'])
        with self.assertRaises(ModelRequestNotSent) as stopped:
            precompaction.prepare(restored, 256000)
        self.assertEqual(stopped.exception.status, 'compaction_commit_unknown')
        self.assertEqual(len(self.transport.requests), 1)
        self.assertEqual(restored.budget.reported_tokens, 30)

    def test_prepared_metadata_tampering_does_not_dispatch_another_request(self):
        self.start()
        self.finish()
        self.loop.state['precompaction_pending']['reservation'] += 1
        with self.assertRaisesRegex(ModelRequestNotSent, 'metadata changed'):
            precompaction.prepare(self.loop, 256000)
        self.assertEqual(len(self.transport.requests), 1)

    def test_background_does_not_change_checkpoint_before_safe_boundary(self):
        self.start()
        reserved_state = deepcopy(self.loop.state)
        reserved_budget = asdict(self.loop.budget)
        self.finish()
        self.assertEqual(self.loop.state, reserved_state)
        self.assertEqual(asdict(self.loop.budget), reserved_budget)
        self.assertTrue((self.root / 'requests/call-0001/response.json').exists())

    def test_saved_summary_is_deduplicated_and_verified_in_durable_checkpoints(self):
        from agent.storage.history import LocalHistory

        store = LocalHistory(self.root)
        self.addCleanup(store.close)
        self.loop.history_store = store
        with store.lease():
            self.start()
            self.finish()
            self.assertTrue(precompaction.reconcile(self.loop))
            self.loop.persist('another_safe_boundary')
            loaded = store.load()
            row = store.connection.execute(
                'SELECT path FROM commits ORDER BY seq DESC LIMIT 1'
            ).fetchone()
        saved = json.loads((self.root / row[0]).read_text('utf-8'))['state']
        self.assertNotIn('response', saved['precompaction_pending'])
        self.assertIn('response_ref', saved['precompaction_pending'])
        self.assertEqual(
            loaded['precompaction_pending']['response'],
            self.loop.state['precompaction_pending']['response'],
        )


def estimate(loop):
    from agent.model.payload import estimate_input_tokens

    return estimate_input_tokens(compaction.messages(loop), loop.tools)


class DurableWorkPrecompactionTests(unittest.TestCase):
    def test_exception_cancels_background_before_releasing_task_lease(self):
        from agent.execution.local import LocalEnvironment
        from agent.storage import recovery

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / 'repo'
            repo.mkdir()
            loop = Loop(root / 'state')
            loop.environment = LocalEnvironment(repo, artifact_root=root / 'runtime')
            loop.task_token_limit, loop.task_timeout = None, 7200
            loop.explicit_state, loop.artifacts_root = loop.root, root
            loop.persistence = True
            events = []
            done, started = Event(), Event()
            signal = control.Signal()

            def background():
                started.set()
                with signal.subscribe(done.set):
                    done.wait(3)

            worker = Thread(target=background)
            job = SimpleNamespace(worker=worker, signal=signal, done=done)

            def crash(*args):
                loop._precompaction_job = job
                worker.start()
                self.assertTrue(started.wait(2))
                raise InterruptedError('controlled failure while summary is running')

            loop._run_new = crash

            class History:
                @contextmanager
                def lease(self):
                    events.append('lease_acquired')
                    try:
                        yield
                    finally:
                        events.append('lease_released')

                def load(self):
                    return None

                def close(self):
                    events.append('history_closed')

            original_close = precompaction.close

            def joined(owner, *, settle=True):
                original_close(owner, settle=settle)
                self.assertFalse(worker.is_alive())
                events.append('background_cancel_and_join')

            with (
                patch('agent.storage.history.LocalHistory', return_value=History()),
                patch('agent.source.lifecycle.close'),
                patch.object(precompaction, 'close', side_effect=joined),
                self.assertRaises(InterruptedError),
            ):
                recovery.run(loop, [], 'task', StepBudget(), None, None)
            self.assertLess(events.index('background_cancel_and_join'),
                            events.index('lease_released'))
            self.assertLess(events.index('lease_released'), events.index('history_closed'))

    def test_task_runner_compacts_then_preserves_frozen_acceptance_and_reuses_result(self):
        from agent.execution.local import LocalEnvironment
        from agent.storage.history import LocalHistory
        from agent.task.runner import TaskRunner
        from support import python_command
        from test_compaction import SDK
        from test_task_runner import CODE

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / 'repo'
            repo.mkdir()
            (repo / 'sample.py').write_text('def value():\n    return 0\n')
            environment = LocalEnvironment(repo, artifact_root=root / 'runtime')
            sdk = SDK([
                {'kind': 'inspect', 'note': 'read', 'steps': [
                    {'id': 'r', 'op': 'read', 'path': 'sample.py'},
                ]},
                {'kind': 'inspect', 'note': 'reread', 'steps': [
                    {'id': 'r', 'op': 'read', 'path': 'sample.py', 'query': 'reread: check'},
                ]},
            ])
            model = ConfiguredModel(sdk)

            def engine(**options):
                return TaskRunner(model, environment, state_dir=root / 'state', **options)

            first = engine(pause_after_steps=2).run([], 'value must return four', StepBudget())
            self.assertEqual(first['status'], 'paused', first)
            sdk.api_base = 'https://api.deepseek.com/beta'
            sdk.plans = iter([{
                'kind': 'candidate', 'note': 'implement fix', 'steps': [{
                    'id': 'fix', 'op': 'shell', 'command': python_command(
                        "from pathlib import Path; "
                        "Path('sample.py').write_text('def value():\\n    return 4\\n')"
                    ),
                }], 'test_code': CODE,
            }])
            transport = Transport()
            transport.release.set()
            with (
                patch.object(precompaction, '_independent',
                             return_value=(transport, transport.closed.set)),
                patch.dict(compaction.CONFIG, {
                    'retain_ratio': 0.00005,
                    'deepseek_work': {'soft_input_tokens': 15000, 'hard_input_tokens': 30000},
                }),
            ):
                result = engine().run([], 'value must return four', StepBudget())
                self.assertEqual(result['status'], 'verified_success', result)
                self.assertEqual(len(result['compactions']), 1)
                again = engine().run([], 'value must return four', StepBudget())
            self.assertTrue(again['result_reused'])
            self.assertEqual(len(transport.requests), 1)
            self.assertTrue(transport.closed.is_set())
            store = LocalHistory(root / 'state')
            try:
                with store.lease():
                    saved = store.load()
            finally:
                store.close()
            summaries = [row for row in saved['model_usage'] if row['purpose'] == 'compaction']
            self.assertEqual(len(summaries), 1)
            self.assertEqual(summaries[0]['usage']['total_tokens'], 30)
            self.assertFalse(saved.get('precompaction_pending'))
