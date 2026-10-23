"""无界面回调的内核同样遵守暂停、请求归档和用量结算边界。"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.model.client import ConfiguredModel
from agent.model.errors import ModelInterrupted
from agent.task import control
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from support import interrupt_after_persist
from test_model_interrupts import WaitingClient
import test_compaction as compaction_fixtures
import test_local_repair as repairs
import test_task_runner as fixtures


class KernelModelCancellationTests(unittest.TestCase):
    def test_invoke_without_display_checks_pause_before_calling_sdk(self):
        client = WaitingClient()
        with control.active('kernel', 'presend') as signal:
            signal.interrupt('pause')
            with self.assertRaises(ModelInterrupted) as caught:
                ConfiguredModel(client).invoke([])
        self.assertFalse(caught.exception.sent)
        self.assertFalse(client.calls)

    def test_pause_between_async_scheduling_and_sdk_dispatch_is_not_sent(self):
        calls = []

        class ImmediateClient:
            async def astream(self, messages, **options):
                calls.append(messages)
                yield AIMessageChunk(content='complete', response_metadata={'finish_reason': 'stop'})

        def pause_before_start(coroutine, signal):
            signal.interrupt('pause')
            return asyncio.run(coroutine)

        with control.active('kernel', 'scheduling'), patch(
            'agent.model.streaming.run', side_effect=pause_before_start
        ):
            with self.assertRaises(ModelInterrupted) as caught:
                ConfiguredModel(ImmediateClient()).invoke_with_progress([], lambda text: None)
        self.assertFalse(caught.exception.sent)
        self.assertFalse(calls)

    def test_cancelled_queued_coroutine_does_not_count_as_sent(self):
        client = WaitingClient()

        def cancel_while_queued(coroutine, signal):
            coroutine.close()
            signal.interrupt('pause')
            raise ModelInterrupted('pause')

        with control.active('kernel', 'queued'), patch(
            'agent.model.streaming.run', side_effect=cancel_while_queued
        ):
            with self.assertRaises(ModelInterrupted) as caught:
                ConfiguredModel(client).invoke([])
        self.assertFalse(caught.exception.sent)
        self.assertFalse(client.calls)

    def test_invoke_without_display_cancels_headers_and_body_and_closes_stream(self):
        for before_headers in (True, False):
            with self.subTest(before_headers=before_headers), ThreadPoolExecutor() as pool:
                client = WaitingClient(before_headers)

                def invoke():
                    with control.active('kernel', 'stream'):
                        return ConfiguredModel(client).invoke([])

                future = pool.submit(invoke)
                try:
                    self.assertTrue(client.started.wait(3))
                finally:
                    control.interrupt('kernel', 'stream', 'pause')
                with self.assertRaises(ModelInterrupted) as caught:
                    future.result(timeout=3)
                self.assertTrue(caught.exception.sent)
                self.assertTrue(client.closed.is_set())
                self.assertEqual(len(client.calls), 1)


class KernelRequestAccountingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def runner(self, model, purpose):
        loop = TaskRunner(model, None)
        loop.root = self.root / purpose
        loop.root.mkdir()
        loop.budget = StepBudget(max_steps=3)
        loop.prior_elapsed, loop.run_started = 0, time.monotonic()
        if purpose == 'compaction':
            loop.state['compaction_pending'] = {'stage': 'prepared'}
        return loop

    def test_pause_after_reservation_prevents_sdk_call_and_unknown_usage(self):
        for purpose in ('plan', 'plan_repair', 'compaction'):
            with self.subTest(purpose=purpose):
                # 替身没有取消接口，必须由执行器自身拦住发送。
                client = repairs.Model([])
                loop = self.runner(client, purpose)
                persist = loop.persist
                with control.active('kernel', purpose) as signal:

                    def pause_after_saved_request(event, payload=None):
                        persist(event, payload)
                        if event == 'model_request_started':
                            signal.interrupt('pause')

                    with patch.object(loop, 'persist', side_effect=pause_after_saved_request):
                        with self.assertRaises(ModelInterrupted) as caught:
                            loop.request_model([HumanMessage(content='offline')], purpose)
                self.assertFalse(caught.exception.sent)
                self.assertFalse(client.calls)
                self.assertEqual(loop.budget.used_steps, 0)
                self.assertEqual(loop.budget.unknown_usage_steps, 0)
                self.assertEqual(loop.budget.inflight_reserved_tokens, 0)
                self.assertIsNone(loop.state['request_pending'])
                self.assertIsNone(loop.state.get('compaction_pending'))
                record = json.loads(
                    (loop.root / 'requests/call-0001/interruption.json').read_text('utf-8')
                )
                self.assertEqual(record['usage'], 'not_sent')
                self.assertFalse(record['request_sent'])

    def test_compaction_without_display_cancels_and_keeps_original_view(self):
        client = WaitingClient()
        loop = self.runner(ConfiguredModel(client), 'compaction')
        loop.state['context_view'] = {'generation': 1}
        with ThreadPoolExecutor() as pool:

            def request():
                with control.active('kernel', 'compaction'):
                    return loop.request_model([HumanMessage(content='offline')], 'compaction')

            future = pool.submit(request)
            try:
                self.assertTrue(client.started.wait(3))
            finally:
                control.interrupt('kernel', 'compaction', 'pause')
            with self.assertRaises(ModelInterrupted):
                future.result(timeout=3)
        self.assertTrue(client.closed.is_set())
        self.assertEqual(loop.state['context_view'], {'generation': 1})
        self.assertIsNone(loop.state['compaction_pending'])
        self.assertIsNone(loop.state['request_pending'])
        self.assertEqual(loop.budget.used_steps, 1)
        self.assertEqual(loop.budget.unknown_usage_steps, 1)
        self.assertEqual(loop.budget.inflight_reserved_tokens, 0)
        self.assertEqual(loop.state['interrupted_requests'][0]['purpose'], 'compaction')
        self.assertFalse((loop.root / 'requests/call-0001/response.json').exists())


class KernelPauseBoundaryTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp

    def test_complete_reply_is_saved_before_pause_and_never_starts_plan_repair(self):
        with control.active('kernel', 'reply') as signal:

            def complete_then_pause(messages):
                signal.interrupt('pause')
                return fixtures.reply({'kind': 'inspect', 'note': 'invalid', 'steps': []})

            model = repairs.Model([
                complete_then_pause,
                fixtures.reply({'kind': 'blocked', 'note': 'must not be requested'}),
            ])
            loop = TaskRunner(model, self.env, state_dir=self.root / 'state')
            result = loop.run([], 'offline boundary', StepBudget(max_steps=3))
        self.assertEqual(result['status'], 'paused', result)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(loop.state['repair_count'], 0)
        self.assertIsNone(loop.state.get('rejection_pending'))
        self.assertIsNotNone(loop.state['response_pending'])
        self.assertEqual(loop.budget.used_steps, 1)
        self.assertEqual(loop.budget.unknown_usage_steps, 0)
        self.assertTrue((loop.root / 'requests/call-0001/response.json').exists())
        self.assertFalse((loop.root / 'requests/call-0002').exists())
        self.assertFalse(list(loop.root.glob('plans/*/plan.json')))

        # 明确恢复后复用已归档回复，只发出需要的纠错调用。
        with control.active('kernel', 'resume'):
            resumed = TaskRunner(model, self.env, state_dir=self.root / 'state').run(
                [], 'offline boundary', StepBudget(max_steps=3)
            )
        self.assertEqual(resumed['status'], 'task_cancelled', resumed)
        self.assertEqual(resumed['calls_this_invocation'], 1)
        self.assertEqual(len(model.calls), 2)


class KernelSavedRejectionPauseTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    run_replies = repairs.LocalRepairTests.run_replies
    load_state = repairs.LocalRepairTests.load_state

    def test_pause_preserves_prepared_rejection_prefix_for_explicit_resume(self):
        interruption, error = interrupt_after_persist('plan_rejection_prepared')
        response = fixtures.reply({'kind': 'inspect', 'note': 'invalid', 'steps': []})
        with interruption, self.assertRaises(error):
            self.run_replies([response])
        before = self.load_state()
        self.assertIsNotNone(before['rejection_pending'])

        with control.active('saved-kernel', 'rejection') as signal:
            signal.interrupt('pause')
            paused = self.run_replies([])
        self.assertEqual(paused['status'], 'paused', paused)
        self.assertEqual(paused['calls_this_invocation'], 0)
        self.assertFalse(self.model.calls)
        after = self.load_state()
        self.assertEqual(after['history'], before['history'])
        self.assertEqual(after['rejection_pending'], before['rejection_pending'])
        self.assertEqual(after['response_pending'], before['response_pending'])

        resumed = self.run_replies([
            fixtures.reply({'kind': 'blocked', 'note': 'explicitly resumed'})
        ])
        self.assertEqual(resumed['status'], 'task_cancelled', resumed)
        self.assertEqual(resumed['calls_this_invocation'], 1)
        self.assertEqual(self.engine.state['repair_count'], 2)  # 原拒收与后来的受阻报告各配对一次。
        self.assertIsNone(self.engine.state.get('rejection_pending'))

    def test_message_applies_after_saved_rejection_pairing_without_prefix_conflict(self):
        interruption, error = interrupt_after_persist('plan_rejection_prepared')
        response = fixtures.reply({'kind': 'inspect', 'note': 'invalid', 'steps': []})
        with interruption, self.assertRaises(error):
            self.run_replies([response])
        before = self.load_state()
        incoming = {'id': 'saved-update', 'kind': 'message', 'text': '只解释剩余工作'}
        from agent.task.requirements import confirmed

        original = before['requirements']
        incoming['confirmed_requirements'] = confirmed(
            before['request'] + '\n\n' + incoming['text'], original['targets'],
            sources=[*original['sources'], {'id': incoming['id'], 'role': 'user', 'text': incoming['text']}],
            confirmation={'id': 'confirmed-update', 'text': 'Confirm the updated request'}, parent=original,
        )

        with control.active('saved-kernel', 'rejection-message') as signal:
            signal.interrupt('message', incoming['id'])

            def controls():
                # 与应用桥接层相同，只在消息已持久化应用后消费信号。
                if incoming['id'] in self.engine.state.get('applied_inputs', []):
                    signal.consume_input(incoming['id'])
                return [incoming]

            result = self.run_replies([
                fixtures.reply({'kind': 'blocked', 'note': 'new input inspected'})
            ], control_source=controls)

        self.assertEqual(result['status'], 'task_cancelled', result)
        self.assertEqual(result['calls_this_invocation'], 1)
        after = self.load_state()
        self.assertEqual(after['history'][:len(before['history'])], before['history'])
        self.assertEqual(after['repair_count'], 2)  # 新输入之后的受阻报告继续走返工反馈。
        self.assertIsNone(after.get('rejection_pending'))
        self.assertEqual(after['applied_inputs'].count(incoming['id']), 1)
        self.assertIn(incoming['text'], repr(self.model.calls[0]))


class KernelSavedCompactionPauseTests(unittest.TestCase):
    setUp = compaction_fixtures.DurableCompactionTests.setUp
    engine = compaction_fixtures.DurableCompactionTests.engine
    load_state = repairs.LocalRepairTests.load_state

    def check_saved_summary(self, event, stage):
        interruption, error = interrupt_after_persist(event)
        with patch.dict(compaction_fixtures.compaction.CONFIG, self.policy), interruption:
            with self.assertRaises(error):
                self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        before = self.load_state()
        self.assertEqual(before['compaction_pending']['stage'], stage)
        calls = len(self.sdk.calls)

        with control.active('saved-kernel', 'summary') as signal:
            signal.interrupt('pause')
            with patch.dict(compaction_fixtures.compaction.CONFIG, self.policy):
                paused = self.engine().run(
                    [], 'value must return four', StepBudget(max_steps=10)
                )
        self.assertEqual(paused['status'], 'paused', paused)
        self.assertEqual(paused['calls_this_invocation'], 0)
        self.assertEqual(len(self.sdk.calls), calls)
        after = self.load_state()
        self.assertEqual(after['history'], before['history'])
        self.assertEqual(after['compaction_pending'], before['compaction_pending'])
        self.assertEqual(after.get('context_view'), before.get('context_view'))

        self.sdk.plans = iter([{'kind': 'blocked', 'note': 'explicitly resumed'}])
        with patch.dict(compaction_fixtures.compaction.CONFIG, self.policy):
            resumed = self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        self.assertEqual(resumed['status'], 'task_cancelled', resumed)
        self.assertEqual(len(resumed['compactions']), 1)
        self.assertEqual(
            sum(options.get('tool_choice') == 'none' for _, options in self.sdk.calls), 1
        )

    def test_pause_preserves_prepared_summary_prefix_for_explicit_resume(self):
        self.check_saved_summary('compaction_started', 'prepared')

    def test_pause_preserves_saved_summary_response_for_explicit_resume(self):
        self.check_saved_summary('model_response_received', 'response_saved')


if __name__ == '__main__':
    unittest.main()
