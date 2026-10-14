"""被拒绝的交付保持原样、成对持久保存，且绝不执行。"""

import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from agent.model.payload import RAW_TOOL_CALLS, wire_input
from agent.task.runner import TaskRunner
from support import interrupt_after_persist
import test_task_runner as f
import test_local_repair as repairs


class RejectedContinuationTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp
    run_replies = repairs.LocalRepairTests.run_replies
    load_state = repairs.LocalRepairTests.load_state

    def rejected(self, malformed=False):
        args = '{\n "kind": "inspect", "note": "Retained draft", "steps": [], "extra": 1\n}'
        if malformed:
            args += '}'
        call = {
            'id': 'rejected-call',
            'type': 'function',
            'function': {'name': 'submit_plan', 'arguments': args},
        }
        options = (
            {
                'invalid_tool_calls': [
                    {'id': call['id'], 'name': 'submit_plan', 'args': args, 'error': 'bad JSON'}
                ]
            }
            if malformed
            else {
                'tool_calls': [{'id': call['id'], 'name': 'submit_plan', 'args': json.loads(args)}]
            }
        )
        return AIMessage(
            content='I will correct the submitted fields.',
            additional_kwargs={
                RAW_TOOL_CALLS: [call],
                'reasoning_content': 'Exact retained reasoning 中文.',
            },
            response_metadata={'finish_reason': 'tool_calls'},
            **options
        )

    def stop(self, messages):
        return f.reply({'kind': 'blocked', 'note': 'Offline continuation inspected'})

    def assert_cycle(self, calls, response):
        payload = wire_input(calls, self.engine.tools)
        ai, result = payload['messages'][-2:]
        self.assertEqual(ai['tool_calls'], response.additional_kwargs[RAW_TOOL_CALLS])
        self.assertEqual(ai['reasoning_content'], response.additional_kwargs['reasoning_content'])
        self.assertEqual(ai['content'], response.content)
        self.assertEqual(result['role'], 'tool')
        self.assertEqual(result['tool_call_id'], 'rejected-call')
        self.assertIn('no actions executed', result['content'])
        self.assertEqual(
            sum(
                m['role'] == 'assistant' and m.get('tool_calls') == ai['tool_calls']
                for m in payload['messages']
            ),
            1,
        )

    def test_complete_invalid_arguments_are_returned_with_thought_and_original_bytes(self):
        response = self.rejected()
        result = self.run_replies([response, self.stop], interactive=True)
        self.assertEqual(result['status'], 'task_blocked', result)
        self.assert_cycle(self.model.calls[1], response)
        self.assertEqual(self.engine.state['plan_count'], 0)
        self.assertEqual(
            self.engine.state['history'][: len(self.model.calls[0])],
            self.load_state()['history'][: len(self.model.calls[0])],
        )
        self.assertEqual(
            [
                r['content']
                for r in self.engine.state['public_replies']
                if r['content'] == response.content
            ],
            [response.content],
        )

    def test_malformed_json_is_returned_without_fabrication(self):
        response = self.rejected(malformed=True)
        result = self.run_replies([response, self.stop])
        self.assertEqual(result['status'], 'task_blocked', result)
        self.assert_cycle(self.model.calls[1], response)
        self.assertEqual(self.engine.state['plan_count'], 0)

    def test_resume_between_preparation_and_pairing_does_not_repeat_reply_or_call(self):
        response = self.rejected()
        interrupt, error = interrupt_after_persist('plan_rejection_prepared')
        with interrupt, self.assertRaises(error):
            self.run_replies([response])
        self.assertIsNotNone(self.load_state().get('rejection_pending'))
        result = self.run_replies([self.stop])
        self.assertEqual(result['calls_this_invocation'], 1, result)
        self.assert_cycle(self.model.calls[0], response)

    def test_pairing_checkpoint_write_failure_stops_without_another_model_call(self):
        original = TaskRunner.persist

        def failing(engine, event, payload=None):
            if event == 'plan_rejected':
                raise OSError('offline checkpoint failure')
            return original(engine, event, payload)

        with patch.object(TaskRunner, 'persist', failing):
            result = self.run_replies([self.rejected(), self.stop])
        self.assertEqual(result['status'], 'runtime_error', result)
        self.assertEqual(len(self.model.calls), 1)

    def test_incomplete_or_untrusted_identity_stops_with_response_preserved(self):
        response = self.rejected()
        response.response_metadata['finish_reason'] = 'length'
        result = self.run_replies([response, self.stop])
        self.assertEqual(result['status'], 'output_limit_reached', result)
        self.assertIsNotNone(self.engine.state.get('response_pending'))
        self.assertEqual(len(self.model.calls), 1)

    def test_consecutive_rejections_each_pair_once_and_keep_prefix(self):
        first, second = self.rejected(), self.rejected(True)
        second.invalid_tool_calls[0]['id'] = 'second-call'
        second.additional_kwargs[RAW_TOOL_CALLS][0]['id'] = 'second-call'
        second.additional_kwargs['reasoning_content'] = 'Second retained thought.'
        result = self.run_replies([first, second, self.stop])
        self.assertEqual(result['status'], 'task_blocked', result)
        body = wire_input(self.model.calls[2], self.engine.tools)['messages']
        self.assertEqual(
            [m['tool_call_id'] for m in body if m['role'] == 'tool'],
            ['rejected-call', 'second-call'],
        )
        previous = wire_input(self.model.calls[1], self.engine.tools)['messages']
        self.assertEqual(body[: len(previous)], previous)
        self.assertEqual(self.engine.state['plan_count'], 0)

    def test_multiple_calls_rejected_as_one_delivery_all_get_results(self):
        response = self.rejected()
        response.tool_calls.append({'id': 'extra-call', 'name': 'freeze_acceptance', 'args': {}})
        response.additional_kwargs[RAW_TOOL_CALLS].append(
            {
                'id': 'extra-call',
                'type': 'function',
                'function': {'name': 'freeze_acceptance', 'arguments': '{}'},
            }
        )
        result = self.run_replies([response, self.stop])
        self.assertEqual(result['status'], 'task_blocked', result)
        body = wire_input(self.model.calls[1], self.engine.tools)['messages']
        self.assertEqual(
            [m['tool_call_id'] for m in body if m['role'] == 'tool'],
            ['rejected-call', 'extra-call'],
        )
        self.assertEqual(self.engine.state['plan_count'], 0)

    def test_untrusted_call_identity_stops_and_is_not_replaced(self):
        response = self.rejected()
        response.tool_calls[0]['id'] = None
        result = self.run_replies([response, self.stop])
        self.assertEqual(result['status'], 'invalid_message_history', result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertIsNotNone(self.engine.state.get('response_pending'))

    def test_pause_after_preparation_keeps_pair_and_public_reply_on_resume(self):
        response = self.rejected()
        interrupt, error = interrupt_after_persist('plan_rejection_prepared')
        with interrupt, self.assertRaises(error):
            self.run_replies([response], interactive=True)
        result = self.run_replies(
            [self.stop],
            interactive=True,
            control_source=lambda: [{'id': 'pause-once', 'kind': 'pause'}],
        )
        self.assertEqual(result['status'], 'paused', result)
        self.assertEqual(len(self.model.calls), 0)
        result = self.run_replies([self.stop], interactive=True)
        self.assertEqual(result['status'], 'task_blocked', result)
        self.assert_cycle(self.model.calls[0], response)
        self.assertEqual(
            [
                r['content']
                for r in self.engine.state['public_replies']
                if r['content'] == response.content
            ],
            [response.content],
        )

    def test_compaction_recognizes_invalid_call_cycle_as_balanced(self):
        from agent.model.compaction import balanced_boundaries
        from langchain_core.messages import message_to_dict, ToolMessage

        response = self.rejected(True)
        self.assertEqual(
            balanced_boundaries(
                [
                    message_to_dict(response),
                    message_to_dict(ToolMessage(content='rejected', tool_call_id='rejected-call')),
                ]
            ),
            [0, 2],
        )

    def test_unrunnable_freeze_rejection_keeps_review_call_and_never_modifies_source(self):
        import test_acceptance_review as reviews

        def rejected_review(messages):
            response = reviews.review(messages)
            response.content = 'Retain review commentary.'
            response.additional_kwargs['reasoning_content'] = 'Retain review thought.'
            return response

        candidate = f.TaskRunnerTests.candidate
        self.shell = f.TaskRunnerTests.shell.__get__(self)
        self.model = reviews.ScriptModel(
            [
                f.reply(candidate(self, test_code='import missing_fixture\n' + f.CODE)),
                rejected_review,
                self.stop,
            ]
        )
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / 'state')
        from agent.task.budget import StepBudget

        result = self.engine.run([], 'return 4', StepBudget(max_steps=4))
        self.assertEqual(result['status'], 'task_blocked', result)
        body = wire_input(self.model.calls[-1], self.engine.tools)['messages']
        self.assertEqual(body[-2]['tool_calls'][0]['function']['name'], 'freeze_acceptance')
        self.assertEqual(body[-2]['reasoning_content'], 'Retain review thought.')
        self.assertEqual(body[-1]['tool_call_id'], 'freeze_acceptance-call')
        self.assertTrue(body[-1]['content'].startswith('[Acceptance review rejected'))
        self.assertIn('return 0', (self.repo / 'sample.py').read_text())
        self.assertFalse((self.root / 'state/frozen-acceptance.json').exists())

    def test_last_available_call_can_finish_local_pairing_on_resume(self):
        from agent.task.budget import StepBudget

        interrupt, error = interrupt_after_persist('plan_rejection_prepared')
        model = repairs.Model([self.rejected()])
        engine = TaskRunner(model, self.env, state_dir=self.root / 'state')
        with interrupt, self.assertRaises(error):
            engine.run([], 'return 4', StepBudget(max_steps=1))
        model = repairs.Model([])
        engine = TaskRunner(model, self.env, state_dir=self.root / 'state')
        result = engine.run([], 'return 4', StepBudget(max_steps=1))
        self.assertEqual(result['status'], 'budget_exhausted', result)
        self.assertEqual(len(model.calls), 0)
        state = self.load_state()
        self.assertIsNone(state.get('rejection_pending'))
        self.assertEqual(state['history'][-1]['data']['tool_call_id'], 'rejected-call')
