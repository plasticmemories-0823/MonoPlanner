"""Exercise real request views, paired tools and durable compaction recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict

from agent.model import compaction
from agent.model.client import ConfiguredModel
from agent.model.context import admit
from agent.model.errors import ModelInterrupted, ModelRequestNotSent, ModelCallError, is_context_overflow
from agent.model.payload import wire_input
from agent.model.tools import WORK_TOOLS
from agent.source.paging import visible_reads
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from agent.execution.local import LocalEnvironment
from test_task_runner import reply, confirmation, CODE
from support import python_command


def summary(text='Known failure remains unresolved. Original code and tests remain in their records.'):
    return message_to_dict(AIMessage(content=text, response_metadata={'finish_reason': 'stop'},
        additional_kwargs={'reasoning_content': 'private summary reasoning'},
        usage_metadata={'input_tokens': 12, 'output_tokens': 8, 'total_tokens': 20}))


def cycle(call_id, path, reasoning=''):
    data = {'path': path, 'sha256': path, 'text': '1: exact source\n', 'visible_ranges': [[0, 1]]}
    return [message_to_dict(AIMessage(content='inspection', additional_kwargs={'reasoning_content': reasoning},
                tool_calls=[{'id': call_id, 'name': 'submit_plan', 'args': {'kind': 'inspect'}}])),
            message_to_dict(ToolMessage(tool_call_id=call_id, content=json.dumps(
                {'steps': [{'op': 'read', 'status': 'succeeded', 'data': data}]})))]


class Loop:
    def __init__(self, root):
        self.root, self.rules, self.history_store = root, 'fixed rules', None
        self.state = {'history': [message_to_dict(HumanMessage(content='original request')),
                      *cycle('old', 'old.py', 'old reasoning ' * 14000), *cycle('recent', 'recent.py', 'recent reasoning')],
                      'rules': self.rules, 'request': 'fix behavior', 'acceptance': 'fixed requirement',
                      'verify_command': 'fixed command', 'source': 'source-v1'}
        self.calls, self.events = [], []
        self.answer = summary()
        self.answers = None
        self.crash_on = None

    def persist(self, event, payload=None):
        self.events.append(event)
        if event == self.crash_on:
            raise InterruptedError(event)

    def request_model(self, messages, purpose):
        self.calls.append((messages, purpose))
        if self.answers is not None:
            self.answer = next(self.answers)
        self.state['compaction_pending'].update(stage='response_saved', response=copy.deepcopy(self.answer), call=len(self.calls))
        self.persist('model_response_received')
        return self.answer


class CompactionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.loop = Loop(self.root)
        self.policy = {'threshold_ratio': .01, 'headroom_tokens': 0, 'retain_ratio': .00005}

    def compress(self):
        with patch.dict(compaction.CONFIG, self.policy):
            compaction.prepare(self.loop, 65536)

    def test_default_limits_and_no_change_below_threshold(self):
        self.assertEqual(compaction.limits(65536), (800000, 149514))
        before = wire_input(compaction.messages(self.loop), WORK_TOOLS)
        compaction.prepare(self.loop, 65536)
        self.assertEqual(before, wire_input(compaction.messages(self.loop), WORK_TOOLS))
        self.assertFalse(self.loop.calls)

    def test_summary_replaces_old_span_preserves_archive_and_recent_reasoning(self):
        before = copy.deepcopy(self.loop.state['history'])
        self.compress()
        self.assertEqual(before, self.loop.state['history'])
        self.assertEqual(len(self.loop.calls), 1)
        request = self.loop.calls[0][0]
        self.assertTrue(request[-1].content.startswith(compaction.INSTRUCTION))
        self.assertIn('<execution-state-at-checkpoint>', request[-1].content)
        self.assertIn('old reasoning', request[2].additional_kwargs['reasoning_content'])
        actual = wire_input(compaction.messages(self.loop), WORK_TOOLS)
        self.assertNotIn('old reasoning', json.dumps(actual))
        self.assertIn('recent reasoning', json.dumps(actual))
        self.assertIn('fixed requirement', json.dumps(actual))
        self.assertNotIn('private summary reasoning', json.dumps(actual))
        self.assertLess(self.loop.state['compactions'][0]['after_tokens'], self.loop.state['compactions'][0]['before_tokens'])
        compaction.prepare(self.loop, 65536)
        self.assertEqual(len(self.loop.calls), 1)

    def test_only_committed_rollover_allows_new_prefix_and_tools_stay_fixed(self):
        options = {'max_tokens': 65536}
        admit(self.loop.state, compaction.messages(self.loop), WORK_TOOLS, options)
        self.compress()
        changed = compaction.messages(self.loop)
        admit(self.loop.state, changed, WORK_TOOLS, options)
        admit(self.loop.state, [*changed, HumanMessage(content='append')], WORK_TOOLS, options)
        with self.assertRaises(ModelRequestNotSent):
            admit(self.loop.state, changed, WORK_TOOLS, options)
        with self.assertRaises(ModelRequestNotSent):
            admit(self.loop.state, changed, WORK_TOOLS[:-1], options)

    def test_original_history_and_summary_tampering_are_rejected(self):
        self.compress()
        self.loop.state['history'][1]['data']['content'] = 'tampered'
        with self.assertRaises(ModelRequestNotSent):
            compaction.prepare(self.loop, 65536)
        self.loop = Loop(self.root / 'second')
        self.compress()
        path = self.loop.root / self.loop.state['context_view']['record']['path']
        path.write_text('{}')
        with self.assertRaises(ModelRequestNotSent):
            compaction.messages(self.loop)

    def test_compacted_read_is_not_visible_but_recent_read_is(self):
        self.assertIn(('old.py', 'old.py'), visible_reads(self.loop.state['history'], []))
        self.compress()
        visible = visible_reads(compaction.history(self.loop.state, self.root), [])
        self.assertNotIn(('old.py', 'old.py'), visible)
        self.assertIn(('recent.py', 'recent.py'), visible)

    def test_program_snapshot_uses_actual_records_without_llm_completion_judgment(self):
        directory = self.root / 'plans/p0001'; directory.mkdir(parents=True)
        result = {'outcome': 'verification_failed', 'steps': [{'id': 'fix', 'op': 'shell',
            'status': 'succeeded', 'record': 'plans/p0001/fix/result.json'}]}
        (directory / 'result.json').write_text(json.dumps(result), encoding='utf-8')
        (self.root / 'frozen-acceptance.json').write_text('{"fixed": true}', encoding='utf-8')
        self.loop.state.update(status='needs_plan', plan_count=1, last_plan='p0001', last_result=result,
                               verification={'status': 'failed'})
        self.loop.answer = summary('The historical command succeeded; no current completion claim.')
        snapshot = compaction.execution_snapshot(self.loop)
        self.assertEqual(snapshot['verification_status'], 'failed')
        self.assertEqual(snapshot['operations'][0]['status'], 'succeeded')
        self.assertEqual(snapshot['latest_result']['path'], 'plans/p0001/result.json')
        self.assertEqual(snapshot['latest_result']['sha256'], __import__('hashlib').sha256((directory / 'result.json').read_bytes()).hexdigest())
        self.compress()
        body = compaction.checked(self.root, self.loop.state['context_view']['record'])
        self.assertEqual(body['execution_snapshot'], snapshot)
        self.assertIn(compaction.snapshot_text(snapshot), self.loop.calls[0][0][-1].content)
        self.assertIn(compaction.snapshot_text(snapshot), body['message']['data']['content'])
        self.assertEqual(self.loop.state['verification']['status'], 'failed')

    def test_recent_request_and_results_follow_background_without_rewriting_checkpoint(self):
        self.loop.state['history'].append(message_to_dict(HumanMessage(content='NEW request: read fresh.py, old reads do not satisfy it')))
        self.compress()
        view = compaction.history(self.loop.state, self.root)
        self.assertIn('Later requests and actual tool results take precedence', view[1]['data']['content'])
        self.assertEqual(view[-1]['data']['content'], 'NEW request: read fresh.py, old reads do not satisfy it')
        original = copy.deepcopy(view[:2])
        self.loop.state.update(status='needs_plan', verification={'status': 'failed'}, source='source-v2')
        self.loop.state['history'].extend(cycle('new', 'fresh.py'))
        updated = compaction.history(self.loop.state, self.root)
        self.assertEqual(updated[:2], original)
        self.assertEqual(updated[-1]['data']['tool_call_id'], 'new')

    def test_saved_snapshot_does_not_change_when_response_is_reused(self):
        self.loop.state.update(status='needs_plan')
        self.loop.crash_on = 'model_response_received'
        with self.assertRaises(InterruptedError):
            self.compress()
        snapshot = copy.deepcopy(self.loop.state['compaction_pending']['execution_snapshot'])
        self.assertEqual(snapshot['verification_status'], '尚未运行')
        self.loop.crash_on = None
        self.compress()
        body = compaction.checked(self.root, self.loop.state['context_view']['record'])
        self.assertEqual(body['execution_snapshot'], snapshot)
        self.assertEqual(len(self.loop.calls), 1)

    def test_user_updates_append_without_changing_prior_summary_and_survive_next_compaction(self):
        self.compress()
        pinned = copy.deepcopy(compaction.history(self.loop.state, self.root)[:2])
        self.loop.state['user_updates'] = [{'id': 'new', 'text': '新增要求逐字保留'}]
        self.loop.state['history'].append(message_to_dict(HumanMessage(content='新增要求逐字保留', id='new')))
        self.assertEqual(compaction.history(self.loop.state, self.root)[:2], pinned)
        self.loop.state['history'].extend(cycle('large', 'large.py', 'large reasoning ' * 14000))
        self.loop.state['history'].extend(cycle('last', 'last.py'))
        self.compress()
        current = compaction.checked(self.root, self.loop.state['context_view']['record'])
        self.assertEqual(current['requirements']['user_updates'], self.loop.state['user_updates'])

    def test_unfinished_tool_pair_is_retained(self):
        self.loop.state['history'].extend(cycle('open', 'next.py')[:1])
        with patch.dict(compaction.CONFIG, self.policy):
            request, cutoff, _ = compaction.select(self.loop, 0, 65536)
        self.assertEqual(cutoff, len(self.loop.state['history']) - 1)
        self.assertNotIn('open', [c['id'] for m in request for c in getattr(m, 'tool_calls', [])])

    def test_incomplete_empty_tool_and_expanding_summaries_do_not_replace_history(self):
        for answer in (summary(''), summary('x' * 400000),
                       message_to_dict(AIMessage(content='truncated', response_metadata={'finish_reason': 'length'})),
                       message_to_dict(AIMessage(content='bad', tool_calls=[{'id': 'bad', 'name': 'submit_plan', 'args': {}}],
                                                response_metadata={'finish_reason': 'stop'}))):
            with self.subTest(answer=answer['data']['content'][:15]):
                self.loop = Loop(self.root / str(len(self.loop.calls)) / str(len(answer['data']['content'])))
                self.loop.answer = answer
                before = copy.deepcopy(self.loop.state['history'])
                with self.assertRaises(ModelRequestNotSent):
                    self.compress()
                self.assertEqual(before, self.loop.state['history'])
                self.assertNotIn('context_view', self.loop.state)
                self.assertEqual(len(self.loop.calls), 1)

    def test_saved_response_is_reused_after_interruption(self):
        self.loop.crash_on = 'model_response_received'
        with self.assertRaises(InterruptedError):
            self.compress()
        self.loop.crash_on = None
        self.compress()
        self.assertEqual(len(self.loop.calls), 1)
        self.assertEqual(len(self.loop.state['compactions']), 1)

    def test_commit_unknown_does_not_replay(self):
        self.loop.crash_on = 'compaction_commit_started'
        with self.assertRaises(InterruptedError):
            self.compress()
        self.loop.crash_on = None
        with self.assertRaisesRegex(ModelRequestNotSent, 'Interrupted compaction commit'):
            self.compress()
        self.assertEqual(len(self.loop.calls), 1)
        self.assertNotIn('context_view', self.loop.state)

    def test_later_compaction_merges_prior_summary_and_keeps_original_archive(self):
        self.compress()
        first = self.loop.state['context_view'].copy()
        self.loop.state['history'].extend(cycle('next', 'next.py', 'second old reasoning ' * 14000))
        self.loop.state['history'].extend(cycle('last', 'last.py'))
        self.compress()
        self.assertEqual(self.loop.state['context_view']['generation'], 2)
        self.assertGreater(self.loop.state['context_view']['cutoff'], first['cutoff'])
        self.assertIn('Known failure', str(self.loop.calls[1][0]))
        self.assertEqual(len(self.loop.state['history']), 9)

    def test_compaction_options_do_not_change_normal_call(self):
        model = ConfiguredModel(object())
        self.assertNotIn('tool_choice', model.request_options)
        self.assertEqual(model.for_purpose('compaction').request_options['tool_choice'], 'none')
        self.assertEqual(model.for_purpose('compaction').request_options['reasoning_effort'], 'high')

    def test_two_effective_summaries_and_no_third_attempt(self):
        self.loop.answers = iter([summary('x' * 45000), summary()])
        self.compress()
        self.assertEqual(len(self.loop.calls), 2)
        self.assertEqual(len(self.loop.state['compactions']), 2)
        self.assertNotIn('compaction_trigger', self.loop.state)
        self.loop = Loop(self.root / 'still-large')
        self.loop.answers = iter([summary('x' * 45000), summary('x' * 44000)])
        with self.assertRaisesRegex(ModelRequestNotSent, 'after two summaries'):
            self.compress()
        self.assertEqual(len(self.loop.calls), 2)

    def test_pending_operation_prevents_compaction(self):
        self.loop.state['pending'] = {'cursor': 0}
        with self.assertRaisesRegex(ModelRequestNotSent, 'unfinished'):
            self.compress()
        self.assertFalse(self.loop.calls)

    def test_explicit_context_error_only(self):
        class Rejected(Exception):
            status_code = 400
            body = {'error': {'code': 'context_length_exceeded'}}
        self.assertTrue(is_context_overflow(Rejected()))
        self.assertTrue(is_context_overflow(ModelCallError('rejected', context_overflow=True)))
        self.assertFalse(is_context_overflow(TimeoutError('context length exceeded')))
        error = Rejected()
        error.body = {'error': {'message': 'Missing reasoning_content; invalid request'}}
        self.assertFalse(is_context_overflow(error))


class SDK:
    def __init__(self, plans):
        self.plans, self.calls = iter(plans), []
        self.failures = 0

    def bind_tools(self, tools, strict=True):
        self.asserted_tools = tools
        return self

    def invoke(self, messages, **options):
        self.calls.append((messages, options))
        if options.get('tool_choice') == 'none':
            return AIMessage(content='Evidence in saved source records. Value must return four; work remains pending.',
                response_metadata={'finish_reason': 'stop'},
                usage_metadata={'input_tokens': 12, 'output_tokens': 8, 'total_tokens': 20})
        if self.failures:
            self.failures -= 1
            raise ModelCallError('Explicit context rejection', context_overflow=True)
        result = confirmation(messages)
        if result:
            return result
        plan = next(self.plans)
        result = reply(plan)
        result.additional_kwargs['reasoning_content'] = 'old evidence reasoning ' * (10000 if len(self.calls) == 1 else 1)
        return result


class DurableCompactionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / 'repo'; self.repo.mkdir()
        (self.repo / 'sample.py').write_text('def value():\n    return 0\n')
        self.env = LocalEnvironment(self.repo, artifact_root=self.root / 'runtime')
        plans = [{'kind': 'inspect', 'note': 'read source', 'steps': [{'id': 'r', 'op': 'read', 'path': 'sample.py'}]},
                 {'kind': 'inspect', 'note': 'confirm exact source', 'steps': [{'id': 'r', 'op': 'read', 'path': 'sample.py',
                                                                           'query': 'reread: verify'}]}]
        self.sdk = SDK(plans)
        self.model = ConfiguredModel(self.sdk)
        first = self.engine(pause_after_steps=2).run([], 'value must return four', StepBudget(max_steps=10))
        self.assertEqual(first['status'], 'paused', first)
        self.policy = {'threshold_ratio': .01, 'headroom_tokens': 0, 'retain_ratio': .00005}

    def engine(self, **options):
        return TaskRunner(self.model, self.env, state_dir=self.root / 'state', **options)

    def test_full_execution_budget_and_frozen_acceptance_after_saved_summary(self):
        self.sdk.plans = iter([{'kind': 'candidate', 'note': 'fix value', 'steps': [{'id': 'fix', 'op': 'shell',
            'command': python_command("from pathlib import Path; Path('sample.py').write_text('def value():\\n    return 4\\n')")}],
            'test_code': CODE}])
        with patch.dict(compaction.CONFIG, self.policy):
            result = self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(result['planner_calls'], 5)
        self.assertEqual(len(result['compactions']), 1)
        self.assertEqual(sum(options.get('tool_choice') == 'none' for _, options in self.sdk.calls), 1)
        again = self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        self.assertTrue(again['result_reused'])
        self.assertEqual(len(self.sdk.calls), 5)

    def test_unknown_summary_request_is_not_resent(self):
        original = TaskRunner.persist
        def crash(loop, event, payload=None):
            original(loop, event, payload)
            if event == 'model_request_started' and loop.state['request_pending']['purpose'] == 'compaction':
                raise InterruptedError('before response')
        with patch.dict(compaction.CONFIG, self.policy), patch.object(TaskRunner, 'persist', crash):
            self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        calls = len(self.sdk.calls)
        result = self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'model_request_unknown')
        self.assertEqual(len(self.sdk.calls), calls)

    def test_user_interrupted_summary_keeps_original_view_and_clears_pending(self):
        from agent.storage.history import LocalHistory
        before = LocalHistory(self.root/'state')
        with before.lease():
            previous = before.load()
        before.close()
        original = self.sdk.invoke
        def interrupt_summary(messages, **options):
            if options.get('tool_choice') == 'none':
                raise ModelInterrupted('pause')
            return original(messages, **options)
        with patch.dict(compaction.CONFIG, self.policy), patch.object(self.sdk, 'invoke', interrupt_summary):
            result = self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'paused')
        after = LocalHistory(self.root/'state')
        with after.lease():
            state = after.load()
        after.close()
        self.assertIsNone(state['compaction_pending'])
        self.assertIsNone(state['request_pending'])
        self.assertEqual(state.get('context_view'), previous.get('context_view'))
        self.assertEqual(state['history'][:len(previous['history'])], previous['history'])
        self.assertEqual(state['interrupted_requests'][0]['purpose'], 'compaction')
        self.assertFalse(list((self.root/'state'/'compaction').glob('**/*.json')))

    def test_unknown_commit_is_not_repeated(self):
        original = TaskRunner.persist
        def crash(loop, event, payload=None):
            original(loop, event, payload)
            if event == 'compaction_commit_started':
                raise InterruptedError('commit interrupted')
        with patch.dict(compaction.CONFIG, self.policy), patch.object(TaskRunner, 'persist', crash):
            self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        calls = len(self.sdk.calls)
        result = self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'compaction_commit_unknown')
        self.assertEqual(len(self.sdk.calls), calls)

    def test_saved_summary_response_is_reused_with_real_checkpoint(self):
        class PowerLoss(BaseException):
            pass
        original = TaskRunner.persist
        def crash(loop, event, payload=None):
            original(loop, event, payload)
            if event == 'model_response_received' and loop.state.get('compaction_pending'):
                raise PowerLoss()
        with patch.dict(compaction.CONFIG, self.policy), patch.object(TaskRunner, 'persist', crash), self.assertRaises(PowerLoss):
            self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        self.sdk.plans = iter([{'kind': 'blocked', 'note': 'fixture complete'}])
        with patch.dict(compaction.CONFIG, self.policy):
            result = self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'task_blocked')
        self.assertEqual(len(result['compactions']), 1)
        self.assertEqual(sum(o.get('tool_choice') == 'none' for _, o in self.sdk.calls), 1)

    def test_explicit_overflow_compresses_and_retries_only_once(self):
        self.sdk.plans = iter([{'kind': 'blocked', 'note': 'fixture complete'}])
        self.sdk.failures = 1
        result = self.engine().run([], 'value must return four', StepBudget(max_steps=10))
        self.assertEqual(result['status'], 'task_blocked', result)
        self.assertEqual(result['compactions'][0]['trigger'], 'context-overflow')
        self.assertEqual(len(self.sdk.calls), 5)  # two completed calls, rejection, summary, continuation
        self.assertEqual(self.sdk.calls[-1][1].get('tool_choice'), None)
        self.assertEqual(len(result['compactions']), 1)
        self.assertEqual(result['planner_calls'], 5)
