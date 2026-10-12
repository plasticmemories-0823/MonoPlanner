"""Incremental edits are real work; only complete frozen acceptance establishes success."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest

from langchain_core.messages import AIMessage
from agent.acceptance import revisions
from agent.model.payload import wire_input
from agent.model.tools import CODING_TOOLS
from agent.task.operations import final_acceptance
from support import interrupt_after_persist
import test_acceptance_review as a
import test_task_runner as f


class IncrementalWorkTests(unittest.TestCase):
    setUp = a.AcceptanceReviewTests.setUp
    candidate = a.AcceptanceReviewTests.candidate
    shell = a.AcceptanceReviewTests.shell
    run_replies = a.AcceptanceReviewTests.run_replies
    state = a.AcceptanceReviewTests.state

    def partial(self, value=4, **extra):
        return self.candidate(value, test_code="", **extra)

    def final(self):
        return self.candidate(steps=[])

    def test_real_edits_and_local_checks_precede_complete_acceptance(self):
        first = self.partial(2)
        first['steps'].append(self.shell('from sample import value; assert value() == 2', 'partial-check'))

        def continue_work(messages):
            result = json.loads(messages[-1].content)
            self.assertEqual(result['outcome'], 'partial_work_executed')
            self.assertEqual(len(result['steps']), 2)
            self.assertNotIn('verification_spec', self.engine.state)
            self.assertEqual(self.engine.state['rounds'], [])
            return f.reply(self.partial())

        def finish(messages):
            self.assertEqual(json.loads(messages[-1].content)['outcome'], 'partial_work_executed')
            return f.reply(self.final())

        def freeze(messages):
            report = json.loads(messages[-1].content)
            self.assertEqual(report['baseline']['status'], 'failed')
            self.assertIn('return 0', (self.engine.baseline.root/'sample.py').read_text())
            self.assertIn('return 4', (self.repo/'sample.py').read_text())
            return a.review(messages)

        result = self.run_replies([f.reply(first), continue_work, finish, freeze])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(result['rounds']), 1)
        self.assertTrue(self.engine.baseline.verify_integrity()['valid'])
        patch_text = (self.root/'state/candidate.patch').read_text()
        self.assertIn('-    return 0', patch_text)
        self.assertIn('+    return 4', patch_text)
        self.assertNotIn('trial_acceptance', [t['function']['name'] for t in wire_input([], CODING_TOOLS)['tools']])

    def test_partial_checks_and_text_cannot_claim_task_success(self):
        result = self.run_replies([f.reply(self.partial()), AIMessage(content='Done')], interactive=True)
        self.assertEqual(result['status'], 'awaiting_user')
        self.assertEqual(result['rounds'], [])
        self.assertFalse((self.root/'state/frozen-acceptance.json').exists())
        with self.assertRaisesRegex(ValueError, 'Complete frozen acceptance'):
            final_acceptance(self.engine)

    def test_failed_local_check_stops_later_edits_without_final_round(self):
        plan = self.partial(2)
        plan['steps'] += [self.shell('from sample import value; assert value() == 4', 'check'),
                          self.shell("from pathlib import Path; Path('forbidden').write_text('x')", 'later')]
        result = self.run_replies([f.reply(plan), f.reply({'kind': 'blocked', 'note': 'Local check failed'})])
        self.assertEqual(result['status'], 'task_blocked')
        self.assertEqual(result['rounds'], [])
        self.assertFalse((self.repo/'forbidden').exists())
        self.assertIn('return 2', (self.repo/'sample.py').read_text())

    def test_final_failure_still_requires_source_repair_under_same_tests(self):
        result = self.run_replies([f.reply(self.partial(2)), f.reply(self.final()), a.review,
                                  f.reply(self.partial())])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual([r['verification']['status'] for r in result['rounds']], ['failed', 'passed'])
        self.assertEqual(self.state()['verification_spec']['test_code'], f.CODE)

    def test_incremental_replacement_invalidates_old_pending_approval(self):
        old = []

        def incremental(messages):
            old.append(json.loads(messages[-1].content)['trial_id'])
            return f.reply(self.partial())

        def stale_freeze(messages):
            return AIMessage(content='', tool_calls=[{'name': 'freeze_acceptance', 'id': 'stale',
                'args': {'trial_id': old[0], 'reason': 'Old approval must not release old edits'}}])

        def final_delivery(messages):
            self.assertIn('current acceptance trial_id', messages[-1].content)
            self.assertIn('return 4', (self.repo/'sample.py').read_text())
            return f.reply(self.final())

        result = self.run_replies([f.reply(self.candidate(2)), incremental, stale_freeze, final_delivery, a.review])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(self.state()['acceptance_trials']), 2)
        self.assertIn('return 4', (self.repo/'sample.py').read_text())
        events = [json.loads(line) for line in (self.root/'state/events.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['kind'] == 'acceptance_trial_superseded' for e in events), 1)

    def test_resume_does_not_replay_completed_incremental_edit(self):
        plan = self.partial()
        plan['steps'].insert(0, self.shell("from pathlib import Path; p=Path('counter'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')", 'count'))
        interrupt, error = interrupt_after_persist('step_finished')
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(plan)])
        result = self.run_replies([f.reply(self.final()), a.review])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual((self.repo/'counter').read_text(), '1')
        self.assertEqual(result['calls_this_invocation'], 2)

    def test_unknown_trial_stops_before_incremental_work_or_model_replay(self):
        interrupt, error = interrupt_after_persist('acceptance_trial_started')
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.final())])
        with patch.object(revisions, 'run_report', side_effect=AssertionError('must not retry')):
            result = self.run_replies([f.reply(self.partial())])
        self.assertEqual(result['status'], 'execution_unknown')
        self.assertEqual(len(self.model.calls), 0)
        self.assertIn('return 0', (self.repo/'sample.py').read_text())

    def test_driver_without_complete_test_is_rejected_without_edits(self):
        result = self.run_replies([f.reply(self.partial(test_driver=['-m', 'pytest'])),
                                  f.reply({'kind': 'blocked', 'note': 'Rejected invalid driver'})])
        self.assertEqual(result['status'], 'task_blocked')
        self.assertIn('test_driver requires complete test_code', self.model.calls[-1][-1].content)
        self.assertIn('return 0', (self.repo/'sample.py').read_text())


if __name__ == '__main__':
    unittest.main()
