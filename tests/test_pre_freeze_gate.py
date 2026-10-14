"""恢复后的候选方案交付仍须在完整验收审查之后才能修改。"""

import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from agent.acceptance import revisions
from agent.model.payload import wire_input
from agent.model.tools import CODING_TOOLS
from support import interrupt_after_persist
import test_acceptance_review as a
import test_task_runner as f


class PreFreezeGateTests(unittest.TestCase):
    setUp = a.AcceptanceReviewTests.setUp
    candidate = a.AcceptanceReviewTests.candidate
    shell = a.AcceptanceReviewTests.shell
    run_replies = a.AcceptanceReviewTests.run_replies
    state = a.AcceptanceReviewTests.state

    def blocked(self):
        return f.reply({'kind': 'blocked', 'note': 'Preserve the rejected delivery'})

    def test_missing_complete_acceptance_cannot_execute_edits(self):
        result = self.run_replies([f.reply(self.candidate(test_code='')), self.blocked()])
        self.assertEqual(result['status'], 'task_blocked')
        self.assertIn('验收代码不能为空', self.model.calls[-1][-1].content)
        self.assertIn('return 0', (self.repo / 'sample.py').read_text())
        self.assertFalse(result['rounds'])
        self.assertFalse(list((self.root / 'state').glob('plans/*/*/result.json')))

    def test_whitespace_acceptance_and_driver_cannot_bypass_gate(self):
        result = self.run_replies(
            [f.reply(self.candidate(test_code=' \n', test_driver=['-m', 'pytest'])), self.blocked()]
        )
        self.assertEqual(result['status'], 'task_blocked')
        self.assertIn('验收代码不能为空', self.model.calls[-1][-1].content)
        self.assertIn('return 0', (self.repo / 'sample.py').read_text())

    def test_inspection_without_complete_acceptance_can_advance(self):
        inspection = {
            'kind': 'inspect',
            'note': 'Read the implementation before deciding',
            'steps': [{'id': 'source', 'op': 'read', 'path': 'sample.py'}],
        }

        def candidate(messages):
            value = json.loads(messages[-1].content)
            self.assertEqual(value['outcome'], 'evidence_ready')
            self.assertIn('return 0', (self.repo / 'sample.py').read_text())
            self.assertFalse(self.engine.state.get('verification_spec'))
            return f.reply(self.candidate())

        result = self.run_replies([f.reply(inspection), candidate, a.review])
        self.assertEqual(result['status'], 'verified_success')
        self.assertEqual(len(result['rounds']), 1)
        self.assertNotIn(
            'trial_acceptance',
            [t['function']['name'] for t in wire_input([], CODING_TOOLS)['tools']],
        )

    def test_original_trial_and_explicit_freeze_precede_edits(self):
        def freeze(messages):
            report = json.loads(messages[-1].content)
            self.assertEqual(report['baseline']['status'], 'failed')
            self.assertIn('No proposed modification has run', report['next'])
            self.assertIn('return 0', (self.repo / 'sample.py').read_text())
            self.assertFalse(self.engine.state.get('verification_spec'))
            return a.review(messages)

        result = self.run_replies([f.reply(self.candidate()), freeze])
        self.assertEqual(result['status'], 'verified_success')
        self.assertEqual((self.repo / 'sample.py').read_text(), 'def value():\n    return 4\n')

    def test_invalid_partial_replacement_preserves_pending_complete_candidate(self):
        def invalid(messages):
            self.trial = json.loads(messages[-1].content)['trial_id']
            return f.reply(self.candidate(test_code=''))

        def freeze(messages):
            self.assertIn('验收代码不能为空', messages[-1].content)
            self.assertIn('return 0', (self.repo / 'sample.py').read_text())
            self.assertEqual(self.engine.state['acceptance_trial']['ref']['id'], self.trial)
            return AIMessage(
                content='',
                tool_calls=[
                    {
                        'name': 'freeze_acceptance',
                        'id': 'freeze',
                        'args': {
                            'trial_id': self.trial,
                            'reason': 'Freeze the unchanged valid candidate',
                        },
                    }
                ],
            )

        result = self.run_replies(
            [f.reply(self.candidate(2)), invalid, freeze, f.reply(self.candidate(test_code=''))]
        )
        self.assertEqual(result['status'], 'verified_success')
        self.assertEqual(
            [r['verification']['status'] for r in result['rounds']], ['failed', 'passed']
        )
        self.assertEqual(len(self.state()['acceptance_trials']), 1)
        self.assertEqual(self.state()['verification_spec']['test_code'], f.CODE)

    def test_real_failure_repair_keeps_frozen_assertions(self):
        result = self.run_replies(
            [f.reply(self.candidate(2)), a.review, f.reply(self.candidate(test_code=''))]
        )
        self.assertEqual(result['status'], 'verified_success')
        self.assertEqual(
            [r['verification']['status'] for r in result['rounds']], ['failed', 'passed']
        )
        self.assertEqual(self.state()['verification_spec']['test_code'], f.CODE)

    def test_resume_after_frozen_edit_does_not_replay_completed_operation(self):
        plan = self.candidate()
        plan['steps'].insert(
            0,
            self.shell(
                "from pathlib import Path; p=Path('counter'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1')",
                'count',
            ),
        )
        interrupt, error = interrupt_after_persist('step_finished')
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(plan), a.review])
        self.assertEqual((self.repo / 'counter').read_text(), '1')
        result = self.run_replies([])
        self.assertEqual(result['status'], 'verified_success')
        self.assertEqual((self.repo / 'counter').read_text(), '1')
        self.assertFalse(self.model.calls)

    def test_unknown_original_trial_never_replays_request_or_edits(self):
        interrupt, error = interrupt_after_persist('acceptance_trial_started')
        with interrupt, self.assertRaises(error):
            self.run_replies([f.reply(self.candidate())])
        with patch.object(revisions, 'run_report', side_effect=AssertionError('must not retry')):
            result = self.run_replies([])
        self.assertEqual(result['status'], 'execution_unknown')
        self.assertFalse(self.model.calls)
        self.assertIn('return 0', (self.repo / 'sample.py').read_text())


if __name__ == '__main__':
    unittest.main()
