"""假模型走真实试跑、冻结、版本比较和断点恢复；不调用付费模型。"""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent.acceptance import revisions
from agent.acceptance.evidence import QUOTE_ERROR, RULE_VERSION
from agent.storage.recovery import code_key
from agent.source.snapshots import SourceSnapshot
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from test_acceptance_review import ScriptModel, review
import test_acceptance_review as review_tests
import test_task_runner as f


WRONG = f.CODE.replace('value(), 4', 'value(), 2')


class EvidenceAcceptanceFlow(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp
    candidate = f.TaskRunnerTests.candidate
    shell = f.TaskRunnerTests.shell
    state = review_tests.AcceptanceReviewTests.state

    def run_replies(self, replies, **kw):
        self.model = ScriptModel(replies)
        self.engine = TaskRunner(self.model, self.env, state_dir=self.root / 'state', **kw)
        return self.engine.run([], 'return 4', StepBudget(max_steps=12))

    def correct(self, messages, *, code=f.CODE, quote='return 4', source='task_request',
                before='value(), 2', after='value(), 4', **changes):
        state = self.engine.state
        ref = state['acceptance_trial']['ref'] if state.get('acceptance_trial') else state['active_acceptance']
        args = dict(trial_id=ref['id'], reason='The independent original requires 4',
                    test_code=code, test_driver=[],
                    evidence=[{'id': 'requirement', 'source': source, 'quote': quote}],
                    corrections=[{'before': before, 'after': after,
                                  'evidence_ids': ['requirement'],
                                  'reason': 'The original task says return 4, not 2'}])
        args.update(changes)
        return AIMessage(content='', tool_calls=[{
            'name': 'revise_acceptance', 'id': 'correct-call', 'args': args,
        }], usage_metadata={'input_tokens': 10, 'output_tokens': 10, 'total_tokens': 20})

    def initial(self, code=WRONG):
        return [f.reply(self.candidate(test_code=code)), review]

    def comparison(self, result):
        ref = result['verification']['acceptance_comparison']['record'][8:]
        return json.loads((self.root / 'state' / ref).read_text('utf-8'))

    def test_wrong_frozen_expectation_corrected_retrialled_and_refrozen(self):
        result = self.run_replies(self.initial() + [
            self.correct, review, f.reply(self.candidate(test_code='')),
        ])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(result['planner_calls'], 6)
        self.assertEqual(len(result['rounds']), 2)  # 修订不抹掉已消耗的验收轮次。
        self.assertEqual(len(self.state()['acceptance_versions']), 2)
        data = self.comparison(result)
        old, new = data['versions']
        self.assertEqual(old['candidate']['status'], 'failed')
        self.assertEqual(old['effective_candidate']['status'], 'passed')
        self.assertEqual(old['superseded_acceptance_by'], [new['trial_id']])
        self.assertEqual(data['conflicts'], [])
        version = revisions.checked(self.engine.root, self.state()['active_acceptance'])
        correction = version['acceptance_correction']
        self.assertEqual(correction['rule_version'], RULE_VERSION)
        self.assertFalse(correction['semantic_correctness_proven'])
        self.assertEqual(correction['evidence'][0]['quote'], 'return 4')
        self.assertEqual(revisions.checked(self.engine.root, version['baseline_result'])['status'], 'failed')

    def test_bad_quote_can_retry_without_changing_acceptance_or_workspace(self):
        saved = {}

        def wrong(messages):
            saved.update(spec=deepcopy(self.engine.state['verification_spec']),
                         source=(self.repo / 'sample.py').read_bytes(),
                         trials=len(self.engine.state['acceptance_trials']))
            return self.correct(messages, quote='return  4')

        def retry(messages):
            self.assertIn(QUOTE_ERROR, messages[-1].content)
            self.assertEqual(self.engine.state['verification_spec'], saved['spec'])
            self.assertEqual((self.repo / 'sample.py').read_bytes(), saved['source'])
            self.assertEqual(len(self.engine.state['acceptance_trials']), saved['trials'])
            return self.correct(messages)

        result = self.run_replies(self.initial() + [wrong, retry, review,
                                                  f.reply(self.candidate(test_code=''))])
        self.assertEqual(result['status'], 'verified_success', result)

    def test_reject_undeclared_change_missing_reference_and_driver(self):
        def rejected(messages):
            old_state = deepcopy(self.engine.state)
            old_file = (self.repo / 'sample.py').read_bytes()
            for kw in [dict(code=f.CODE + '# undeclared\n'),
                       dict(corrections=[{'before': '2', 'after': '4', 'evidence_ids': ['missing'],
                                          'reason': 'wrong reference'}]),
                       dict(test_driver=['other.py', '{test_module}']),
                       dict(code=f.CODE.replace('value()', 'value(4)'))]:
                response = self.correct(messages, **kw)
                decision = revisions.Review(decision='revise', **response.tool_calls[0]['args'])
                with self.subTest(kw=kw), self.assertRaises(ValueError):
                    revisions.review(self.engine, decision, {})
                self.assertEqual(self.engine.state, old_state)
                self.assertEqual((self.repo / 'sample.py').read_bytes(), old_file)
            return self.correct(messages)

        result = self.run_replies(self.initial() + [rejected, review,
                                                  f.reply(self.candidate(test_code=''))])
        self.assertEqual(result['status'], 'verified_success', result)

    def test_evidence_cannot_change_user_fixed_command(self):
        def blocked(messages):
            decision = revisions.Review(decision='revise', **self.correct(messages).tool_calls[0]['args'])
            self.engine.state['verify_command'] = 'fixed user command'
            try:
                with self.assertRaisesRegex(ValueError, 'User fixed'):
                    revisions.review(self.engine, decision, {})
            finally:
                self.engine.state['verify_command'] = None
            return f.reply({'kind': 'blocked', 'note': 'fixed command was preserved'})

        result = self.run_replies(self.initial() + [blocked])
        self.assertEqual(result['status'], 'task_blocked')

    def test_correction_does_not_hide_unchanged_failure(self):
        extra = '\n    def test_unchanged(self):\n        self.assertEqual(value(), 5)\n'
        result = self.run_replies(self.initial(WRONG + extra) + [
            lambda m: self.correct(m, code=f.CODE + extra), review,
            f.reply(self.candidate(test_code='')),
            f.reply({'kind': 'blocked', 'note': 'Unchanged check still fails'}),
        ])
        self.assertEqual(result['status'], 'task_blocked')
        self.assertEqual(result['verification']['status'], 'failed')
        self.assertIn('test_unchanged', str(result['verification']))

    def test_correction_does_not_remove_undeclared_setup_failure(self):
        setup = "from sample import value\nif value() != 0:\n    raise RuntimeError('setup conflict')\n"
        result = self.run_replies(self.initial(setup + WRONG) + [
            lambda m: self.correct(m, code=setup + f.CODE), review,
            f.reply(self.candidate(test_code='')),
            f.reply({'kind': 'blocked', 'note': 'Unchanged setup still fails'}),
        ])
        self.assertEqual(result['status'], 'task_blocked', result)
        data = self.comparison(result)
        self.assertIn(setup, data['versions'][0]['effective_spec']['test_code'])
        self.assertNotEqual(data['versions'][0]['effective_candidate']['status'], 'passed')

    def test_multiple_corrections_and_setup_revision_keep_supersession_chain(self):
        def to_three(messages):
            return self.correct(messages, code=f.CODE.replace('4)', '3)'), after='value(), 3')

        result = self.run_replies(self.initial() + [to_three, review,
            lambda m: self.correct(m, before='value(), 3'), review,
            f.reply(self.candidate(test_code=''))])
        self.assertEqual(result['status'], 'verified_success', result)
        data = self.comparison(result)
        self.assertEqual(len(data['versions']), 3)
        self.assertEqual(len(data['versions'][0]['superseded_acceptance_by']), 2)
        self.assertFalse(data['conflicts'])

    def test_pending_correction_and_frozen_version_resume_without_replay(self):
        class Interrupted(BaseException):
            pass

        persist = TaskRunner.persist

        def stop(engine, event, payload=None):
            persist(engine, event, payload)
            if event == 'acceptance_trial_finished' and engine.state.get('verification_spec'):
                raise Interrupted()

        with patch.object(TaskRunner, 'persist', stop), self.assertRaises(Interrupted):
            self.run_replies(self.initial() + [self.correct])
        self.assertEqual(self.state()['verification_spec']['test_code'], WRONG)
        result = self.run_replies([review, f.reply(self.candidate(test_code=''))])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(result['calls_this_invocation'], 2)
        self.assertEqual(result['planner_calls'], 6)
        self.assertEqual(len(self.state()['acceptance_trials']), 2)

    def test_evidence_fingerprint_tampering_stops_before_freeze(self):
        def tamper(messages):
            trial = self.engine.state['acceptance_trial']
            path = self.engine.root / trial['ref']['record']
            record = json.loads(path.read_text('utf-8'))
            record['acceptance_correction']['evidence'][0]['quote'] = 'return 2'
            path.write_text(json.dumps(record), encoding='utf-8')
            return review(messages)

        result = self.run_replies(self.initial() + [self.correct, tamper])
        self.assertEqual(result['status'], 'state_conflict', result)
        self.assertIn('Acceptance record changed', result['stop_reason'])
        self.assertEqual(self.state()['verification_spec']['test_code'], WRONG)

    def test_new_validator_modules_participate_in_restore_identity(self):
        from agent.settings import PROJECT_ROOT

        original = code_key()
        for name in ['agent/acceptance/evidence.py', 'agent/acceptance/corrections.py']:
            changed = self.root / 'changed.py'
            changed.write_bytes((PROJECT_ROOT / name).read_bytes() + b'\n')

            class Root:
                def __truediv__(self, item):
                    return changed if item == name else PROJECT_ROOT / item

            with patch('agent.storage.recovery.PROJECT_ROOT', Root()):
                self.assertNotEqual(code_key(), original)

    def test_original_file_evidence_survives_live_workspace_edit(self):
        (self.repo / 'spec.txt').write_bytes(b'The value must be 4.\r\n')

        def use_original(messages):
            return self.correct(messages, source='original_file:spec.txt',
                                quote='The value must be 4.\r\n')

        plan = self.candidate(test_code=WRONG)
        plan['steps'].append(self.shell(
            "from pathlib import Path; Path('spec.txt').write_text('a candidate output says 2')", 'edit-doc'
        ))
        result = self.run_replies([f.reply(plan), review, use_original, review,
                                  f.reply(self.candidate(test_code=''))])
        self.assertEqual(result['status'], 'verified_success', result)

    def test_unknown_correction_trial_is_not_replayed_or_frozen(self):
        class Interrupted(BaseException):
            pass

        persist = TaskRunner.persist

        def stop(engine, event, payload=None):
            persist(engine, event, payload)
            if event == 'acceptance_trial_started' and engine.state.get('verification_spec'):
                raise Interrupted()

        with patch.object(TaskRunner, 'persist', stop), self.assertRaises(Interrupted):
            self.run_replies(self.initial() + [self.correct])
        with patch('agent.acceptance.revisions.run_report', side_effect=AssertionError('no replay')):
            result = self.run_replies([])
        self.assertEqual(result['status'], 'execution_unknown', result)
        self.assertFalse(self.model.calls)
        self.assertEqual(self.state()['verification_spec']['test_code'], WRONG)

    def test_replaced_snapshot_with_self_consistent_manifest_is_not_original_evidence(self):
        def replace_baseline(messages):
            self.engine.baseline = SourceSnapshot.capture(self.repo, self.root / 'substitute')
            return self.correct(messages)

        result = self.run_replies(self.initial() + [replace_baseline])
        self.assertEqual(result['status'], 'state_conflict', result)
        self.assertIn('snapshot identity changed', result['stop_reason'])
        self.assertEqual(len(self.state()['acceptance_versions']), 1)


if __name__ == '__main__':
    unittest.main()
