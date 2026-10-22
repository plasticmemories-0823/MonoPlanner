"""独立证据可纠正自拟验收，不能更改用户确认的目标。"""

from copy import deepcopy
import json
import unittest

from agent.acceptance import revisions
import test_evidence_acceptance_flow as flow
import test_task_runner as task


FIXTURE_WRONG = task.CODE + "        data = {'lower': 4}\n        self.assertEqual(data['UPPER'], 4)\n"
FIXTURE_FIXED = FIXTURE_WRONG.replace("{'lower': 4}", "{'UPPER': 4}")
ASSERTION_FIXED = FIXTURE_WRONG.replace(
    "self.assertEqual(data['UPPER'], 4)", "self.assertNotIn('UPPER', data)"
)


class AcceptanceRevisionScope(unittest.TestCase):
    setUp = flow.EvidenceAcceptanceFlow.setUp
    candidate = flow.EvidenceAcceptanceFlow.candidate
    shell = flow.EvidenceAcceptanceFlow.shell
    state = flow.EvidenceAcceptanceFlow.state
    run_replies = flow.EvidenceAcceptanceFlow.run_replies
    initial = flow.EvidenceAcceptanceFlow.initial
    correct = flow.EvidenceAcceptanceFlow.correct
    comparison = flow.EvidenceAcceptanceFlow.comparison

    def add_evidence(self):
        (self.repo / 'contract.txt').write_bytes(b'Keys are case sensitive. Uppercase and lowercase names are distinct.\n')

    def fix_case(self, messages, *, code=FIXTURE_FIXED, before="{'lower': 4}",
                 after="{'UPPER': 4}", **changes):
        return self.correct(
            messages, code=code, before=before, after=after,
            source='original_file:contract.txt',
            quote='Keys are case sensitive. Uppercase and lowercase names are distinct.\n',
            **changes,
        )

    def test_fixture_correction_retrials_and_preserves_the_original_failure(self):
        self.add_evidence()
        result = self.run_replies(self.initial(FIXTURE_WRONG) + [
            self.fix_case, flow.review, task.reply(self.candidate(test_code='')),
        ])
        self.assertEqual(result['status'], 'verified_success', result)
        old, new = self.comparison(result)['versions']
        self.assertEqual(old['candidate']['status'], 'error')
        self.assertEqual(old['effective_candidate']['status'], 'passed')
        self.assertEqual(new['candidate']['status'], 'passed')
        self.assertEqual(len(self.state()['acceptance_versions']), 2)
        self.assertEqual(len(result['rounds']), 2)

    def test_assertion_type_can_be_corrected_with_original_evidence(self):
        self.add_evidence()
        result = self.run_replies(self.initial(FIXTURE_WRONG) + [
            lambda m: self.fix_case(
                m, code=ASSERTION_FIXED,
                before="self.assertEqual(data['UPPER'], 4)",
                after="self.assertNotIn('UPPER', data)",
            ), flow.review, task.reply(self.candidate(test_code='')),
        ])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertFalse(self.comparison(result)['conflicts'])

    def test_missing_evidence_returns_actionable_guidance_without_a_trial(self):
        self.add_evidence()
        saved = {}

        def no_evidence(messages):
            saved.update(trials=len(self.engine.state['acceptance_trials']),
                         spec=deepcopy(self.engine.state['verification_spec']))
            return self.fix_case(messages, evidence=[], corrections=[])

        def corrected(messages):
            self.assertIn('Provide valid original evidence', messages[-1].content)
            self.assertIn('original requirements', messages[-1].content.lower())
            self.assertEqual(len(self.engine.state['acceptance_trials']), saved['trials'])
            self.assertEqual(self.engine.state['verification_spec'], saved['spec'])
            return self.fix_case(messages)

        result = self.run_replies(self.initial(FIXTURE_WRONG) + [
            no_evidence, corrected, flow.review, task.reply(self.candidate(test_code='')),
        ])
        self.assertEqual(result['status'], 'verified_success', result)

    def test_generated_driver_and_execution_details_can_be_revised(self):
        driver = (
            "import sys, unittest\n"
            "suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])\n"
            "result = unittest.TextTestRunner().run(suite)\n"
            "sys.exit(0 if result.wasSuccessful() else 1)\n"
        )
        (self.repo / 'run_selected.py').write_bytes(driver.encode())
        checks = [deepcopy(task.EXECUTION_CHECK)]
        checks[0]['requirement_id'] = ''
        # 名称为模型自拟；原始需求绑定的入口、运行环境和参数保持不变。
        checks[0]['id'] = 'corrected_public_value'
        new_driver = ['run_selected.py', '{test_module}']

        def correct_materials(messages):
            old = self.engine.state['verification_spec']
            return self.correct(messages, test_driver=new_driver, execution_checks=checks,
                evidence=[{'id': 'requirement', 'source': 'task_request', 'quote': 'return 4'},
                          {'id': 'driver', 'source': 'original_file:run_selected.py', 'quote': driver}],
                corrections=[
                    {'field': 'test_code', 'before': 'value(), 2', 'after': 'value(), 4',
                     'evidence_ids': ['requirement'], 'reason': 'Preserve the original expected result'},
                    {'field': 'test_driver', 'before': json.dumps(old['test_driver']),
                     'after': json.dumps(new_driver), 'evidence_ids': ['driver'],
                     'reason': 'Use the recorded repository test runner'},
                    {'field': 'execution_checks', 'before': json.dumps(old['execution_checks']),
                     'after': json.dumps(checks), 'evidence_ids': ['requirement'],
                     'reason': 'Name the check for the same confirmed public behavior'},
                ])

        result = self.run_replies(self.initial() + [
            correct_materials, flow.review,
            task.reply(self.candidate(test_code='', execution_checks=[])),
        ])
        self.assertEqual(result['status'], 'verified_success', result)
        spec = self.state()['verification_spec']
        self.assertEqual(spec['test_driver'], new_driver)
        self.assertEqual(spec['execution_checks'], checks)
        old = self.comparison(result)['versions'][0]
        self.assertEqual(old['effective_candidate']['status'], 'passed')

    def test_generated_regression_command_can_be_corrected_without_budget_reset(self):
        old_command = self.shell('raise SystemExit(1)')['command']
        new_command = self.shell('from sample import value; assert value() == 4')['command']

        def correct_regression(messages):
            return self.correct(messages, code=task.CODE, regression_command=new_command,
                corrections=[{'field': 'regression_command', 'before': old_command,
                              'after': new_command, 'evidence_ids': ['requirement'],
                              'reason': 'Check the required public result instead of unconditional failure'}])

        plan = self.candidate(regression_command=old_command)
        result = self.run_replies([task.reply(plan), flow.review, correct_regression,
                                  flow.review, task.reply(self.candidate(test_code=''))])
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(self.state()['regression_command'], new_command)
        self.assertEqual(len(result['rounds']), 2)
        old = self.comparison(result)['versions'][0]
        self.assertEqual(old['effective_regression'], new_command)

    def test_original_target_cannot_change_even_with_a_valid_quote(self):
        saved = {}

        def wrong_target(messages):
            saved.update(spec=deepcopy(self.engine.state['verification_spec']),
                         trials=len(self.engine.state['acceptance_trials']),
                         source=(self.repo / 'sample.py').read_bytes())
            checks = deepcopy(saved['spec']['execution_checks'])
            checks[0]['entry_file'] = 'substitute.py'
            return self.correct(messages, execution_checks=checks,
                corrections=[{'field': 'test_code', 'before': 'value(), 2', 'after': 'value(), 4',
                              'evidence_ids': ['requirement'], 'reason': 'Correct result'},
                             {'field': 'execution_checks',
                              'before': json.dumps(saved['spec']['execution_checks']),
                              'after': json.dumps(checks), 'evidence_ids': ['requirement'],
                              'reason': 'This must not authorize a substitute target'}])

        def stop(messages):
            self.assertIn('confirmed', messages[-1].content)
            self.assertIn('cannot be changed', messages[-1].content)
            self.assertEqual(self.engine.state['verification_spec'], saved['spec'])
            self.assertEqual(len(self.engine.state['acceptance_trials']), saved['trials'])
            self.assertEqual((self.repo / 'sample.py').read_bytes(), saved['source'])
            return task.reply({'kind': 'blocked', 'note': 'Original target remains protected'})

        result = self.run_replies(self.initial() + [wrong_target, stop])
        self.assertEqual(result['status'], 'task_blocked', result)

    def test_evidence_does_not_allow_zero_case_acceptance_to_freeze(self):
        empty = task.CODE + '\ndef load_tests(loader, tests, pattern):\n    return unittest.TestSuite()\n'
        result = self.run_replies(self.initial() + [
            lambda m: self.correct(m, code=empty, before=flow.WRONG, after=empty),
            flow.review, task.reply({'kind': 'blocked', 'note': 'Zero tests are not acceptance'}),
        ])
        self.assertEqual(result['status'], 'task_blocked', result)
        self.assertEqual(len(self.state()['acceptance_versions']), 1)
        self.assertEqual(self.state()['verification_spec']['test_code'], flow.WRONG)
        self.assertIn('not runnable', self.model.calls[-1][-1].content)


if __name__ == '__main__':
    unittest.main()
