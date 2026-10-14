"""验证行为约定：使用真实输出，绝不为匹配模板而改写输出。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from agent.acceptance.artifact_checks import load_artifact_checks
from agent.acceptance.behavior import declaration_only
from agent.acceptance.c_driver import build_c_driver
from agent.acceptance.runner import AcceptanceRun
from agent.acceptance.contracts import VerificationSpec
from agent.execution.local import LocalEnvironment

CHECK = {
    'id': 'value',
    'kind': 'c_statement',
    'imports': 'from generator import render',
    'setup': '',
    'expression': 'render()',
    'variables': ['x'],
    'result_variable': 'y',
    'cases': [{'values': [0.0], 'expected': 1.0}],
}


def declaration(check=None):
    return 'ARTIFACT_CHECKS = ' + repr([CHECK if check is None else check]) + '\n'


class StatementContractTests(unittest.TestCase):
    def test_statement_contract_and_declaration_only_are_valid(self):
        self.assertEqual(load_artifact_checks(declaration()), [CHECK])
        VerificationSpec(
            ('numeric C output',), test_code=declaration(), required_consumer='c_expression'
        )
        self.assertTrue(declaration_only(declaration()))

    def test_expression_contract_is_backward_compatible(self):
        c = {k: v for k, v in CHECK.items() if k != 'result_variable'}
        c['kind'] = 'c_expression'
        self.assertEqual(load_artifact_checks(declaration(c)), [c])
        self.assertIn('return (\nx + 1\n);', build_c_driver(c, 'x + 1'))

    def test_output_identifier_is_required_and_bounded(self):
        for output in (None, [], 'x', 'double', 'NAN', 'a;return 1;', '_private'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                load_artifact_checks(declaration({**CHECK, 'result_variable': output}))
        with self.assertRaises(ValueError):
            load_artifact_checks(
                declaration({k: v for k, v in CHECK.items() if k != 'result_variable'})
            )

    def test_transforming_producer_output_is_still_forbidden(self):
        with self.assertRaises(ValueError):
            load_artifact_checks(
                declaration({**CHECK, 'expression': "render().replace('bad', 'good')"})
            )

    def test_statement_is_embedded_verbatim_and_unset_output_is_nan(self):
        for raw in ('y = x ? 2 : 1;', 'if (x) { y = 2; } else { y = 1; }'):
            built = build_c_driver(CHECK, raw)
            self.assertEqual(built.count(raw), 1)
            self.assertIn('double y = NAN;\n' + raw + '\nreturn y;', built)

    def test_code_outside_declaration_is_not_silently_ignored(self):
        for extra in ('\nprint(1)', '\nother = 1', '\nimport unittest'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                VerificationSpec(('C output',), test_code=declaration() + extra)
        self.assertFalse(declaration_only('other = ARTIFACT_CHECKS = []'))


class BehaviorOnlyRunnerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'generator.py').write_text('def render():\n    return "y = x + 1;"\n')
        env = LocalEnvironment(self.repo, artifact_root=self.root / 'runtime')
        self.runner = AcceptanceRun(
            self.repo,
            VerificationSpec(('C output',), test_code=declaration()),
            artifacts_root=self.root / 'results',
            environment=env,
        )

    @staticmethod
    def rows(compile_status='passed', case_status='passed'):
        return [
            {'id': 'artifact:value:compile', 'status': compile_status, 'detail': ''},
            {'id': 'artifact:value:case:0', 'status': case_status, 'detail': 'expected=1 actual=2'},
        ]

    def test_success_requires_real_consumer_result_not_dummy_unittest(self):
        with patch(
            'agent.acceptance.behavior.run_artifact_checks', return_value=self.rows()
        ) as consumer:
            report = self.runner.run('candidate')
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['tests_run'], 2)
        self.assertTrue(report['tests_ready'])
        self.assertEqual(consumer.call_count, 1)
        self.assertEqual(report['acceptance_mode'], 'artifact_behavior_only')

    def test_wrong_value_fails(self):
        with patch(
            'agent.acceptance.behavior.run_artifact_checks',
            return_value=self.rows(case_status='failed'),
        ):
            report = self.runner.run('candidate')
        self.assertEqual(report['status'], 'failed')
        self.assertTrue(report['tests_ready'])
        self.assertEqual(report['failures'], 1)

    def test_compile_failure_is_recorded_and_cases_are_not_run(self):
        with patch(
            'agent.acceptance.behavior.run_artifact_checks',
            return_value=self.rows('failed', 'not_run'),
        ):
            report = self.runner.run('candidate')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['tests_run'], 1)
        self.assertTrue(report['tests_ready'])

    def test_environment_failure_cannot_pass(self):
        with patch(
            'agent.acceptance.behavior.run_artifact_checks', side_effect=RuntimeError('no compiler')
        ):
            report = self.runner.run('candidate')
        self.assertEqual(report['status'], 'error')
        self.assertFalse(report['tests_ready'])

    def test_frozen_contract_tamper_prevents_execution(self):
        self.runner.test_path.write_text('ARTIFACT_CHECKS = []')
        with patch('agent.acceptance.behavior.run_artifact_checks') as consumer:
            report = self.runner.run('candidate')
        consumer.assert_not_called()
        self.assertFalse(report['integrity_ok'])
        self.assertEqual(report['status'], 'error')


if __name__ == '__main__':
    unittest.main()
