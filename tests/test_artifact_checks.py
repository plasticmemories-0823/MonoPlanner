"""Fixed consumer checks must defeat a green but misleading generated test suite."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.execution.local import LocalEnvironment
from agent.acceptance.artifact_checks import load_artifact_checks, required_consumer, requested_calls
from agent.task.budget import StepBudget
from agent.acceptance.runner import AcceptanceRun
from agent.acceptance.contracts import VerificationSpec


CHECK = {'id': 'generated', 'kind': 'c_expression',
         'imports': 'from generator import render', 'setup': '', 'expression': 'render()',
         'variables': ['x'], 'cases': [{'values': [0.0], 'expected': 1.0}]}
TEST = '''import unittest
from generator import render
class OutputTests(unittest.TestCase):
    def test_output(self):
        ARTIFACT_CHECKS.clear()  # Cannot alter the program's already-frozen checks.
        self.assertEqual(render().replace('Missing(x)', 'x + 1'), 'x + 1')
'''


def code(check=CHECK):
    return 'ARTIFACT_CHECKS = ' + repr([check]) + '\n' + TEST


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'generator.py').write_text('def render():\n    return "Missing(x)"\n', encoding='utf-8')
        self.environment = LocalEnvironment(self.repo, artifact_root=self.root/'commands')

    def test_only_explicit_c_output_requires_supported_consumer(self):
        self.assertEqual(required_consumer('Fix generated C code from render()'), 'c_expression')
        self.assertEqual(required_consumer('Output: Not supported in C'), 'c_expression')
        self.assertIsNone(required_consumer('修复 calculator.add，保留接口。'))

    def test_missing_checks_are_rejected_before_writing_or_ready(self):
        with self.assertRaisesRegex(ValueError, 'ARTIFACT_CHECKS'):
            VerificationSpec(('C output',), test_code=TEST, required_consumer='c_expression')

    def test_no_replacement_wrapper_producer_override_or_invalid_values(self):
        for fields in ({'expression': "render().replace('Missing(x)', 'x + 1')"},
                       {'setup': "render = lambda: 'x + 1'"},
                       {'variables': [[]]}, {'kind': 'pretend_c'},
                       {'imports': 'from generator import *'},
                       {'expression': 'render('}, {'cases': [{'values': [], 'expected': 1.0}]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                load_artifact_checks(code({**CHECK, **fields}), 'c_expression')
        self.assertEqual(load_artifact_checks(code()), [CHECK])

    def test_requested_call_chain_must_reach_the_raw_producer(self):
        required = requested_calls("render(Node(x)) doesn't work; generated C code is invalid")
        self.assertEqual(set(required), {'render', 'Node'})
        with self.assertRaisesRegex(ValueError, '调用链'):
            load_artifact_checks(code(), 'c_expression', required)
        valid = {**CHECK, 'imports':'from generator import render, Node as N',
                 'setup':'input_node = N(0)', 'expression':'render(input_node)'}
        self.assertEqual(load_artifact_checks(code(valid), 'c_expression', required), [valid])
        irrelevant = {**valid, 'expression':'render()'}
        with self.assertRaisesRegex(ValueError, '调用链'):
            load_artifact_checks(code(irrelevant), 'c_expression', required)

    def test_input_comparisons_and_unpacking_do_not_transform_raw_output(self):
        check = {**CHECK, 'setup':'x, y = (1, 2)', 'expression':'render(x < y)'}
        self.assertEqual(load_artifact_checks(code(check)), [check])

    @staticmethod
    def consumer(checks, environment, directory, timeout):
        assert checks == [CHECK]  # The model test cleared its own list, not this copy.
        valid = 'Missing' not in (environment.repo_root/'generator.py').read_text(encoding='utf-8')
        return [{'id':'artifact:generated:compile', 'status':'passed' if valid else 'failed',
                 'detail':'' if valid else "implicit declaration of function 'Missing'"},
                {'id':'artifact:generated:case:0', 'status':'passed' if valid else 'not_run', 'detail':''}]

    def test_raw_failure_overrides_green_model_tests(self):
        spec = VerificationSpec(('Valid raw C output',), test_code=code(), required_consumer='c_expression')
        run = AcceptanceRun(self.repo, spec, self.root/'acceptance', environment=self.environment)
        with patch('agent.acceptance.runner.run_artifact_checks', side_effect=self.consumer):
            report = run.run('candidate')
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['test_results'][0]['status'], 'passed')
        self.assertEqual(report['artifact_results'][0]['status'], 'failed')
        self.assertIn('Missing', report['artifact_results'][0]['detail'])
        self.assertTrue(report['tests_ready'], 'A recorded compile failure is usable candidate evidence')

    def test_unexplained_consumer_not_run_does_not_make_tests_ready(self):
        for compile_status in ('passed', 'error'):
            with self.subTest(compile_status=compile_status):
                rows = [{'id':'artifact:generated:compile', 'status':compile_status, 'detail':''},
                        {'id':'artifact:generated:case:0', 'status':'not_run', 'detail':''}]
                run = AcceptanceRun(self.repo, VerificationSpec(('C output',), test_code=code()),
                                    self.root/'acceptance', environment=self.environment)
                with patch('agent.acceptance.runner.run_artifact_checks', return_value=rows):
                    report = run.run('baseline')
                self.assertFalse(report['tests_ready'])

    def test_missing_consumer_result_cannot_reuse_green_status(self):
        run = AcceptanceRun(self.repo, VerificationSpec(('C output',), test_code=code()),
                            self.root/'acceptance', environment=self.environment)
        with patch('agent.acceptance.runner.run_artifact_checks', side_effect=RuntimeError('consumer missing')):
            report = run.run('candidate')
        self.assertEqual(report['status'], 'error')
        self.assertFalse(report['test_evidence_valid'])



if __name__ == '__main__':
    unittest.main()
