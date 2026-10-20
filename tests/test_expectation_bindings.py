"""预期变量的 with 块、顺序重绑定及保守拒绝边界；全程离线。"""

import unittest
from unittest.mock import patch

from agent.acceptance.expectations import validate_change
from agent.task.runner import TaskRunner
from test_expectation_corrections import module
import test_evidence_acceptance_flow as flow
import test_task_runner as task_tests


class ExpectedBindings(unittest.TestCase):
    def test_with_block_local_expected_literal(self):
        code = module(
            "        with context() as resource:\n"
            "            expected = [1, 2]\n"
            "            self.assertEqual(value(resource), expected)\n"
        )
        validate_change(code, code.replace("[1, 2]", "[2, 1]"))

    def test_nested_with_and_async_with(self):
        for header in ["with context():", "async with context():"]:
            code = module(
                "        with outer():\n"
                f"            {header}\n"
                "                expected = {'order': (1, 2)}\n"
                "                self.assertEqual(value(), expected)\n"
            )
            if header.startswith("async"):
                code = code.replace("def test_value", "async def test_value")
            with self.subTest(header=header):
                validate_change(code, code.replace("(1, 2)", "(2, 1)"))

    def test_each_sequential_binding_owns_its_expectation(self):
        code = module(
            "        expected = [1, 2]\n"
            "        self.assertEqual(first(), expected)\n"
            "        expected = {'pair': (3, 4)}\n"
            "        self.assertEqual(second(), expected)\n"
        )
        for revised in [code.replace("[1, 2]", "[2, 1]"),
                        code.replace("(3, 4)", "(4, 3)"),
                        code.replace("[1, 2]", "[2, 1]").replace("(3, 4)", "(4, 3)")]:
            with self.subTest(revised=revised):
                validate_change(code, revised)

    def test_repeated_binding_inside_with_and_python_assert(self):
        code = (
            "def test_value():\n"
            "    with context():\n"
            "        expected = [1, 2]\n"
            "        assert first() == expected\n"
            "        expected = (3, 4)\n"
            "        assert expected == second()\n"
        )
        validate_change(code, code.replace("[1, 2]", "[2, 1]").replace("(3, 4)", "(4, 3)"))

    def test_independent_with_blocks_and_unrelated_statement(self):
        code = module(
            "        with first_context():\n"
            "            expected = [1, 2]\n"
            "            actual = first()\n"
            "            self.assertEqual(actual, expected)\n"
            "        with second_context():\n"
            "            expected = [3, 4]\n"
            "            self.assertEqual(second(), expected)\n"
        )
        validate_change(code, code.replace("[1, 2]", "[2, 1]").replace("[3, 4]", "[4, 3]"))

    def test_unused_overwritten_binding_cannot_be_changed(self):
        code = module(
            "        expected = [1, 2]\n"
            "        expected = [3, 4]\n"
            "        self.assertEqual(value(), expected)\n"
        )
        with self.assertRaises(ValueError):
            validate_change(code, code.replace("[1, 2]", "[2, 1]"))
        validate_change(code, code.replace("[3, 4]", "[4, 3]"))

    def test_input_alias_mutation_and_closure_still_reject(self):
        for extra in ["use(expected)", "alias = expected", "expected.append(3)",
                      "callback = lambda: expected", "print(locals())"]:
            code = module(
                "        with context():\n"
                "            expected = [1, 2]\n"
                "            self.assertEqual(first(), expected)\n"
                f"            {extra}\n"
                "            expected = [3, 4]\n"
                "            self.assertEqual(second(), expected)\n"
            )
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_change(code, code.replace("[1, 2]", "[2, 1]"))

    def test_ambiguous_control_flow_and_other_bindings_still_reject(self):
        for extra in [
            "        if condition():\n            expected = [3, 4]\n",
            "        for expected in values():\n            pass\n",
            "        try:\n            value()\n        except Exception as expected:\n            pass\n",
            "        with context() as expected:\n            pass\n",
            "        expected = make_expected()\n",
            "        expected += [3]\n",
            "        del expected\n",
            "        import other as expected\n",
        ]:
            code = module("        expected = [1, 2]\n" + extra
                          + "        self.assertEqual(value(), expected)\n")
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_change(code, code.replace("[1, 2]", "[2, 1]"))

    def test_with_exit_does_not_guess_a_reaching_assignment(self):
        for body in [
            "        expected = [1, 2]\n"
            "        with context():\n"
            "            expected = [3, 4]\n"
            "            self.assertEqual(first(), expected)\n"
            "        self.assertEqual(second(), expected)\n",
            "        with context():\n"
            "            expected = [1, 2]\n"
            "            self.assertEqual(first(), expected)\n"
            "        use(expected)\n",
            "        self.assertEqual(first(), expected)\n"
            "        expected = [1, 2]\n"
            "        self.assertEqual(second(), expected)\n",
        ]:
            code = module(body)
            with self.subTest(body=body), self.assertRaises(ValueError):
                validate_change(code, code.replace("[1, 2]", "[2, 1]"))

    def test_two_literal_variables_are_ambiguous(self):
        code = module(
            "        with context():\n"
            "            actual = [1, 2]\n"
            "            expected = [3, 4]\n"
            "            self.assertEqual(actual, expected)\n"
        )
        with self.assertRaises(ValueError):
            validate_change(code, code.replace("[3, 4]", "[4, 3]"))

    def test_input_assertion_and_with_setup_cannot_change(self):
        code = module(
            "        with context(1):\n"
            "            expected = [1, 2]\n"
            "            self.assertEqual(value(1), expected)\n"
            "            expected = [3, 4]\n"
            "            self.assertEqual(other(), expected)\n"
        )
        for revised in [code.replace("context(1)", "context(2)"),
                        code.replace("value(1)", "value(2)"),
                        code.replace("assertEqual", "assertNotEqual"),
                        code.replace("            self.assertEqual(other(), expected)\n", "")]:
            with self.subTest(revised=revised), self.assertRaises(ValueError):
                validate_change(code, revised)


WITH_WRONG = (
    "import unittest\nfrom contextlib import nullcontext\nfrom sample import value\n"
    "class ValueTests(unittest.TestCase):\n"
    "    def test_value(self):\n"
    "        with nullcontext():\n"
    "            expected = 2\n"
    "            self.assertEqual(value(), expected)\n"
    "            expected = (2, 2)\n"
    "            self.assertEqual((value(), value()), expected)\n"
)
WITH_CORRECT = WITH_WRONG.replace("expected = 2", "expected = 4").replace("(2, 2)", "(4, 4)")


class BindingAcceptanceFlow(unittest.TestCase):
    setUp = flow.EvidenceAcceptanceFlow.setUp
    candidate = flow.EvidenceAcceptanceFlow.candidate
    shell = flow.EvidenceAcceptanceFlow.shell
    state = flow.EvidenceAcceptanceFlow.state
    run_replies = flow.EvidenceAcceptanceFlow.run_replies
    initial = flow.EvidenceAcceptanceFlow.initial
    correct = flow.EvidenceAcceptanceFlow.correct
    comparison = flow.EvidenceAcceptanceFlow.comparison

    def correct_bindings(self, messages):
        corrections = [
            {"before": before, "after": after, "evidence_ids": ["requirement"],
             "reason": "The original request specifies 4 for every call"}
            for before, after in [("expected = 2", "expected = 4"), ("(2, 2)", "(4, 4)")]
        ]
        return self.correct(messages, code=WITH_CORRECT, corrections=corrections)

    def test_with_rebindings_retrial_freeze_and_compare_preserve_old_failure(self):
        result = self.run_replies(self.initial(WITH_WRONG) + [
            self.correct_bindings, flow.review, task_tests.reply(self.candidate(test_code="")),
        ])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["planner_calls"], 5)
        self.assertEqual(len(result["rounds"]), 2)
        versions = self.comparison(result)["versions"]
        self.assertEqual(len(versions), 2)
        self.assertEqual(versions[0]["candidate"]["status"], "failed")
        self.assertEqual(versions[0]["effective_candidate"]["status"], "passed")
        self.assertEqual(versions[0]["superseded_expectations_by"], [versions[1]["trial_id"]])

    def test_with_rebinding_proposal_resumes_without_replaying_trial(self):
        class Interrupted(BaseException):
            pass

        persist = TaskRunner.persist

        def stop(engine, event, payload=None):
            persist(engine, event, payload)
            if event == "acceptance_trial_finished" and engine.state.get("verification_spec"):
                raise Interrupted()

        with patch.object(TaskRunner, "persist", stop), self.assertRaises(Interrupted):
            self.run_replies(self.initial(WITH_WRONG) + [self.correct_bindings])
        self.assertEqual(self.state()["verification_spec"]["test_code"], WITH_WRONG)
        result = self.run_replies([flow.review, task_tests.reply(self.candidate(test_code=""))])
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["calls_this_invocation"], 2)
        self.assertEqual(result["planner_calls"], 5)
        self.assertEqual(len(self.state()["acceptance_trials"]), 2)


if __name__ == "__main__":
    unittest.main()
