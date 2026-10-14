import unittest

from eval.recording import TokenLedger
from eval.swebench_runner import parse_args
from agent.task.budget import StepBudget
from agent.model.errors import ModelRequestNotSent


class UncappedEvaluationTests(unittest.TestCase):
    def test_zero_limits_disable_both_ceilings(self):
        args = parse_args(
            [
                "--tasks",
                "tasks.json",
                "--images",
                "images.json",
                "--run-dir",
                "run",
                "--token-limit",
                "0",
                "--task-token-limit",
                "0",
            ]
        )
        self.assertIsNone(args.token_limit)
        self.assertIsNone(args.task_token_limit)

    def test_unlimited_ledger_still_accounts_and_reserves_unknown_usage(self):
        ledger = TokenLedger(limit=None)
        ledger.reserve(2_000_000)
        ledger.settle(
            {"total_tokens": 1_200_000, "input_tokens": 1_100_000, "output_tokens": 100_000},
            2_000_000,
        )
        ledger.reserve(3_000_000)
        self.assertEqual(ledger.reported_tokens, 1_200_000)
        self.assertEqual(ledger.reserved_unknown_tokens, 3_000_000)
        self.assertEqual(ledger.unknown_calls, 1)
        budget = StepBudget(max_steps=1000, max_tokens=None)
        budget.reserve_request(10_000_000)
        self.assertEqual(budget.used_steps, 1)

    def test_finite_limit_remains_enforced(self):
        with self.assertRaises(ModelRequestNotSent):
            TokenLedger(limit=100).reserve(101)


if __name__ == "__main__":
    unittest.main()
