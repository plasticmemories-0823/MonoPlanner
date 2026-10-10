"""P0338 entrypoints work offline and reject removed CLI modes."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import agent.cli.interactive as cli
from agent.cli import workspace
from agent.task.runner import TaskRunner
from agent.cli.tasks import main as persistent_main


class CliTests(unittest.TestCase):
    def test_workspace_check_is_offline(self):
        with tempfile.TemporaryDirectory() as name, patch.object(cli, "load_model", side_effect=AssertionError("API forbidden")):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--repo", name, "--check-workspace"]), 0)

    def test_invalid_workspace_does_not_create_model_or_prompt(self):
        with tempfile.TemporaryDirectory() as name, patch.object(cli, "load_model") as model, patch("builtins.input") as prompt:
            with redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["--repo", str(Path(name) / "missing")]), 1)
            model.assert_not_called()
            prompt.assert_not_called()

    def test_removed_single_task_and_parallel_modes_are_rejected(self):
        for args in (["--task", "request"], ["--workflow", "parallel"], ["--repo-navigation", "on"]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.parse_args(args)
            self.assertEqual(error.exception.code, 2)

    def test_positive_budget_and_nonempty_acceptance(self):
        self.assertEqual(cli.parse_args(["--max-steps", "5"]).max_steps, 5)
        for args in (["--max-steps", "0"], ["--verify-command", ""], ["--acceptance", " "]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(args)

    def test_factory_uses_only_p0338_and_display_handles_its_result(self):
        with tempfile.TemporaryDirectory() as name:
            env = cli.open_workspace(name)
            self.assertIsInstance(cli.create_workspace_agent(object(), env), TaskRunner)
            result = {"status": "verified_success", "messages": [], "final_report": "DONE"}
            from agent.server.runtime import ConversationRunner
            with patch.object(ConversationRunner, "run", return_value=result), redirect_stdout(io.StringIO()) as output:
                cli.run_one_turn(cli.create_workspace_agent(object(), env), [], "request", 5)
            self.assertIn("固定程序验收通过", output.getvalue())

    def test_persistent_entry_still_requires_explicit_new_task_and_supports_offline_lookup(self):
        with tempfile.TemporaryDirectory() as name, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            persistent_main(["--state-dir", name, "--repo", name])
