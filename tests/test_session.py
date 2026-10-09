"""Workspace switching, one-use acceptance and append-only conversation remain usable."""
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from langchain_core.messages import HumanMessage, AIMessage
import agent.cli.interactive as cli
from agent.cli import workspace


class SessionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.a, self.b = self.root / "A", self.root / "B"
        self.a.mkdir()
        self.b.mkdir()
        self.state = self.root / "preferences.json"
        guard = patch.object(workspace, "STATE_PATH", self.state)
        guard.start()
        self.addCleanup(guard.stop)

    def run_cli(self, args, inputs):
        calls = []
        def turn(agent, history, text, max_steps, verify_command=None, acceptance=None):
            calls.append((text, verify_command, acceptance, agent.environment.repo_root, list(history)))
            return {"status": "verified_success", "messages": [*history, HumanMessage(content=text), AIMessage(content="done")]}
        with patch.object(cli, "load_model", return_value=object()) as loader, patch.object(cli, "run_one_turn", side_effect=turn), \
                patch("builtins.input", side_effect=inputs), redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            result = cli.main(args)
        return result, calls, loader, out.getvalue(), err.getvalue()

    def test_history_appends_and_clear_resets(self):
        result, calls, loader, _, errors = self.run_cli(["--repo", str(self.a)], ["one", "two", "/clear", "three", "/exit"])
        self.assertEqual(result, 0, errors)
        self.assertEqual([m.content for m in calls[1][4]], ["one", "done"])
        self.assertEqual(calls[2][4], [])
        self.assertEqual(loader.call_count, 1)

    def test_acceptance_is_one_use_and_switches_only_after_valid_repo(self):
        result, calls, _, _, errors = self.run_cli(["--repo", str(self.a), "--verify-command", "INITIAL", "--acceptance", "RULE"],
            ["first", "second", "/verify-command CUSTOM", "/acceptance EXTRA", "/repo " + str(self.root / "missing"),
             "third", "/acceptance STALE", "/repo " + str(self.b), "fourth", "/exit"])
        self.assertEqual(result, 0, errors)
        self.assertEqual([(t, c, a, r) for t, c, a, r, h in calls],
            [("first", "INITIAL", "RULE", self.a), ("second", None, None, self.a),
             ("third", "CUSTOM", "EXTRA", self.a), ("fourth", None, None, self.b)])
        self.assertEqual(calls[-1][4], [])
        self.assertEqual(Path(workspace.load_saved_repo()), self.b)

    def test_restart_remembers_repository_but_not_conversation(self):
        first = self.run_cli([], [str(self.root / "missing"), '"' + str(self.a) + '"', "old", "/exit"])
        self.assertEqual(first[0], 0)
        second = self.run_cli([], ["fresh", "/exit"])
        self.assertEqual(second[0], 0)
        self.assertEqual(second[1][0][4], [])

    def test_corrupt_preferences_can_be_replaced(self):
        self.state.write_text("{broken", encoding="utf-8")
        result, calls, loader, _, errors = self.run_cli([], [str(self.b), "/help", "/repo", "/unknown", "/exit"])
        self.assertEqual(result, 0, errors)
        loader.assert_not_called()
        self.assertEqual(calls, [])
        self.assertEqual(Path(workspace.load_saved_repo()), self.b)

    def test_check_without_saved_repo_does_not_prompt_or_change_preferences(self):
        with patch("builtins.input", side_effect=AssertionError("must not prompt")), redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["--check-workspace"]), 1)
        self.assertFalse(self.state.exists())
