"""Offline contracts for the adapter; these are NOT SWE-bench success scores."""

from dataclasses import asdict
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage

from eval.recording import RecordingModel, TokenLedger
from eval.swebench_runner import export_patch, load_tasks, prepare_base_checkout, save_predictions
from eval import swebench_runner as runner


class FakeModel:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class ManifestTests(unittest.TestCase):
    def test_default_and_explicit_task_use_one_planner(self):
        with tempfile.TemporaryDirectory() as name:
            for extra in ([], ["--workflow", "task"]):
                args = runner.parse_args(["--tasks", "tasks.json", "--images", "images.json", "--run-dir", name, *extra])
                self.assertEqual(args.workflow, "task")
                from agent.task.runner import TaskRunner
                selected = runner.create_task_controller(object(), object(), Path(name), args)
                self.assertIsInstance(selected, TaskRunner)


    def test_manifest_selection_errors_stop_before_model_or_archive(self):
        task = {"instance_id": "example__repo-1", "repo": "example/repo",
                "base_commit": "a" * 40, "problem_statement": "Fix the issue"}
        for selection, images, message in (
            (["--instance-id", "missing"], [], "Unknown instance id"),
            ([], [], "Missing image mapping"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                (root / "tasks.json").write_text(json.dumps([task]), encoding="utf-8")
                (root / "images.json").write_text(json.dumps(images), encoding="utf-8")
                with (
                    patch.object(runner, "load_settings") as settings,
                    patch.object(runner, "create_model") as model,
                    patch.object(runner, "archive_source") as archive,
                    redirect_stderr(io.StringIO()) as output,
                    self.assertRaises(SystemExit) as error,
                ):
                    runner.main(["--tasks", str(root / "tasks.json"), "--images",
                                 str(root / "images.json"), "--run-dir", str(root / "run"),
                                 *selection])
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, output.getvalue())
                self.assertFalse((root / "run").exists())
                settings.assert_not_called()
                model.assert_not_called()
                archive.assert_not_called()

    def test_answer_material_is_rejected_before_model_setup(self):
        task = {"instance_id": "example__repo-1", "repo": "example/repo",
                "base_commit": "a" * 40, "problem_statement": "Fix the issue"}
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "tasks.json"
            path.write_text(json.dumps([task]), encoding="utf-8")
            self.assertEqual(load_tasks(path), [task])
            for private in ("patch", "test_patch", "FAIL_TO_PASS", "hints_text"):
                path.write_text(json.dumps([{**task, private: "PRIVATE"}]), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_tasks(path)

    def test_predictions_keep_failed_and_empty_patches(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            save_predictions(directory, [
                {"instance_id": "r__r-1", "agent_status": "api_error", "model_patch": ""},
                {"instance_id": "r__r-2", "agent_status": "verified_success", "model_patch": "diff"},
            ], "fake-offline-only")
            rows = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["instance_id"] for row in rows], ["r__r-1", "r__r-2"])
            self.assertEqual(rows[0]["model_patch"], "")


class RecordingTests(unittest.TestCase):
    def test_context_rejection_survives_wrapper_and_missing_usage_is_not_zero(self):
        from agent.model.errors import ModelCallError, is_context_overflow
        class Rejected(Exception):
            status_code = 400
            body = {'error': {'code': 'context_length_exceeded', 'message': 'Explicit context rejection'}}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            ledger = TokenLedger()
            model = RecordingModel(FakeModel(Rejected()), root / 'calls', ledger, root / 'ledger.json')
            with self.assertRaises(ModelCallError) as caught:
                model.invoke([HumanMessage(content='task')])
            self.assertTrue(is_context_overflow(caught.exception))
            self.assertEqual(ledger.unknown_calls, 1)
            self.assertGreater(ledger.reserved_unknown_tokens, 0)
            record = json.loads((root / 'calls/call-0001.response.json').read_text())
            self.assertTrue(record['context_overflow'])
            self.assertIsNone(record['usage'])

    def test_usage_and_complete_messages_are_saved_without_secret(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            ledger = TokenLedger()
            model = RecordingModel(FakeModel(AIMessage(content="answer", usage_metadata={
                "input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
            })), directory / "calls", ledger, directory / "ledger.json", secrets=("SECRET",))
            model.invoke([HumanMessage(content="task SECRET")])
            self.assertEqual(ledger.reported_tokens, 5)
            self.assertEqual(ledger.unknown_calls, 0)
            self.assertEqual(ledger.reserved_unknown_tokens, 0)
            request = (directory / "calls" / "call-0001.request.json").read_text()
            self.assertNotIn("SECRET", request)
            self.assertIn("[REDACTED]", request)

    def test_unknown_failed_call_retains_reservation_and_redacts_error(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            ledger = TokenLedger()
            wrapped = RecordingModel(FakeModel(RuntimeError("key SECRET")), directory,
                                     ledger, directory / "ledger.json", secrets=("SECRET",))
            with self.assertRaisesRegex(RuntimeError, "REDACTED"):
                wrapped.invoke([HumanMessage(content="hello")])
            self.assertEqual(ledger.reported_tokens, 0)
            self.assertEqual(ledger.unknown_calls, 1)
            self.assertGreater(ledger.reserved_unknown_tokens, 0)
            disk = json.loads((directory / "ledger.json").read_text())
            self.assertEqual(disk, asdict(ledger))

    def test_token_guard_blocks_before_sending(self):
        with tempfile.TemporaryDirectory() as name:
            raw = FakeModel(AIMessage(content="unused"))
            model = RecordingModel(raw, name, TokenLedger(limit=10), Path(name) / "ledger.json")
            with self.assertRaisesRegex(RuntimeError, "allowance"):
                model.invoke([HumanMessage(content="hello")])
            self.assertEqual(raw.calls, 0)

    def test_each_task_gets_a_separate_allowance_under_shared_total(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            ledger = TokenLedger(limit=20000)
            response = AIMessage(content="done", usage_metadata={
                "input_tokens": 4500, "output_tokens": 300, "total_tokens": 4800})
            first_raw = FakeModel(response)
            first = RecordingModel(first_raw, root / "first", ledger, root / "ledger.json",
                                   max_output_tokens=10, task_token_limit=5000)
            first.invoke([HumanMessage(content="task one")])
            with self.assertRaisesRegex(RuntimeError, "per-task"):
                first.invoke([HumanMessage(content="continue")])
            second = RecordingModel(FakeModel(response), root / "second", ledger,
                                    root / "ledger.json", max_output_tokens=10, task_token_limit=5000)
            second.invoke([HumanMessage(content="task two")])
            self.assertEqual(ledger.calls, 2)
            self.assertEqual(ledger.reported_tokens, 9600)
            self.assertEqual(ledger.last_guard_reason, "task_token_allowance")


class PatchTests(unittest.TestCase):
    def test_setup_mode_commit_is_normalized_but_content_changes_still_fail(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            repo = root / "repo"
            repo.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-c", "core.autocrlf=false", *args],
                                               cwd=repo, stderr=subprocess.STDOUT)
            git("init")
            git("config", "user.name", "Offline Test")
            git("config", "user.email", "offline@example.invalid")
            source = repo / "code.py"
            source.write_text("value = 1\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-m", "base")
            base = git("rev-parse", "HEAD").decode().strip()
            git("update-index", "--chmod=+x", "code.py")
            git("commit", "-m", "image chmod setup")
            prepare_base_checkout(repo, base, root / "mode-logs")
            self.assertEqual(source.read_text(), "value = 1\n")
            self.assertEqual(git("rev-parse", "HEAD").decode().strip(), base)
            source.write_text("value = 999\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source content differs"):
                prepare_base_checkout(repo, base, root / "content-logs")
            self.assertEqual(source.read_text(), "value = 999\n")

    def test_export_includes_new_staged_unstaged_deleted_without_changing_real_index(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            repo = root / "repo"
            repo.mkdir()
            options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

            def git(*args):
                return subprocess.check_output(["git", "-c", "core.autocrlf=false", *args],
                                               cwd=repo, stderr=subprocess.STDOUT, **options)

            git("init")
            git("config", "user.email", "offline@example.invalid")
            git("config", "user.name", "Offline Test")
            (repo / "edit.py").write_text("x = 1\n", encoding="utf-8")
            (repo / "delete.py").write_text("gone = True\n", encoding="utf-8")
            (repo / ".gitignore").write_text("generated.py\n", encoding="utf-8")
            git("add", ".")
            git("commit", "-m", "base")
            base = git("rev-parse", "HEAD").decode().strip()
            (repo / "edit.py").write_text("x = 2\n", encoding="utf-8")
            git("add", "edit.py")
            (repo / "edit.py").write_text("x = 3\n", encoding="utf-8")
            (repo / "delete.py").unlink()
            (repo / "new.py").write_text("new = True\n", encoding="utf-8")
            (repo / "generated.py").write_text("required = 1\n", encoding="utf-8")
            git("add", "--force", "generated.py")
            (repo / "generated.py").write_text("required = 2\n", encoding="utf-8")
            (repo / ".env").write_text("TEST_SECRET=notreal\n", encoding="utf-8")
            (repo / "preexisting.log").write_text("setup log\n", encoding="utf-8")
            staged_before = git("diff", "--cached")
            patch = export_patch(repo, base, root / "patch", ["preexisting.log"])
            self.assertIn("+x = 3", patch)
            self.assertIn("new.py", patch)
            self.assertIn("generated.py", patch)
            self.assertIn("+required = 2", patch)
            self.assertIn("deleted file mode", patch)
            self.assertNotIn("TEST_SECRET", patch)
            self.assertNotIn("preexisting.log", patch)
            self.assertEqual(git("diff", "--cached"), staged_before)

    def test_backend_export_never_runs_host_git_and_maps_its_private_index(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)

            class Backend:
                artifact_root = root / "export-runtime"

                def __init__(self):
                    self.commands = []

                def host_to_container(self, path):
                    return "/agent-artifacts/" + Path(path).relative_to(self.artifact_root).as_posix()

                def run_process(self, command, *, stdout_path, stderr_path, timeout, shell):
                    self.commands.append(command)
                    if "ls-tree" in command or "ls-files" in command and "--deleted" not in command:
                        output = b"code.py\0"
                    elif "diff" in command:
                        output = b"diff --git a/code.py b/code.py\n"
                    else:
                        output = b""
                    Path(stdout_path).write_bytes(output)
                    Path(stderr_path).write_bytes(b"")
                    return {"exit_code": 0, "timed_out": False}

            backend = Backend()
            with patch("eval.swebench_runner.run_logged", side_effect=AssertionError("Host Git forbidden")):
                exported = export_patch(root / "repo", "a" * 40, backend.artifact_root / "patch",
                                        environment=backend)
            self.assertIn("diff --git", exported)
            self.assertGreater(len(backend.commands), 5)
            self.assertTrue(all(command[0] == "/usr/bin/env" for command in backend.commands))
            index_commands = [command for command in backend.commands if "read-tree" in command]
            self.assertEqual(index_commands[0][1],
                             "GIT_INDEX_FILE=/agent-artifacts/patch/patch.index")
            self.assertTrue(all(str(root) not in " ".join(command) for command in backend.commands))


if __name__ == "__main__":
    unittest.main()
