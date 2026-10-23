"""P0338 入口可离线运行，并拒绝已移除的 CLI 模式。"""

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
    def test_persistent_output_limit_requires_explicit_run_not_status_lookup(self):
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as name:
            (Path(name) / 'history.sqlite3').touch()
            stored = {
                'status': 'output_limit_reached',
                'repo': name,
                'request': 'saved task',
                'execution_binding': {'backend': 'docker'},
            }
            history = Mock()
            history.load.return_value = stored
            with patch('agent.cli.tasks.LocalHistory', return_value=history), patch(
                'agent.cli.tasks.run_work'
            ) as run, redirect_stdout(io.StringIO()):
                self.assertEqual(persistent_main(['--state-dir', name, '--status']), 0)
                run.assert_not_called()
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    persistent_main(['--state-dir', name])
                run.assert_not_called()
                self.assertFalse((Path(name) / 'bridge-job.json').exists())

    def test_persistent_new_task_uses_work_bridge_and_saves_stable_original(self):
        import json

        with tempfile.TemporaryDirectory() as name:
            repo = Path(name) / 'repo'
            repo.mkdir()
            root = Path(name) / 'records'
            with patch('agent.cli.tasks.run_work', return_value={'status': 'paused'}) as run, patch(
                'agent.cli.tasks._history', return_value=None
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(persistent_main([
                    '--state-dir', str(root), '--repo', str(repo), '--task', 'original request',
                    '--max-steps', '5', '--pause-after-steps', '1',
                ]), 0)
                raw = (root / 'bridge-job.json').read_bytes()
                saved = json.loads(raw)
                self.assertEqual(run.call_args.args[0]['request'], 'original request')
                self.assertEqual(run.call_args.args[0]['budget']['max_steps'], 5)
                self.assertEqual(run.call_args.args[0]['pause_after_steps'], 1)
                self.assertEqual(saved['state_root'], str(root.resolve()))
                self.assertNotIn('api_key', saved['settings'])
            with patch('agent.cli.tasks.run_work', return_value={'status': 'paused'}) as run, patch(
                'agent.cli.tasks._history', return_value={'status': 'output_limit_reached'}
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(persistent_main(['--state-dir', str(root)]), 0)
                run.assert_not_called()
                self.assertEqual(persistent_main(['--state-dir', str(root), '--continue']), 0)
                self.assertEqual(run.call_args.args[0]['task_id'], saved['task_id'])
                self.assertEqual([row['kind'] for row in run.call_args.args[2]()], ['resume'])
                self.assertEqual((root / 'bridge-job.json').read_bytes(), raw)

    def test_unified_continue_routes_old_bound_kernel_pause_without_new_task(self):
        import json
        from agent.application.legacy_stop import STATUS

        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / 'records'
            repo = Path(name) / 'repo'
            repo.mkdir()
            with patch('agent.cli.tasks.run_work', return_value={'status': 'paused'}), patch(
                'agent.cli.tasks._history', return_value=None
            ), redirect_stdout(io.StringIO()):
                persistent_main(['--state-dir', str(root), '--repo', str(repo), '--task', 'task'])
            original = (root / 'bridge-job.json').read_bytes()
            stored = {'status': STATUS, 'acceptance_limit': {
                'active': True, 'pause_id': 'saved-pause', 'round_increment': 3,
                'round_limit': 3, 'completed_rounds': 3, 'last_verification_status': 'failed',
            }}
            with patch('agent.cli.tasks._history', return_value=stored), patch(
                'agent.cli.tasks.run_work', return_value={'status': STATUS}
            ) as run, redirect_stdout(io.StringIO()):
                persistent_main(['--state-dir', str(root), '--status'])
                run.assert_not_called()
                persistent_main(['--state-dir', str(root)])
                self.assertEqual(run.call_args.args[2](), [])
                persistent_main(['--state-dir', str(root), '--continue'])
                controls = run.call_args.args[2]()
                self.assertEqual(len(controls), 1)
                self.assertEqual(controls[0]['acceptance_pause_id'], 'saved-pause')
                self.assertEqual(controls[0]['kind'], 'resume')
                self.assertEqual(run.call_args.args[0]['task_id'], json.loads(original)['task_id'])
                self.assertEqual((root / 'bridge-job.json').read_bytes(), original)

    def test_persistent_records_cannot_be_written_inside_work_source(self):
        with tempfile.TemporaryDirectory() as name, patch('agent.cli.tasks.run_work') as run:
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                persistent_main(['--state-dir', str(Path(name) / 'records'),
                                 '--repo', name, '--task', 'request'])
            run.assert_not_called()
            self.assertFalse((Path(name) / 'records').exists())

    def test_continue_after_crash_before_first_checkpoint_keeps_explicit_control(self):
        import json

        with tempfile.TemporaryDirectory() as name:
            root, repo = Path(name) / 'records', Path(name) / 'repo'
            repo.mkdir()
            with patch('agent.cli.tasks.run_work', return_value={'status': 'paused'}), patch(
                'agent.cli.tasks._history', return_value=None
            ), redirect_stdout(io.StringIO()):
                persistent_main(['--state-dir', str(root), '--repo', str(repo), '--task', 'task'])
            original = (root / 'bridge-job.json').read_bytes()
            for stored in (None, {'status': 'executing', 'request_sequence': 4}):
                with self.subTest(stored=stored), patch('agent.cli.tasks._history', return_value=stored), patch(
                    'agent.cli.tasks.run_work', return_value={'status': 'paused'}
                ) as run, redirect_stdout(io.StringIO()):
                    persistent_main(['--state-dir', str(root), '--continue'])
                    first = run.call_args.args[2]()
                    self.assertEqual(len(first), 1)
                    self.assertEqual(first[0]['kind'], 'resume')
                    persistent_main(['--state-dir', str(root), '--continue'])
                    self.assertEqual(run.call_args.args[2](), first)
                    self.assertEqual(run.call_args.args[0]['task_id'], json.loads(original)['task_id'])
            self.assertEqual((root / 'bridge-job.json').read_bytes(), original)

    def test_continue_while_native_owner_runs_returns_status_without_dispatch(self):
        import json
        from agent.application.lease import ServiceLease

        with tempfile.TemporaryDirectory() as name:
            root, repo = Path(name) / 'records', Path(name) / 'repo'
            repo.mkdir()
            with patch('agent.cli.tasks.run_work', return_value={'status': 'paused'}), patch(
                'agent.cli.tasks._history', return_value=None
            ), redirect_stdout(io.StringIO()):
                persistent_main(['--state-dir', str(root), '--repo', str(repo), '--task', 'task'])
            job = json.loads((root / 'bridge-job.json').read_bytes())
            folder = root / 'tasks' / job['task_id']
            folder.mkdir(parents=True)
            owner = ServiceLease(folder)
            owner.acquire()
            try:
                output = io.StringIO()
                with patch('agent.cli.tasks._history', return_value={'status': 'executing'}), patch(
                    'agent.runtime.bridge.Session', side_effect=AssertionError('No second worker')
                ), redirect_stdout(output):
                    self.assertEqual(persistent_main(['--state-dir', str(root), '--continue']), 0)
                self.assertEqual(json.loads(output.getvalue())['status'], 'running')
            finally:
                owner.close()

    def test_workspace_check_is_offline(self):
        with tempfile.TemporaryDirectory() as name, patch.object(
            cli, "load_model", side_effect=AssertionError("API forbidden")
        ):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--repo", name, "--check-workspace"]), 0)

    def test_invalid_workspace_does_not_create_model_or_prompt(self):
        with tempfile.TemporaryDirectory() as name, patch.object(cli, "load_model") as model, patch(
            "builtins.input"
        ) as prompt:
            with redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["--repo", str(Path(name) / "missing")]), 1)
            model.assert_not_called()
            prompt.assert_not_called()

    def test_removed_single_task_and_parallel_modes_are_rejected(self):
        for args in (
            ["--task", "request"],
            ["--workflow", "parallel"],
            ["--repo-navigation", "on"],
        ):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ) as error:
                cli.parse_args(args)
            self.assertEqual(error.exception.code, 2)

    def test_positive_budget_and_nonempty_acceptance(self):
        self.assertEqual(cli.parse_args(["--max-steps", "5"]).max_steps, 5)
        for args in (["--max-steps", "0"], ["--verify-command", ""], ["--acceptance", " "]):
            with self.subTest(args=args), redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ):
                cli.parse_args(args)

    def test_factory_uses_application_adapter_and_display_handles_its_result(self):
        from agent.application.cli import ConsoleConversation

        with tempfile.TemporaryDirectory() as name:
            env = cli.open_workspace(name)
            result = {"status": "verified_success", "messages": [], "final_report": "DONE"}
            with patch.object(ConsoleConversation, '__init__', return_value=None), patch.object(
                ConsoleConversation, "run", return_value=result
            ), redirect_stdout(io.StringIO()) as output:
                adapter = cli.create_workspace_agent(object(), env)
                self.assertIsInstance(adapter, ConsoleConversation)
                cli.run_one_turn(adapter, [], "request", 5)
            self.assertIn("固定程序验收通过", output.getvalue())

    def test_persistent_entry_still_requires_explicit_new_task_and_supports_offline_lookup(self):
        with tempfile.TemporaryDirectory() as name, redirect_stderr(
            io.StringIO()
        ), self.assertRaises(SystemExit):
            persistent_main(["--state-dir", name, "--repo", name])
