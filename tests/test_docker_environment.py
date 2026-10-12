"""Offline Docker CLI contract tests; no daemon, image pull, or paid API needed."""

import base64
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agent.execution.docker import DockerEnvironment


class DockerEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.repo = self.base / "repo space"
        self.repo.mkdir()
        self.artifacts = self.base / "evidence space"
        self.environment = DockerEnvironment(self.repo, "benchmark:test", "agent-test",
                                             artifact_root=self.artifacts)
        self.cid = "a" * 64

    def _successful_run(self, arguments, **kwargs):
        if arguments[1] == "run":
            return subprocess.CompletedProcess(arguments, 0, self.cid + "\n", "")
        if arguments[1] == "stop":
            return subprocess.CompletedProcess(arguments, 0, self.cid + "\n", "")
        payload = json.loads(base64.b64decode(arguments[-1]))
        filename = Path(payload["status_path"]).name
        (self.artifacts / filename).write_text(json.dumps({
            "exit_code": 0, "timed_out": False, "pid": 321,
        }), encoding="utf-8")
        kwargs["stdout"].write(b"COMMAND_STDOUT\n")
        kwargs["stderr"].write(b"COMMAND_STDERR\n")
        return subprocess.CompletedProcess(arguments, 0)

    @patch("agent.execution.docker.subprocess.run")
    def test_lifecycle_mounts_only_task_and_evidence_and_stops_owned_id(self, run):
        run.side_effect = self._successful_run
        self.environment.start()
        self.environment.start()
        self.environment.close()
        self.environment.close()
        self.assertEqual(run.call_count, 2)
        args = run.call_args_list[0].args[0]
        self.assertEqual(args[args.index("--network") + 1], "none")
        mounts = [args[i + 1] for i, arg in enumerate(args) if arg == "--mount"]
        self.assertEqual(len(mounts), 2)
        self.assertIn(f"source={self.repo.resolve()},target=/testbed", mounts[0])
        self.assertNotIn("docker.sock", " ".join(args))
        self.assertNotIn("--env-file", args)
        self.assertEqual(run.call_args_list[-1].args[0][-1], self.cid)
        self.assertNotIn("rm", run.call_args_list[-1].args[0])
        self.assertTrue((self.artifacts / "container.json").is_file())

    @patch("agent.execution.docker.subprocess.run")
    def test_readonly_probe_can_execute_temporary_compiler_products(self, run):
        run.side_effect = self._successful_run
        self.environment.repo_read_only = self.environment.root_read_only = True
        self.environment.start()
        args = run.call_args.args[0]
        self.assertIn('--read-only', args)
        self.assertIn('/tmp:rw,exec,nosuid,nodev,size=268435456', args)
        mounts = [args[i+1] for i, arg in enumerate(args) if arg == '--mount']
        self.assertTrue(mounts[0].endswith(',readonly'))
        self.assertEqual(args[args.index('--network')+1], 'none')
        self.environment.close()

    @patch("agent.execution.docker.subprocess.run")
    def test_name_collision_is_reported_without_stopping_existing_container(self, run):
        run.return_value = subprocess.CompletedProcess([], 125, "", "name already in use")
        with self.assertRaisesRegex(RuntimeError, "name already in use"):
            self.environment.start()
        self.environment.close()
        self.assertEqual(run.call_count, 1)

    def test_host_mapping_rejects_other_files_and_handles_spaces(self):
        self.assertEqual(self.environment.host_to_container(self.repo), "/testbed")
        self.assertEqual(self.environment.host_to_container(self.repo / "a b.py"),
                         "/testbed/a b.py")
        self.assertEqual(self.environment.host_to_container(self.artifacts / "run" / "test.py"),
                         "/agent-artifacts/run/test.py")
        with self.assertRaises(ValueError):
            self.environment.host_to_container(self.base / "credentials.env")

    @patch("agent.execution.docker.subprocess.run")
    def test_execute_preserves_complete_logs_and_returns_result(self, run):
        run.side_effect = self._successful_run
        result = self.environment.execute("python -c 'print(123)'")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["stdout"], "COMMAND_STDOUT\n")
        self.assertEqual(Path(result["stdout_log"]).read_bytes(), b"COMMAND_STDOUT\n")
        payload = json.loads(base64.b64decode(run.call_args_list[1].args[0][-1]))
        self.assertEqual(payload["argv"][-1], "python -c 'print(123)'")
        self.assertIn("activate testbed", payload["argv"][2])
        self.assertTrue(list(self.artifacts.glob("process-*.host.json")))

    @patch("agent.execution.docker.subprocess.run")
    def test_container_timeout_result_is_preserved(self, run):
        def response(arguments, **kwargs):
            if arguments[1] == "run":
                return self._successful_run(arguments, **kwargs)
            payload = json.loads(base64.b64decode(arguments[-1]))
            path = self.artifacts / Path(payload["status_path"]).name
            path.write_text(json.dumps({"exit_code": 124, "timed_out": True, "pid": 42}))
            kwargs["stdout"].write(b"OUTPUT_BEFORE_TIMEOUT")
            return subprocess.CompletedProcess(arguments, 124)
        run.side_effect = response
        result = self.environment.execute("sleep 1000")
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["exit_code"], 124)
        self.assertEqual(result["stdout"], "OUTPUT_BEFORE_TIMEOUT")

    @patch("agent.execution.docker.subprocess.run")
    def test_transport_timeout_attempts_cleanup_inside_same_container(self, run):
        run.side_effect = [
            subprocess.CompletedProcess([], 0, self.cid, ""),
            subprocess.TimeoutExpired("docker", 135),
            subprocess.CompletedProcess([], 0, "", ""),
        ]
        result = self.environment.execute("sleep 1000")
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["process_cleanup_confirmed"])
        cleanup_args = run.call_args_list[-1].args[0]
        self.assertEqual(cleanup_args[:3], ["docker", "exec", self.cid])
        self.assertIn("os.killpg", cleanup_args[-2])

    @patch("agent.execution.docker.subprocess.run")
    def test_zero_docker_exit_without_process_record_is_not_success(self, run):
        run.side_effect = [subprocess.CompletedProcess([], 0, self.cid, ""),
                           subprocess.CompletedProcess([], 0)]
        result = self.environment.execute("echo no-result")
        self.assertIn("没有有效的容器执行结果", result["error"])
        self.assertIsNone(result["exit_code"])

    def test_list_argv_keeps_arguments_and_shell_code_separate(self):
        command = ["python", "-c", "print('quote; $x')", "a b"]
        argv = self.environment._activated_argv(command, False)
        self.assertEqual(argv[4:], command)
        with self.assertRaises(ValueError):
            self.environment._activated_argv("python test.py", False)

    def test_long_output_is_bounded_but_source_log_is_complete(self):
        path = self.artifacts / "large.log"
        path.write_text("a" * 20_000 + "LAST", encoding="utf-8")
        summary = self.environment._summary(path)
        self.assertIn("前文已截断", summary)
        self.assertTrue(summary.endswith("LAST"))
        self.assertLess(len(summary), 6100)
        self.assertEqual(path.stat().st_size, 20_004)
        self.assertEqual(summary, "[前文已截断；完整内容见日志]\n" +
                         path.read_bytes()[-6000:].decode("utf-8", errors="replace"))


if __name__ == "__main__":
    unittest.main()
