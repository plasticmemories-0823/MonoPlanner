"""验证独立的源码和测试执行视图；不使用真实模型或 Docker 守护进程。"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from agent.execution.docker import DockerEnvironment
from agent.execution.local import LocalEnvironment
from agent.source.snapshots import SnapshotError, SourceSnapshot


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "original"
        self.repo.mkdir()
        self.write("module.py", "VALUE = 1\n")

    def write(self, relative, text):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def capture(self, name="snapshot", **kwargs):
        return SourceSnapshot.capture(self.repo, self.base / name, **kwargs)

    def test_baseline_and_candidate_are_independent_of_worker_changes(self):
        start = self.capture("initial")
        self.write("module.py", "VALUE = 2\n")
        candidate = self.capture("candidate")
        self.write("module.py", "VALUE = 3\n")
        self.assertEqual((start.root / "module.py").read_text(), "VALUE = 1\n")
        self.assertEqual((candidate.root / "module.py").read_text(), "VALUE = 2\n")
        self.assertNotEqual(start.digest, candidate.digest)
        self.assertTrue(start.verify_integrity()["valid"])
        self.assertTrue(candidate.verify_integrity()["valid"])
        self.assertEqual(candidate.changes_from(start)["modified"], ["module.py"])

    def test_exclusions_have_reasons_and_source_assets_are_preserved(self):
        for name in (
            ".env",
            ".env.local",
            "server.key",
            "__pycache__/cache.pyc",
            ".venv/installed.py",
            "node_modules/package/index.js",
            "eval/runs/large.log",
            "evidence/run.json",
        ):
            self.write(name, "PRIVATE_OR_DERIVED")
        self.write("config.toml", "[project]\nname = 'example'\n")
        self.write("tests/test_example.py", "assert True\n")
        self.write("notes.txt", "This file mentions -----BEGIN PRIVATE KEY----- in prose.\n")
        self.write("secret.txt", "-----BEGIN PRIVATE KEY-----\nsecret\n-----END PRIVATE KEY-----\n")
        self.write("data/fixture.bin", "\x00RESOURCE")
        snapshot = self.capture(excluded_roots=[self.repo / "evidence"])
        self.assertEqual(
            set(snapshot.files),
            {"module.py", "config.toml", "tests/test_example.py", "data/fixture.bin", "notes.txt"},
        )
        report = json.loads(snapshot.manifest_path.read_text())
        excluded = {entry["path"]: entry["reason"] for entry in report["excluded"]}
        self.assertEqual(excluded[".env"], "environment_credentials")
        self.assertEqual(excluded["secret.txt"], "private_key_content")
        self.assertEqual(excluded["evidence/run.json"], "explicit_artifact_root")
        self.assertFalse((snapshot.root / "secret.txt").exists())
        self.assertEqual((self.repo / ".env").read_text(), "PRIVATE_OR_DERIVED")

    @unittest.skipUnless(shutil.which("git"), "Git is not installed")
    def test_git_selection_keeps_dirty_and_new_files_and_omits_deleted_and_ignored(self):
        def git(*args):
            return subprocess.run(
                ["git", "-C", str(self.repo), *args], check=True, capture_output=True
            )

        git("init", "--quiet")
        self.write("deleted.py", "old\n")
        self.write(".gitignore", "ignored-data/\n")
        git("add", ".")
        self.write("module.py", "VALUE = 42\n")
        self.write("new.py", "NEW = True\n")
        self.write("ignored-data/private.txt", "not part of source")
        (self.repo / "deleted.py").unlink()
        snapshot = self.capture()
        self.assertEqual((snapshot.root / "module.py").read_text(), "VALUE = 42\n")
        self.assertIn("new.py", snapshot.files)
        self.assertNotIn("deleted.py", snapshot.files)
        self.assertFalse((snapshot.root / "ignored-data").exists())
        manifest = json.loads(snapshot.manifest_path.read_text())
        self.assertEqual(manifest["selection"], "git")
        self.assertIn(
            {"path": "deleted.py", "reason": "deleted_tracked_file"}, manifest["excluded"]
        )

    @unittest.skipUnless(shutil.which("git"), "Git is not installed")
    def test_tracked_source_overrides_generated_names_and_integrity_still_checks_it(self):
        subprocess.run(
            ["git", "-C", str(self.repo), "init", "--quiet"], check=True, capture_output=True
        )
        self.write("build/tool.py", "TOOL = 1\n")
        self.write("fixtures/expected.log", "EXPECTED OUTPUT\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True, capture_output=True)
        self.write("build/untracked.py", "derived\n")
        snapshot = self.capture()
        self.assertIn("build/tool.py", snapshot.files)
        self.assertIn("fixtures/expected.log", snapshot.files)
        self.assertNotIn("build/untracked.py", snapshot.files)
        self.assertTrue(snapshot.verify_integrity()["valid"])
        (snapshot.root / "build/tool.py").write_text("TOOL = 2\n")
        self.assertEqual(snapshot.verify_integrity()["modified"], ["build/tool.py"])

    @unittest.skipUnless(shutil.which("git"), "Git is not installed")
    def test_git_ignored_directory_is_not_traversed_for_secret_discovery(self):
        subprocess.run(
            ["git", "-C", str(self.repo), "init", "--quiet"], check=True, capture_output=True
        )
        self.write(".gitignore", "validation/\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True, capture_output=True)
        self.write("validation/pnpm-store/link/private.txt", "ignored cache")
        from agent.source.snapshots import _link

        with patch(
            "agent.source.snapshots._link", side_effect=lambda p: p.name == "link" or _link(p)
        ):
            snapshot = self.capture()
        self.assertIn("module.py", snapshot.files)
        self.assertFalse((snapshot.root / "validation").exists())

    @unittest.skipUnless(shutil.which("git"), "Git is not installed")
    def test_tracked_source_inside_ignored_directory_still_rejects_links(self):
        subprocess.run(
            ["git", "-C", str(self.repo), "init", "--quiet"], check=True, capture_output=True
        )
        self.write(".gitignore", "validation/\n")
        self.write("validation/tracked.py", "VALUE = 1\n")
        subprocess.run(
            ["git", "-C", str(self.repo), "add", "-f", "validation/tracked.py"],
            check=True,
            capture_output=True,
        )
        snapshot = self.capture("tracked")
        self.assertIn("validation/tracked.py", snapshot.files)
        from agent.source.snapshots import _link

        with patch(
            "agent.source.snapshots._link", side_effect=lambda p: p.name == "tracked.py" or _link(p)
        ):
            with self.assertRaisesRegex(SnapshotError, "Links"):
                self.capture("linked")

    @unittest.skipUnless(shutil.which("git"), "Git is not installed")
    def test_tracked_secrets_in_generated_directories_still_take_exclusion_priority(self):
        subprocess.run(
            ["git", "-C", str(self.repo), "init", "--quiet"], check=True, capture_output=True
        )
        paths = ("build/.env", "dist/.env.local", "build/server.key", "eval/runs/id_rsa")
        for name in paths:
            self.write(name, "DEMONSTRATION_SECRET_BEFORE\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True, capture_output=True)
        initial = self.capture("initial")
        for name in paths:
            with self.subTest(path=name):
                self.assertNotIn(name, initial.files)
                self.assertFalse((initial.root / name).exists())
                self.assertIn(name, initial.excluded_files)
                self.write(name, "DEMONSTRATION_SECRET_AFTER\n")
        updated = self.capture("updated")
        self.assertEqual(updated.excluded_changes_from(initial)["modified"], sorted(paths))
        reasons = {
            item["path"]: item["reason"]
            for item in json.loads(initial.manifest_path.read_text())["excluded"]
        }
        self.assertEqual(reasons["build/.env"], "environment_credentials")
        self.assertEqual(reasons["build/server.key"], "possible_private_key")

    @unittest.skipUnless(shutil.which("git"), "Git is not installed")
    def test_excluded_secret_changes_are_fingerprinted_without_copying_content(self):
        subprocess.run(
            ["git", "-C", str(self.repo), "init", "--quiet"], check=True, capture_output=True
        )
        self.write(".env", "TOKEN=SECRET_INITIAL_VALUE\n")
        self.write(".gitignore", ".env.local\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True, capture_output=True)
        self.write(".env.local", "TOKEN=IGNORED_INITIAL_VALUE\n")
        initial = self.capture("initial")
        self.write(".env", "TOKEN=SECRET_NEW_VALUE\n")
        self.write(".env.local", "TOKEN=IGNORED_NEW_VALUE\n")
        updated = self.capture("updated")
        self.assertEqual(updated.excluded_changes_from(initial)["modified"], [".env", ".env.local"])
        self.assertNotEqual(initial.digest, updated.digest)
        self.assertEqual(updated.changes_from(initial)["modified"], [])
        self.assertFalse((updated.root / ".env").exists())
        self.assertFalse((updated.root / ".env.local").exists())
        manifest = updated.manifest_path.read_text()
        self.assertNotIn("SECRET_NEW_VALUE", manifest)
        self.assertNotIn("IGNORED_NEW_VALUE", manifest)
        report = updated.write_patch(initial, self.base / "excluded.patch")
        self.assertFalse(report["complete_text_patch"])
        self.assertEqual(report["excluded_changes"]["modified"], [".env", ".env.local"])
        (self.repo / ".env").unlink()
        deleted = self.capture("deleted")
        self.assertEqual(deleted.excluded_changes_from(updated)["deleted"], [".env"])

    @unittest.skipUnless(shutil.which("git"), "Git is not installed")
    def test_independent_source_under_ignored_parent_directory_is_not_an_empty_git_snapshot(self):
        subprocess.run(
            ["git", "-C", str(self.base), "init", "--quiet"], check=True, capture_output=True
        )
        (self.base / ".gitignore").write_text("original/\n")
        snapshot = self.capture()
        self.assertEqual((snapshot.root / "module.py").read_text(), "VALUE = 1\n")
        self.assertEqual(json.loads(snapshot.manifest_path.read_text())["selection"], "source_walk")

    def test_new_secret_file_inside_snapshot_cannot_evade_integrity(self):
        snapshot = self.capture()
        (snapshot.root / ".env").write_text("TOKEN=ADDED_DURING_TESTING\n")
        integrity = snapshot.verify_integrity()
        self.assertFalse(integrity["valid"])
        self.assertEqual(integrity["added"], [".env"])

    def test_nested_or_existing_destination_is_rejected_without_touching_source(self):
        with self.assertRaisesRegex(SnapshotError, "outside"):
            SourceSnapshot.capture(self.repo, self.repo / "snapshot")
        existing = self.base / "existing"
        existing.mkdir()
        (existing / "keep.txt").write_text("keep")
        with self.assertRaisesRegex(SnapshotError, "overwrite"):
            SourceSnapshot.capture(self.repo, existing)
        self.assertEqual((existing / "keep.txt").read_text(), "keep")
        with self.assertRaisesRegex(SnapshotError, "entire"):
            self.capture(excluded_roots=[self.base])

    def test_external_link_is_rejected_and_a_failed_manifest_remains(self):
        outside = self.base / "outside.txt"
        outside.write_text("DO NOT COPY")
        link = self.repo / "linked.txt"
        try:
            link.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"Symlink creation is unavailable: {exc}")
        with self.assertRaisesRegex(SnapshotError, "Links"):
            self.capture()
        manifest = json.loads((self.base / "snapshot/manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse((self.base / "snapshot/source/linked.txt").exists())

    @unittest.skipUnless(os.name == "nt", "Windows junction check")
    def test_windows_external_junction_is_not_followed(self):
        outside = self.base / "external"
        outside.mkdir()
        (outside / "private.txt").write_text("secret")
        junction = self.repo / "junction"
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"New-Item -ItemType Junction -Path {quote(junction)} -Target {quote(outside)} | Out-Null",
            ],
            capture_output=True,
        )
        if result.returncode:
            self.skipTest("Junction creation is unavailable")
        # rmdir 只移除此处创建的目录链接，绝不删除外部目标。
        self.addCleanup(lambda: os.rmdir(junction) if junction.exists() else None)
        with self.assertRaisesRegex(SnapshotError, "reparse"):
            self.capture()
        self.assertEqual((outside / "private.txt").read_text(), "secret")

    def test_changes_during_copy_fail_instead_of_saving_mixed_content(self):
        copy = shutil.copystat

        def mutate(source, destination):
            copied = copy(source, destination)
            Path(source).write_text("changed during capture\n")
            return copied

        with patch("agent.source.snapshots.shutil.copystat", side_effect=mutate):
            with self.assertRaisesRegex(SnapshotError, "changed during capture"):
                self.capture()
        manifest = json.loads((self.base / "snapshot/manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")

    def test_source_tampering_is_detected_but_python_caches_are_allowed(self):
        snapshot = self.capture()
        (snapshot.root / "__pycache__").mkdir()
        (snapshot.root / "__pycache__/module.pyc").write_bytes(b"derived")
        self.assertTrue(snapshot.verify_integrity()["valid"])
        (snapshot.root / "module.py").write_text("VALUE = 0\n")
        (snapshot.root / "extra.py").write_text("injected\n")
        check = snapshot.verify_integrity()
        self.assertFalse(check["valid"])
        self.assertEqual(check["modified"], ["module.py"])
        self.assertEqual(check["added"], ["extra.py"])
        (snapshot.root / "module.py").unlink()
        self.assertEqual(snapshot.verify_integrity()["deleted"], ["module.py"])

    def test_local_backend_executes_inside_snapshot(self):
        snapshot = self.capture()
        original = LocalEnvironment(self.repo, timeout=7, artifact_root=self.base / "original-logs")
        with snapshot.environment_like(original, self.base / "snapshot-logs") as environment:
            stdout, stderr = (
                environment.artifact_root / "out.txt",
                environment.artifact_root / "err.txt",
            )
            result = environment.run_process(
                [
                    environment.python_executable,
                    "-c",
                    "from pathlib import Path; print(Path('module.py').read_text()); Path('new.py').write_text('new')",
                ],
                stdout_path=stdout,
                stderr_path=stderr,
            )
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(environment.timeout, 7)
            self.assertEqual(environment.repo_root, snapshot.root)
        self.assertFalse((self.repo / "new.py").exists())
        self.assertIn("new.py", snapshot.verify_integrity()["added"])

    def test_docker_clone_reuses_image_but_owns_its_lifecycle(self):
        snapshot = self.capture()
        original = DockerEnvironment(
            self.repo,
            "fixture:image",
            "original-task",
            timeout=9,
            artifact_root=self.base / "original-logs",
        )
        original._container_id = "a" * 64
        events = []

        def start(environment):
            events.append(("start", environment))
            environment._container_id = "b" * 64
            return environment

        def close(environment):
            events.append(("close", environment))

        with patch.object(DockerEnvironment, "start", start), patch.object(
            DockerEnvironment, "close", close
        ):
            with self.assertRaisesRegex(RuntimeError, "test failed"):
                with snapshot.environment_like(original, self.base / "snapshot-logs") as child:
                    self.assertEqual(child.image, original.image)
                    self.assertEqual(child.timeout, 9)
                    self.assertEqual(child.repo_root, snapshot.root)
                    self.assertNotEqual(child.container_name, original.container_name)
                    raise RuntimeError("test failed")
        self.assertEqual([event[0] for event in events], ["start", "close"])
        self.assertIs(events[0][1], events[1][1])
        self.assertIsNot(events[1][1], original)
        self.assertEqual(original._container_id, "a" * 64)

    def test_patch_records_text_and_explicitly_reports_binary_omissions(self):
        self.write("removed.py", "REMOVE = 1\n")
        start = self.capture("start")
        self.write("module.py", "VALUE = 42\n")
        self.write("added.py", "ADD = 1")
        self.write("data.bin", "\x00BINARY")
        (self.repo / "removed.py").unlink()
        candidate = self.capture("candidate")
        path = self.base / "candidate.patch"
        report = candidate.write_patch(start, path)
        text = path.read_text()
        self.assertIn("-VALUE = 1", text)
        self.assertIn("+VALUE = 42", text)
        self.assertIn("+++ b/added.py", text)
        self.assertIn("--- a/removed.py", text)
        self.assertIn("\\ No newline at end of file", text)
        self.assertEqual(report["omitted"], [{"path": "data.bin", "reason": "binary file"}])
        self.assertFalse(report["complete_text_patch"])
        self.assertTrue(path.with_suffix(".patch.manifest.json").exists())
        (candidate.root / "module.py").write_text("tampered")
        with self.assertRaisesRegex(SnapshotError, "unverified"):
            candidate.write_patch(start, self.base / "bad.patch")


if __name__ == "__main__":
    unittest.main()
