"""验证生产源码解析缓存；查询行为在统一检索入口覆盖。"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.source.index import RepositoryNavigator


FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/navigation.json"


class RepositoryMapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for relative, content in self.data["files"].items():
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        self.navigator = RepositoryNavigator(self.repo, self.root / "cache")


    def test_aliases_relative_imports_and_call_locations_are_retained(self):
        self.navigator.refresh()
        entries = self.navigator.entries
        symbol = entries['src/tinycalc/arithmetic.py']['symbols'][0]
        self.assertEqual(symbol['signature'], 'def add(left: float, right: float) -> float')
        self.assertEqual((symbol['line'], symbol['end_line']), (1, 3))
        alias = entries['tests/test_arithmetic.py']['imports'][0]
        self.assertEqual(alias['binding'], 'sum_pair')
        self.assertEqual(alias['target'], 'tinycalc.arithmetic.add')
        call = entries['src/tinycalc/api.py']['calls'][0]
        self.assertEqual(call['targets'], ['tinycalc.arithmetic.add'])
        self.assertEqual(call['line'], 7)

    def test_class_async_signature_and_doc_are_parsed_without_executing_source(self):
        marker = self.root / "SHOULD_NOT_EXIST"
        (self.repo / "client.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
            "class Client:\n    async def send(self, text: str) -> int:\n"
            "        \"\"\"Send one text.\"\"\"\n        return len(text)\n",
            encoding="utf-8",
        )
        self.navigator.refresh()
        self.assertFalse(marker.exists())
        symbols = self.navigator.entries["client.py"]["symbols"]
        method = next(s for s in symbols if s["name"] == "Client.send")
        self.assertEqual(method["signature"], "async def send(self, text: str) -> int")
        self.assertEqual(method["doc"], "Send one text.")
        self.assertEqual((method["line"], method["end_line"]), (4, 6))

    def test_content_hash_detects_same_size_same_mtime_changes_and_reuses_other_parses(self):
        first = self.navigator.refresh()
        version = self.navigator.version
        path = self.repo / "src/tinycalc/arithmetic.py"
        before = path.stat()
        path.write_text(
            path.read_text(encoding="utf-8").replace("left - right", "left + right"),
            encoding="utf-8",
        )
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        second = self.navigator.refresh()
        self.assertNotEqual(version, self.navigator.version)
        self.assertEqual(second["parsed_files"], 1)
        self.assertEqual(second["reused_files"], first["indexed_files"] - 1)
        warm = RepositoryNavigator(self.repo, self.root / "cache")
        stats = warm.refresh()
        self.assertEqual(warm.version, self.navigator.version)
        self.assertEqual(stats["parsed_files"], 0)

    def test_deleted_and_broken_files_never_return_stale_symbols(self):
        self.navigator.refresh()
        (self.repo / "src/tinycalc/names.py").unlink()
        (self.repo / "src/tinycalc/arithmetic.py").write_text("def add(:\n", encoding="utf-8")
        result = self.navigator.refresh()
        entry = self.navigator.entries["src/tinycalc/arithmetic.py"]
        self.assertEqual(entry["symbols"], [])
        self.assertIn("SyntaxError", entry["error"])
        self.assertTrue(result["incomplete"])
        self.assertNotIn("src/tinycalc/names.py", self.navigator.entries)

    def test_baseline_and_live_paths_do_not_share_mutable_cache(self):
        import shutil

        copy = self.root / "baseline"
        shutil.copytree(self.repo, copy)
        baseline = RepositoryNavigator(copy, self.root / "cache")
        baseline.refresh()
        self.navigator.refresh()
        self.assertEqual(baseline.version, self.navigator.version)
        path = self.repo / "src/tinycalc/arithmetic.py"
        path.write_text("def add(a, b, strict=False):\n    return a + b\n", encoding="utf-8")
        self.navigator.refresh()
        baseline.refresh()
        self.assertNotEqual(self.navigator.version, baseline.version)
        self.assertNotEqual(baseline.cache_path, self.navigator.cache_path)

    def test_corrupt_cache_is_rebuilt(self):
        self.navigator.cache_path.write_text("{", encoding="utf-8")
        rebuilt = RepositoryNavigator(self.repo, self.root / "cache")
        stats = rebuilt.refresh()
        self.assertGreater(stats["parsed_files"], 0)
        self.assertIn("src/tinycalc/names.py", rebuilt.entries)

    def test_derived_and_secret_files_are_excluded(self):
        for relative in (
            ".venv/library.py",
            ".env.local.py",
            ".git/hook.py",
            "eval/runs/old.py",
            "node_modules/lib.py",
            "__pycache__/generated.py",
        ):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("def EXCLUDED_MARKER(): pass\n", encoding="utf-8")
        self.navigator.refresh()
        self.assertNotIn("EXCLUDED_MARKER", str(self.navigator.entries))

    def test_reparse_points_are_not_read(self):
        target = self.repo / "linked.py"
        target.write_text("def LINK_TARGET(): pass\n", encoding="utf-8")
        # 验证重解析点排除规则，无需 Windows 符号链接权限。
        from agent.source.index import _link

        with patch("agent.source.index._link", side_effect=lambda p: p == target or _link(p)):
            self.navigator.refresh()
            self.assertNotIn("linked.py", self.navigator.entries)

    def test_file_and_total_byte_caps_are_reported_as_partial(self):
        with patch("agent.source.index.MAX_FILE_BYTES", 60):
            result = self.navigator.refresh()
        self.assertTrue(result["incomplete"])
        self.assertTrue(result["skipped"])
        with patch("agent.source.index.MAX_FILES", 2):
            result = self.navigator.refresh()
        self.assertEqual(result["indexed_files"], 2)
        self.assertTrue(result["incomplete"])
        with patch("agent.source.index.MAX_TOTAL_BYTES", 100):
            result = self.navigator.refresh()
        self.assertLessEqual(result["hashed_bytes"], 100)
        self.assertTrue(result["incomplete"])

    def test_cache_cannot_pollute_source(self):
        with self.assertRaises(ValueError):
            RepositoryNavigator(self.repo, self.repo / "cache")


if __name__ == "__main__":
    unittest.main()
