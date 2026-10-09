"""Offline navigation contracts. These do not measure model task success."""

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

    def test_inspection_returns_real_definition_and_related_test_in_one_tool_result(self):
        result = self.navigator.inspect_code("add", "src/tinycalc/arithmetic.py")
        item = result["candidates"][0]
        self.assertIn("left - right", item["definition"]["text"])
        self.assertEqual(item["test_example"]["path"], "tests/test_arithmetic.py")
        self.assertTrue(item["definition"]["source_sha256"])
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 10000)

    def test_inspection_marks_partial_code_and_refreshes_changed_source(self):
        path = self.repo / "long.py"
        path.write_text("def long_function():\n" + "    value = 1\n" * 180 + "    return value\n")
        before = self.navigator.inspect_code("long_function")["candidates"][0]["definition"]
        self.assertTrue(before["truncated"])
        self.assertLessEqual(before["end_line"] - before["start_line"] + 1, 70)
        self.assertEqual(before["read_hint"]["offset"], before["end_line"])
        path.write_text("def long_function():\n    return 'CHANGED'\n")
        after = self.navigator.inspect_code("long_function")["candidates"][0]["definition"]
        self.assertFalse(after["truncated"])
        self.assertIn("CHANGED", after["text"])
        self.assertNotEqual(before["source_sha256"], after["source_sha256"])
        with self.assertRaises(ValueError):
            self.navigator.inspect_code("long_function", "../outside")

    def test_readme_is_read_before_source_index_and_both_have_length_limits(self):
        (self.repo / "README.md").write_text("PROJECT_OVERVIEW\n" * 2000, encoding="utf-8")
        events = []
        read = self.navigator._text_excerpt
        refresh = self.navigator.refresh
        def track_read(path, chars):
            events.append(path.name)
            return read(path, chars)
        def track_refresh():
            events.append("INDEX")
            return refresh()
        with patch.object(self.navigator, "_text_excerpt", side_effect=track_read), \
             patch.object(self.navigator, "refresh", side_effect=track_refresh):
            result = self.navigator.overview("修复 add")
        self.assertLess(events.index("README.md"), events.index("INDEX"))
        self.assertTrue(result["readme"]["excerpt"].startswith("PROJECT_OVERVIEW"))
        self.assertTrue(result["readme"]["truncated"])
        self.assertLessEqual(len(result["readme"]["excerpt"]), 2400)
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 8000)
        self.assertEqual(result["test_config_hints"][0]["line"], 4)
        self.assertIn("[tool.pytest.ini_options]", result["test_config_hints"][0]["excerpt"])

    def test_missing_readme_and_natural_language_miss_leave_module_navigation(self):
        (self.repo / "README.md").unlink()
        result = self.navigator.overview("修复空白昵称的行为")
        self.assertIsNone(result["readme"]["path"])
        self.assertEqual(result["candidates"], [])
        self.assertIn("src/tinycalc/names.py", result["module_paths"])
        self.assertIn("不是语义检索", result["note"])

    def test_nested_request_prefers_consumer_not_argument_definitions(self):
        (self.repo / "consumer.py").write_text("def render(value):\n    return str(value)\n", encoding="utf-8")
        (self.repo / "arguments.py").write_text("def sin(x): return x\ndef wrap(x): return x\n", encoding="utf-8")
        result = self.navigator.overview("Please implement a code fix for this repository issue:\n\nrender(sin(wrap(x))) returns invalid output")
        self.assertEqual(result["candidates"][0]["path"], "consumer.py")
        self.assertEqual(result["candidates"][0]["symbols"][0]["name"], "render")

    def test_related_tests_follow_requested_symbol_not_entire_module_or_unbound_method(self):
        (self.repo / "consumer.py").write_text("def render(x): return x\ndef unrelated(x): return x\nclass Other:\n    def run(self): pass\n", encoding="utf-8")
        (self.repo / "test_relevant.py").write_text("from consumer import render as convert\ndef test_render(): assert convert(1) == 1\n", encoding="utf-8")
        (self.repo / "test_unrelated.py").write_text("from consumer import unrelated\ndef test_other(): assert unrelated(1) == 1\n", encoding="utf-8")
        (self.repo / "test_method.py").write_text("def test_run(obj): obj.run()\n", encoding="utf-8")
        result = self.navigator.query("render", path="consumer.py")["candidates"][0]
        self.assertEqual([s["name"] for s in result["symbols"]], ["render"])
        self.assertEqual([t["path"] for t in result["related"]["tests"]], ["test_relevant.py"])
        result = self.navigator.query("Other", path="consumer.py")["candidates"][0]
        self.assertNotIn("test_method.py", [t["path"] for t in result["related"]["tests"]])

    def test_every_frozen_localization_case_finds_source_and_related_test(self):
        for case in self.data["cases"]:
            with self.subTest(case=case["id"]):
                result = self.navigator.query(case["query"], limit=3)
                self.assertEqual(result["candidates"][0]["path"], case["expected_file"])
                self.assertIn(case["expected_test"],
                              [t["path"] for t in result["candidates"][0]["related"]["tests"]])
                self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 10000)

    def test_aliases_relative_imports_and_fixtures_have_explicit_static_evidence(self):
        result = self.navigator.query("add", path="./src/tinycalc/arithmetic.py")
        item = result["candidates"][0]
        self.assertEqual(item["symbols"][0]["signature"], "def add(left: float, right: float) -> float")
        self.assertEqual(item["symbols"][0]["line"], 1)
        self.assertEqual(item["symbols"][0]["end_line"], 3)
        self.assertEqual(item["read_hint"]["offset"], 0)
        self.assertEqual(item["related"]["tests"][0]["path"], "tests/test_arithmetic.py")
        self.assertIn("sum_pair", item["related"]["tests"][0]["evidence"])
        self.assertEqual(item["related"]["tests"][0]["certainty"], "static_candidate")
        self.assertIn("tests/conftest.py", item["related"]["fixture_files"])
        self.assertIn("src/tinycalc/api.py", [c["path"] for c in item["related"]["callers"]])
        dependencies = self.navigator.query("total", path="src/tinycalc/api.py")["candidates"][0]["related"]["dependencies"]
        self.assertIn("src/tinycalc/arithmetic.py", [d["path"] for d in dependencies])
        decoy = self.navigator.query("add", path="src/unrelated/arithmetic.py")["candidates"][0]
        self.assertEqual(decoy["related"]["tests"], [])

    def test_class_async_signature_and_doc_are_parsed_without_executing_source(self):
        marker = self.root / "SHOULD_NOT_EXIST"
        (self.repo / "client.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
            "class Client:\n    async def send(self, text: str) -> int:\n"
            "        \"\"\"Send one text.\"\"\"\n        return len(text)\n", encoding="utf-8")
        result = self.navigator.query("Client", path="client.py")
        self.assertFalse(marker.exists())
        symbols = result["candidates"][0]["symbols"]
        method = next(s for s in symbols if s["name"] == "Client.send")
        self.assertEqual(method["signature"], "async def send(self, text: str) -> int")
        self.assertEqual(method["doc"], "Send one text.")
        self.assertEqual((method["line"], method["end_line"]), (4, 6))

    def test_content_hash_detects_same_size_same_mtime_changes_and_reuses_other_parses(self):
        first = self.navigator.query("add")
        path = self.repo / "src/tinycalc/arithmetic.py"
        before = path.stat()
        path.write_text(path.read_text(encoding="utf-8").replace("left - right", "left + right"), encoding="utf-8")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        second = self.navigator.query("add")
        self.assertNotEqual(first["source_version"], second["source_version"])
        self.assertEqual(second["index"]["parsed_files"], 1)
        self.assertEqual(second["index"]["reused_files"], first["index"]["indexed_files"] - 1)
        warm = RepositoryNavigator(self.repo, self.root / "cache").query("add")
        self.assertEqual(warm["source_version"], second["source_version"])
        self.assertEqual(warm["index"]["parsed_files"], 0)

    def test_deleted_and_broken_files_never_return_stale_symbols(self):
        self.navigator.refresh()
        (self.repo / "src/tinycalc/names.py").unlink()
        (self.repo / "src/tinycalc/arithmetic.py").write_text("def add(:\n", encoding="utf-8")
        result = self.navigator.query("", path="src/tinycalc/arithmetic.py")
        self.assertEqual(result["candidates"][0]["symbols"], [])
        self.assertIn("SyntaxError", result["candidates"][0]["parse_error"])
        self.assertTrue(result["index"]["incomplete"])
        self.assertNotIn("src/tinycalc/names.py", self.navigator.entries)

    def test_baseline_and_live_paths_do_not_share_mutable_cache(self):
        import shutil
        copy = self.root / "baseline"
        shutil.copytree(self.repo, copy)
        baseline = RepositoryNavigator(copy, self.root / "cache")
        baseline_before = baseline.query("add")
        live_before = self.navigator.query("add")
        self.assertEqual(baseline_before["source_version"], live_before["source_version"])
        path = self.repo / "src/tinycalc/arithmetic.py"
        path.write_text("def add(a, b, strict=False):\n    return a + b\n", encoding="utf-8")
        live_after = self.navigator.query("add")
        self.assertNotEqual(live_after["source_version"], baseline.query("add")["source_version"])
        self.assertNotEqual(baseline.cache_path, self.navigator.cache_path)

    def test_corrupt_cache_is_rebuilt(self):
        self.navigator.cache_path.write_text("{", encoding="utf-8")
        rebuilt = RepositoryNavigator(self.repo, self.root / "cache").query("normalize_name")
        self.assertGreater(rebuilt["index"]["parsed_files"], 0)
        self.assertEqual(rebuilt["candidates"][0]["path"], "src/tinycalc/names.py")

    def test_derived_and_secret_files_are_excluded(self):
        for relative in (".venv/library.py", ".env.local.py", ".git/hook.py", "eval/runs/old.py",
                         "node_modules/lib.py", "__pycache__/generated.py"):
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("def EXCLUDED_MARKER(): pass\n", encoding="utf-8")
        self.assertEqual(self.navigator.query("EXCLUDED_MARKER")["candidates"], [])

    def test_reparse_points_are_not_read(self):
        target = self.repo / "linked.py"
        target.write_text("def LINK_TARGET(): pass\n", encoding="utf-8")
        # Exercise the reparse exclusion without requiring Windows symlink privilege.
        from agent.source.index import _link
        with patch("agent.source.index._link", side_effect=lambda p: p == target or _link(p)):
            self.assertEqual(self.navigator.query("LINK_TARGET")["candidates"], [])

    def test_file_and_total_byte_caps_are_reported_as_partial(self):
        with patch("agent.source.index.MAX_FILE_BYTES", 60):
            result = self.navigator.query("total")
        self.assertTrue(result["index"]["incomplete"])
        self.assertTrue(result["index"]["skipped"])
        with patch("agent.source.index.MAX_FILES", 2):
            result = self.navigator.query("total")
        self.assertEqual(result["index"]["indexed_files"], 2)
        self.assertTrue(result["index"]["incomplete"])
        with patch("agent.source.index.MAX_TOTAL_BYTES", 100):
            result = self.navigator.query("total")
        self.assertLessEqual(result["index"]["hashed_bytes"], 100)
        self.assertTrue(result["index"]["incomplete"])


    def test_invalid_scope_is_rejected_and_cache_cannot_pollute_source(self):
        for path in ("../x", "/tmp/x", "C:/source/x", "src/../../x", "x" * 601, None):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.navigator.query(path=path)
        for limit in (0, 7, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.navigator.query(limit=limit)
        with self.assertRaises(ValueError):
            self.navigator.query("x" * 1201)
        with self.assertRaises(ValueError):
            RepositoryNavigator(self.repo, self.repo / "cache")


if __name__ == "__main__":
    unittest.main()
