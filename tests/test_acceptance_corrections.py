"""原文证据、逐项修订和非需求材料边界；不访问模型或网络。"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agent.acceptance import corrections
from agent.acceptance.evidence import Evidence, Correction, QUOTE_ERROR, verify_sources
from agent.source.snapshots import SourceSnapshot


def correction(before, after, field="test_code"):
    return Correction(field=field, before=before, after=after,
                      evidence_ids=["requirement"], reason="Supported by the quoted original")


class MaterialCorrections(unittest.TestCase):
    def test_structural_fixture_and_assertion_edits_are_supported(self):
        code = "data = {'small': 1}\nassert data['LARGE'] == 1\n"
        rows = [correction("{'small': 1}", "{'LARGE': 1}"),
                correction("assert data['LARGE'] == 1", "assert 'small' not in data")]
        self.assertEqual(corrections.rewrite(code, rows),
                         "data = {'LARGE': 1}\nassert 'small' not in data\n")

    def test_unique_context_is_required(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            corrections.rewrite("value(2) == 2", [correction("2", "4")])
        self.assertEqual(corrections.rewrite("value(2) == 2", [correction("== 2", "== 4")]),
                         "value(2) == 4")

    def test_overlapping_changes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            corrections.rewrite("items = [1, 2]", [correction("[1, 2]", "[]"),
                                                correction("1, 2", "2, 1")])

    def test_undeclared_unicode_whitespace_and_crlf_are_preserved(self):
        code = "# 中文\r\nassert item == '旧'\r\n"
        self.assertEqual(corrections.rewrite(code, [correction("'旧'", "'新'")]),
                         "# 中文\r\nassert item == '新'\r\n")

    def test_full_array_corrections_are_checked_and_can_clear_generated_driver(self):
        old = {"test_code": "assert value() == 4", "test_driver": ["driver.py", "{test_file}"],
               "execution_checks": [{"id": "target"}], "regression_command": ""}
        row = correction(json.dumps(old["test_driver"]), "[]", "test_driver")
        self.assertEqual(corrections.apply(old, [row])["test_driver"], [])
        self.assertEqual(old["test_driver"], ["driver.py", "{test_file}"])
        with self.assertRaisesRegex(ValueError, "before"):
            corrections.apply(old, [correction("[]", '["x"]', "test_driver")])
        with self.assertRaisesRegex(ValueError, "JSON"):
            corrections.apply(old, [correction("not json", "[]", "test_driver")])

    def test_empty_generated_regression_can_be_added_and_removed(self):
        old = {"regression_command": ""}
        row = correction("", "python checks.py", "regression_command")
        added = corrections.apply(old, [row])
        self.assertEqual(added["regression_command"], "python checks.py")
        self.assertEqual(corrections.apply(added, [correction("python checks.py", "", "regression_command")]), old)

    def test_repeated_empty_insertions_and_noops_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            corrections.rewrite("", [correction("", "a"), correction("", "b")])
        with self.assertRaisesRegex(ValueError, "change"):
            corrections.rewrite("abc", [correction("abc", "abc")])

    def test_correction_schema_cannot_target_original_requirement(self):
        with self.assertRaises(ValueError):
            correction("old", "new", "criteria")
        with self.assertRaises(ValueError):
            correction("old", "new", "requirement_digest")

    def test_history_projection_keeps_unrelated_checks_even_with_the_same_literal(self):
        base = {"test_code": "assert current() == 2\n"}
        different = {"test_code": "assert historical() == 2\n"}
        rows = [correction("2", "4")]
        with self.assertRaisesRegex(ValueError, "does not match"):
            corrections.project(different, base, rows)
        prefix = "raise RuntimeError('independent setup')\n"
        projected = corrections.project({"test_code": prefix + base["test_code"]}, base, rows)
        self.assertEqual(projected["test_code"], prefix + "assert current() == 4\n")


class EvidenceOrigins(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        (self.repo / 'requirements.txt').write_bytes(b'expected: 4\r\norder: a, b\r\n')
        snapshot = SourceSnapshot.capture(self.repo, self.root / 'baseline')
        self.loop = SimpleNamespace(baseline=snapshot, state={
            'request': 'return 4', 'user_updates': [{'id': 'human-1', 'text': 'Keep order a, b'}]
        })

    def evidence(self, source='task_request', quote='return 4'):
        return Evidence(id='requirement', source=source, quote=quote)

    def test_all_three_origins_are_bound_to_fingerprints(self):
        for source, quote in [('task_request', 'return 4'),
                              ('user_update:human-1', 'Keep order a, b'),
                              ('original_file:requirements.txt', 'expected: 4\r\n')]:
            rows = verify_sources(self.loop, [self.evidence(source, quote)])
            self.assertEqual(rows[0]['quote'], quote)
            self.assertEqual(len(rows[0]['source_sha256']), 64)
            self.assertEqual(rows[0]['source'], source)

    def test_exact_quotes_missing_origins_and_invalid_sources(self):
        for source, quote in [('task_request', ''), ('task_request', 'return  4'), ('task_request', 'Return 4'),
                              ('task_request', 'return 4 '), ('user_update:missing', 'return 4'),
                              ('original_file:missing.txt', 'return 4'),
                              ('original_file:requirements.txt', 'expected: 4\n')]:
            with self.subTest(source=source, quote=quote), self.assertRaisesRegex(ValueError, QUOTE_ERROR):
                verify_sources(self.loop, [self.evidence(source, quote)])
        for source in ['workspace:requirements.txt', '$record/output.txt',
                       'original_file:../repo/requirements.txt', 'https://example.com']:
            with self.subTest(source=source), self.assertRaises(ValueError):
                verify_sources(self.loop, [self.evidence(source)])

    def test_live_edits_and_generated_files_cannot_unlock(self):
        (self.repo / 'requirements.txt').write_text('expected: 2')
        (self.repo / 'model-output.txt').write_text('expected: 2')
        for source in ['original_file:requirements.txt', 'original_file:model-output.txt']:
            with self.assertRaisesRegex(ValueError, QUOTE_ERROR):
                verify_sources(self.loop, [self.evidence(source, 'expected: 2')])

    def test_snapshot_file_and_manifest_tampering(self):
        base = self.loop.baseline
        file = base.root / 'requirements.txt'
        raw = file.read_bytes()
        file.write_bytes(b'changed')
        with self.assertRaisesRegex(Exception, 'snapshot'):
            verify_sources(self.loop, [self.evidence()])
        file.write_bytes(raw)
        manifest = json.loads(base.manifest_path.read_text())
        manifest['files']['requirements.txt'] = '0' * 64
        base.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(Exception, 'snapshot'):
            verify_sources(self.loop, [self.evidence()])


if __name__ == '__main__':
    unittest.main()
