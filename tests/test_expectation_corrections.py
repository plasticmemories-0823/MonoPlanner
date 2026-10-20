"""原文身份与预期字面量边界；不访问模型或网络。"""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from agent.acceptance.evidence import Evidence, Correction, QUOTE_ERROR, verify_sources
from agent.acceptance.expectations import rewrite, validate_change
from agent.source.snapshots import SourceSnapshot


def module(body):
    return "import unittest\nclass Checks(unittest.TestCase):\n    def test_value(self):\n" + body


def correction(before, after, **kw):
    return Correction(before=before, after=after, evidence_ids=["requirement"],
                      reason="The original requirement specifies this expectation", **kw)


class ExpectationBoundaries(unittest.TestCase):
    def test_literals_and_order_changes(self):
        for old, new in [("2", "4"), ("[1, 2]", "[2, 1]"),
                         ("{'a': (1, 2)}", "{'a': (2, 1)}"),
                         ("-2", "-4"), ("'错误'", "'正确'")]:
            with self.subTest(old=old):
                code = module(f"        self.assertEqual(value(), {old})\n")
                revised = code.replace(old, new)
                self.assertEqual(rewrite(code, [correction(old, new)]), revised)
                validate_change(code, revised)

    def test_unique_context_can_disambiguate_repeated_literal(self):
        code = module("        self.assertEqual(value(2), 2)\n")
        with self.assertRaisesRegex(ValueError, "unique"):
            rewrite(code, [correction("2", "4")])
        before = "self.assertEqual(value(2), 2)"
        after = "self.assertEqual(value(2), 4)"
        self.assertEqual(rewrite(code, [correction(before, after)]), code.replace(before, after))

    def test_only_local_expected_variable(self):
        code = module("        expected = [1, 2]\n        actual = value()\n"
                      "        self.assertEqual(actual, expected)\n"
                      "        self.assertEqual(other(), expected)\n")
        validate_change(code, code.replace("[1, 2]", "[2, 1]"))
        for extra in ["        value(expected)\n", "        expected.append(3)\n",
                      "        alias = expected\n", "        print(locals())\n",
                      "        expected = [1, 2]\n"]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_change(code + extra, (code + extra).replace("[1, 2]", "[2, 1]"))

    def test_reverse_expected_and_python_assert(self):
        for line in ["self.assertEqual([1, 2], value())", "assert value() == [1, 2]"]:
            code = module("        " + line + "\n")
            validate_change(code, code.replace("[1, 2]", "[2, 1]"))

    def test_custom_assertions_and_rebound_locals_are_not_proven_expectations(self):
        for code in [module("        self.assertEqual(value(), 2)\n").replace(
                'unittest.TestCase', 'CustomCase'),
                module("        expected = 2\n        import other as expected\n"
                       "        self.assertEqual(value(), expected)\n"),
                module("        expected = 2\n        try:\n            value()\n"
                       "        except Exception as expected:\n            pass\n"
                       "        self.assertEqual(value(), expected)\n")]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate_change(code, code.replace('2', '4'))

    def test_cannot_change_input_call_assertion_test_set_or_setup(self):
        code = module("        self.assertEqual(value(1), [1, 2])\n"
                      "    def test_other(self):\n        self.assertEqual(other(), 3)\n")
        changes = [code.replace("value(1)", "value(2)"),
                   code.replace("value(1)", "other(1)"),
                   code.replace("assertEqual", "assertNotEqual"),
                   code.split("    def test_other")[0],
                   code.replace("import unittest", "import unittest\nprint('setup')"),
                   code.replace("[1, 2]", "value()"),
                   code.replace("[1, 2]", "[value()]"),
                   code.replace("[1, 2]", "[*items]"),
                   code.replace("[1, 2]", "[1, 2]  # unrelated")]
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_change(code, changed)

    def test_ambiguous_literal_inputs_and_global_variable_are_rejected(self):
        for code in [module("        self.assertEqual(1, 2)\n"),
                     "expected = 2\n" + module("        self.assertEqual(value(), expected)\n"),
                     module("        expected = 2\n        self.assertEqual(value(expected), 4)\n")]:
            with self.subTest(code=code), self.assertRaises(ValueError):
                validate_change(code, code.replace("2", "3"))

    def test_overlapping_patches_are_rejected(self):
        code = module("        self.assertEqual(value(), [1, 2])\n")
        with self.assertRaisesRegex(ValueError, "overlap"):
            rewrite(code, [correction("[1, 2]", "[2, 1]"), correction("1, 2", "2, 1")])

    def test_unicode_and_crlf_are_not_normalized(self):
        code = module("        # 中文\n        self.assertEqual(value(), '错')\n").replace("\n", "\r\n")
        validate_change(code, code.replace("'错'", "'对'"))
        with self.assertRaises(ValueError):
            validate_change(code, code.replace("'错'", "'对'").replace("\r\n", "\n"))


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
