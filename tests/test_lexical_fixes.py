"""Keep the S6 lexical fixes independently of the removed vector backend."""
from pathlib import Path
import tempfile
import unittest
from agent.source.index import RepositoryNavigator, query_identifiers

class LexicalFixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
    def navigator(self):
        return RepositoryNavigator(self.repo, self.repo.parent / "cache")
    def test_compaction_keeps_candidate_paths_without_mutating_ast(self):
        candidates = [{"path": f"p{i}.py", "symbols": [{"name": "name", "doc": "x" * 250} for _ in range(5)],
                       "related": {"tests": [{"path": "test.py"}]}} for i in range(3)]
        value = RepositoryNavigator._pack({"candidates": candidates}, 1500)
        self.assertEqual(len(value["candidates"]), 3)
        self.assertEqual(len(candidates[0]["symbols"]), 5)
    def test_class_name_does_not_match_lowercase_property_or_literal(self):
        (self.repo / "property.py").write_text("class Domain:\n    def symbol(self): pass\n", encoding="utf8")
        (self.repo / "symbol.py").write_text("class Symbol: pass\n", encoding="utf8")
        query = "Symbol instances have __dict__\n`pkg.Symbol('s').__dict__` now exists (unexpectedly)."
        identifiers = query_identifiers(query)
        self.assertIn("Symbol", identifiers)
        self.assertNotIn("s", identifiers)
        self.assertNotIn("exists", identifiers)
        self.assertEqual(self.navigator().query(query, limit=1)["candidates"][0]["path"], "symbol.py")
    def test_requested_interface_precedes_incidental_example_names(self):
        (self.repo / "matrix.py").write_text("class Matrix:\n    def H(self): pass\n", encoding="utf8")
        (self.repo / "helper.py").write_text("class Piecewise: pass\ndef Ne(): pass\n", encoding="utf8")
        (self.repo / "printer.py").write_text("def ccode(expr): pass\n", encoding="utf8")
        query = "ccode(sinc(x)) fails\nUse `math.h`; expected example: Piecewise(Ne(x, 0), sin(x))."
        self.assertNotIn("h", query_identifiers(query))
        self.assertEqual(self.navigator().query(query, limit=1)["candidates"][0]["path"], "printer.py")
