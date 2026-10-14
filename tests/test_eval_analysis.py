import json
from pathlib import Path
import tempfile
import unittest

from eval.analysis.summary import summarize


class AnalysisTests(unittest.TestCase):
    def test_internal_acceptance_cannot_become_official_success(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "summary.json").write_text(
                json.dumps(
                    [
                        {"instance_id": "a", "agent_status": "verified_success"},
                        {"instance_id": "b", "agent_status": "api_error"},
                    ]
                )
            )
            unscored = summarize(root)
            self.assertIsNone(unscored["complete_sample_resolve_rate"])
            self.assertEqual(unscored["officially_scored_tasks"], 0)
            report = root / "official.json"
            report.write_text(
                json.dumps({"resolved_ids": [], "unresolved_ids": ["a"], "error_ids": ["b"]})
            )
            partial = summarize(root, [report])
            self.assertIsNone(partial["complete_sample_resolve_rate"])
            self.assertEqual(partial["false_accepts_against_official_tests"], 1)
            self.assertIsNone(partial["tasks"][1]["official_resolved"])


if __name__ == "__main__":
    unittest.main()
