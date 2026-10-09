"""Source journals retain original versions, changed bytes and deletions."""
import json
from pathlib import Path
import tempfile
import unittest

from agent.execution.local import LocalEnvironment
from agent.source.snapshots import SourceSnapshot
from agent.source.versions import SourceVersions


class SourceVersionsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        (self.repo / "module.py").write_text("value = 1\n", encoding="utf-8")
        self.env = LocalEnvironment(self.repo, artifact_root=self.root / "logs")
        self.baseline = SourceSnapshot.capture(self.repo, self.root / "baseline")
        self.evidence = SourceVersions(self.env, self.env.artifact_root / "tests", baseline=self.baseline)


    def test_versions_retain_changed_bytes_and_deletions_without_copying_baseline(self):
        (self.repo / "module.py").write_bytes(b"value = 2\n")
        edited = self.evidence.record("edit")
        manifest = json.loads((self.evidence.root / "states" / (edited + ".json")).read_text(encoding="utf-8"))
        self.assertEqual((self.evidence.root / "blobs" / manifest["files"]["module.py"]).read_bytes(), b"value = 2\n")
        self.assertFalse((self.evidence.root / "blobs" / self.baseline.files["module.py"]).exists())
        (self.repo / "module.py").unlink()
        deleted = self.evidence.record("delete")
        self.assertEqual(json.loads((self.evidence.root / "states" / (deleted + ".json")).read_text())["files"], {})


if __name__ == "__main__":
    unittest.main()
