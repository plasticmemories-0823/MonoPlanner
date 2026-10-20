"""即使用户输入来自管道，源码搜索也必须使用仓库原始字节。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PipedInputSearchTests(unittest.TestCase):
    def test_native_search_does_not_consume_input(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            repo = root / "repo"
            repo.mkdir()
            (repo / "target.py").write_text("REPOSITORY_ONLY_MARKER\n", encoding="utf-8")
            script = """
import json, sys
from pathlib import Path
from agent.execution.local import LocalEnvironment
from agent.source.snapshots import SourceSnapshot
from agent.source.versions import SourceVersions
from agent.execution.executor import OperationExecutor
from agent.task.contracts import Operation
repo = Path(sys.argv[1])
env = LocalEnvironment(repo, artifact_root=repo.parent / 'records')
base = SourceSnapshot.capture(repo, env.artifact_root / 'baseline')
evidence = SourceVersions(env, env.artifact_root / 'evidence', baseline=base)
executor = OperationExecutor(env, evidence, env.artifact_root)
print(json.dumps({q: executor.search(Operation(id='s', op='search', query=q)) for q in
                 ('REPOSITORY_ONLY_MARKER', 'PIPE_ONLY_MARKER')}))
"""
            result = subprocess.run(
                [sys.executable, "-B", "-X", "utf8", "-c", script, str(repo)],
                input="PIPE_ONLY_MARKER\n",
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
            self.assertEqual(data["REPOSITORY_ONLY_MARKER"]["hits"][0]["path"], "target.py")
            self.assertEqual(data["PIPE_ONLY_MARKER"]["hits"], [])
