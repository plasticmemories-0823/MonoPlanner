"""验收证据不可变；编译副本允许派生物，但仍保护源码和显式输入。"""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.acceptance.contracts import VerificationSpec
from agent.acceptance.final import verify_final
from agent.execution.local import LocalEnvironment
from agent.source.snapshots import SnapshotError, SourceSnapshot
from agent.storage import continuation
from agent.storage.records import digest, save


@pytest.fixture
def candidate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Proof.v").write_text("Goal True. Proof. exact I. Qed.\n")
    (repo / ".Proof.aux").write_text("old auxiliary data")
    (repo / "Proof.glob").write_text("old symbol data")
    (repo / "_CoqProject").write_text("-Q . Top\nProof.v\n")
    return SourceSnapshot.capture(repo, tmp_path / "candidate", runtime_files=["Proof.v", "_CoqProject"])


@pytest.mark.parametrize("status", ["passed", "failed"])
def test_final_runs_on_copy_and_retains_actual_compile_outcome(candidate, tmp_path, status):
    roots = []

    def compile_fixture(runner, phase):
        roots.append(runner.repo_root)
        assert runner.repo_root == runner.environment.repo_root
        (runner.repo_root / ".Proof.aux").write_text("new auxiliary data")
        (runner.repo_root / "Proof.glob").write_text("new symbol data")
        (runner.repo_root / "Proof.vo").write_bytes(b"compiled object")
        return {"status": status, "exit_code": 0 if status == "passed" else 1}

    with patch("agent.acceptance.runner.AcceptanceRun.run", compile_fixture):
        result = verify_final(LocalEnvironment(tmp_path / "repo"), candidate,
                              VerificationSpec(["Compile the proof"], command="coqc Proof.v"),
                              None, tmp_path / "verification", 30)
    assert roots[0] != candidate.root
    assert result["status"] == status
    assert result["source_integrity"]["valid"]
    assert result["source_integrity"]["build_outputs"]["modified"] == [".Proof.aux", "Proof.glob"]
    assert candidate.verify_integrity()["valid"]
    assert (candidate.root / ".Proof.aux").read_text() == "old auxiliary data"


@pytest.mark.parametrize("path,action", [
    ("Proof.v", "modify"), ("_CoqProject", "delete"), ("injected.v", "add"),
    ("unrelated.glob", "add"), (".env", "add"),
])
def test_build_copy_does_not_hide_source_or_unknown_file_changes(candidate, tmp_path, path, action):
    build = candidate.build_copy(tmp_path / "build")
    if action == "delete":
        (build.root / path).unlink()
    else:
        (build.root / path).write_text("changed")
    assert not build.verify_build_integrity()["valid"]
    assert candidate.verify_integrity()["valid"]


@pytest.mark.parametrize("kind", ["runtime", "tracked", "legacy_git"])
def test_declared_or_tracked_compiler_named_inputs_are_protected(candidate, tmp_path, kind):
    metadata = json.loads(candidate.manifest_path.read_bytes())
    if kind == "runtime":
        metadata["runtime_files"].append("Proof.glob")
    else:
        metadata["selection"] = "git"
        if kind == "tracked":
            metadata["tracked_files"] = ["Proof.glob"]
        else:
            metadata.pop("tracked_files", None)
    save(candidate.manifest_path, metadata)
    build = candidate.build_copy(tmp_path / "build")
    (build.root / "Proof.glob").write_text("must not change an explicit input")
    assert not build.verify_build_integrity()["valid"]


def test_generated_name_does_not_allow_secret_content_or_links(candidate, tmp_path):
    build = candidate.build_copy(tmp_path / "build")
    (build.root / "Proof.glob").write_text("-----BEGIN PRIVATE KEY-----\nfixture\n")
    assert not build.verify_build_integrity()["valid"]
    with patch("agent.source.snapshots._link", side_effect=lambda path: path.name == ".Proof.aux"):
        assert not build.verify_build_integrity()["valid"]


@pytest.mark.skipif(not shutil.which("git"), reason="Git is not installed")
def test_build_copy_keeps_exact_manifest_even_under_generated_directory(candidate, tmp_path):
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    (repo / "build").mkdir()
    (repo / "build/tool.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    snapshot = SourceSnapshot.capture(repo, tmp_path / "git-candidate")
    build = snapshot.build_copy(tmp_path / "build")
    assert build.files == snapshot.files
    assert (build.root / "build/tool.py").read_text() == "VALUE = 1\n"
    assert json.loads(snapshot.manifest_path.read_bytes())["tracked_files"]


def test_copy_rejects_changed_evidence_and_existing_or_nested_destinations(candidate, tmp_path):
    with pytest.raises(SnapshotError):
        candidate.build_copy(candidate.root / "nested")
    with pytest.raises(SnapshotError):
        candidate.build_copy(tmp_path / "repo")
    (candidate.root / "Proof.v").write_text("changed evidence")
    with pytest.raises(SnapshotError):
        candidate.build_copy(tmp_path / "build")


def test_copy_checks_cancellation_and_storage_errors(candidate, tmp_path):
    with patch("agent.storage.resources.owner_cancelled", return_value=True):
        with pytest.raises(InterruptedError):
            candidate.build_copy(tmp_path / "cancelled")
    from agent.source.errors import SourceStorageError

    with patch("agent.storage.resources.preflight_space", side_effect=OSError("disk full")):
        with pytest.raises(SourceStorageError):
            candidate.build_copy(tmp_path / "full")


def test_recovery_rejects_tampered_build_despite_intact_evidence(candidate, tmp_path):
    folder = tmp_path / "feedback/p0001"
    folder.mkdir(parents=True)
    # The candidate evidence itself remains intact; the saved build report must still block.
    shutil.copytree(candidate.manifest_path.parent, folder / "candidate")
    report = {"status": "error", "source_integrity": {"valid": False, "modified": ["Proof.v"]}}
    save(folder / "result.json", report)
    save(folder / "completion.json", {"report_digest": digest(report),
         "candidate_digest": candidate.digest, "acceptance_digest": digest({})})
    loop = SimpleNamespace(root=tmp_path, state={"verification_running": True,
                           "verification_spec": {}, "pending": {"id": "p0001"}},
                           evidence=SimpleNamespace(view=lambda: SimpleNamespace(files=candidate.files)))
    with pytest.raises((ValueError, SnapshotError), match="integrity|source"):
        continuation._verification(loop)
