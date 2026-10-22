"""开工复制失败保留原件；既不冒充执行未知，也不重新捕获同一任务。"""

import errno
import json
from unittest.mock import patch

import pytest

from agent.application import work_bridge as bridge
from agent.execution.projects import EnvironmentRequired
from test_work_bridge import OfflineEnvironment, RecordedModel


def job_for(repo):
    return {
        "task_id": "8" * 32, "repo": str(repo), "request": "开始",
        "execution_binding": {"image": "sha256:" + "b" * 64, "runtime": {}},
    }


def test_copy_failure_is_reported_once_and_preserves_partial_scene(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("original")
    root = tmp_path / "state"
    job = job_for(repo)
    model = RecordedModel()

    def full_disk(source, target, **kwargs):
        target.mkdir()
        (target / "source").mkdir()
        (target / "source" / "partial.zip").write_bytes(b"partial evidence")
        (target / "manifest.json").write_text('{"status":"failed"}')
        raise OSError(errno.ENOSPC, "No space left on device")

    with patch.object(bridge, "require_binding", side_effect=lambda repo, value: value), patch(
        "agent.source.snapshots.SourceSnapshot.capture", side_effect=full_disk
    ) as capture:
        result = bridge.run_work(job, lambda *args: None, lambda: [], model=model,
                                 environment_factory=OfflineEnvironment, root=root)
        before = {str(p.relative_to(root)): p.read_bytes()
                  for p in root.rglob("*") if p.is_file()}
        again = bridge.run_work(job, lambda *args: None, lambda: [], model=model,
                                environment_factory=OfflineEnvironment, root=root)
        after = {str(p.relative_to(root)): p.read_bytes()
                 for p in root.rglob("*") if p.is_file()}
    assert result["status"] == again["status"] == "preparation_failed"
    assert "空间" in result["stop_reason"]
    assert capture.call_count == 1
    assert model.calls == []
    assert before == after
    assert (repo / "sample.py").read_text() == "original"


def test_environment_failure_is_saved_without_model_or_container(tmp_path):
    job = job_for(tmp_path)
    job.pop("execution_binding")
    with patch.object(bridge, "prepare_binding", side_effect=EnvironmentRequired("环境不完整")), patch(
        "agent.application.work_bridge.create_model"
    ) as create, patch.object(OfflineEnvironment, "start") as start:
        result = bridge.run_work(job, lambda *args: None, lambda: [], root=tmp_path,
                                 environment_factory=OfflineEnvironment)
    assert result["status"] == "preparation_failed"
    assert "环境不完整" in result["stop_reason"]
    create.assert_not_called()
    start.assert_not_called()
    saved = json.loads((tmp_path / "tasks" / job["task_id"] /
                        "preparation-failed.json").read_text(encoding="utf-8"))
    assert saved["status"] == "preparation_failed"


def test_failure_after_work_state_exists_remains_critical(tmp_path):
    job = job_for(tmp_path / "repo")
    (tmp_path / "repo").mkdir()

    def fail_running(runner, *args, **kwargs):
        runner.state = {"status": "executing", "running_step": "write"}
        raise OSError(errno.ENOSPC, "Critical result not saved")

    with patch.object(bridge, "require_binding", side_effect=lambda repo, value: value), patch.object(
        bridge.WorkRunner, "run", fail_running
    ), pytest.raises(OSError, match="Critical result"):
        bridge.run_work(job, lambda *args: None, lambda: [], model=RecordedModel(),
                        environment_factory=OfflineEnvironment, root=tmp_path / "state")
    assert not (tmp_path / "state/tasks" / job["task_id"] / "preparation-failed.json").exists()


def test_unwritable_failure_record_does_not_claim_preparation_result_saved(tmp_path):
    job = job_for(tmp_path)
    job.pop("execution_binding")
    with patch.object(bridge, "prepare_binding", side_effect=EnvironmentRequired("failure")), patch.object(
        bridge, "save", side_effect=OSError(errno.ENOSPC, "record full")
    ), pytest.raises(OSError, match="record full"):
        bridge.run_work(job, lambda *args: None, lambda: [], root=tmp_path,
                        environment_factory=OfflineEnvironment)
    assert (str(tmp_path.resolve()), job["task_id"]) not in bridge._owners


def test_old_task_environment_failure_keeps_recovery_protection(tmp_path):
    job = job_for(tmp_path)
    job.pop("execution_binding")
    directory = tmp_path / "tasks" / job["task_id"]
    directory.mkdir(parents=True)
    (directory / "state.json").write_text('{"running_step":"old write"}')
    with patch.object(bridge, "prepare_binding", side_effect=EnvironmentRequired("old failure")), (
        pytest.raises(ValueError, match="no valid index")
    ):
        bridge.run_work(job, lambda *args: None, lambda: [], root=tmp_path,
                        environment_factory=OfflineEnvironment)
    assert not (directory / "preparation-failed.json").exists()


def test_pause_still_kills_owned_container_when_preparation_index_is_full(tmp_path):
    with patch("agent.execution.preparation.jobs", side_effect=OSError(errno.ENOSPC, "full")), patch(
        "agent.storage.resources.cancel_owned_containers",
        return_value=[{"container": "owned", "stopped": True}],
    ) as terminate:
        report = bridge.pause_work("9" * 32, "emergency", root=tmp_path)
    terminate.assert_called_once_with("task:" + "9" * 32, root=tmp_path.resolve())
    assert report["termination"][0]["stopped"] is False
    assert report["termination"][1] == {"container": "owned", "stopped": True}


def test_history_connection_closes_when_background_copy_cleanup_fails(tmp_path):
    from agent.task.runner import TaskRunner
    from agent.task.budget import StepBudget

    repo = tmp_path / "repo"
    repo.mkdir()
    loop = TaskRunner(RecordedModel(), OfflineEnvironment(repo, "fixture", tmp_path / "run"),
                      state_dir=tmp_path / "state")
    with patch.object(loop, "_run_new", return_value={"status": "task_blocked"}), patch(
        "agent.source.lifecycle.close", side_effect=OSError(errno.ENOSPC, "cleanup full")
    ), pytest.raises(OSError, match="cleanup full"):
        loop.run([], "fixture", StepBudget(max_steps=1))
    assert loop.history_store.connection is None
