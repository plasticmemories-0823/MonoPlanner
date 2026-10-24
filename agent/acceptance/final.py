"""候选方案的全部操作完成后，统一执行最终验收。"""

import time
from pathlib import Path
from agent.acceptance.runner import AcceptanceRun
from agent.source.snapshots import SnapshotError


def require_source_integrity(report):
    """包括恢复路径在内，源码被改动的验收不能推进已保存游标。"""
    for field in ("source_integrity", "snapshot_integrity"):
        if field in report and not report[field]["valid"]:
            raise SnapshotError("Acceptance source integrity failed; retained evidence requires inspection.")


def verify_final(
    environment, snapshot, spec, regression_command, directory, timeout, repository_baseline=None
):
    """每份候选只在操作全部完成后验收，不逐步调用。"""
    started = time.monotonic()
    build = snapshot.build_copy(Path(directory) / "build-workspace")
    with build.environment_like(environment, directory) as backend:
        runner = AcceptanceRun(
            build.root,
            spec,
            artifacts_root=directory,
            environment=backend,
            timeout=timeout,
            repository_baseline=repository_baseline,
        )
        report = runner.run("candidate")
        if report["status"] == "passed" and regression_command:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                report.update(status="error", reason="Final acceptance time budget exhausted")
            else:
                extra = AcceptanceRun(
                    build.root,
                    type(spec)(spec.criteria, command=regression_command),
                    artifacts_root=directory,
                    environment=backend,
                    timeout=remaining,
                ).run("regression")
                report["regression"] = extra
                if extra["status"] != "passed":
                    report.update(
                        status=extra["status"], reason="Frozen repository regression did not pass"
                    )
        integrity = build.verify_build_integrity()
        report["source_integrity"] = integrity
        report["snapshot_integrity"] = snapshot.verify_integrity()
        report["build_manifest"] = str(build.manifest_path)
        if not integrity["valid"] or not report["snapshot_integrity"]["valid"]:
            report.update(
                status="error", reason="Acceptance modified candidate source; result invalid"
            )
    report["seconds"] = time.monotonic() - started
    return report
