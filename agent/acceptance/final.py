"""候选方案的全部操作完成后，统一执行最终验收。"""

import time
from agent.acceptance.runner import AcceptanceRun


def verify_final(
    environment, snapshot, spec, regression_command, directory, timeout, repository_baseline=None
):
    """每份候选只在操作全部完成后验收，不逐步调用。"""
    started = time.monotonic()
    with snapshot.environment_like(environment, directory) as backend:
        runner = AcceptanceRun(
            snapshot.root,
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
                    snapshot.root,
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
        integrity = snapshot.verify_integrity()
        report["source_integrity"] = integrity
        if not integrity["valid"]:
            report.update(
                status="error", reason="Acceptance modified candidate source; result invalid"
            )
    report["seconds"] = time.monotonic() - started
    return report
