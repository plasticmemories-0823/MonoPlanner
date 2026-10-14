"""观察仓库 Python 测试驱动的执行，不替换其配置或数据库。"""

import json
import inspect
from pathlib import Path
import runpy
import sys
import traceback
import unittest
import acceptance_runner as evidence

test_path, repo_root, output_path, driver_json = sys.argv[1:]
evidence.test_path, evidence.repo_root = test_path, repo_root
sys.path[:0] = [str(Path(test_path).parent), repo_root]
driver = [
    arg.replace("{test_file}", test_path)
    .replace("{test_module}", Path(test_path).stem)
    .replace("{test_dir}", str(Path(test_path).parent))
    for arg in json.loads(driver_json)
]
items, collected, generated, executed = {}, set(), set(), set()


def generated_case(test):
    method = getattr(test, getattr(test, "_testMethodName", ""), None)
    code = getattr(inspect.unwrap(method), "__code__", None) if method else None
    return code and Path(code.co_filename).resolve() == Path(test_path).resolve()


original_run = unittest.TextTestRunner.run


def observe_run(runner, suite):
    ids = list(evidence.test_ids(suite))
    if len(ids) != len(set(ids)) or collected.intersection(ids):
        raise ValueError("Repeated test IDs; repository evidence is ambiguous")
    collected.update(ids)

    def discover(node):
        if isinstance(node, unittest.TestSuite):
            for child in node:
                discover(child)
        elif generated_case(node):
            generated.add(node.id())

    discover(suite)
    native = runner.resultclass
    runner.resultclass = type("ObservedResult", (evidence.EvidenceResult, native), {})
    try:
        result = original_run(runner, suite)
    finally:
        runner.resultclass = native
    items.update(result.items)
    executed.update(key for key in ids if key in result.items)
    return result


unittest.TextTestRunner.run = observe_run


class PytestEvidence:
    def pytest_collection_modifyitems(self, items):
        ids = [item.nodeid for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated pytest IDs")
        collected.update(ids)
        for item in items:
            if Path(str(item.fspath)).resolve() == Path(test_path).resolve():
                generated.add(item.nodeid)

    def pytest_runtest_makereport(self, item, call):
        if call.excinfo:
            err = (call.excinfo.type, call.excinfo.value, call.excinfo.tb)
            record = items.setdefault(
                item.nodeid, {"id": item.nodeid, "status": "not_run", "detail": ""}
            )
            record.setdefault("exceptions", []).append(
                evidence.exception_evidence(None, err, "test" if call.when == "call" else call.when)
            )

    def pytest_runtest_logreport(self, report):
        key = report.nodeid
        record = items.setdefault(key, {"id": key, "status": "not_run", "detail": ""})
        if report.when == "call":
            executed.add(key)
        status = (
            ("expected_failure" if report.skipped else "unexpected_success")
            if hasattr(report, "wasxfail")
            else (
                "skipped"
                if report.skipped
                else (
                    "failed"
                    if report.failed and report.when == "call"
                    else (
                        "error"
                        if report.failed
                        else "passed" if report.when == "call" else "not_run"
                    )
                )
            )
        )
        order = {
            "not_run": 0,
            "passed": 1,
            "skipped": 2,
            "expected_failure": 3,
            "unexpected_success": 4,
            "failed": 5,
            "error": 6,
        }
        if order[status] >= order[record["status"]]:
            record.update(status=status, detail=str(getattr(report, "longrepr", "") or "")[:1500])


try:
    import pytest
except ImportError:
    pass
else:
    original_pytest = pytest.main

    def observed_pytest(args=None, plugins=None):
        return original_pytest(args, plugins=[*(plugins or []), PytestEvidence()])

    pytest.main = observed_pytest

report = {
    "status": "error",
    "reason": "Repository test runner did not finish",
    "tests_run": 0,
    "failures": 0,
    "errors": 0,
    "skipped": 0,
    "expected_failures": 0,
    "unexpected_successes": 0,
    "test_ids": [],
    "test_results": [],
}
native_exit, fault = 0, None
try:
    if driver[0] == "-m":
        sys.argv = [driver[1], *driver[2:]]
        if driver[1] == "pytest":
            native_exit = int(pytest.main(driver[2:]))
        else:
            runpy.run_module(driver[1], run_name="__main__", alter_sys=True)
    else:
        entry = Path(repo_root, driver[0]).resolve()
        entry.relative_to(Path(repo_root).resolve())
        sys.path.insert(0, str(entry.parent))
        sys.argv = [str(entry), *driver[1:]]
        runpy.run_path(str(entry), run_name="__main__")
except SystemExit as exc:
    native_exit = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
except BaseException as exc:
    traceback.print_exc()
    fault = evidence.exception_evidence(None, sys.exc_info(), "load")

rows = [
    items.get(key, {"id": key, "status": "not_run", "detail": ""})
    for key in sorted(collected | set(items))
]
report.update(
    test_ids=sorted(collected),
    test_results=rows,
    tests_run=len(executed),
    generated_test_ids=sorted(generated),
    native_exit_code=native_exit,
)
for status, name in (
    ("failed", "failures"),
    ("error", "errors"),
    ("skipped", "skipped"),
    ("expected_failure", "expected_failures"),
    ("unexpected_success", "unexpected_successes"),
):
    report[name] = sum(row["status"] == status for row in rows)
if fault:
    report["execution_exception"] = fault
elif not generated or not generated.issubset(executed):
    report["reason"] = (
        "Generated acceptance cases were not actually executed by the repository runner"
    )
elif any(
    row["status"] in {"not_run", "expected_failure"}
    or row["status"] == "skipped"
    and row["id"] in generated
    for row in rows
):
    report["reason"] = "Repository test run is incomplete, skipped, or expected to fail"
elif report["errors"]:
    report["reason"] = "Repository test setup or execution errored"
elif report["failures"] or report["unexpected_successes"]:
    report.update(status="failed", reason="Repository acceptance assertions failed")
elif native_exit:
    report["reason"] = "Repository driver returned nonzero despite recorded passing cases"
else:
    report.update(
        status="passed", reason="Generated acceptance passed using the repository test runner"
    )
Path(output_path).write_text(json.dumps(report, ensure_ascii=False), encoding="utf8")
sys.exit(0 if report["status"] == "passed" else 1)
