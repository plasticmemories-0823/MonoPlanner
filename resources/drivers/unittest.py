
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import unittest

def exception_evidence(test, err, phase=None):
    """Facts only: source origin is not proof that a test's expectation is correct."""
    frames = []
    for item in traceback.extract_tb(err[2]):
        path = Path(item.filename).resolve()
        if item.filename.startswith("<"):
            scope, name = "external", item.filename
        elif path == Path(test_path).resolve():
            scope, name = "test", path.name
        else:
            try:
                name = str(path.relative_to(Path(repo_root).resolve()))
                scope = "source"
            except ValueError:
                scope, name = "external", str(path)
        frames.append({"file": name, "line": item.lineno,
                       "function": item.name, "scope": scope})
    if phase is None:
        method = getattr(test, "_testMethodName", None)
        fixtures = {"setUp": "setup", "setUpClass": "setup", "setUpModule": "setup",
                    "tearDown": "teardown", "tearDownClass": "teardown", "tearDownModule": "teardown"}
        phase = next((fixtures[f["function"]] for f in frames if f["function"] in fixtures), None)
        if phase is None:
            phase = "test" if any(f["scope"] == "test" and f["function"] == method for f in frames) else "unknown"
    return {"type": err[0].__name__, "message": str(err[1]), "phase": phase,
            "origin": frames[-1]["scope"] if frames else "unknown",
            "ordinary_exception": issubclass(err[0], Exception), "frames": frames}

class EvidenceResult(unittest.TextTestResult):
    """保留每个实际测试的结果；子测试失败必须反映到父测试。"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.items = {}

    def record(self, test, status, detail="", err=None):
        test = getattr(test, "test_case", test)
        key = test.id()
        order = {"not_run": 0, "passed": 1, "skipped": 2, "expected_failure": 3,
                 "unexpected_success": 4, "failed": 5, "error": 6}
        old = self.items.get(key, {"id": key, "status": "not_run", "detail": ""})
        exceptions = list(old.get("exceptions", []))
        if err is not None:
            exceptions.append(exception_evidence(test, err))
        if order[status] >= order[old["status"]]:
            self.items[key] = {"id": key, "status": status, "detail": str(detail)[:1500]}
        if exceptions:
            self.items[key]["exceptions"] = exceptions

    def addSuccess(self, test):
        super().addSuccess(test)
        self.record(test, "passed")

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.record(test, "failed", f"{err[0].__name__}: {err[1]}", err)

    def addError(self, test, err):
        super().addError(test, err)
        self.record(test, "error", f"{err[0].__name__}: {err[1]}", err)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.record(test, "skipped", reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self.record(test, "expected_failure", str(err[1]))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self.record(test, "unexpected_success")

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            status = "failed" if issubclass(err[0], test.failureException) else "error"
            self.record(test, status, f"{subtest}: {err[0].__name__}: {err[1]}", err)

def test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_ids(item)
        else:
            yield item.id()

def main():
    global test_path, repo_root
    test_path, repo_root, output_path = sys.argv[1:]
    sys.path.insert(0, repo_root)
    report = {"status": "error", "reason": "验收程序未完成。",
              "tests_run": None, "failures": None, "errors": None, "skipped": None,
              "test_ids": [], "test_results": []}
    phase = "load"
    try:
        spec = importlib.util.spec_from_file_location("agent_acceptance_tests", test_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        phase = "collect"
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        collected = sorted(test_ids(suite))
        if len(set(collected)) != len(collected):
            raise ValueError("测试 ID 重复，无法比较逐项结果")
        report["test_ids"] = collected
        phase = "run"
        result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
        report["test_results"] = [result.items.get(key, {"id": key, "status": "not_run", "detail": ""})
                                  for key in sorted(set(collected) | set(result.items))]
        report.update(tests_run=result.testsRun, failures=len(result.failures),
                      errors=len(result.errors), skipped=len(result.skipped),
                      expected_failures=len(result.expectedFailures),
                      unexpected_successes=len(result.unexpectedSuccesses))
        if not result.testsRun:
            report.update(status="error", reason="没有实际执行任何测试。")
        elif result.skipped or result.expectedFailures:
            report.update(status="error", reason="存在跳过或预期失败的测试，验收不完整。")
        elif result.errors:
            report.update(status="error", reason="验收测试运行出错，请检查错误日志和测试环境。")
        elif result.failures or result.unexpectedSuccesses:
            report.update(status="failed", reason="验收断言未通过。")
        else:
            report.update(status="passed", reason="本次冻结的验收测试全部通过。")
    except BaseException as exc:
        traceback.print_exc()
        report.update(status="error", reason=f"无法运行验收测试：{type(exc).__name__}: {exc}")
        report["execution_exception"] = exception_evidence(None, sys.exc_info(), phase)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    sys.exit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
