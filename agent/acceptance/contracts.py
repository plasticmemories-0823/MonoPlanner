"""Validate frozen acceptance inputs and completeness of real results."""
import ast
from dataclasses import dataclass
from agent.acceptance.oracles import oracle_issues
from agent.acceptance.artifact_checks import load_artifact_checks
from agent.acceptance.behavior import declaration_only


def _validate_test_code(code: str, required_consumer=None, required_calls=()) -> None:
    try:
        tree = ast.parse(code, filename="test_acceptance.py")
        compile(tree, "test_acceptance.py", "exec")
    except SyntaxError as exc:
        raise ValueError(f"验收测试有语法错误：{exc}") from exc
    issues = oracle_issues(tree)
    errors = ["生成的验收缺少独立预期：" + "；".join(issues)] if issues else []
    try:
        checks = load_artifact_checks(code, required_consumer, required_calls)
    except ValueError as exc:
        errors.append(str(exc))
        checks = []
    if errors:
        raise ValueError("\n".join(errors))
    if checks and declaration_only(code):
        return
    if checks and not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                          and node.name.startswith("test_") for node in ast.walk(tree)):
        raise ValueError("产物声明之外的代码需要实际测试，不能被忽略。")

    # Collection and execution, rather than a particular imported base class,
    # establish whether this module contains real runnable tests.


def validate_driver(driver):
    import re
    from pathlib import PurePosixPath
    if not isinstance(driver, (list, tuple)) or any(
            not isinstance(arg, str) or not arg or "\x00" in arg for arg in driver):
        raise ValueError("test_driver must be a Python entry and argument list")
    if not driver:
        return ()
    entry = driver[1] if driver[0] == "-m" and len(driver) > 1 else driver[0]
    if driver[0] == "-m":
        valid = bool(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", entry))
    else:
        path = PurePosixPath(entry)
        valid = (not path.is_absolute() and ".." not in path.parts and ":" not in entry
                 and "\\" not in entry and path.suffix == ".py")
    if not valid or not any("{test_file}" in arg or "{test_module}" in arg for arg in driver[1:]):
        raise ValueError("test_driver needs a repository Python entry (or -m module) and {test_file}/{test_module}")
    return tuple(driver)


@dataclass(frozen=True)
class VerificationSpec:
    criteria: tuple[str, ...]
    command: str | None = None
    test_code: str | None = None
    required_consumer: str | None = None
    required_calls: tuple[str, ...] = ()
    test_driver: tuple[str, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.criteria, (tuple, list)) or not self.criteria
                or any(not isinstance(item, str) or not item.strip()
                       for item in self.criteria)):
            raise ValueError("验收要求必须是非空的文本列表。")
        object.__setattr__(self, "criteria", tuple(item.strip() for item in self.criteria))
        errors = []
        try:
            object.__setattr__(self, "test_driver", validate_driver(self.test_driver))
        except ValueError as exc:
            errors.append(str(exc))
        if (self.command is None) == (self.test_code is None):
            raise ValueError("验收命令和生成的验收代码必须且只能指定一个。")
        if self.command is not None:
            if self.test_driver:
                raise ValueError("Fixed command acceptance cannot specify a generated test_driver")
            if not isinstance(self.command, str) or not self.command.strip():
                raise ValueError("验收命令不能为空。")
        else:
            if not isinstance(self.test_code, str) or not self.test_code.strip():
                errors.append("验收代码不能为空。")
            else:
                try:
                    _validate_test_code(self.test_code, self.required_consumer, self.required_calls)
                except ValueError as exc:
                    errors.append(str(exc))
        if errors:
            raise ValueError("\n".join(errors))


def tests_ready_for_candidate(report):
    """A runnable test can expose a source exception and still be ready to freeze.

    This checks execution evidence, not the semantic correctness of its oracle.
    Unknown origins, fixtures, missing cases and infrastructure faults stay blocked.
    Consumer cases blocked by a recorded compilation failure remain failed evidence.
    """
    ids, rows = report.get("test_ids"), report.get("test_results")
    if (report.get("generated_test_ids") and isinstance(ids, list) and isinstance(rows, list)
            and type(report.get("tests_run")) is int):
        # Native runners may collect their own conditional tests alongside ours.
        # Exclude only their recorded skips from readiness, never generated skips.
        skipped = {r["id"] for r in rows if r["status"] == "skipped"
                   and r["id"] not in report["generated_test_ids"]}
        ids = [key for key in ids if key not in skipped]
        rows = [row for row in rows if row["id"] not in skipped]
        report = {**report, "tests_run": report.get("tests_run", 0) - len(skipped)}
    artifacts = {r["id"]: r for r in report.get("artifact_results", [])}
    blocked_cases = set()
    for identifier, row in artifacts.items():
        prefix, separator, index = identifier.rpartition(":case:")
        if (row["status"] == "not_run" and separator and index.isdigit()
                and artifacts.get(prefix + ":compile", {}).get("status") == "failed"):
            blocked_cases.add(identifier)
    if (not report.get("test_evidence_valid") or not report.get("integrity_ok")
            or report.get("status") not in {"passed", "failed", "error"}
            or not isinstance(ids, list) or not ids or not isinstance(rows, list) or not rows
            or type(report.get("tests_run")) is not int
            or report["tests_run"] != len(ids) - len(blocked_cases)
            or report.get("error_kind") not in (None, "test_error")
            or len(rows) != len(ids) or {r["id"] for r in rows} != set(ids)):
        return False
    for row in rows:
        status, exceptions = row["status"], row.get("exceptions", [])
        if status == "not_run" and row["id"] in blocked_cases:
            continue
        if status not in {"passed", "failed", "error"}:
            return False
        if not isinstance(exceptions, list) or any(not isinstance(e, dict) for e in exceptions):
            return False
        if any(e.get("phase") != "test" for e in exceptions):
            return False
        if status == "error" and (not exceptions or any(
                e.get("origin") != "source" or not e.get("ordinary_exception") for e in exceptions)):
            return False
    return True
