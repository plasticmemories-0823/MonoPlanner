"""校验冻结的验收输入，以及真实执行结果是否完整。"""

import ast
from dataclasses import dataclass
from agent.acceptance.oracles import oracle_issues
from agent.acceptance.execution import checked_execution
from agent.task.repair import ValidationErrors


def _validate_test_code(code: str) -> None:
    try:
        tree = ast.parse(code, filename="test_acceptance.py")
        compile(tree, "test_acceptance.py", "exec")
    except SyntaxError as exc:
        raise ValueError(
            f"验收测试有语法错误：{exc}; category=python_syntax; "
            f"source line {exc.lineno}, column {exc.offset} "
            "(1-based, Python source coordinates)"
        ) from exc
    issues = oracle_issues(tree)
    if issues:
        raise ValueError("生成的验收缺少独立预期：" + "；".join(issues))

    # 是否存在真实可运行的测试，由用例收集与执行结果确定，
    # 不能仅凭模块导入了某个测试基类判断。


def validate_driver(driver):
    import re
    from pathlib import PurePosixPath

    if not isinstance(driver, (list, tuple)) or any(
        not isinstance(arg, str) or not arg or "\x00" in arg for arg in driver
    ):
        raise ValueError("test_driver must be a Python entry and argument list")
    if not driver:
        return ()
    entry = driver[1] if driver[0] == "-m" and len(driver) > 1 else driver[0]
    if driver[0] == "-m":
        valid = bool(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", entry))
    else:
        path = PurePosixPath(entry)
        valid = (
            not path.is_absolute()
            and ".." not in path.parts
            and ":" not in entry
            and "\\" not in entry
            and path.suffix == ".py"
        )
    if not valid or not any("{test_file}" in arg or "{test_module}" in arg for arg in driver[1:]):
        raise ValueError(
            "test_driver needs a repository Python entry (or -m module) and {test_file}/{test_module}"
        )
    return tuple(driver)


@dataclass(frozen=True)
class VerificationSpec:
    criteria: tuple[str, ...]
    command: str | None = None
    test_code: str | None = None
    test_driver: tuple[str, ...] = ()
    execution_checks: tuple[dict, ...] = ()
    requirement_digest: str = ""

    def __post_init__(self):
        if (
            not isinstance(self.criteria, (tuple, list))
            or not self.criteria
            or any(not isinstance(item, str) or not item.strip() for item in self.criteria)
        ):
            raise ValueError("验收要求必须是非空的文本列表。")
        object.__setattr__(self, "criteria", tuple(item.strip() for item in self.criteria))
        errors = []
        try:
            object.__setattr__(self, "test_driver", validate_driver(self.test_driver))
        except ValueError as exc:
            errors.append((["/test_driver"], exc))
        if (self.command is None) == (self.test_code is None):
            raise ValueError("验收命令和生成的验收代码必须且只能指定一个。")
        if self.command is not None:
            if self.execution_checks:
                raise ValueError("Fixed command acceptance cannot specify execution_checks")
            if self.test_driver:
                raise ValueError("Fixed command acceptance cannot specify a generated test_driver")
            if not isinstance(self.command, str) or not self.command.strip():
                raise ValueError("验收命令不能为空。")
        else:
            if not isinstance(self.test_code, str) or not self.test_code.strip():
                errors.append((["/test_code"], ValueError("验收代码不能为空。")))
            else:
                try:
                    _validate_test_code(self.test_code)
                except ValueError as exc:
                    errors.append((["/test_code"], exc))
            try:
                object.__setattr__(self, "execution_checks", checked_execution(
                    self.execution_checks, self.criteria,
                ))
            except (ValueError, SyntaxError) as exc:
                errors.append((["/execution_checks"], ValueError("execution_checks: " + str(exc))))
        if errors:
            raise ValidationErrors(errors)


def tests_ready_for_candidate(report):
    """可运行的测试即使暴露源码异常，也可能已经满足冻结条件。

    此处检查执行证据，不证明断言在语义上正确。
    来源不明、夹具问题、用例缺失和基础环境故障仍会阻止冻结。
    """
    execution = report.get("execution")
    if execution is not None and not report.get("execution_ready", False):
        return False
    ids, rows = report.get("test_ids"), report.get("test_results")
    if (
        report.get("generated_test_ids")
        and isinstance(ids, list)
        and isinstance(rows, list)
        and type(report.get("tests_run")) is int
    ):
        # 仓库自己的驱动可能同时收集带条件的原有用例。
        # 就绪检查只排除这些原有用例的已记录跳过，不排除生成用例的跳过。
        skipped = {
            r["id"]
            for r in rows
            if r["status"] == "skipped" and r["id"] not in report["generated_test_ids"]
        }
        ids = [key for key in ids if key not in skipped]
        rows = [row for row in rows if row["id"] not in skipped]
        report = {**report, "tests_run": report.get("tests_run", 0) - len(skipped)}
    if (
        not report.get("test_evidence_valid")
        or not report.get("integrity_ok")
        or report.get("status") not in {"passed", "failed", "error"}
        or not isinstance(ids, list)
        or not ids
        or not isinstance(rows, list)
        or not rows
        or type(report.get("tests_run")) is not int
        or report["tests_run"] != len(ids)
        or report.get("error_kind") not in (None, "test_error")
        or len(rows) != len(ids)
        or {r["id"] for r in rows} != set(ids)
    ):
        return False
    for row in rows:
        status, exceptions = row["status"], row.get("exceptions", [])
        if status not in {"passed", "failed", "error"}:
            return False
        if not isinstance(exceptions, list) or any(not isinstance(e, dict) for e in exceptions):
            return False
        if any(e.get("phase") != "test" for e in exceptions):
            return False
        if status == "error" and (
            not exceptions
            or any(
                e.get("origin") != "source" or not e.get("ordinary_exception") for e in exceptions
            )
        ):
            return False
    return True
