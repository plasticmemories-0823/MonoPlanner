"""校验显式真实执行声明；不按题号、函数名或题面关键词选择检查。"""

import ast
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EXECUTION_DESCRIPTION = (
    "Required for first generated acceptance: artifact type, real entry and consuming runtime. "
    "[] only for inspection, fixed user commands, or reuse of frozen checks. "
    "Compile/syntax-check and execute the actual artifact. Wine CMD instead parses and "
    "executes the unchanged BAT/CMD once; only exercised paths are validated in compatibility mode. "
    "Invocation-only call and unchecked executable consumers are not supported. "
    "No task-name detection or self-reported pass."
)

def parsed(text, mode="exec"):
    try:
        return ast.parse(text, mode=mode)
    except SyntaxError as error:
        raise ValueError("execution_checks contains invalid Python syntax: " + str(error)) from error


class NumericCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    values: list[float] = Field(max_length=8)
    expected: float

    @model_validator(mode="after")
    def finite_values(self):
        if not all(math.isfinite(value) for value in [*self.values, self.expected]):
            raise ValueError("execution_checks require finite numeric inputs and expectations")
        return self


class ExecutionCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    requirement_id: str = Field(default="", max_length=32)
    requirement_quote: str = Field(
        default="", max_length=12000,
        description="Verbatim original requirement scope; empty binds the complete original request.",
    )
    artifact_type: Literal["generated_code", "repository_entry"]
    imports: str = Field(default="", max_length=2000)
    setup: str = Field(default="", max_length=4000)
    expression: str = Field(default="", max_length=4000)
    consumer: Literal["command", "c_expression", "c_statement"]
    entry_file: str = Field(default="", max_length=1024)
    interpreter: Literal["", "python", "node", "shell", "wine-cmd"] = ""
    argv: list[str] = Field(default_factory=list, max_length=32)
    variables: list[str] = Field(default_factory=list, max_length=8)
    result_variable: str = ""
    cases: list[NumericCase] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def direct_producer(self):
        if self.consumer == "command":
            if self.artifact_type != "repository_entry":
                raise ValueError("Generated code must compile its producer output, not a repository command")
            path = PurePosixPath(self.entry_file)
            if (not self.entry_file or path.is_absolute() or ".." in path.parts
                    or ":" in self.entry_file or "\\" in self.entry_file
                    or "\x00" in self.entry_file or not self.interpreter):
                raise ValueError("execution_checks command requires a relative repository entry_file and interpreter")
            if (self.imports or self.setup or self.expression or self.variables
                    or self.cases or self.result_variable):
                raise ValueError("execution_checks command cannot mix function or numeric consumer fields")
            if any("\x00" in value or len(value) > 4000 for value in self.argv):
                raise ValueError("execution_checks command arguments are invalid or too large")
            if self.interpreter == "wine-cmd":
                from resources.drivers.execution import wine_command
                wine_command("/testbed/" + self.entry_file, self.argv)
            return self
        if self.artifact_type != "generated_code":
            raise ValueError("Code consumers require the generated_code artifact type")
        if self.entry_file or self.interpreter or self.argv:
            raise ValueError("execution_checks non-command consumer cannot supply command fields")
        if not self.imports or not self.expression:
            raise ValueError("execution_checks requires imported producer and direct expression")
        imports = parsed(self.imports)
        names = set()
        for node in imports.body:
            if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                raise ValueError("execution_checks.imports requires absolute from-imports")
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_"):
                    raise ValueError("execution_checks cannot import wildcard/private names")
                name = alias.asname or alias.name
                if name in names:
                    raise ValueError("execution_checks contains duplicate imported bindings")
                names.add(name)
        call = parsed(self.expression, mode="eval").body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            raise ValueError("execution_checks.expression must directly call an imported entry")
        if (call.func.id not in names or any(keyword.arg is None for keyword in call.keywords)
                or any(isinstance(arg, ast.Starred) for arg in call.args)):
            raise ValueError("execution_checks entry must be imported; expanded keywords are unsupported")
        for node in parsed(self.setup).body:
            if (not isinstance(node, ast.Assign) or len(node.targets) != 1
                    or not isinstance(node.targets[0], ast.Name)):
                raise ValueError("execution_checks.setup only constructs named input values")
            if node.targets[0].id in names or node.targets[0].id.startswith("_"):
                raise ValueError("execution_checks cannot replace imported entries")
            if any(isinstance(item, (ast.Lambda, ast.NamedExpr)) for item in ast.walk(node.value)):
                raise ValueError("execution_checks.setup cannot define or replace behavior")
        if any(isinstance(item, (ast.Lambda, ast.NamedExpr)) for item in ast.walk(call)):
            raise ValueError("execution_checks.expression cannot define replacement behavior")
        if not self.cases:
            raise ValueError("execution_checks numeric consumers require actual input/expected cases")
        if (len(set(self.variables)) != len(self.variables)
                or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
                       for name in self.variables)):
            raise ValueError("execution_checks variables must be distinct plain identifiers")
        if any(len(case.values) != len(self.variables) for case in self.cases):
            raise ValueError("execution_checks numeric input arity differs from variables")
        if self.consumer == "c_statement":
            if (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.result_variable)
                    or self.result_variable in self.variables):
                raise ValueError("execution_checks statement result must be a separate identifier")
        elif self.result_variable:
            raise ValueError("execution_checks expression consumer has no result_variable")
        return self


def checked_execution(checks, criteria, *, required=False):
    """引文证明来源，不能证明选择的入口或预期在语义上充分。"""
    if not isinstance(checks, (list, tuple)) or len(checks) > 8:
        raise ValueError("execution_checks must contain at most eight declarations")
    if required and not checks:
        raise ValueError("execution_checks requires a real public-entry execution declaration")
    result, identifiers = [], set()
    for raw in checks:
        check = raw if isinstance(raw, ExecutionCheck) else ExecutionCheck.model_validate(raw)
        if check.id in identifiers:
            raise ValueError("execution_checks IDs must be unique")
        if check.requirement_quote and not any(check.requirement_quote in text for text in criteria):
            raise ValueError("execution_checks.requirement_quote must match original requirements exactly")
        identifiers.add(check.id)
        result.append(check.model_dump())
    return tuple(result)


def validate_execution_result(path, checks, *, repository=None, map_path=None):
    """只接收完整、逐项对应的执行回执，缺失或不明结果不放行。"""
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("checks")
    if not isinstance(rows, list) or len(rows) != len(checks):
        raise ValueError("execution_checks missing complete execution receipts")
    for check, row in zip(checks, rows):
        check = ExecutionCheck.model_validate(check).model_dump()
        if row.get("id") != check["id"] or row.get("consumer") != check["consumer"]:
            raise ValueError("execution_checks receipt identity differs from frozen declaration")
        runtime = check["interpreter"] if check["consumer"] == "command" else "c"
        if row.get("artifact_type") != check["artifact_type"] or row.get("target_runtime") != runtime:
            raise ValueError("execution_checks artifact/runtime differs from frozen declaration")
        if row.get("status") not in {"passed", "failed", "error"}:
            raise ValueError("execution_checks unknown execution state")
        if row["status"] == "passed":
            if (row.get("producer_called") is not True
                    or not re.fullmatch(r"[0-9a-f]{64}", row.get("producer_source_sha256", ""))):
                raise ValueError("execution_checks has no real repository producer receipt")
            if runtime == "wine-cmd":
                from resources.drivers.execution import WINE_SCOPE
                if (row.get("validation_scope") != WINE_SCOPE
                        or row.get("runtime_exit_code") != 0 or not row.get("runtime_argv")
                        or not str(row.get("runtime_version", "")).startswith("wine-")
                        or row.get("executed_source_sha256") != row["producer_source_sha256"]
                        or any(key.startswith("compile") for key in row)):
                    raise ValueError("execution_checks lacks honest Wine CMD compatibility evidence")
            elif (row.get("compile_exit_code") != 0 or not row.get("compile_argv")
                    or not re.fullmatch(r"[0-9a-f]{64}", row.get("compiled_source_sha256", ""))):
                raise ValueError("execution_checks lacks successful compilation of the actual artifact")
            if check["consumer"] == "command":
                if (row.get("exit_code") != 0 or not row.get("argv")
                        or (runtime != "wine-cmd" and
                            row["compiled_source_sha256"] != row["producer_source_sha256"])):
                    raise ValueError("execution_checks lacks successful repository command evidence")
            else:
                runs = row.get("runs", [])
                if (not re.fullmatch(r"[0-9a-f]{64}", row.get("artifact_sha256", ""))
                        or row["compiled_source_sha256"] != row.get("consumer_source_sha256")
                        or len(runs) != len(check["cases"])
                        or any(run.get("exit_code") != 0 or run.get("passed") is not True
                               for run in runs)):
                    raise ValueError("execution_checks lacks complete consumed-artifact evidence")
            if repository is not None:
                validate_originals(path, check, row, Path(repository), map_path)
    return value


def validate_originals(output, check, row, repository, map_path):
    """宿主复核真实原件和输出，不仅相信 JSON 中的 passed 字段。"""
    from agent.source.snapshots import _safe_file

    def content(root, relative, expected):
        try:
            raw = _safe_file(root, Path(relative)).read_bytes()
        except (OSError, RuntimeError) as error:
            raise ValueError('Execution receipt has no intact original artifact') from error
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Execution receipt differs from original artifact bytes')
        return raw

    mapped_root = str(map_path(repository)).replace('\\', '/').rstrip('/')
    producer_path = str(row.get('producer_source', '')).replace('\\', '/')
    try:
        relative = PurePosixPath(producer_path).relative_to(PurePosixPath(mapped_root))
    except ValueError as error:
        raise ValueError('Execution producer receipt escapes the confirmed repository') from error
    content(repository, relative, row['producer_source_sha256'])
    folder = output.parent / (output.stem + '-' + check['id'])
    # 编译日志必须来自本次独占执行目录，不能借另一份报告凑字段。
    logs = ('runtime.stdout', 'runtime.stderr') if check['interpreter'] == 'wine-cmd' else (
        'compile.stdout', 'compile.stderr')
    for name in logs:
        try:
            _safe_file(output.parent, Path(folder.name) / name)
        except (OSError, RuntimeError) as error:
            raise ValueError('Compilation receipt has no original process logs') from error
    if check['consumer'] == 'command':
        if relative.as_posix() != check['entry_file']:
            raise ValueError('Execution receipt names a different repository entry')
        content(folder, 'command.stdout', row.get('producer_result_sha256'))
        content(folder, 'command.stderr', row.get('stderr_sha256'))
        if check['interpreter'] == 'wine-cmd':
            from resources.drivers.execution import wine_command
            version = content(folder, 'runtime.stdout', row.get('runtime_stdout_sha256'))
            content(folder, 'runtime.stderr', row.get('runtime_stderr_sha256'))
            runtime_argv = row['runtime_argv']
            if (len(runtime_argv) != 2 or runtime_argv[1] != '--version'
                    or not runtime_argv[0].startswith('/')
                    or PurePosixPath(runtime_argv[0]).name != 'wine'
                    or version.decode('utf-8', 'replace').strip() != row['runtime_version']):
                raise ValueError('Wine runtime identity receipt changed')
            expected = [runtime_argv[0], 'cmd.exe', '/d', '/v:off', '/u', '/s', '/c',
                        '%MONOPLANER_CMD_INVOCATION%']
            invocation = wine_command(map_path(repository / check['entry_file']), check['argv'])
            if (row['argv'] != expected or row.get('cmd_invocation') != invocation
                    or row.get('cmd_builtin_output_encoding') != 'utf-16-le'):
                raise ValueError('Wine CMD receipt differs from the unchanged entry and literal arguments')
            return
        expected_tail = [str(map_path(repository / check['entry_file'])), *check['argv']]
        if row['argv'][-len(expected_tail):] != expected_tail:
            raise ValueError('Execution receipt command differs from confirmed entry and arguments')
        return
    raw = content(folder, 'raw.txt', row['artifact_sha256'])
    compiled = content(folder, 'consumer.c', row['compiled_source_sha256'])
    # 复用随产品冻结的驱动模板；生成物必须逐字进入实际编译的驱动。
    from resources.drivers.execution import c_source

    if compiled != c_source(check, raw.decode('utf-8')).encode('utf-8'):
        raise ValueError('The compiled consumer did not preserve the raw generated artifact')
    content(folder, 'consumer.exe' if row.get('executable_name') == 'consumer.exe' else 'consumer',
            row.get('executable_sha256'))
    for index, (case, run) in enumerate(zip(check['cases'], row['runs'])):
        stdout = content(folder, f'run-{index}.stdout', run.get('stdout_sha256'))
        content(folder, f'run-{index}.stderr', run.get('stderr_sha256'))
        try:
            actual = float(stdout.strip())
        except ValueError as error:
            raise ValueError('Execution output does not establish the claimed result') from error
        if (run.get('values') != case['values'] or run.get('expected') != case['expected']
                or not math.isfinite(actual)
                or not math.isclose(actual, case['expected'], rel_tol=1e-9, abs_tol=1e-12)):
            raise ValueError('Execution receipt does not satisfy the frozen behavior assertion')
