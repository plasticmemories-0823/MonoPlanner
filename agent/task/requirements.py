"""保存人类确认的交付边界；工作模型只能实现，不能改写这些约束。"""

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.storage.records import digest, save


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Quote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=12000)


class DeliveryTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    artifact_type: Literal["generated_code", "repository_entry", "python_api", "file_artifact"]
    target_runtime: str = Field(min_length=1, max_length=40)
    entry: str = Field(min_length=1, max_length=1024)
    behavior: str = Field(min_length=1, max_length=4000)
    input_scope: str = Field(min_length=1, max_length=4000)
    imports: str = Field(default="", max_length=2000)
    setup: str = Field(default="", max_length=4000)
    expression: str = Field(default="", max_length=4000)
    variables: list[str] = Field(default_factory=list, max_length=8)
    required_values: list[list[float]] = Field(default_factory=list, max_length=8)
    argv: list[str] = Field(default_factory=list, max_length=32)
    python_module: str = Field(
        default="", max_length=1024,
        description="Optional dotted module for Python -m; entry binds the module .py or package __main__.py. Empty preserves file invocation.",
    )
    quotes: list[Quote] = Field(min_length=1, max_length=16)

    @field_validator("imports", "setup", "expression")
    @classmethod
    def invocation_syntax(cls, value, info):
        # 这里只解析语法；中文说明等非法参数必须留在可纠正的字段拒收路径。
        if value:
            try:
                ast.parse(value, mode="eval" if info.field_name == "expression" else "exec")
            except SyntaxError as exc:
                raise ValueError(
                    f"{info.field_name}: line {exc.lineno or 1}, column {exc.offset or 1}: "
                    "provide valid Python code, not a prose description."
                ) from exc
        return value

    @model_validator(mode="after")
    def interface(self):
        if self.artifact_type in {"python_api", "file_artifact"}:
            if (self.target_runtime != "python" or self.imports or self.setup
                    or self.expression or self.variables or self.required_values
                    or self.argv or self.python_module):
                raise ValueError("Library and file deliveries use Python unittest consumption, not a command")
            if self.artifact_type == "python_api":
                from agent.acceptance.execution import validate_python_api
                validate_python_api(self.entry, allow_scope=True)
            else:
                path = PurePosixPath(self.entry)
                if (path.is_absolute() or ".." in path.parts or ":" in self.entry
                        or "\\" in self.entry or "\x00" in self.entry):
                    raise ValueError("A file artifact requires its original relative delivery path")
            return self
        if self.artifact_type == "generated_code":
            module, separator, name = self.entry.partition(":")
            if (not separator or not module or not name.isidentifier() or not self.expression
                    or not self.imports):
                raise ValueError("A generated target requires module:entry and its original invocation")
            if self.argv or self.python_module:
                raise ValueError("Generated targets cannot substitute a repository command")
        else:
            path = PurePosixPath(self.entry)
            if path.is_absolute() or ".." in path.parts or ":" in self.entry or "\\" in self.entry:
                raise ValueError("A repository target requires a relative entry file")
            if self.imports or self.setup or self.expression or self.variables or self.required_values:
                raise ValueError("Repository targets cannot contain generated-code input fields")
            from agent.acceptance.execution import validate_python_module
            validate_python_module(self.python_module, self.entry, self.target_runtime)
        if any(len(row) != len(self.variables) or any(not math.isfinite(n) for n in row)
               for row in self.required_values):
            raise ValueError('Confirmed required inputs must be finite and match the input arity')
        return self


def confirmed(request, targets, *, sources=None, confirmation=None, parent=None):
    """仅供可信入口调用；不作为工作模型工具公开，也不从模型回复猜测授权。"""
    sources = deepcopy(sources or [{"id": "task_request", "role": "user", "text": request}])
    if not sources or any(row.get("role") != "user" for row in sources):
        raise ValueError("Original requirement sources must be saved user messages")
    originals = {}
    for row in sources:
        identity, text = row["id"], row["text"]
        if not isinstance(text, str) or not text or identity in originals:
            raise ValueError("Requirement source identities must be unique and complete")
        originals[identity] = text
        row["sha256"] = text_hash(text)
    values = [DeliveryTarget.model_validate(item).model_dump() for item in targets]
    if not values or len(values) > 8 or len({item["id"] for item in values}) != len(values):
        raise ValueError("Confirmed requirements need 1..8 distinct delivery targets")
    for item in values:
        for quote in item["quotes"]:
            if quote["source_id"] not in originals or quote["text"] not in originals[quote["source_id"]]:
                raise ValueError("Delivery requirement quote is not an exact original user excerpt")
    if not confirmation or not confirmation.get("id") or not confirmation.get("text"):
        raise ValueError("A saved explicit user confirmation is required")
    value = {
        "format": "monoplanner-confirmed-requirements-v1",
        "revision": parent["revision"] + 1 if parent else 1,
        "parent_digest": parent["digest"] if parent else "",
        "request_sha256": text_hash(request), "sources": sources,
        "targets": values, "confirmation": deepcopy(confirmation),
    }
    value["digest"] = digest(value)
    return value


def checked(value, request=None):
    if not isinstance(value, dict) or value.get("format") != "monoplanner-confirmed-requirements-v1":
        raise ValueError("Confirmed original delivery requirements are missing; inspection only")
    core = {key: item for key, item in value.items() if key != "digest"}
    if value.get("digest") != digest(core):
        raise ValueError("Confirmed delivery requirements changed after confirmation")
    fields = {'format', 'revision', 'parent_digest', 'request_sha256', 'sources', 'targets', 'confirmation'}
    if (set(core) != fields or type(value['revision']) is not int or value['revision'] < 1
            or (value['revision'] == 1) != (value['parent_digest'] == '')):
        raise ValueError('Invalid confirmed requirement version')
    if request is not None and value["request_sha256"] != text_hash(request):
        raise ValueError("Confirmed delivery requirements belong to a different original request")
    for source in value["sources"]:
        if source["sha256"] != text_hash(source["text"]):
            raise ValueError("Original requirement source changed")
    originals = {source['id']: source['text'] for source in value['sources']}
    if (len(originals) != len(value['sources']) or not originals
            or any(source.get('role') != 'user' for source in value['sources'])
            or not value['confirmation'].get('id') or not value['confirmation'].get('text')):
        raise ValueError('Incomplete original requirement or human confirmation')
    targets = [DeliveryTarget.model_validate(target) for target in value['targets']]
    if not 1 <= len(targets) <= 8 or len({target.id for target in targets}) != len(targets):
        raise ValueError('Invalid confirmed delivery target identities')
    for target in targets:
        if any(q.source_id not in originals or q.text not in originals[q.source_id] for q in target.quotes):
            raise ValueError('Delivery reference differs from original requirement')
    return value


def active(loop):
    if loop.state.get("requirements_pending"):
        raise ValueError("A human requirement change is awaiting confirmation; inspection only")
    value = checked(loop.state.get("requirements"))
    return value


def model_view(value):
    if value is None:
        return None
    checked(value)
    return {key: deepcopy(value[key]) for key in ("revision", "digest", "targets")}


def same_code(left, right, mode="exec"):
    return ast.dump(ast.parse(left, mode=mode)) == ast.dump(ast.parse(right, mode=mode))


def check_execution(loop, checks):
    """逐项绑定已确认的入口、输入构造和运行时，并保留每个错误的字段位置。"""
    from agent.acceptance.execution import ExecutionCheck
    from agent.task.repair import ValidationErrors, encode

    contract = active(loop)
    if loop.state.get("verify_command"):
        return contract['digest']
    targets = {item["id"]: item for item in contract["targets"]}
    for target in targets.values():
        supported = ({'c'} if target['artifact_type'] == 'generated_code' else
                     {'python'} if target['artifact_type'] in {'python_api', 'file_artifact'} else
                     {'python', 'node', 'shell', 'wine-cmd'})
        if target['target_runtime'] not in supported:
            raise ValueError('The confirmed target runtime has no execution adapter; report the capability gap without downgrade')
    covered, errors = set(), []

    def issue(paths, message):
        errors.append((paths, ValueError(message)))

    def mismatch(base, field, expected, actual, reason):
        # 已确认值仅供定位差异，不代模型补写验收预期或修改保留草稿。
        issue([base + "/" + field], reason + "; expected=" + encode(expected)[:400]
              + "; actual=" + encode(actual)[:400])

    for index, raw in enumerate(checks):
        base = f"/execution_checks/{index}"
        try:
            check = ExecutionCheck.model_validate(
                raw.model_dump() if hasattr(raw, "model_dump") else raw
            ).model_dump()
        except ValueError as error:
            issue([base], str(error))
            continue
        identity = check.get("requirement_id", "")
        if not identity and len(targets) == 1:
            identity = next(iter(targets))
        if identity not in targets:
            mismatch(base, "requirement_id", list(targets), identity,
                     "execution_checks must reference a confirmed delivery requirement")
            continue
        target = targets[identity]
        covered.add(identity)
        runtime = check["interpreter"] if check["consumer"] in {"command", "python_tests"} else "c"
        if (target["artifact_type"] != check["artifact_type"]
                or target["target_runtime"] != runtime):
            issue([base], "Execution artifact or runtime differs from the confirmed original "
                  "requirement; expected=" + encode({"artifact_type": target["artifact_type"],
                  "target_runtime": target["target_runtime"]}) + "; actual=" + encode({
                  "artifact_type": check["artifact_type"], "target_runtime": runtime}))
            continue
        if target["artifact_type"] in {"python_api", "file_artifact"}:
            field = "api_entry" if target["artifact_type"] == "python_api" else "entry_file"
            matches = check[field] == target['entry']
            if target['artifact_type'] == 'python_api' and target['entry'].endswith(':*'):
                scope = target['entry'][:-2]
                module = check[field].split(':', 1)[0]
                matches = module == scope or module.startswith(scope + '.')
            if not matches:
                mismatch(base, field, target['entry'], check[field],
                         "Consumption target differs from the confirmed original delivery")
        elif target["artifact_type"] == "generated_code":
            imported = {}
            for node in ast.parse(check["imports"]).body:
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported[alias.asname or alias.name] = node.module + ":" + alias.name
            call = ast.parse(check["expression"], mode="eval").body
            producer = imported.get(getattr(call.func, "id", None))
            if producer != target["entry"]:
                issue([base + "/imports", base + "/expression"],
                      "Execution producer differs from the confirmed public entry; expected="
                      + encode(target["entry"]) + "; actual=" + encode(producer))
            for field in ("imports", "expression", "setup", "variables"):
                equal = (
                    check[field] == target[field] if field == "variables" else
                    same_code(check[field], target[field],
                              "eval" if field == "expression" else "exec")
                )
                if not equal:
                    mismatch(
                        base, field, target[field], check[field],
                        "Execution invocation or input construction narrows/changes confirmed scope",
                    )
            # 变量数已不匹配时，同时指出受影响的用例，避免只修变量后再遭遇结构错误。
            if any(len(row["values"]) != len(target["variables"]) for row in check["cases"]):
                issue([base + "/cases"], "Execution cases must use the confirmed variable arity; "
                      f"expected_arity={len(target['variables'])}; actual_arities="
                      + encode([len(row["values"]) for row in check["cases"]])
                      + ". Supply actual inputs and independently grounded expectations.")
            actual = [row["values"] for row in check["cases"]]
            missing = [values for values in target["required_values"] if values not in actual]
            if missing:
                mismatch(base, "cases", missing, actual,
                         "Execution omits a confirmed required input; retain all required inputs")
        else:
            for field, original in (("entry_file", "entry"), ("argv", "argv"),
                                    ("python_module", "python_module")):
                expected = target.get(original, '')
                if check[field] != expected:
                    mismatch(base, field, expected, check[field],
                             "Execution entry or arguments differ from the confirmed delivery")
    if covered != set(targets):
        issue(
            ["/execution_checks"],
            "Execution does not cover every confirmed delivery target; missing="
            + encode(sorted(set(targets) - covered)),
        )
    if errors:
        raise ValidationErrors(errors)
    return contract["digest"]


def output_paths(loop):
    """只有人类确认的文件交付物可以在执行副本中被生成或更新。"""
    return tuple(sorted({target['entry'] for target in active(loop)['targets']
                         if target['artifact_type'] == 'file_artifact'}))


def guard(loop, spec=None):
    contract = active(loop)
    if spec is not None:
        spec = spec if isinstance(spec, dict) else vars(spec)
        if spec.get("requirement_digest") != contract["digest"]:
            raise ValueError("Acceptance belongs to an earlier requirement revision")
        if tuple(spec.get('output_paths', ())) != output_paths(loop):
            raise ValueError('Acceptance output mutability differs from confirmed file deliveries')
        check_execution(loop, spec.get("execution_checks", ()))


def replace_requirements(loop, value):
    """需求变更创建新验收版本；旧验收、累计轮数和历史失败留存。"""
    previous = loop.state.get("requirements")
    if value is not None:
        checked(value)
        if value["parent_digest"] != (previous["digest"] if previous else ""):
            raise ValueError("Requirement update does not follow the active confirmed revision")
        if previous:
            old_sources = {row['id']: row for row in previous['sources']}
            new_sources = {row['id']: row for row in value['sources']}
            if any(new_sources.get(key) != row for key, row in old_sources.items()):
                raise ValueError('A requirement update must retain all original human sources')
    archived = {
        key: deepcopy(loop.state.get(key)) for key in (
            "requirements", "verification_spec", "active_acceptance", "acceptance_versions",
            "acceptance_trial", "acceptance_trials", "verification", "regression_command",
        )
    }
    identity = digest(archived)
    save(loop.root / "requirements" / "history" / (identity + ".json"), archived)
    loop.state.setdefault("retired_requirements", []).append(identity)
    for key in ("verification_spec", "active_acceptance", "acceptance_trial",
                "acceptance_release", "acceptance_draft", "verification"):
        loop.state.pop(key, None)
    loop.state["acceptance_versions"] = []
    loop.state["acceptance_trials"] = []
    loop.state["requirements"] = deepcopy(value if value is not None else previous)
    loop.state["requirements_pending"] = value is None
    loop.persist("requirements_updated", {
        "digest": value["digest"] if value else None, "previous": identity,
    })


def validate_state(loop):
    """恢复时校验当前绑定和退休原件，不把旧验收套到新目标上。"""
    from agent.acceptance.revisions import checked as checked_record
    from agent.source.snapshots import _safe_file

    if loop.state.get("requirements"):
        checked(loop.state["requirements"])
    if loop.state.get("verification_spec"):
        guard(loop, loop.state["verification_spec"])

    def references(value):
        if isinstance(value, dict):
            if {"record", "sha256"} <= set(value):
                checked_record(loop.root, value)
            else:
                for child in value.values():
                    references(child)
        elif isinstance(value, list):
            for child in value:
                references(child)

    for identity in loop.state.get("retired_requirements", []):
        name = "requirements/history/" + identity + ".json"
        value = json.loads(_safe_file(loop.root, Path(name)).read_text("utf-8"))
        if digest(value) != identity:
            raise ValueError("Retained requirement history changed")
        references(value)
