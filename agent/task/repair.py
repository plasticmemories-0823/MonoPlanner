"""被拒草稿保留在本地，原子应用明确且绑定源码的编辑。"""

from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from agent.model.errors import ModelRequestNotSent
from agent.storage.records import save, digest, clip
from agent.source.snapshots import _safe_file


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    operation: Literal["set", "remove", "edit_text"]
    before: str
    after: str


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    draft_id: str
    edits: list[Edit] = Field(min_length=1, max_length=24)


# ========================================================================
# 严格 JSON 与字段路径处理
# ========================================================================
def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=object_pairs,
        parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Invalid JSON number: " + v)),
    )


def location(value, path):
    if not path.startswith("/") or re.search(r"~(?![01])", path):
        raise ValueError("Expected a JSON pointer to a field: " + path)
    parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
    parent = value
    for part in parts[:-1]:
        try:
            parent = (
                parent[int(part)] if isinstance(parent, list) and part.isdecimal() else parent[part]
            )
        except (KeyError, IndexError, TypeError):
            raise ValueError("Correction parent does not exist: " + path) from None
    if not isinstance(parent, (dict, list)):
        raise ValueError("Correction parent is not an object or list: " + path)
    key = int(parts[-1]) if isinstance(parent, list) and parts[-1].isdecimal() else parts[-1]
    if isinstance(parent, list) and not isinstance(key, int):
        raise ValueError("Expected a list position: " + path)
    return parent, key


# ========================================================================
# 按明确允许的路径应用修正
# ========================================================================
def literal(edit, index, field):
    """只补充字段和解析位置，不修复 JSON 或改写提交的字符串。"""
    try:
        return loads(getattr(edit, field))
    except ValueError as exc:
        position = ""
        if isinstance(exc, json.JSONDecodeError):
            position = (
                f"line {exc.lineno}, column {exc.colno} (1-based), "
                f"character {exc.pos} (0-based) inside this field JSON, not Python source; "
            )
        raise ValueError(
            f"category=correction_json; field=/edits/{index}/{field}; "
            f"operation={edit.operation}; target={edit.path}; {position}{exc}. "
            "set before/after and remove before require JSON literals; "
            "encode string values as JSON strings."
        ) from exc


def apply(value, edits, allowed):
    result = deepcopy(value)
    for index, edit in enumerate(edits):
        if not any(
            edit.path == p or (p != "$raw" and edit.path.startswith(p + "/")) for p in allowed
        ):
            raise ValueError("Correction outside reported fields: " + edit.path)
        if edit.path == "$raw":
            if edit.operation != "edit_text":
                raise ValueError("Raw JSON correction requires edit_text")
            parent, key = {"raw": result}, "raw"
        else:
            parent, key = location(result, edit.path)
        present = (
            key in parent
            if isinstance(parent, dict)
            else isinstance(key, int) and 0 <= key < len(parent)
        )
        current = parent[key] if present else None
        if edit.operation == "edit_text":
            matches = current.count(edit.before) if isinstance(current, str) and edit.before else 0
            if not isinstance(current, str) or not edit.before or matches != 1:
                length = len(current) if isinstance(current, str) else "not-a-string"
                raise ValueError(
                    "Text correction requires one exact nonempty match: " + edit.path
                    + f"; category=correction_text_match; field=/edits/{index}/before; "
                    f"match_count={matches}; before_length={len(edit.before)}; "
                    f"target_length={length}. No unique source location is available; "
                    "before must be copied exactly from the retained original string."
                )
            parent[key] = current.replace(edit.before, edit.after, 1)
        else:
            if (not present and edit.before != "") or (
                present and encode(current) != encode(literal(edit, index, "before"))
            ):
                raise ValueError("Original value does not match: " + edit.path)
            if edit.operation == "remove":
                if not present or edit.after:
                    raise ValueError(
                        "remove requires an existing field and empty after: " + edit.path
                    )
                del parent[key]
            else:
                if isinstance(parent, list) and not present:
                    raise ValueError("List position does not exist: " + edit.path)
                parent[key] = literal(edit, index, "after")
        if edit.path == "$raw":
            result = parent[key]
    return loads(result) if isinstance(result, str) else result


# ========================================================================
# 聚合校验错误并生成局部预览
# ========================================================================
def fields(error, value):
    if isinstance(error, ValidationErrors):
        return error.paths
    if isinstance(value, str):
        return ["$raw"]
    if hasattr(error, "errors"):
        paths = [
            "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in e["loc"])
            for e in error.errors()
            if e["loc"]
        ]
        if paths:
            return sorted(set(paths))
    message = str(error)
    paths = []
    if "test_driver" in message:
        paths.append("/test_driver")
    if "execution_checks" in message:
        paths.append("/execution_checks")
    if "regression command" in message:
        paths.append("/regression_command")
    if any(word in message for word in ("验收", "产物", "预期", "Fixed task tests", "test_code")):
        paths.append("/test_code")
    if paths:
        return paths
    if any(word in message for word in ("Step", "from_step", "inspect", "blocked")):
        return ["/steps"]
    return []


class ValidationErrors(ValueError):
    """集中报告独立检查结果，不执行被拒绝的工作。"""

    def __init__(self, errors):
        self.paths = sorted({path for paths, _ in errors for path in paths})
        self.issues = []
        for paths, error in errors:
            if isinstance(error, ValidationErrors):
                self.issues.extend(error.issues)
            elif hasattr(error, "errors"):
                for issue in error.errors():
                    location = "/" + "/".join(map(str, issue["loc"]))
                    self.issues.append(
                        f"{location}: {issue['msg']}; actual={encode(issue.get('input'))[:100]}"
                    )
            else:
                self.issues.append(", ".join(paths) + ": " + str(error))
        self.issues = list(dict.fromkeys(self.issues))
        super().__init__("\n".join(self.issues))


def preview(value, paths, error):
    snippets = {}
    for path in paths:
        if path == "$raw":
            text = value
            start = max(0, getattr(error, "pos", 0) - 100)
            snippets[path] = text[start : start + 400]
            continue
        try:
            parent, key = location(value, path)
            text = encode(parent[key])
            if path == "/test_code" and isinstance(parent[key], str):
                rows = parent[key].splitlines(keepends=True)
                cause = error.__cause__
                match = re.search(r":(\d+)\b", str(error))
                line = getattr(cause, "lineno", None) or (int(match[1]) if match else None)
                indices = [max(0, line - 2)] if line else []
                text = (
                    "".join("".join(rows[i : i + 3]) for i in indices[:3])
                    if indices
                    else parent[key][:600]
                )
            snippets[path] = text[:1200]
        except (KeyError, IndexError, TypeError, ValueError):
            snippets[path] = "<missing>"
    return snippets


# ========================================================================
# 被拒草稿的保存、读取与合并
# ========================================================================
def targets_current_draft(owner, response=None):
    """只把当前单个修正调用绑定到仍有效的草稿，不借用其它回复的错误。"""
    ref = owner.state.get("repair_draft")
    data = (response or owner.state.get("response_pending") or {}).get("data", {})
    calls = data.get("tool_calls", [])
    return bool(
        ref and len(calls) == 1 and not data.get("invalid_tool_calls")
        and calls[0].get("name") == "repair_work"
        and isinstance(calls[0].get("args"), dict)
        and calls[0]["args"].get("draft_id") == ref["id"]
    )


def reject(owner, error):
    value = getattr(owner, "rejected_input", None)
    if value is not None:
        paths = fields(error, value)
        record = {
            "value": value,
            "source": owner.state["source"],
            "paths": paths,
            "error": str(error),
            "snippets": preview(value, paths, error),
        }
        if isinstance(error, ValidationErrors):
            record["issues"] = error.issues
        raw = encode(record).encode()
        identifier = hashlib.sha256(raw).hexdigest()
        relative = "drafts/" + identifier + ".json"
        path = owner.root / relative
        if not path.exists():
            save(path, record)
        ref = {
            "id": identifier,
            "record": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": record["source"],
        }
        read(owner, ref)
        owner.state["repair_draft"] = ref
    ref = owner.state.get("repair_draft")
    if value is None and not targets_current_draft(owner):
        ref = None
    if not ref:
        return (
            "[Plan rejected; no actions executed] "
            + str(error)[:1800]
            + "\nSubmit a valid work request."
        )
    record = read(owner, ref)
    notice = {
        "draft_id": ref["id"],
        "error": str(error)[:1800],
        "allowed_paths": record["paths"],
        "original_fragments": record["snippets"],
        "original_record": "$record/" + ref["record"],
        "origin": "Your rejected tool arguments; unchanged work is retained locally, not generated by the host.",
    }
    if record.get("issues"):
        notice["issues"] = record["issues"]
    if str(error) != record["error"]:
        notice["original_error"] = record["error"][:1800]
    instruction = (
        "Use repair_work for these fields; preserve all other work."
        if record["paths"]
        else "No field correction is available; submit a valid work request."
    )
    return "[Plan rejected; no actions executed] " + encode(notice) + "\n" + instruction


def read(owner, ref):
    try:
        raw = _safe_file(owner.root, Path(ref["record"])).read_bytes()
        record = loads(raw.decode())
    except (OSError, ValueError):
        raise ModelRequestNotSent(
            "state_conflict", "Rejected draft unavailable; no correction applied."
        ) from None
    if (
        not isinstance(record, dict)
        or hashlib.sha256(raw).hexdigest() != ref["sha256"]
        or ref["source"] != owner.state["source"]
        or record.get("source") != ref["source"]
        or hashlib.sha256(encode(record).encode()).hexdigest() != ref["id"]
    ):
        raise ModelRequestNotSent(
            "state_conflict", "Rejected draft changed; no correction or operation applied."
        )
    return record


def merge(owner, arguments):
    correction = Correction.model_validate(arguments)
    ref = owner.state.get("repair_draft")
    if not ref or correction.draft_id != ref["id"]:
        raise ValueError("repair_work requires the current draft_id")
    record = read(owner, ref)
    return apply(record["value"], correction.edits, record["paths"])


# ========================================================================
# 助手回复与拒绝结果的完整配对提交
# ========================================================================
def complete_empty_reply(owner, message):
    """已完整结算且完全没有调用的空回复，与身份损坏或在途请求严格区分。"""
    usage = message.usage_metadata or {}
    tokens = [usage.get(key) for key in ('input_tokens', 'output_tokens', 'total_tokens')]
    cached = (usage.get('input_token_details') or {}).get('cache_read')
    budget = getattr(owner, 'budget', None)
    return bool(
        message.content == ''
        and message.response_metadata.get('finish_reason') == 'stop'
        and all(type(value) is int and value >= 0 for value in tokens)
        and tokens[2] == tokens[0] + tokens[1]
        and type(cached) is int and 0 <= cached <= tokens[0]
        and budget is not None
        and not budget.unknown_usage_steps
        and not budget.inflight_reserved_tokens
        and not budget.unknown_reserved_tokens
        and not budget.unknown_service_cost_micros
        and not budget.inflight_service_cost_micros
        and not any(owner.state.get(key) for key in (
            'pending', 'acceptance_release', 'web_pending',
            'request_pending', 'running_step', 'verification_running',
            'acceptance_running', 'assistant_running', 'web_running',
            'web_request_pending', 'compaction_pending',
        ))
    )


def prepare_rejection(owner, error):
    """提交配对的拒绝结果前，先记录完整回复。"""
    from langchain_core.messages import messages_from_dict
    from agent.model.payload import original_calls

    response = owner.state.get('response_pending')
    if not response:
        raise ModelRequestNotSent(
            'invalid_message_history', 'Rejected response is unavailable; no retry.'
        )
    message = messages_from_dict([response])[0]
    if message.response_metadata.get('finish_reason') in {
        'length',
        'content_filter',
        'aborted',
        'insufficient_system_resource',
    }:
        raise ModelRequestNotSent(
            'incomplete_model_response',
            'Incomplete reply retained; no operations executed or automatic retry.',
        )
    calls = original_calls(message)
    if not calls:
        if not complete_empty_reply(owner, message):
            raise ModelRequestNotSent(
                'invalid_message_history', 'No trustworthy tool identity; reply retained.'
            )
        notice = (
            '[Host protocol feedback; no actions executed] '
            'The complete reply contained no executable tool call. '
            'Submit exactly one valid work tool call. No task requirements have changed.'
        )
    else:
        notice = (
            '[Acceptance review rejected; no workspace edits] ' + str(error)
            if getattr(owner, 'reviewing', False)
            else reject(owner, error)
        )
        if (getattr(owner, 'reviewing', False) and len(calls) == 1
                and calls[0].get('function', {}).get('name') == 'revise_acceptance'):
            from agent.acceptance.evidence import REVISION_GUIDANCE

            if REVISION_GUIDANCE not in notice:
                notice += '. ' + REVISION_GUIDANCE
    history = owner.state['history']
    owner.state['rejection_pending'] = {
        'response': response,
        'calls': calls,
        'notice': notice,
        'reason': str(error),
        'history_length': len(history),
        'prefix_digest': digest(history),
    }
    if not calls:
        owner.state['rejection_pending']['kind'] = 'empty_response'
    owner.persist('plan_rejection_prepared')
    finish_rejection(owner)


def finish_rejection(owner):
    from langchain_core.messages import HumanMessage, ToolMessage, message_to_dict, messages_from_dict
    from agent.model.payload import original_calls
    from agent.task import dialogue

    pending = owner.state['rejection_pending']
    history = owner.state['history']
    if (
        len(history) != pending['history_length']
        or digest(history) != pending['prefix_digest']
        or owner.state.get('response_pending') != pending['response']
        or original_calls(messages_from_dict([pending['response']])[0]) != pending['calls']
    ):
        raise ModelRequestNotSent(
            'state_conflict', 'Rejection continuation changed; no request sent.'
        )
    if owner.interactive:
        dialogue.publish(owner, pending['response'])
    if pending.get('kind') == 'empty_response':
        if (pending['calls'] or not complete_empty_reply(owner, messages_from_dict([pending['response']])[0])):
            raise ModelRequestNotSent(
                'state_conflict', 'Empty reply correction changed; no request sent.'
            )
        # 没有工具调用时只追加宿主协议反馈，不伪造调用身份或工具执行结果。
        results = [message_to_dict(HumanMessage(content=pending['notice']))]
        owner.state['empty_response_repairs'] = owner.state.get('empty_response_repairs', 0) + 1
    else:
        if not pending['calls']:
            raise ModelRequestNotSent('state_conflict', 'Rejected tool reply has no call identity.')
        results = [
            message_to_dict(ToolMessage(
                content=pending['notice'], name=call['function']['name'],
                tool_call_id=call['id'], status='error',
            ))
            for call in pending['calls']
        ]
    # 单个检查点保存整个交互周期，不留下未配对的助手消息。
    history.extend([pending['response'], *results])
    owner.state.update(
        response_pending=None,
        repair_count=owner.state['repair_count'] + 1,
        plan_error=clip(pending['reason'], 1800),
    )
    owner.state.pop('rejection_pending')
    owner.persist(
        'plan_rejected',
        {'reason': pending['reason'], 'call_ids': [call['id'] for call in pending['calls']]},
    )
