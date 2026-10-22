"""为保留的 P0338 测试提供少量离线辅助函数。"""

import os
import shlex
import subprocess
import sys
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool


def python_command(code, executable=None):
    argv = [executable or sys.executable, "-B", "-X", "utf8", "-c", code]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def interrupt_after_persist(event_name, *, tool_name=None):
    """模拟在指定检查点完整保存后断电。"""
    from unittest.mock import patch
    from agent.task.runner import TaskRunner

    class Interrupted(BaseException):
        pass

    original = TaskRunner.persist

    def persist(engine, event, payload=None):
        original(engine, event, payload)
        calls = (engine.state.get("response_pending") or {}).get("data", {}).get("tool_calls", [])
        if event == event_name and (
            tool_name is None or any(call["name"] == tool_name for call in calls)
        ):
            raise Interrupted()

    return patch.object(TaskRunner, "persist", persist), Interrupted


@tool
def read_record(name: str) -> str:
    """Read an offline record by name."""
    return name


def tool_group(number, content, calls=1):
    requests = [
        {
            "name": "read_record",
            "args": {"name": str(index)},
            "id": f"call-{number}-{index}",
            "type": "tool_call",
        }
        for index in range(calls)
    ]
    return [
        AIMessage(content="", id=f"ai-{number}", tool_calls=requests),
        *[
            ToolMessage(
                content=content,
                name="read_record",
                id=f"tool-{number}-{index}",
                tool_call_id=request["id"],
            )
            for index, request in enumerate(requests)
        ],
    ]


def boundary_aware_reply(model, messages, produce):
    """离线模型先走真实词典读取轮次，再提交保留的首份候选。"""
    import json

    pending = getattr(model, "_boundary_candidate", None)
    if pending is not None:
        model._boundary_candidate = None
        return pending
    result = produce()
    calls = getattr(result, "tool_calls", [])
    invalid = getattr(result, "invalid_tool_calls", [])
    candidate = (
        len(calls) == 1 and (
            calls[0]["name"] == "repair_work" or (
                calls[0]["name"] == "submit_plan"
                and calls[0]["args"].get("kind") == "candidate"
            )
        )
    )
    if len(invalid) == 1 and invalid[0]["name"] == "submit_plan":
        try:
            decoded, _ = json.JSONDecoder().raw_decode(invalid[0]["args"].lstrip())
            candidate = isinstance(decoded, dict) and decoded.get("kind") == "candidate"
        except (ValueError, TypeError):
            pass
    if not candidate:
        return result
    for message in messages:
        try:
            content = json.loads(message.content)
        except (ValueError, TypeError):
            continue
        if not isinstance(content, dict):
            continue
        if (content.get("verify_command")
                or (content.get("outcome") == "boundary_hints"
                    and "general" in content.get("entries", {}))):
            model._boundary_known = True
    if getattr(model, "_boundary_known", False):
        return result
    model._boundary_known = True
    model._boundary_candidate = result
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "get_boundary_hints", "id": "fixture-boundary-call",
            "args": {"keys": ["general"]},
        }],
        usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    )


def fixture_delivery(request, entry="sample.py"):
    """离线夹具明确确认自己的固定交付入口，不从模型候选推断需求。"""
    from agent.task.requirements import confirmed

    return confirmed(request, [{
        "id": "fixture_delivery", "artifact_type": "repository_entry",
        "target_runtime": "python", "entry": entry,
        "behavior": request, "input_scope": "The complete fixture request",
        "setup": "", "expression": "", "variables": [], "required_values": [], "argv": [],
        "quotes": [{"source_id": "task_request", "text": request}],
    }], confirmation={"id": "fixture-user-confirmation", "text": "The test explicitly confirms this entry"})


def confirmed_fixture(test, entry="sample.py"):
    """只用于已有执行回归的测试输入；缺少确认的产品路径另有独立测试。"""
    from unittest.mock import patch
    from agent.task.runner import TaskRunner

    original = TaskRunner.run

    def run(owner, history, text, budget, verify_command=None, acceptance=None):
        if owner.confirmed_requirements is None:
            owner.confirmed_requirements = fixture_delivery(text, entry)
        return original(owner, history, text, budget, verify_command, acceptance)

    mocked = patch.object(TaskRunner, "run", run)
    mocked.start()
    test.addCleanup(mocked.stop)
