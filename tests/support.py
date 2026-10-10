"""Small offline helpers for the retained P0338 tests."""
import os
import shlex
import subprocess
import sys
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool


def python_command(code, executable=None):
    argv = [executable or sys.executable, "-B", "-X", "utf8", "-c", code]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


@tool
def read_record(name: str) -> str:
    """Read an offline record by name."""
    return name


def tool_group(number, content, calls=1):
    requests = [{"name": "read_record", "args": {"name": str(index)},
                 "id": f"call-{number}-{index}", "type": "tool_call"}
                for index in range(calls)]
    return [AIMessage(content="", id=f"ai-{number}", tool_calls=requests), *[
        ToolMessage(content=content, name="read_record", id=f"tool-{number}-{index}",
                    tool_call_id=request["id"])
        for index, request in enumerate(requests)
    ]]
