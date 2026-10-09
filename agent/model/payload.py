"""One provider-facing representation for transmission, estimates and audit logs."""

import json
from math import ceil

from langchain_core.messages import AIMessage, convert_to_openai_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.model.errors import ModelRequestNotSent


RAW_TOOL_CALLS = "_provider_tool_calls"


def _original_calls(message):
    raw = message.additional_kwargs.get(RAW_TOOL_CALLS)
    if raw is None or not message.tool_calls or message.invalid_tool_calls:
        return None  # Old archives and simulated models still use parsed calls.
    try:
        if not isinstance(raw, list) or len(raw) != len(message.tool_calls):
            raise ValueError("call count changed")
        for original, parsed in zip(raw, message.tool_calls, strict=True):
            function = original["function"]
            if (original.get("type") != "function" or original["id"] != parsed["id"]
                    or function["name"] != parsed["name"] or not isinstance(function["arguments"], str)):
                raise ValueError("call identity changed")
            decoded = json.loads(function["arguments"])
            if json.dumps(decoded, sort_keys=True) != json.dumps(parsed["args"], sort_keys=True):
                raise ValueError("parsed arguments changed")
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelRequestNotSent("invalid_message_history",
                                  "原始工具参数与执行参数不一致，停止回放以保留可核查记录。") from exc
    return [{"id": call["id"], "type": "function", "function": {
        "name": call["function"]["name"], "arguments": call["function"]["arguments"]}} for call in raw]


def wire_input(messages, tools=()):
    """DeepSeek Chat Completions fields only; archive metadata is not input."""
    messages = list(messages)
    converted = convert_to_openai_messages(messages)
    for original, item in zip(messages, converted, strict=True):
        if item["role"] == "tool":
            if isinstance(item["content"], list):
                item["content"] = json.dumps(item["content"], ensure_ascii=False)
            item.pop("name", None)
        if isinstance(original, AIMessage):
            raw_calls = _original_calls(original)
            if raw_calls is not None:
                item["tool_calls"] = raw_calls
            if isinstance(item["content"], list):
                item["content"] = "".join(block.get("text", "") for block in item["content"]
                                          if isinstance(block, dict) and block.get("type") == "text")
            # Required by DeepSeek thinking + tools. Never drop it from a retained
            # assistant/tool cycle, and never count metadata that is not sent.
            reasoning = original.additional_kwargs.get("reasoning_content")
            if tools and isinstance(reasoning, str):
                item["reasoning_content"] = reasoning
    result = {"messages": converted}
    if tools:
        result["tools"] = [convert_to_openai_tool(item, strict=True) for item in tools]
    return result


def estimate_input_tokens(messages, tools=()):
    payload = wire_input(messages, tools)
    # LangChain's approximate counter handles text/tool schemas, but ignores
    # the vendor reasoning extension. Add only that transmitted field once.
    total = count_tokens_approximately(payload["messages"], tools=payload.get("tools", []))
    total += sum(ceil(len(item.get("reasoning_content", "")) / 4)
                 for item in payload["messages"])
    return total


def request_reservation(messages, tools, output_cap):
    """Conservative spending admission, not an exact provider token count."""
    return len(json.dumps(wire_input(messages, tools), ensure_ascii=False).encode("utf-8")) + output_cap + 4096
