"""One provider-facing representation for transmission, estimates and audit logs."""

import json
from copy import deepcopy
from math import ceil

from langchain_core.messages import AIMessage, convert_to_openai_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.model.errors import ModelRequestNotSent


RAW_TOOL_CALLS = "_provider_tool_calls"


def retain_provider_calls(message, calls):
    """Keep provider bytes; classify JSON rejected by the strict decoder.

    The SDK parses tool arguments with strict=False, accepting literal control
    characters inside JSON strings. Such input must be a rejected tool call,
    not executable arguments followed by a fatal history mismatch. Never fix
    escapes or fabricate arguments, names or call identities here.
    """
    message.additional_kwargs[RAW_TOOL_CALLS] = deepcopy(calls)
    for original in calls:
        function = original.get('function', {})
        arguments = function.get('arguments')
        if not isinstance(arguments, str):
            continue  # The existing identity/original-argument guard handles it.
        try:
            json.loads(arguments)
        except json.JSONDecodeError as exc:
            parsed = next((call for call in message.tool_calls
                           if call.get('id') == original.get('id')
                           and call.get('name') == function.get('name')), None)
            if parsed is not None:
                message.tool_calls.remove(parsed)
                message.invalid_tool_calls.append({'type': 'invalid_tool_call',
                    'id': parsed['id'], 'name': parsed['name'], 'args': arguments,
                    'error': str(exc)})


def original_calls(message):
    """Validate call identities, retaining even rejected argument strings verbatim."""
    raw = message.additional_kwargs.get(RAW_TOOL_CALLS)
    parsed = [*message.tool_calls, *message.invalid_tool_calls]
    if raw is None and not parsed:
        return []
    try:
        by_id = {}
        for call in parsed:
            if (not isinstance(call.get('id'), str) or not call['id'].strip()
                    or not isinstance(call.get('name'), str) or not call['name'].strip()
                    or call['id'] in by_id):
                raise ValueError('missing or duplicate call identity')
            by_id[call['id']] = call
        if raw is None:
            # Legacy archives/fake models have no provider representation.
            raw = [{'id': c['id'], 'type': 'function', 'function': {
                'name': c['name'], 'arguments': c['args'] if c in message.invalid_tool_calls
                else json.dumps(c['args'], ensure_ascii=False, sort_keys=True)}} for c in parsed]
        if not isinstance(raw, list) or len(raw) != len(parsed):
            raise ValueError("call count changed")
        seen = set()
        for original in raw:
            function = original["function"]
            call = by_id[original['id']]
            if (original.get("type") != "function" or original['id'] in seen
                    or function["name"] != call["name"] or not isinstance(function["arguments"], str)):
                raise ValueError("call identity changed")
            seen.add(original['id'])
            if call in message.invalid_tool_calls:
                if function['arguments'] != call['args']:
                    raise ValueError('rejected arguments changed')
            else:
                decoded = json.loads(function["arguments"])
                if json.dumps(decoded, sort_keys=True) != json.dumps(call["args"], sort_keys=True):
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
            raw_calls = original_calls(original)
            if raw_calls:
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
    # The SDK counter reparses tool JSON and cannot handle a retained rejected
    # string. Count those exact transmitted strings separately, without parsing.
    texts = [{k: v for k, v in item.items() if k != 'tool_calls'} for item in payload['messages']]
    total = count_tokens_approximately(texts, tools=payload.get("tools", []))
    total += sum(ceil(len(json.dumps(item['tool_calls'], ensure_ascii=False)) / 4)
                 for item in payload['messages'] if item.get('tool_calls'))
    total += sum(ceil(len(item.get("reasoning_content", "")) / 4)
                 for item in payload["messages"])
    return total


def request_reservation(messages, tools, output_cap):
    """Conservative spending admission, not an exact provider token count."""
    return len(json.dumps(wire_input(messages, tools), ensure_ascii=False).encode("utf-8")) + output_cap + 4096
