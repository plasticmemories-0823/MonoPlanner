"""发送内容、用量估算和审计日志共用同一份服务商请求表示。"""

import json
from copy import deepcopy
from math import ceil

from langchain_core.messages import AIMessage, convert_to_openai_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent.model.errors import ModelRequestNotSent


RAW_TOOL_CALLS = "_provider_tool_calls"


# ========================================================================
# 保留并校验服务商原始工具参数
# ========================================================================
def retain_provider_calls(message, calls):
    """保留服务商原始字节，并分类严格解码器拒绝的 JSON。

    SDK 使用 strict=False 解析工具参数，允许 JSON 字符串包含原始控制字符。
    这类输入应被拒绝为工具调用，而不能先执行参数再因历史不一致而失败。
    此处不得修复转义，也不得伪造参数、名称或调用标识。"""
    message.additional_kwargs[RAW_TOOL_CALLS] = deepcopy(calls)
    for original in calls:
        function = original.get('function', {})
        arguments = function.get('arguments')
        if not isinstance(arguments, str):
            continue  # 交由现有调用标识与原始参数检查处理。
        try:
            json.loads(arguments)
        except json.JSONDecodeError as exc:
            parsed = next(
                (
                    call
                    for call in message.tool_calls
                    if call.get('id') == original.get('id')
                    and call.get('name') == function.get('name')
                ),
                None,
            )
            if parsed is not None:
                message.tool_calls.remove(parsed)
                message.invalid_tool_calls.append(
                    {
                        'type': 'invalid_tool_call',
                        'id': parsed['id'],
                        'name': parsed['name'],
                        'args': arguments,
                        'error': str(exc),
                    }
                )


def original_calls(message):
    """校验调用标识，即使参数字符串被拒绝也保留其原文。"""
    raw = message.additional_kwargs.get(RAW_TOOL_CALLS)
    parsed = [*message.tool_calls, *message.invalid_tool_calls]
    if raw is None and not parsed:
        return []
    try:
        by_id = {}
        for call in parsed:
            if (
                not isinstance(call.get('id'), str)
                or not call['id'].strip()
                or not isinstance(call.get('name'), str)
                or not call['name'].strip()
                or call['id'] in by_id
            ):
                raise ValueError('missing or duplicate call identity')
            by_id[call['id']] = call
        if raw is None:
            # 旧归档和假模型没有服务商原始表示。
            raw = [
                {
                    'id': c['id'],
                    'type': 'function',
                    'function': {
                        'name': c['name'],
                        'arguments': (
                            c['args']
                            if c in message.invalid_tool_calls
                            else json.dumps(c['args'], ensure_ascii=False, sort_keys=True)
                        ),
                    },
                }
                for c in parsed
            ]
        if not isinstance(raw, list) or len(raw) != len(parsed):
            raise ValueError("call count changed")
        seen = set()
        for original in raw:
            function = original["function"]
            call = by_id[original['id']]
            if (
                original.get("type") != "function"
                or original['id'] in seen
                or function["name"] != call["name"]
                or not isinstance(function["arguments"], str)
            ):
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
        raise ModelRequestNotSent(
            "invalid_message_history", "原始工具参数与执行参数不一致，停止回放以保留可核查记录。"
        ) from exc
    return [
        {
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["function"]["name"],
                "arguments": call["function"]["arguments"],
            },
        }
        for call in raw
    ]


# ========================================================================
# 实际发送字段与 Token 预算估算
# ========================================================================
def wire_input(messages, tools=()):
    """仅输出 DeepSeek Chat Completions 字段；归档元数据不作为输入。"""
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
                item["content"] = "".join(
                    block.get("text", "")
                    for block in item["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            # DeepSeek 的思考与工具调用需要此字段，保留的助手与工具调用周期中
            # 不能删除它，也不能计入实际未发送的元数据。
            reasoning = original.additional_kwargs.get("reasoning_content")
            if tools and isinstance(reasoning, str):
                item["reasoning_content"] = reasoning
    result = {"messages": converted}
    if tools:
        result["tools"] = [convert_to_openai_tool(item, strict=True) for item in tools]
    return result


def estimate_input_tokens(messages, tools=()):
    payload = wire_input(messages, tools)
    # LangChain 的近似计数涵盖文本和工具结构，但忽略服务商的
    # 思考扩展字段；仅将这个实际传输的字段补计一次。
    # SDK 计数器会重新解析工具 JSON，无法处理保留下来的被拒绝字符串。
    # 这些实际传输的字符串单独计数，不再解析。
    texts = [{k: v for k, v in item.items() if k != 'tool_calls'} for item in payload['messages']]
    total = count_tokens_approximately(texts, tools=payload.get("tools", []))
    total += sum(
        ceil(len(json.dumps(item['tool_calls'], ensure_ascii=False)) / 4)
        for item in payload['messages']
        if item.get('tool_calls')
    )
    total += sum(ceil(len(item.get("reasoning_content", "")) / 4) for item in payload["messages"])
    return total


def request_reservation(messages, tools, output_cap):
    """保守检查预算是否允许请求，不代表服务商的精确 Token 计数。"""
    return (
        len(json.dumps(wire_input(messages, tools), ensure_ascii=False).encode("utf-8"))
        + output_cap
        + 4096
    )
