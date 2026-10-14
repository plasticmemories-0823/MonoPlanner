"""为已安装 LangChain SDK 遗漏的服务商字段提供专用适配。"""

from langchain_deepseek import ChatDeepSeek
from agent.model.payload import retain_provider_calls, wire_input


class DeepSeekClient(ChatDeepSeek):
    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        # 应用的生产路径使用非流式 invoke/ainvoke。
        # 解析前保留参数字符串，避免丢失空白与转义。
        for generation, choice in zip(result.generations, raw.get("choices", [])):
            calls = choice.get("message", {}).get("tool_calls")
            if calls:
                retain_provider_calls(generation.message, calls)
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        payload["messages"] = wire_input(messages, payload.get("tools", ()))["messages"]
        return payload
