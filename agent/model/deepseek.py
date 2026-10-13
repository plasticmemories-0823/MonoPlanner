"""Narrow adapter for a provider field omitted by the installed LangChain SDK."""

from langchain_deepseek import ChatDeepSeek
from agent.model.payload import retain_provider_calls, wire_input


class DeepSeekClient(ChatDeepSeek):
    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        # Non-streaming invoke/ainvoke is this application's production path.
        # Preserve argument strings before parsing loses whitespace and escapes.
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
