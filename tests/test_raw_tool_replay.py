import json
import unittest

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_from_dict, messages_to_dict
from agent.model.deepseek import DeepSeekClient
from agent.model.client import ConfiguredModel
from agent.model.errors import ModelRequestNotSent
from agent.model.payload import RAW_TOOL_CALLS, wire_input
from support import read_record


class RawToolReplayTests(unittest.TestCase):
    def test_real_sdk_response_archive_and_next_http_request_keep_argument_bytes(self):
        arguments = '{\n  "name" : "\\u4e2d\\u6587"\n}'
        calls = [{"id": "raw-call", "type": "function",
                  "function": {"name": "read_record", "arguments": arguments}}]
        requests = []
        def http(request):
            requests.append(json.loads(request.content))
            message = {"role": "assistant", "content": "", "reasoning_content": "retained reasoning",
                       "tool_calls": calls} if len(requests) == 1 else {"role": "assistant", "content": "DONE"}
            return httpx.Response(200, json={"id": "fixture-response", "object": "chat.completion",
                "created": 1, "model": "deepseek-flash", "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if len(requests) == 1 else "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
        with httpx.Client(transport=httpx.MockTransport(http)) as transport:
            model = ConfiguredModel(DeepSeekClient(model="deepseek-flash", api_key="offline-placeholder",
                                                   http_client=transport, max_retries=0)).bind_tools([read_record])
            history = [HumanMessage(content="Read a record")]
            response = model.invoke(history)
            self.assertEqual(response.tool_calls[0]["args"], {"name": "中文"})
            self.assertEqual(response.additional_kwargs[RAW_TOOL_CALLS], calls)
            saved = json.loads(json.dumps(messages_to_dict([response]), ensure_ascii=False))
            restored = messages_from_dict(saved)[0]
            model.invoke([*history, restored, ToolMessage(content="RESULT", tool_call_id="raw-call")])
        replay = requests[1]["messages"][1]
        self.assertEqual(replay["tool_calls"], calls)
        self.assertEqual(replay["tool_calls"][0]["function"]["arguments"].encode(), arguments.encode())
        self.assertEqual(replay["reasoning_content"], "retained reasoning")
        self.assertNotIn(RAW_TOOL_CALLS, replay)

    def test_changed_execution_arguments_cannot_silently_replay_original_values(self):
        message = AIMessage(content="", tool_calls=[{"name": "read_record", "args": {"name": "new"},
                                                     "id": "call", "type": "tool_call"}],
            additional_kwargs={RAW_TOOL_CALLS: [{"id": "call", "type": "function", "function": {
                "name": "read_record", "arguments": '{"name":"old"}'}}]})
        with self.assertRaises(ModelRequestNotSent) as raised:
            wire_input([message], [read_record])
        self.assertEqual(raised.exception.status, "invalid_message_history")

    def test_old_archives_without_raw_arguments_remain_usable(self):
        message = AIMessage(content="", tool_calls=[{"name": "read_record", "args": {"name": "old"},
                                                     "id": "call", "type": "tool_call"}])
        payload = wire_input([message], [read_record])
        self.assertEqual(json.loads(payload["messages"][0]["tool_calls"][0]["function"]["arguments"]),
                         {"name": "old"})


if __name__ == "__main__":
    unittest.main()
