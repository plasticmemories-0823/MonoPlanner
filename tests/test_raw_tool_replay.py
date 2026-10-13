import json
import unittest

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_from_dict, messages_to_dict
from agent.model.deepseek import DeepSeekClient
from agent.model.client import ConfiguredModel
from agent.model.errors import ModelRequestNotSent
from agent.model.payload import RAW_TOOL_CALLS, wire_input, estimate_input_tokens
from support import read_record


class RawToolReplayTests(unittest.TestCase):
    def test_sdk_replays_malformed_arguments_and_thought_without_parsing(self):
        arguments = '{\n "name": "中文", broken'
        call = {'id': 'bad-json', 'type': 'function', 'function': {
            'name': 'read_record', 'arguments': arguments}}
        requests = []
        def http(request):
            requests.append(json.loads(request.content))
            message = {'role': 'assistant', 'content': 'Correct the arguments.',
                'reasoning_content': 'Retain this thought.', 'tool_calls': [call]} if len(requests) == 1 else {
                    'role': 'assistant', 'content': 'DONE'}
            return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion',
                'created': 1, 'model': 'deepseek-flash', 'choices': [{'index': 0, 'message': message,
                'finish_reason': 'tool_calls' if len(requests) == 1 else 'stop'}]})
        with httpx.Client(transport=httpx.MockTransport(http)) as transport:
            model = ConfiguredModel(DeepSeekClient(model='deepseek-flash', api_key='offline-placeholder',
                http_client=transport, max_retries=0)).bind_tools([read_record])
            response = model.invoke([HumanMessage(content='Read')])
            self.assertEqual(response.invalid_tool_calls[0]['args'], arguments)
            restored = messages_from_dict(json.loads(json.dumps(messages_to_dict([response]))))[0]
            history = [HumanMessage(content='Read'), restored, ToolMessage(content='No actions executed',
                                                                 tool_call_id='bad-json')]
            self.assertGreater(estimate_input_tokens(history, [read_record]), 0)
            model.invoke(history)
        replay = requests[1]['messages'][1]
        self.assertEqual(replay['tool_calls'], [call])
        self.assertEqual(replay['reasoning_content'], 'Retain this thought.')

    def test_mixed_calls_keep_provider_order_and_reject_changed_invalid_strings(self):
        raw = [{'id': 'invalid', 'type': 'function', 'function': {'name': 'read_record', 'arguments': '{bad'}},
               {'id': 'valid', 'type': 'function', 'function': {'name': 'read_record', 'arguments': '{ "name": "ok" }'}}]
        message = AIMessage(content='', additional_kwargs={RAW_TOOL_CALLS: raw},
            tool_calls=[{'id': 'valid', 'name': 'read_record', 'args': {'name': 'ok'}}],
            invalid_tool_calls=[{'id': 'invalid', 'name': 'read_record', 'args': '{bad', 'error': 'bad'}])
        self.assertEqual(wire_input([message], [read_record])['messages'][0]['tool_calls'], raw)
        message.invalid_tool_calls[0]['args'] = '{changed'
        with self.assertRaises(ModelRequestNotSent):
            wire_input([message], [read_record])

    def test_missing_and_duplicate_identities_are_not_fabricated(self):
        for calls in ([{'name': 'read_record', 'args': {}, 'id': None}],
                      [{'name': 'read_record', 'args': {}, 'id': 'same'}] * 2):
            with self.subTest(calls=calls), self.assertRaises(ModelRequestNotSent):
                wire_input([AIMessage(content='', tool_calls=calls)], [read_record])

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
