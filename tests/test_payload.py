"""The production SDK payload is checked without opening a network connection."""
import json
import unittest
from unittest.mock import patch
from langchain_core.messages import HumanMessage
from agent.model.client import create_model, configure_call
from agent.model.payload import wire_input, estimate_input_tokens
from support import read_record, tool_group

class ProviderPayloadTests(unittest.TestCase):

    def test_summary_keeps_tools_and_reasoning_but_disables_tool_choice(self):
        model = create_model({"model_name": "deepseek-flash", "api_key": "offline-placeholder"})
        self.addCleanup(model.client.http_client.close)
        model = configure_call(model.bind_tools([read_record]), 'compaction')
        messages = [HumanMessage(content='task'), *tool_group('1', 'result'), HumanMessage(content='summarize')]
        messages[1].additional_kwargs['reasoning_content'] = 'PRESERVED_REASONING'
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
            raise RuntimeError('offline HTTP boundary')
        with patch.object(model.client.client, 'with_raw_response') as boundary:
            boundary.create.side_effect = capture
            with self.assertRaises(RuntimeError):
                model.invoke(messages)
        self.assertEqual(captured['tool_choice'], 'none')
        self.assertEqual(captured['tools'], wire_input(messages, [read_record])['tools'])
        self.assertIn('PRESERVED_REASONING', json.dumps(captured['messages']))

    def test_real_sdk_http_boundary_preserves_retained_reasoning(self):
        model = create_model({"model_name": "deepseek-flash", "api_key": "offline-placeholder"})
        model = model.bind_tools([read_record])
        messages = [HumanMessage(content="task"), *tool_group("1", "result")]
        messages[1].additional_kwargs["reasoning_content"] = "RETAINED_REASONING"
        messages[1].response_metadata["private_archive_only"] = "not-on-wire"
        payload = wire_input(messages, [read_record])
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("HTTP is intentionally blocked offline")
        with patch.object(model.client.client, "with_raw_response") as boundary:
            boundary.create.side_effect = capture
            with self.assertRaises(RuntimeError):
                model.invoke(messages)
        self.assertEqual(captured["messages"], payload["messages"])
        self.assertEqual(captured["tools"], payload["tools"])
        self.assertIn("RETAINED_REASONING", json.dumps(captured))
        self.assertNotIn("private_archive_only", json.dumps(captured))

    def test_estimate_counts_reasoning_once_and_ignores_archive_metadata(self):
        messages = [HumanMessage(content="task"), *tool_group("1", "result")]
        base = estimate_input_tokens(messages, [read_record])
        messages[1].additional_kwargs["reasoning_content"] = "r" * 4000
        extra = estimate_input_tokens(messages, [read_record])
        self.assertEqual(extra - base, 1000)
        messages[1].additional_kwargs["tool_calls"] = "duplicate" * 5000
        messages[1].response_metadata["private"] = "metadata" * 5000
        self.assertEqual(estimate_input_tokens(messages, [read_record]), extra)
        self.assertNotIn("reasoning_content", json.dumps(wire_input(messages, [])))
