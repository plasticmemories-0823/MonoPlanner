"""Token 容量计入输出预留，同时保留已有历史记录。"""

import copy
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from agent.model.context import admit, CONTEXT_WINDOW_TOKENS
from agent.model.errors import ModelRequestNotSent
from agent.model.payload import wire_input
from agent.model.tools import submit_plan


class ContextCapacityTests(unittest.TestCase):
    def test_large_byte_payload_is_admitted_and_prefix_still_protected(self):
        state = {}
        messages = [HumanMessage(content='x' * 600000)]
        tools = [submit_plan]
        options = {'max_tokens': 65536}
        before = wire_input(messages, tools)
        self.assertGreater(len(json.dumps(before).encode()), 512000)
        admit(state, messages, tools, options)
        self.assertEqual(wire_input(messages, tools), before)
        admit(state, [*messages, HumanMessage(content='continue')], tools, options)
        with self.assertRaisesRegex(ModelRequestNotSent, 'Previous input changed'):
            admit(state, [HumanMessage(content='changed')], tools, options)

    def test_output_reservation_limits_total_and_rejection_keeps_admission(self):
        self.assertEqual(CONTEXT_WINDOW_TOKENS, 1000000)
        state, messages, options = {}, [HumanMessage(content='task')], {'max_tokens': 65536}
        with patch('agent.model.context.estimate_input_tokens', return_value=934464):
            admit(state, messages, [submit_plan], options)
        admitted = copy.deepcopy(state)
        with patch('agent.model.context.estimate_input_tokens', return_value=934465):
            with self.assertRaisesRegex(ModelRequestNotSent, 'reserved output 65536'):
                admit(state, messages, [submit_plan], options)
        self.assertEqual(state, admitted)

    def test_retained_reasoning_counts_towards_capacity(self):
        messages = [
            HumanMessage(content='task'),
            AIMessage(content='done', additional_kwargs={'reasoning_content': 'x' * 4000000}),
        ]
        state = {}
        with self.assertRaisesRegex(ModelRequestNotSent, '1000000 tokens'):
            admit(state, messages, [submit_plan], {'max_tokens': 65536})
        self.assertEqual(state, {})


if __name__ == '__main__':
    unittest.main()
