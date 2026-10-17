"""在真实 SDK 边界验证严格工具请求，不发起网络调用。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from eval.swebench_runner import archive_source
from agent.model.client import create_model
from agent.model.payload import wire_input
from agent.task.contracts import WorkRequest
from agent.model.tools import WORK_SCHEMA, SCHEMA_PATH, submit_plan
from agent.storage.recovery import code_key


class PlanSchemaTests(unittest.TestCase):
    def test_chat_retrieval_uses_supported_strict_schema_fields(self):
        from agent.model.tools import retrieve_evidence

        parameters = convert_to_openai_tool(retrieve_evidence, strict=True)['function']['parameters']

        def check(value):
            if isinstance(value, dict):
                self.assertFalse(set(value) & {
                    'minLength', 'maxLength', 'minItems', 'maxItems', 'default',
                })
                if value.get('type') == 'object':
                    self.assertEqual(set(value['required']), set(value['properties']))
                    self.assertIs(value['additionalProperties'], False)
                for child in value.values():
                    check(child)
            elif isinstance(value, list):
                for child in value:
                    check(child)

        check(parameters)

    def test_resource_matches_local_contract_and_supported_strict_rules(self):
        # 在相同且受支持的投影处理之后，比较现有约定。
        expected = convert_to_openai_tool(WorkRequest, strict=True)["function"]["parameters"]

        def project(value):
            if isinstance(value, dict):
                for key in ("minLength", "maxLength", "minItems", "maxItems", "title", "default"):
                    value.pop(key, None)
                for child in value.values():
                    project(child)
                if value.get("type") == "object":
                    self.assertEqual(set(value["required"]), set(value["properties"]))
                    self.assertIs(value["additionalProperties"], False)
            elif isinstance(value, list):
                for child in value:
                    project(child)

        project(expected)
        actual = deepcopy(WORK_SCHEMA)
        project(actual)
        self.assertEqual(actual, WORK_SCHEMA)
        self.assertEqual(actual, expected)
        self.assertTrue(wire_input([], [submit_plan])["tools"][0]["function"]["strict"])

    def test_real_http_payload_uses_beta_high_and_preserves_code_and_tool_history(self):
        code = 'print("中文\\n")\npath = "C:\\\\quoted\\\\file"\n# brackets { [ and a quote \'\n'
        plan = WorkRequest(
            kind="candidate",
            note="说明里的 {多余括号、不闭合 JSON 不影响字符串}",
            test_code=code,
            steps=[{"id": "edit", "op": "shell", "command": code}],
        ).model_dump()
        arguments = json.dumps(plan, ensure_ascii=False, indent=2)
        requests = []

        def respond(request):
            requests.append((str(request.url), json.loads(request.content)))
            return httpx.Response(
                200,
                json={
                    "id": "offline",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "existing P0338 thinking",
                                "tool_calls": [
                                    {
                                        "id": "plan-call",
                                        "type": "function",
                                        "function": {"name": "submit_plan", "arguments": arguments},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                },
            )

        with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
            with patch(
                "agent.model.network.model_http_clients", return_value={"http_client": transport}
            ):
                model = create_model(
                    {"model_name": "deepseek-flash", "api_key": "offline-placeholder"}
                ).bind_tools([submit_plan])
            history = [HumanMessage(content="original request")]
            first = model.invoke(history)
            self.assertEqual(first.tool_calls[0]["args"]["test_code"], code)
            self.assertEqual(first.tool_calls[0]["args"]["steps"][0]["command"], code)
            model.invoke(
                [*history, first, ToolMessage(content="execution result", tool_call_id="plan-call")]
            )
        for url, body in requests:
            self.assertEqual(url, "https://api.deepseek.com/beta/chat/completions")
            self.assertEqual(body["reasoning_effort"], "high")
            self.assertNotIn("response_format", body)
            self.assertNotIn("tool_choice", body)
            self.assertEqual(body["tools"], wire_input([], [submit_plan])["tools"])
        self.assertEqual(requests[1][1]["messages"][:1], requests[0][1]["messages"])
        self.assertEqual(
            requests[1][1]["messages"][1]["tool_calls"][0]["function"]["arguments"], arguments
        )

    def test_local_limits_and_step_relations_remain_enforced(self):
        invalid = [
            {"kind": "inspect", "note": "x", "steps": []},
            {"kind": "blocked", "note": "x" * 1601},
            {"kind": "candidate", "note": "x", "test_code": "x" * 32769},
            {
                "kind": "candidate",
                "note": "x",
                "steps": [{"id": f"s{i}", "op": "read"} for i in range(13)],
            },
            {
                "kind": "candidate",
                "note": "x",
                "steps": [{"id": "r", "op": "read", "from_step": "missing"}],
            },
            {"kind": "candidate", "note": "x", "steps": [{"id": "s", "op": "search", "query": ""}]},
        ]
        for value in invalid:
            with self.subTest(value=str(value)[:60]), self.assertRaises(ValidationError):
                WorkRequest.model_validate(value)

    def test_schema_bytes_are_in_restore_fingerprint_and_evaluated_archive(self):
        original = code_key()
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            changed = folder / "schema.json"
            changed.write_bytes(SCHEMA_PATH.read_bytes() + b"\n")
            with patch("agent.storage.recovery.SCHEMA_PATH", changed):
                self.assertNotEqual(code_key(), original)
            archive_source(folder)
            hashes = json.loads((folder / "source-sha256.json").read_text())
            self.assertEqual(
                hashes["resources/plan.schema.json"],
                hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest(),
            )
            self.assertIn("launcher/Program.cs", hashes)
            self.assertIn("scripts/start-app.ps1", hashes)


if __name__ == "__main__":
    unittest.main()
