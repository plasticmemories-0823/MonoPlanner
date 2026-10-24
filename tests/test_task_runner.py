"""任务执行测试进行真实修改和最终测试，不发起任何网络请求。"""

import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage
from agent.execution.local import LocalEnvironment
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from agent.task.contracts import WorkRequest
from support import python_command, boundary_aware_reply

CODE = "import unittest\nfrom sample import value\nclass ValueTests(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(value(), 4)\n"
EXECUTION_CHECK = {
    "id": "public_value", "requirement_quote": "", "artifact_type": "repository_entry",
    "imports": "", "setup": "", "expression": "", "consumer": "command",
    "entry_file": "sample.py", "interpreter": "python", "argv": [],
    "python_module": "",
    "variables": [], "result_variable": "", "cases": [],
}


def reply(plan):
    return AIMessage(
        content="",
        tool_calls=[{"name": "submit_plan", "id": "plan-call", "args": plan}],
        usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    )


def confirmation(messages):
    """现有测试辅助函数明确确认新增的真实基线试跑。"""
    try:
        result = json.loads(messages[-1].content)
    except (ValueError, TypeError):
        return None
    if result.get("outcome") != "acceptance_preflight":
        return None
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "freeze_acceptance",
                "id": "confirm-call",
                "args": {
                    "trial_id": result["trial_id"],
                    "reason": "Keep the specified behavior after inspecting the baseline failure",
                },
            }
        ],
        usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    )


class Model:
    def __init__(self, plans):
        self.plans, self.calls = iter(plans), []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return boundary_aware_reply(
            self, messages, lambda: confirmation(messages) or reply(next(self.plans)),
        )


class TaskRunnerTests(unittest.TestCase):
    def setUp(self):
        from support import confirmed_fixture

        confirmed_fixture(self)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        (self.repo / "sample.py").write_text("def value():\n    return 0\n")
        self.env = LocalEnvironment(self.repo, artifact_root=self.root / "runtime")

    def shell(self, code, id="edit"):
        return {"id": id, "op": "shell", "command": python_command(code)}

    def candidate(self, value=4, **extra):
        return {
            "kind": "candidate",
            "note": "Implement the specified value",
            "steps": [
                self.shell(
                    "from pathlib import Path; Path('sample.py').write_text('def value():\\n    return %d\\n')"
                    % value
                )
            ],
            "test_code": CODE,
            "execution_checks": [deepcopy(EXECUTION_CHECK)],
            **extra,
        }

    def loop(self, plans, **kw):
        self.model = Model(plans)
        self.agent = TaskRunner(self.model, self.env, self.root / "records", **kw)
        return self.agent

    def test_three_steps_one_plan_one_confirmation_one_final_acceptance(self):
        plan = self.candidate()
        plan["steps"] += [
            self.shell("from pathlib import Path; Path('a.txt').write_text('a')", "second"),
            self.shell("from pathlib import Path; Path('b.txt').write_text('b')", "third"),
        ]
        result = self.loop([plan]).run([], "value must return 4", StepBudget(max_steps=3))
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(self.model.calls), 3)
        self.assertEqual(len(result["rounds"]), 1)
        self.assertEqual(result["executor_model_calls"], 0)
        self.assertEqual(result["verifier_model_calls"], 0)
        events = [
            json.loads(l)
            for l in (Path(result['artifact_dir']) / 'events.jsonl').read_text().splitlines()
        ]
        self.assertEqual(sum(e['kind'] == 'final_acceptance_started' for e in events), 1)

    def test_inspection_search_and_read_need_no_intermediate_model(self):
        inspect = {
            "kind": "inspect",
            "note": "Locate and read function",
            "steps": [
                {"id": "find", "op": "search", "query": "def value"},
                {"id": "read", "op": "read", "from_step": "find"},
            ],
        }
        result = self.loop([inspect, self.candidate()]).run(
            [], "value returns 4", StepBudget(max_steps=4)
        )
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(len(self.model.calls), 4)
        self.assertIn("return 0", str(self.model.calls[1]))

    def test_execution_failure_stops_remaining_steps(self):
        plan = self.candidate()
        plan['steps'] = [
            self.shell('import sys; sys.exit(3)', 'bad'),
            self.shell(
                "from pathlib import Path; Path('must-not-exist').write_text('bad')", 'later'
            ),
        ]
        result = self.loop([plan, {'kind': 'blocked', 'note': 'Observed execution failure'}]).run(
            [], 'return 4', StepBudget(max_steps=4)
        )
        self.assertFalse((self.repo / 'must-not-exist').exists())
        self.assertEqual(result['rounds'], [])
        self.assertEqual(result['status'], 'budget_exhausted')

    def test_failed_final_acceptance_replans_using_same_contract(self):
        result = self.loop([self.candidate(2), self.candidate()]).run(
            [], 'return 4', StepBudget(max_steps=4)
        )
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(
            [r['verification']['status'] for r in result['rounds']], ['failed', 'passed']
        )
        self.assertEqual(len(self.model.calls), 4)
        self.assertIn('AssertionError', str(self.model.calls[3]))

    def test_frozen_test_cannot_be_weakened(self):
        bad = self.candidate(test_code=CODE.replace('value(), 4', 'value(), 2'))
        result = self.loop([self.candidate(2), bad, bad, bad]).run(
            [], 'return 4', StepBudget(max_steps=5)
        )
        self.assertEqual(result['status'], 'budget_exhausted', result)
        frozen = json.loads((Path(result['artifact_dir']) / 'frozen-acceptance.json').read_text())
        self.assertEqual(frozen['spec']['test_code'], CODE)

    def test_invalid_structure_rejected_before_execution(self):
        self.assertEqual(
            WorkRequest.model_validate(
                {'kind': 'inspect', 'note': 'diagnose', 'steps': [self.shell('echo isolated')]}
            ).kind,
            'inspect',
        )
        with self.assertRaises(ValueError):
            WorkRequest.model_validate(
                {
                    'kind': 'candidate',
                    'note': 'not executable',
                    'steps': [{'id': 'w', 'op': 'shell'}],
                }
            )

    def test_executor_and_verifier_have_no_llm_import_or_invoke(self):
        import agent.execution.executor as module

        tree = ast.parse(Path(module.__file__).read_text())
        self.assertFalse(
            any(
                isinstance(n, ast.Attribute) and n.attr in {'invoke', 'bind_tools'}
                for n in ast.walk(tree)
            )
        )

    def test_one_invalid_plan_can_be_corrected_in_total_budget(self):
        result = self.loop(
            [{'kind': 'inspect', 'note': 'empty', 'steps': []}, self.candidate()]
        ).run([], 'return 4', StepBudget(max_steps=4))
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(self.model.calls), 4)


if __name__ == '__main__':
    unittest.main()
