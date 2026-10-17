"""真实图节点调度规划、验收及执行，既有持久化仍只提交一次。"""

import ast
import inspect
import json
import threading
import unittest
from unittest.mock import patch

from langgraph.graph import StateGraph

from agent.application.work_bridge import WorkRunner
from agent.task.budget import StepBudget
from agent.task.dispatch import TaskDispatch
from agent.task.runner import TaskRunner
import test_task_runner as fixtures


class TaskDispatchTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp
    candidate = fixtures.TaskRunnerTests.candidate
    shell = fixtures.TaskRunnerTests.shell

    def runner(self, plans, **options):
        self.model = fixtures.Model(plans)
        return TaskRunner(
            self.model, self.env, state_dir=self.root / "state", **options,
        )

    def test_distinct_graph_nodes_drive_trial_release_rework_and_execution(self):
        visited = []
        add_node = StateGraph.add_node

        def record(builder, name, action, **options):
            def call(state):
                visited.append(name)
                return action(state)

            return add_node(builder, name, call, **options)

        runner = self.runner([self.candidate(2), self.candidate()])
        with patch.object(StateGraph, "add_node", record):
            result = runner.run([], "return 4", StepBudget(max_steps=4))
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(
            [row["verification"]["status"] for row in result["rounds"]], ["failed", "passed"],
        )
        self.assertEqual(visited.count("request_plan"), 3)
        self.assertEqual(visited.count("review_acceptance"), 1)
        self.assertEqual(visited.count("release_candidate"), 1)
        self.assertEqual(visited.count("execute_plan"), 2)
        self.assertEqual(visited.count("finalize"), 1)
        self.assertLess(visited.index("review_acceptance"), visited.index("execute_plan"))

    def test_default_graph_recursion_limit_does_not_reduce_existing_model_budget(self):
        plans = [
            {
                "kind": "inspect", "note": f"Check saved source at stage {index}",
                "steps": [{"id": "read", "op": "read", "path": "sample.py"}],
            }
            for index in range(12)
        ]
        plans.append({"kind": "blocked", "note": "Offline sequence complete"})
        runner = self.runner(plans)
        result = runner.run([], "Inspect only", StepBudget(max_steps=20))
        self.assertEqual(result["status"], "task_blocked", result)
        self.assertEqual(result["planner_calls"], 13)
        self.assertEqual(runner.state["plan_count"], 12)
        self.assertEqual((self.repo / "sample.py").read_text(), "def value():\n    return 0\n")

    def test_one_cycle_yield_preserves_original_confirmation_and_recovery_boundary(self):
        runner = self.runner([self.candidate()], yield_boundary=True)
        result = runner.run([], "return 4", StepBudget(max_steps=3))
        self.assertEqual(result["status"], "needs_plan", result)
        self.assertEqual(len(self.model.calls), 1)
        self.assertIn("return 0", (self.repo / "sample.py").read_text())
        self.assertFalse((self.root / "state/frozen-acceptance.json").exists())

        resumed = self.runner([])
        result = resumed.run([], "return 4", StepBudget(max_steps=3))
        self.assertEqual(result["status"], "verified_success", result)
        self.assertEqual(result["calls_this_invocation"], 1)
        self.assertEqual(len(result["rounds"]), 1)

    def test_graph_commits_on_the_existing_lease_thread_and_has_no_checkpointer(self):
        runner = self.runner([self.candidate()])
        thread = threading.get_ident()
        persist = runner.persist

        def same_thread(*args, **kwargs):
            self.assertEqual(threading.get_ident(), thread)
            return persist(*args, **kwargs)

        with patch.object(runner, "persist", same_thread):
            result = runner.run([], "return 4", StepBudget(max_steps=3))
        self.assertEqual(result["status"], "verified_success", result)
        self.assertIs(runner._task_dispatch.graph.checkpointer, False)
        state = json.loads((self.root / "state/state.json").read_text())
        self.assertEqual(state["format"], "task-history-reference")

    def test_saved_execution_value_error_stops_without_model_repair_or_retry(self):
        runner = self.runner([self.candidate()])
        with patch.object(
            runner, "execute_pending", side_effect=ValueError("runtime boundary failed"),
        ) as execute:
            result = runner.run([], "return 4", StepBudget(max_steps=3))
        self.assertEqual(result["status"], "runtime_error", result)
        self.assertEqual(result["stop_reason"], "ValueError: runtime boundary failed")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(len(self.model.calls), 2)
        self.assertIn("return 0", (self.repo / "sample.py").read_text())

    def test_product_has_one_graph_scheduler_and_no_previous_outer_loops(self):
        for owner in (TaskRunner, WorkRunner):
            source = inspect.getsource(owner)
            tree = ast.parse(source)
            methods = {
                node.name: node for node in tree.body[0].body if isinstance(node, ast.FunctionDef)
            }
            for name in ("_drive", "advance", "continue_after_return"):
                if name in methods:
                    self.assertFalse(
                        any(isinstance(node, ast.While) for node in ast.walk(methods[name]))
                    )
        self.assertFalse(hasattr(TaskRunner, "advance"))
        graph = TaskDispatch(self.runner([])).graph.get_graph()
        self.assertTrue(
            {"request_plan", "review_acceptance", "execute_plan", "reject_reply"}.issubset(graph.nodes)
        )
        self.assertNotIn("advance", graph.nodes)


if __name__ == "__main__":
    unittest.main()
