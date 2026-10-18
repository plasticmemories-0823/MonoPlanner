"""验证恢复后的资料交接和单一执行方持续工作；不调用模型服务。"""

import ast
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import SystemMessage

from agent.execution.executor import OperationExecutor
from agent.task.contracts import Operation
from agent.task.instructions import TASK_RULES

from agent.model.tools import CHAT_TOOLS, CODING_TOOLS, WORK_TOOLS
from agent.model.payload import wire_input
from agent.task.control import Signal
from agent.settings import PROJECT_ROOT


class CodingHandoffTests(unittest.TestCase):

    def test_acceptance_schema_and_task_rules_outside_lookup_protocol_are_unchanged(self):
        # 实验版2只保留检索替换；地图及其余规则逐字对照正式版工作区。
        baseline = 'aebb8240e1113a5ee3f27a64abea4852a2642b24'
        def original(name):
            return subprocess.check_output(['git', 'show', baseline + ':' + name],
                                           cwd=PROJECT_ROOT).decode('utf-8-sig')

        for name in ('plan.schema.json', 'repair.schema.json', 'acceptance-review.schema.json'):
            self.assertEqual(json.loads((PROJECT_ROOT / 'resources' / name).read_bytes()),
                             json.loads(original('resources/' + name)))
        rules = (PROJECT_ROOT / 'resources/prompts/task.txt').read_text(encoding='utf-8')
        changed = ('read uses repository-relative',)
        protected = lambda value: [line for line in value.splitlines()
                                   if not line.startswith(changed)]
        self.assertEqual(protected(rules), protected(original('resources/prompts/task.txt')))
        expected = {
            'read uses repository-relative':
                '6f462cbc31bba65c93fdf91e74e8e3b2f44c23665bbc5f1eb818afb4ef31fe62',
        }
        for prefix, digest in expected.items():
            line = next(line for line in rules.splitlines() if line.startswith(prefix))
            self.assertEqual(hashlib.sha256(line.encode()).hexdigest(), digest)
        self.assertEqual(wire_input([], CODING_TOOLS)['tools'][:4],
                         wire_input([], WORK_TOOLS)['tools'])
        self.assertEqual([tool.name for tool in CODING_TOOLS],
                         [tool.name for tool in WORK_TOOLS] + ['web_search', 'web_fetch'])
        definitions = lambda value: {node.name: ast.dump(node) for node in ast.parse(value).body
                                     if isinstance(node, ast.FunctionDef)}
        before = definitions(original('agent/model/tools.py'))
        after = definitions((PROJECT_ROOT / 'agent/model/tools.py').read_text('utf-8-sig'))
        for tool in WORK_TOOLS:
            self.assertEqual(before[tool.name], after[tool.name])
        self.assertNotIn('submit_plan', [tool.name for tool in CHAT_TOOLS])

    def test_model_visible_lookup_policy_keeps_execution_and_explicit_scopes(self):
        request = wire_input([SystemMessage(content=TASK_RULES)], WORK_TOOLS)
        rules = request['messages'][0]['content']
        self.assertIn('must use search to locate relevant content', rules)
        self.assertIn('Do not use shell commands or scripts as a substitute', rules)
        self.assertIn('Use read for a known path and range', rules)
        self.assertIn('Shell remains available for compilation, tests, execution', rules)
        self.assertIn('runtime or binary diagnostics, candidate edits', rules)
        self.assertIn('processing output from the command being executed', rules)
        self.assertIn('never expand the authorized scope yourself', rules)
        self.assertIn('$container/system_include/usr/include/', rules)
        self.assertIn('$container/toolchain_src/build/gcc', rules)
        frontdesk = (PROJECT_ROOT / 'resources/prompts/frontdesk.txt').read_text('utf-8')
        self.assertIn('Do not run project scripts to search existing source', frontdesk)

    def test_lookup_policy_does_not_intercept_or_rewrite_shell_execution(self):
        # 工作模型应遵循检索分工，但程序不根据命令词汇阻断或改写已提交命令。
        commands = (
            'grep -rn needle src',
            'cc --version',
            'readelf -Ws ./program | grep target_symbol',
            'python -m pytest tests',
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            environment = SimpleNamespace(repo_root=root)
            executor = OperationExecutor(environment, SimpleNamespace(current='v1'), root)
            for command in commands:
                with self.subTest(command=command):
                    step = Operation(id='probe', op='shell', command=command)
                    observed = {'exit_code': 0, 'stdout': 'observed', 'stderr': ''}
                    with patch('agent.execution.probe.run_probe', return_value=observed) as probe:
                        result = executor.execute(step, {}, root / 'probe', 30, isolated=True)
                    self.assertEqual(result['status'], 'succeeded')
                    self.assertEqual(probe.call_args.args[1], command)


class SignalSequenceTests(unittest.TestCase):
    def test_consuming_old_message_cannot_clear_new_message_or_pause(self):
        signal = Signal()
        signal.interrupt('message', 'one')
        signal.interrupt('message', 'two')
        signal.consume_input('one')
        self.assertEqual(signal.kind, 'message')
        signal.interrupt('pause', 'pause')
        signal.consume_input('two')
        self.assertEqual(signal.kind, 'pause')
        signal.consume_input('pause')
        self.assertIsNone(signal.kind)
