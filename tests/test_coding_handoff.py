"""验证恢复后的资料交接和单一执行方持续工作；不调用模型服务。"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import SystemMessage

from agent.execution.executor import OperationExecutor
from agent.task.contracts import Operation
from agent.task.instructions import TASK_RULES

from agent.model.tools import WORK_TOOLS
from agent.model.payload import wire_input
from agent.task.control import Signal
from agent.settings import PROJECT_ROOT


class CodingHandoffTests(unittest.TestCase):


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
        self.assertIn('host-registered read-only namespaces', rules)
        self.assertIn('Use the namespace and authorized path returned by the host.', rules)
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
