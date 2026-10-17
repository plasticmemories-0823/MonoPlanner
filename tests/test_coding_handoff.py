"""验证恢复后的资料交接和单一执行方持续工作；不调用模型服务。"""

import ast
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
        for name in ('plan.schema.json', 'repair.schema.json', 'acceptance-review.schema.json'):
            original = subprocess.check_output(
                ['git', 'show', '98b3851:resources/' + name], cwd=PROJECT_ROOT
            )
            expected = json.loads(original)
            if name == 'plan.schema.json':
                expected['properties']['steps']['items']['properties']['limit']['maximum'] = 2000
                expected['properties']['steps']['items']['properties']['limit'][
                    'description'
                ] = 'read: default 2000, range 1..2000. Other operations: default 80, maximum 1000.'
                for field, description in {
                    'op': 'read: known original ranges; search: locate evidence through RAG; '
                          'shell: editing, execution and runtime diagnostics, never evidence search.',
                    'path': 'Authorized repository scope, $record path or host-registered '
                            '$container namespace. Never widen the registered scope.',
                    'command': 'Shell code for editing, execution or runtime diagnostics; '
                               'do not search existing source, documents or saved logs here.',
                }.items():
                    expected['properties']['steps']['items']['properties'][field][
                        'description'
                    ] = description
            actual = json.loads((PROJECT_ROOT / 'resources' / name).read_bytes())
            if name == 'plan.schema.json':
                step = actual['properties']['steps']['items']
                for field in ('mode', 'cursor', 'intent', 'required_entities', 'required_kinds'):
                    step['properties'].pop(field)
                    step['required'].remove(field)
            self.assertEqual(actual, expected)
        original = subprocess.check_output(
            ['git', 'show', '98b3851:resources/prompts/task.txt'], cwd=PROJECT_ROOT
        ).decode()
        rules = (PROJECT_ROOT / 'resources/prompts/task.txt').read_text(encoding='utf-8')
        expected = original.replace('submit_work', 'submit_plan')
        expected = expected.replace(
            "limit=80, expected_exit=[0].",
            "limit=80, expected_exit=[0]. A read defaults to limit=2000; choose a smaller window when appropriate. Other operation defaults and limits are unchanged.",
        )
        expected = expected.replace(
            'a total line limit (1..1000), automatically paged at 200 lines; search',
            'a total line limit (1..2000), automatically paged at 200 lines. All pages and from_step locations share 50 KiB of UTF-8 text per read step. Truncation returns only an original prefix and continuation.next_offset; a partial long line is not a complete read. search',
        )
        expected = expected.replace(
            'All full plans, responses and logs remain on disk. To read a recorded result use read path=$record/<relative record path>.',
            'All full plans, responses and logs remain on disk. Large tool text may be offloaded: a monoplanner-tool-payload-v1 reference identifies the exact record path, SHA-256, full size and line count; preview is partial, not the full result. Identical text already visible in this context may return only its hash and record reference. Read the required original ranges with read path=$record/<relative record path>, offset and limit; use continuation.next_offset when present. For an oversized payload line, follow its next_query with read query=payload_bytes:<raw byte offset>, offset=0 and limit=2000; this mode returns a bounded byte window, not complete source lines. Source-file reads retain their normal paging limits.',
        )
        lookup_prefixes = ('submit_plan fields are required.', 'read uses repository-relative',
                           'A read may use from_step=', 'Evidence lookup:', 'Container evidence:')
        unchanged = lambda text: [line for line in text.splitlines()
                                  if not line.startswith(lookup_prefixes)]
        self.assertEqual(unchanged(rules), unchanged(expected))
        self.assertEqual(
            wire_input([], CODING_TOOLS)['tools'][:4], wire_input([], WORK_TOOLS)['tools']
        )
        self.assertEqual(
            [t.name for t in CODING_TOOLS],
            [t.name for t in WORK_TOOLS] + ['web_search', 'web_fetch'],
        )
        # 查询协议调整不改变原四工具的名称、描述或函数体。
        baseline = ast.parse(subprocess.check_output(
            ['git', 'show', 'HEAD:agent/model/tools.py'], cwd=PROJECT_ROOT
        ).decode('utf-8-sig'))
        current = ast.parse((PROJECT_ROOT / 'agent/model/tools.py').read_text('utf-8-sig'))
        definitions = lambda tree: {node.name: ast.dump(node) for node in tree.body
                                    if isinstance(node, ast.FunctionDef)}
        for tool in WORK_TOOLS:
            self.assertEqual(definitions(baseline)[tool.name], definitions(current)[tool.name])
        self.assertNotIn('submit_plan', [t.name for t in CHAT_TOOLS])

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
