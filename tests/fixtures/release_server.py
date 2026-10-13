"""Offline release browser fixture; production entrypoints never import this module."""
import shlex
import json
import time
from langchain_core.messages import AIMessage, ToolMessage
from agent.task.contracts import WorkRequest
from tests.fixtures.web_server import OfflineModel, graph, application


class ReleaseModel(OfflineModel):
    def bind_tools(self, tools):
        self.coding = not any(tool.name == 'start_work' for tool in tools)
        return self

    def invoke_with_progress(self, messages, on_reasoning):
        slow = '慢流式' in messages[-1].text
        for part in ('正在确认当前目标。', '继续核对已有证据。', '准备给出公开结果。'):
            on_reasoning(part)
            time.sleep(10 if slow else .2)
        answer = self.invoke(messages)
        answer.additional_kwargs['reasoning_content'] = '正在确认当前目标。继续核对已有证据。准备给出公开结果。'
        return answer

    def invoke(self, messages):
        if not getattr(self, 'coding', False):
            if '发布编码验证' in messages[-1].text:
                return self.call('start_work', {}, '开始验证独立项目的编码流程。')
            return super().invoke(messages)
        previous = messages[-1]
        if isinstance(previous, ToolMessage):
            value = json.loads(previous.content)
            if value.get('trial_id'):
                return self.call('freeze_acceptance', {'trial_id': value['trial_id'],
                    'reason': '原源码真实断言失败且用例已加载，按固定原件验收。'}, '验收已可运行，继续冻结并核验修改。')
        submitted = [call for message in messages if isinstance(message, AIMessage)
                     for call in message.tool_calls if call['name'] == 'submit_plan']
        if not submitted:
            args = {'kind': 'inspect', 'note': '读取实现并验证只读调查临时编译。', 'steps': [
                {'id': 'readSource', 'op': 'read', 'path': 'sample.py'},
                {'id': 'probe', 'op': 'shell', 'command': "printf 'int main(){return 0;}' > /tmp/check.c; if command -v cc >/dev/null; then cc /tmp/check.c -o /tmp/check && /tmp/check; else python -c 'import sample; assert sample.value() == 0'; fi"}]}
        else:
            value = 2 if len(submitted) == 1 else 4
            command = "from pathlib import Path; Path('sample.py').write_text('def value():\\n    return %d\\n')" % value
            args = {'kind': 'candidate', 'note': '候选修复，依据真实失败继续返工。',
                    'steps': [{'id': 'editSource', 'op': 'shell', 'command': 'python -c ' + shlex.quote(command)}],
                    'test_code': "import unittest\nfrom sample import value\nclass Acceptance(unittest.TestCase):\n    def test_answer(self):\n        self.assertEqual(value(), 4)\n",
                    'regression_command': "python -c 'import sample; assert isinstance(sample.value(), int)'"}
        return self.call('submit_plan', WorkRequest.model_validate(args).model_dump(), '继续按实际执行结果核验。')

    @staticmethod
    def call(name, args, text):
        return AIMessage(content=text, tool_calls=[{'name': name, 'args': args, 'id': 'offline-' + name}],
                         usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5,
                                         'input_token_details': {'cache_read': 2}})


application.model_factory = lambda settings=None: ReleaseModel()
