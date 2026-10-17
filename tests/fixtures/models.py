"""无状态假模型和离线执行后端；不包含任何产品调度或会话实现。"""

import os
from pathlib import Path
from langchain_core.messages import AIMessage
from agent.execution.local import LocalEnvironment
from agent.execution.process import run_process


class FixtureDocker(LocalEnvironment):
    """离线图测试环境；真实 Docker 和隔离能力另行验证。"""

    def __init__(
        self, repo, image, task_root, *, runtime=None, read_sources=(), project_scope=None,
    ):
        super().__init__(
            repo, artifact_root=Path(task_root).parent.parent / 'execution' / Path(task_root).name
        )
        self.image, self.runtime, self._container_id = image, runtime, 'f' * 64
        if project_scope is not None:
            from agent.source.scope import project_scope as resolve_scope

            self.project_scope = resolve_scope(self.repo_root, project_scope)

    def start(self):
        return self

    def close(self):
        pass

    def run_process(
        self, command, *, stdout_path, stderr_path, timeout=None, shell=False, on_started=None
    ):
        import re

        env = {
            k: v
            for k, v in os.environ.items()
            if not re.search(r'API_KEY|ACCESS_TOKEN|SECRET|PASSWORD', k, re.I)
        }
        return run_process(
            command,
            cwd=self.repo_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=self.timeout if timeout is None else timeout,
            shell=shell,
            env=env,
            on_started=on_started,
        )


class BlockedModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        return AIMessage(
            content='',
            tool_calls=[
                {
                    'name': 'submit_plan',
                    'id': 'blocked',
                    'args': {'kind': 'blocked', 'note': 'private explanation'},
                }
            ],
            usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2},
        )


class ScriptedModel:
    def __init__(self, replies, hook=None):
        self.replies = iter(replies)
        self.calls = []
        self.hook = hook

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.hook:
            self.hook(len(self.calls))
        return next(self.replies)


def text_reply(text='公开回答'):
    return AIMessage(
        content=text,
        additional_kwargs={'reasoning_content': 'PRIVATE_REASONING'},
        usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5},
    )


def investigation(paths):
    return AIMessage(
        content='先读取必要的源码。',
        tool_calls=[
            {
                'name': 'submit_plan',
                'id': 'investigate',
                'args': {
                    'kind': 'inspect',
                    'note': 'PRIVATE_NOTE',
                    'steps': [
                        {'id': 'read' + str(i), 'op': 'read', 'path': p}
                        for i, p in enumerate(paths)
                    ],
                },
            }
        ],
        usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5},
    )


def reply(name, **args):
    return AIMessage(
        content='开始查询或运行。',
        tool_calls=[{'name': name, 'id': 'call-' + name, 'args': args}],
        additional_kwargs={'reasoning_content': 'PRIVATE_ASSISTANT_REASONING'},
        usage_metadata={
            'input_tokens': 10,
            'output_tokens': 2,
            'total_tokens': 12,
            'input_token_details': {'cache_read': 6},
        },
    )
