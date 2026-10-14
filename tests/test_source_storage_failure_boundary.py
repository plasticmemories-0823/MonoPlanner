"""宿主证据复制失败必须停机，真实命令失败仍交给原来的返工流程。"""

import errno
from pathlib import Path

from langchain_core.messages import AIMessage

from agent.execution.executor import OperationExecutor
from agent.execution.local import LocalEnvironment
from agent.task.runner import TaskRunner
from agent.task.budget import StepBudget


class Model:
    def __init__(self, inspections=2):
        self.calls = 0
        self.inspections = inspections

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        inspect = self.calls <= self.inspections
        return AIMessage(content='', response_metadata={'finish_reason': 'stop'}, tool_calls=[{
            'id': 'call-' + str(self.calls), 'name': 'submit_plan', 'args': {
                'kind': 'inspect' if inspect else 'blocked', 'note': '观察-' + str(self.calls),
                'steps': [{'id': 'read-' + str(self.calls), 'op': 'shell', 'command': 'unused fixture'}]
                if inspect else [], 'test_code': '', 'test_driver': [], 'regression_command': '',
            },
        }])


def runner(tmp_path, model):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'README.md').write_text('small immutable source')
    environment = LocalEnvironment(repo, artifact_root=tmp_path / 'execution')
    return TaskRunner(model, environment, state_dir=tmp_path / 'task', interactive=False)


def test_partial_export_enospc_is_not_a_replanning_signal(tmp_path, monkeypatch):
    model = Model()
    loop = runner(tmp_path, model)
    original_open = Path.open
    copies = []

    class FailingOutput:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, content):
            self.stream.write(content[:1])
            raise OSError(errno.ENOSPC, 'host source destination full')

    def opened(path, mode='r', *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        if mode == 'xb' and 'probe-source' in path.parts and path.name == 'README.md':
            return FailingOutput(stream)
        return stream

    def execute(executor, step, prior, directory, timeout, **kwargs):
        copies.append(directory)
        return executor.evidence.view().export(directory / 'probe-source')

    monkeypatch.setattr(Path, 'open', opened)
    monkeypatch.setattr(OperationExecutor, 'execute', execute)
    result = loop.run([], '测试宿主存储故障边界', StepBudget(max_steps=4))
    assert result['status'] == 'runtime_error'
    assert model.calls == 1
    assert len(copies) == 1
    assert (copies[0] / 'probe-source/source/README.md').read_bytes() == b's'


def test_completed_nonzero_command_keeps_the_existing_feedback_flow(tmp_path, monkeypatch):
    model = Model(inspections=1)
    loop = runner(tmp_path, model)
    monkeypatch.setattr(OperationExecutor, 'execute', lambda *args, **kwargs: {
        'status': 'failed', 'data': {'exit_code': 1, 'executed': True, 'timed_out': False},
        'seconds': 0,
    })
    result = loop.run([], '普通命令失败仍需返回模型诊断', StepBudget(max_steps=4))
    assert result['status'] == 'task_blocked'
    assert model.calls == 2
    assert loop.state['last_result']['outcome'] == 'execution_failed'
