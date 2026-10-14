"""宽松的 SDK 解码不得将模型服务返回的无效 JSON 转为可执行工作。"""

import json
from pathlib import Path
import tempfile

import httpx
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from agent.execution.local import LocalEnvironment
from agent.model.client import ConfiguredModel
from agent.model.deepseek import DeepSeekClient
from agent.model.payload import RAW_TOOL_CALLS, wire_input
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from support import read_record


def sdk(reply):
    return httpx.Response(
        200,
        json={
            'id': 'offline',
            'object': 'chat.completion',
            'created': 1,
            'model': 'deepseek-flash',
            'choices': [
                {
                    'index': 0,
                    'message': reply,
                    'finish_reason': 'tool_calls' if reply.get('tool_calls') else 'stop',
                }
            ],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
        },
    )


def test_nonstream_sdk_rejects_literal_control_character_and_replays_original_bytes():
    arguments = '{"name":"first\nsecond"}'  # 真实换行符，并非 JSON 转义。
    call = {
        'id': 'invalid-json',
        'type': 'function',
        'function': {'name': 'read_record', 'arguments': arguments},
    }
    requests = []

    def receive(request):
        requests.append(json.loads(request.content))
        return (
            sdk(
                {
                    'role': 'assistant',
                    'content': '',
                    'reasoning_content': 'retained',
                    'tool_calls': [call],
                }
            )
            if len(requests) == 1
            else sdk({'role': 'assistant', 'content': 'DONE'})
        )

    with httpx.Client(transport=httpx.MockTransport(receive)) as transport:
        model = ConfiguredModel(
            DeepSeekClient(
                model='deepseek-flash', api_key='offline', http_client=transport, max_retries=0
            )
        ).bind_tools([read_record])
        response = model.invoke([HumanMessage(content='Read')])
        assert not response.tool_calls
        assert response.invalid_tool_calls[0]['args'] == arguments
        assert response.additional_kwargs[RAW_TOOL_CALLS] == [call]
        model.invoke(
            [
                HumanMessage(content='Read'),
                response,
                ToolMessage(
                    content='Rejected JSON; no operation executed', tool_call_id='invalid-json'
                ),
            ]
        )
    assert requests[1]['messages'][1]['tool_calls'] == [call]
    assert requests[1]['messages'][1]['reasoning_content'] == 'retained'


def test_stream_completion_uses_same_strict_classification_without_editing_original():
    arguments = '{"name":"first\nsecond"}'

    class Stream:
        def bind_tools(self, *args, **kwargs):
            return self

        def stream(self, *args, **kwargs):
            yield AIMessageChunk(
                content='',
                additional_kwargs={'reasoning_content': 'thought'},
                tool_call_chunks=[
                    {'name': 'read_record', 'args': arguments, 'id': 'stream-json', 'index': 0}
                ],
                response_metadata={'finish_reason': 'tool_calls'},
            )

    thoughts = []
    response = (
        ConfiguredModel(Stream())
        .bind_tools([read_record])
        .invoke_with_progress([HumanMessage(content='Read')], thoughts.append)
    )
    assert thoughts == ['thought'] and not response.tool_calls
    assert response.invalid_tool_calls[0]['args'] == arguments
    assert (
        wire_input([response], [read_record])['messages'][0]['tool_calls'][0]['function'][
            'arguments'
        ]
        == arguments
    )


def test_real_runner_rejects_bad_json_then_continues_without_any_operation():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        repo = root / 'repo'
        repo.mkdir()
        bad = '{"kind":"inspect","note":"first\nsecond","steps":[{"id":"bad","op":"shell","command":"echo BAD > must-not-exist"}]}'
        replies = [
            {
                'id': 'bad',
                'type': 'function',
                'function': {'name': 'submit_plan', 'arguments': bad},
            },
            {
                'id': 'done',
                'type': 'function',
                'function': {
                    'name': 'submit_plan',
                    'arguments': json.dumps(
                        {'kind': 'blocked', 'note': 'Missing required input', 'steps': []}
                    ),
                },
            },
        ]
        requests = []

        def receive(request):
            requests.append(json.loads(request.content))
            return sdk(
                {
                    'role': 'assistant',
                    'content': '',
                    'reasoning_content': 'saved thought',
                    'tool_calls': [replies[len(requests) - 1]],
                }
            )

        with httpx.Client(transport=httpx.MockTransport(receive)) as transport:
            model = ConfiguredModel(
                DeepSeekClient(
                    model='deepseek-flash', api_key='offline', http_client=transport, max_retries=0
                )
            )
            environment = LocalEnvironment(repo, artifact_root=root / 'runtime')
            runner = TaskRunner(model, environment, root / 'state')
            result = runner.run([], 'Read the task input', StepBudget(max_steps=3))
        assert result['status'] == 'task_blocked' and len(requests) == 2
        assert not (repo / 'must-not-exist').exists()
        messages = requests[1]['messages']
        original = next(m for m in messages if m.get('role') == 'assistant' and m.get('tool_calls'))
        rejection = next(m for m in messages if m.get('role') == 'tool')
        assert (
            original['tool_calls'] == [replies[0]]
            and original['reasoning_content'] == 'saved thought'
        )
        assert rejection['tool_call_id'] == 'bad' and 'no actions executed' in rejection['content']
