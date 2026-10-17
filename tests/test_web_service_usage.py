"""按次收费的联网服务独立计费，未知结果与重启不伪装为模型 Token。"""

from dataclasses import asdict
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import httpx
from langchain_core.messages import AIMessage

from agent.application.chat import Chatbot
from agent.application.store import ApplicationStore
from agent.model.errors import ModelRequestNotSent
from agent.storage.usage import summarize
from agent.storage.recovery import restore_budget
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from tests.fixtures.models import ScriptedModel
import test_task_runner as fixtures


@pytest.mark.parametrize('outcome,cost,reserved,calls', [
    ('succeeded', 5000, 0, 1), ('empty', 0, 0, 1), ('failed', 0, 0, 1),
    ('unknown', None, 5000, 1), ('not_sent', 0, 0, 0),
])
def test_service_settlement_is_separate_from_model_budget(outcome, cost, reserved, calls):
    budget = StepBudget(max_steps=1, max_tokens=1)
    ticket = budget.reserve_service('web_search')
    assert budget.inflight_service_cost_micros == 5000
    record = budget.settle_service(ticket, outcome)
    assert record['cost_micros'] == cost
    assert record['reserved_cost_micros'] == reserved
    assert budget.service_calls == calls
    assert budget.used_steps == budget.reported_tokens == budget.unknown_usage_steps == 0
    assert budget.inflight_service_cost_micros == 0
    usage = summarize({'web_usage': [record]})
    assert usage['service_calls'] == calls
    assert usage['calls'] == usage['unknown_calls'] == 0
    with pytest.raises(ValueError):
        budget.settle_service(ticket, outcome)


def test_unknown_service_fee_remains_reserved_after_reload():
    budget = StepBudget(max_service_cost_micros=5000)
    budget.settle_service(budget.reserve_service('web_fetch'), 'unknown')
    restored = StepBudget(**asdict(budget))
    with pytest.raises(ModelRequestNotSent) as failure:
        restored.reserve_service('web_search')
    assert failure.value.status == 'service_budget_exhausted'
    assert restored.unknown_service_cost_micros == 5000
    assert restored.unknown_usage_steps == 0
    with pytest.raises(ValueError, match='Budget policy changed'):
        restore_budget(SimpleNamespace(budget=StepBudget(max_service_cost_micros=10000)),
                       {'budget': asdict(budget)})


def test_pending_service_never_becomes_unknown_model_usage():
    state = {'web_request_pending': {
        'kind': 'service', 'call': 1, 'purpose': 'web_fetch', 'reserved_cost_micros': 5000,
    }}
    usage = summarize(state)
    assert usage['service_calls'] == usage['service_unknown_calls'] == 1
    assert usage['service_reserved_cost_micros'] == 5000
    assert usage['calls'] == usage['unknown_calls'] == usage['cache_unknown_calls'] == 0
    state['web_usage'] = [{**state['web_request_pending'], 'cost_micros': None}]
    assert summarize(state) == usage


@pytest.mark.parametrize('name,outcome', [
    ('web_search', 'succeeded'), ('web_fetch', 'empty'), ('web_fetch', 'unknown'),
    ('web_search', 'not_sent'),
])
def test_chat_service_events_persist_fees_without_model_calls(tmp_path, monkeypatch, name, outcome):
    store = ApplicationStore(tmp_path / 'state')
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(store.root))
    conversation = store.create(repo=str(tmp_path))['thread_id']

    def invoke(context, call, folder, history):
        ticket = context.budget.reserve_service(name)
        payload = {'call': ticket.call_id, 'purpose': name, 'reserved_cost_micros': 5000}
        context.persist('web_request_started', payload)
        row = context.budget.settle_service(ticket, outcome)
        context.state['web_usage'] = [{**row, 'seconds': 0.5}]
        event = {'unknown': 'web_result_unknown', 'not_sent': 'web_request_not_sent'}.get(
            outcome, 'web_response_saved')
        context.persist(event, payload)
        return {}

    monkeypatch.setattr('agent.application.chat.invoke_assistant', invoke)
    bot = Chatbot(store, lambda *_: pytest.fail('No handoff expected'))
    bot._invoke_tool(conversation, {'name': name, 'args': {}}, 'test-service')
    usage = ApplicationStore(store.root).usage(conversation)['chat']
    assert usage['calls'] == usage['unknown_calls'] == usage['total_tokens'] == 0
    assert usage['service_calls'] == (0 if outcome == 'not_sent' else 1)
    assert usage['service_cost_micros'] == (5000 if outcome == 'succeeded' else 0)
    assert usage['service_reserved_cost_micros'] == (5000 if outcome == 'unknown' else 0)


def test_chat_dispatched_service_without_response_survives_reopen(tmp_path):
    store = ApplicationStore(tmp_path)
    conversation = store.create()['thread_id']
    call = store.begin_call(conversation, 'chat', {
        'kind': 'service', 'purpose': 'web_fetch', 'reserved_cost_micros': 5000,
    })
    store.mark_sent(call)
    usage = ApplicationStore(tmp_path).usage(conversation)['chat']
    assert usage['calls'] == usage['unknown_calls'] == 0
    assert usage['service_reserved_cost_micros'] == 5000
    assert usage['service_unknown_calls'] == 1


@pytest.mark.parametrize('name,args,body', [
    ('web_search', {'query': 'foo.bar'}, {'results': [
        {'url': 'https://example.com', 'title': 'Example', 'snippet': 'foo.bar',
         'summary': 'Example summary', 'content': ''},
    ]}),
    ('web_fetch', {'url': 'https://example.com'}, {'title': 'Example', 'content': '# Body'}),
])
def test_chat_real_provider_with_fake_http_is_recorded_as_service(tmp_path, monkeypatch,
                                                                 name, args, body):
    store = ApplicationStore(tmp_path / 'state')
    monkeypatch.setenv('MONOPLANER_STATE_DIR', str(store.root))
    conversation = store.create(repo=str(tmp_path), runtime_settings={
        'model_name': 'deepseek-flash', 'reasoning_effort': 'high', 'online': True,
    })['thread_id']
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr('agent.execution.web.model_http_clients',
                        lambda _: {'http_client': client})
    monkeypatch.setattr('agent.execution.web.load_settings',
                        lambda _: {'api_key': 'offline-only'})
    bot = Chatbot(store, lambda *_: pytest.fail('No handoff expected'))
    result = bot._invoke_tool(conversation, {'name': name, 'args': args}, 'http-service')
    assert len(requests) == 1 and requests[0].url.path == '/v1/tools/' + name[4:]
    assert result
    usage = ApplicationStore(store.root).usage(conversation)['chat']
    assert usage['service_calls'] == 1
    assert usage['service_cost_micros'] == 5000
    assert usage['service_reserved_cost_micros'] == 0
    assert usage['calls'] == usage['unknown_calls'] == usage['total_tokens'] == 0


class WebServiceRecoveryTests(unittest.TestCase):
    setUp = fixtures.TaskRunnerTests.setUp

    def test_interrupted_service_reserves_once_and_never_resends(self):
        class Crash(BaseException):
            pass

        def interrupted(provider, query, loop, folder):
            ticket = loop.budget.reserve_service('web_search')
            loop.state['web_request_pending'] = {
                'kind': 'service', 'call': ticket.call_id, 'purpose': 'web_search',
                'reserved_cost_micros': ticket.reserved_cost_micros,
            }
            loop.persist('web_request_started', loop.state['web_request_pending'])
            raise Crash()

        response = AIMessage(content='', tool_calls=[{
            'name': 'web_search', 'id': 'web-1', 'args': {'query': 'offline'},
        }], usage_metadata={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12})
        settings = {'online': True}
        runner = TaskRunner(ScriptedModel([response]), self.env, state_dir=self.root / 'state',
                            runtime_settings=settings)
        with patch('agent.execution.web.WebProvider.search', interrupted), self.assertRaises(Crash):
            runner.run([], 'test original request', StepBudget(max_steps=5))
        for _ in range(2):
            model = ScriptedModel([])
            resumed = TaskRunner(model, self.env, state_dir=self.root / 'state',
                                 runtime_settings=settings)
            result = resumed.run([], 'test original request', StepBudget(max_steps=5))
            self.assertEqual(result['status'], 'web_request_unknown')
            self.assertEqual(result['usage']['service_unknown_calls'], 1)
            self.assertEqual(result['usage']['service_reserved_cost_micros'], 5000)
            self.assertEqual(resumed.budget.unknown_service_cost_micros, 5000)
            self.assertEqual(resumed.budget.unknown_usage_steps, 0)
            self.assertEqual(resumed.budget.used_steps, 1)
            self.assertFalse(model.calls)
