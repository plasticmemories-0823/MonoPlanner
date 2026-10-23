"""任务在聊天生成期间完成时，保留原响应但不向用户发布过期状态。"""

import pytest

from agent.application.chat import Chatbot
from tests.test_application_runtime import FakeModel, answer, command
from tests.test_chatbot_work_context import app, job


@pytest.mark.parametrize('tool_on_retry', [False, True])
def test_completion_during_chat_replies_from_new_facts_without_replaying_request(app, tool_on_retry):
    store, conversation, _ = app
    task = job(app, status='needs_plan')
    request = command('现在完成了吗')
    store.submit(conversation, request)
    replies = [answer('还在工作，尚未完成。')]
    if tool_on_retry:
        replies.append(answer(tool_calls=[{'name': 'get_work_status', 'args': {}, 'id': 'fresh-status'}]))
    replies.append(answer('已经完成，验收通过。'))

    class FinishingModel(FakeModel):
        def invoke(self, messages):
            if not self.requests:
                store.work_state(task['task_id'], 'verified_success')
            return super().invoke(messages)

    model = FinishingModel(replies)
    bot = Chatbot(store, lambda *_: pytest.fail('Status query dispatched work'), factory=lambda _: model)
    result = bot.run(conversation, request)
    assert result['output'] == '已经完成，验收通过。'
    assert len(model.requests) == (3 if tool_on_retry else 2)
    visible = [m['content'] for m in store.snapshot(conversation)['messages'] if m['source'] == 'chat']
    assert visible == ['已经完成，验收通过。']
    history = store.model_history(conversation)
    assert any(raw['type'] == 'ai' and raw['data']['content'] == '还在工作，尚未完成。'
               for _, raw in history)
    if tool_on_retry:
        assert sum(raw['type'] == 'tool' for _, raw in history) == 1
    assert store.usage(conversation)['chat']['calls'] == len(model.requests)
    assert store.usage(conversation)['chat']['unknown_calls'] == 0


def test_repeated_state_changes_have_bounded_refresh_and_factual_fallback(app):
    store, conversation, repo = app
    task = job(app, status='needs_plan')
    request = command('目前是什么状态')
    store.submit(conversation, request)

    class ChangingModel(FakeModel):
        def invoke(self, messages):
            status = ['paused', 'executing', 'verified_success'][len(self.requests)]
            store.work_state(task['task_id'], status)
            return super().invoke(messages)

    model = ChangingModel([answer('旧状态') for _ in range(3)])
    result = Chatbot(store, lambda *_: None, factory=lambda _: model).run(conversation, request)
    assert len(model.requests) == 3
    assert '已完成' in result['output'] and str(repo) in result['output']
    assert all(m['content'] != '旧状态' for m in store.snapshot(conversation)['messages'])
    assert store.usage(conversation)['chat']['unknown_calls'] == 0


def test_unknown_usage_does_not_trigger_another_status_request(app):
    store, conversation, _ = app
    task = job(app, status='needs_plan')
    request = command('完成了吗')
    store.submit(conversation, request)
    response = answer('尚未完成')
    response.usage_metadata = None

    class FinishingModel(FakeModel):
        def invoke(self, messages):
            store.work_state(task['task_id'], 'verified_success')
            return super().invoke(messages)

    model = FinishingModel([response])
    result = Chatbot(store, lambda *_: None, factory=lambda _: model).run(conversation, request)
    assert len(model.requests) == 1
    assert '已完成' in result['output']
    assert store.usage(conversation)['chat']['unknown_calls'] == 1
    assert any(raw['type'] == 'ai' and raw['data']['content'] == '尚未完成'
               for _, raw in store.model_history(conversation))
