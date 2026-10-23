"""Chatbot独立请求视图，复用既有压缩与前缀保护，不改工作上下文。"""

from types import SimpleNamespace
import time

from langchain_core.messages import message_to_dict

from agent.model import compaction
from agent.model.client import ModelPolicy, configure_call
from agent.model.context import admit, CONTEXT_WINDOW_TOKENS
from agent.model.errors import ModelRequestNotSent
from agent.model.payload import estimate_input_tokens


CHAT_INTERFACE_REVISION = 6


def prepare_chat(store, conversation, rules, history, model, tools, configuration_revision=0):
    """传入完整原件，返回经过校验的请求消息；摘要永远只改变独立请求视图。"""
    state = store.context(conversation)
    originals = [item[1] for item in history]
    if state.get('request_pending'):
        raise ModelRequestNotSent('model_request_unknown',
                                  'Saved summary request has no confirmed result; it was not retried.')
    old_interface = state.get('interface_revision', 1 if state.get('rules') else CHAT_INTERFACE_REVISION)
    if type(old_interface) is not int or not 1 <= old_interface <= CHAT_INTERFACE_REVISION:
        raise ModelRequestNotSent('context_prefix_changed', 'Unsupported future chat interface revision')
    # 先核验原文与旧摘要，再因明确的应用接口版本升级新建请求视图；不改历史。
    state['history'] = originals
    compaction.guard_archive(state)
    if state.get('context_view'):
        compaction.history(state, store.root / 'application-records' / conversation / 'context')
    old_revision = state.get('configuration_revision', 0)
    if configuration_revision != old_revision or old_interface != CHAT_INTERFACE_REVISION:
        if configuration_revision < old_revision:
            raise ModelRequestNotSent('context_prefix_changed', 'Chat settings revision moved backwards')
        # 用户明确变更设置时开启新请求视图；旧摘要文件和全部原件依然保留。
        state.setdefault('prior_views', []).append({
            'interface_revision': old_interface, 'rules': state.get('rules'),
            'configuration_revision': old_revision, 'context_view': state.get('context_view'),
            'last_admission': state.get('last_admission'),
        })
        for key in ('context_view', 'last_admission', 'context_rollover'):
            state.pop(key, None)
        state['rules'] = rules
    if state.get('rules') and state['rules'] != rules:
        raise ModelRequestNotSent('context_prefix_changed',
                                  'Chat settings changed; start a new conversation for new settings.')
    if compaction.balanced_boundaries(originals)[-1] != len(originals):
        raise ModelRequestNotSent('invalid_message_history',
                                  'A saved tool call still has no confirmed result.')
    state.update(history=originals, rules=rules, status='chat', source=None)
    state['configuration_revision'] = configuration_revision
    state['interface_revision'] = CHAT_INTERFACE_REVISION
    if 'request' not in state:
        state.update(request=next((raw['data']['content'] for raw in originals
                                   if raw['type'] == 'human'), ''),
                     acceptance=None, verify_command=None)
    root = store.root / 'application-records' / conversation / 'context'
    loop = SimpleNamespace(state=state, rules=rules, root=root, tools=tools, history_store=None)
    loop.persist = lambda event, payload=None: store.save_context(conversation, state, event, payload)

    def request_summary(messages, purpose):
        selected = configure_call(model.bind_tools(tools), 'compaction')
        options = getattr(selected, 'request_options', {})
        output = min(getattr(selected, 'max_output_tokens', ModelPolicy().max_output_tokens),
                     compaction.CONFIG['summary_max_tokens'])
        if estimate_input_tokens(messages, tools) + output > CONTEXT_WINDOW_TOKENS:
            raise ModelRequestNotSent('context_capacity_exhausted',
                                      'Summary request exceeds the available context window')
        pending = state['compaction_pending']
        call = store.begin_call(conversation, 'chat', {
            'purpose': purpose, 'message_ids': [item[0] for item in history],
            'context_view': state.get('context_view'), 'prefix_end': pending['prefix_end'],
            'instruction': compaction.summary_instruction(pending['execution_snapshot']),
            'options': options,
        })
        state['request_pending'] = {'call': call, 'purpose': purpose}
        loop.persist('chat_summary_reserved')
        started = time.monotonic()
        try:
            from agent.task.control import current
            from agent.model.errors import ModelInterrupted

            if current() and current().kind:
                raise ModelInterrupted(current().kind, sent=False)
            store.mark_sent(call)
            if callable(getattr(type(selected), 'invoke_with_progress', None)):
                response = selected.invoke_with_progress(messages, lambda text: None)
            else:
                response = selected.invoke(messages)
            raw = message_to_dict(response)
            store.model_message(conversation, 'chat-compaction', call + ':response', raw)
            store.finish_call(call, response_id=call + ':response', usage=response.usage_metadata,
                               seconds=time.monotonic() - started)
            pending.update(response=raw, call=call)
            state['request_pending'] = None
            loop.persist('chat_summary_response_saved')
        except Exception as exc:
            store.finish_call(call, seconds=time.monotonic() - started, error=type(exc).__name__)
            loop.persist('chat_summary_stopped')
            raise

    loop.request_model = request_summary
    selected = configure_call(model.bind_tools(tools), 'conversation')
    cap = getattr(selected, 'max_output_tokens', ModelPolicy().max_output_tokens)
    compaction.prepare(loop, cap)
    messages = compaction.messages(loop)
    options = getattr(selected, 'request_options', {'max_tokens': cap})
    admit(state, messages, tools, options)
    state['context_usage'] = {'used_tokens': estimate_input_tokens(messages, tools),
                              'window_tokens': CONTEXT_WINDOW_TOKENS, 'estimated': True}
    loop.persist('context_admitted')
    return messages, state.get('context_view')
