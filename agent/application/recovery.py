"""应用重启只补齐保存的响应和工具配对，不执行工具或发起模型请求。"""

import json

from langchain_core.messages import ToolMessage, message_to_dict

from agent.model.tool_results import model_result
from agent.source.snapshots import _safe_file
from agent.storage.records import digest


def reconcile_chat(store):
    with store.connection() as db:
        calls = [dict(row) for row in db.execute('SELECT * FROM app_calls')]
    for row in calls:
        request = json.loads(row['request'])
        if request.get('kind') == 'service':
            continue
        lane = request.get('mode', 'chat')
        folder = store.root / 'application-records' / row['conversation'] / row['id']
        response_path = folder / 'response.json'
        if response_path.exists():
            receipt = json.loads(_safe_file(store.root,
                                           response_path.relative_to(store.root)).read_bytes())
            raw = receipt['response']
            if (receipt['call_id'] != row['id'] or receipt['lane'] != lane
                    or receipt['conversation'] != row['conversation']
                    or receipt['digest'] != digest(raw)):
                raise ValueError('Saved chat response identity changed')
            store.model_message(row['conversation'], lane, row['id'] + ':response', raw)
            store.finish_call(row['id'], response_id=row['id'] + ':response',
                              usage=raw['data'].get('usage_metadata'), seconds=receipt['seconds'])
    for conversation in store.list():
        if conversation.get('read_only'):
            continue
        identity = conversation['thread_id']
        history = store.model_history(identity)
        paired = {raw['data']['tool_call_id'] for _, raw in history if raw['type'] == 'tool'}
        for message_id, raw in history:
            if raw['type'] != 'ai' or not message_id.endswith(':response'):
                continue
            call_id = message_id.removesuffix(':response')
            calls = [*raw['data'].get('tool_calls', []), *raw['data'].get('invalid_tool_calls', [])]
            for index, call in enumerate(calls):
                if call['id'] in paired:
                    continue
                folder = store.root / 'application-records' / identity / (call_id + '-' + str(index))
                saved = folder / 'tool-result.json'
                if saved.exists():
                    receipt = json.loads(_safe_file(store.root, saved.relative_to(store.root)).read_bytes())
                    if receipt['tool_call_id'] != call['id'] or receipt['name'] != call['name']:
                        raise ValueError('Saved chat tool result identity changed')
                    result = receipt['result']
                else:
                    dispatched = (folder / 'intent.json').exists()
                    result = {
                        'status': 'unknown' if dispatched else 'cancelled',
                        'reason': ('Service stopped after tool dispatch; effects are unconfirmed. '
                                   'Do not repeat this operation. Inspect saved application records.'
                                   if dispatched else
                                   'Service stopped before tool dispatch; no operation ran.'),
                        **({'effects': 'unconfirmed'} if dispatched else {'actions_executed': False}),
                    }
                visible = model_result(call['name'], result, project=store.get(identity)['repo'])
                if call['name'] == 'retrieve_evidence':
                    from agent.rag.integration import delivery

                    message = delivery(visible, call)
                else:
                    message = message_to_dict(ToolMessage(
                        content=json.dumps(visible, ensure_ascii=False), name=call['name'],
                        tool_call_id=call['id'],
                    ))
                store.model_message(identity, 'chat', call_id + ':tool:' + str(index), message)
                paired.add(call['id'])
