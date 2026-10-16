"""旧会话适配的离线回归参考，不由产品加载。"""
from starlette.responses import JSONResponse
from agent.storage.preferences import state_root
from tests.legacy.conversations import ConversationStore

def control_value(thread, value=None):
    store = ConversationStore(state_root())
    thread = store.thread(thread)
    session = store.session(thread)
    if not session and value is None:
        return JSONResponse({'error': 'Conversation has no task'}, status_code=404)
    if value is None:
        pending = store.pending(thread)
        return JSONResponse(
            {
                'status': session['status'],
                'task_id': session['task_id'],
                'runtime_settings': session.get('runtime_settings'),
                'workspace_bound': session.get('workspace_bound', False),
                'read_roots': session.get('read_roots', []),
                'read_sources': session.get('read_sources', []),
                'repo': session['repo'],
                'messages': [
                    {'id': m['data'].get('id'), 'type': m['type'], 'content': m['data']['content']}
                    for m in session.get('public_messages', [])
                ],
                'pending': len(pending),
                'pending_messages': [
                    {'id': row['id'], 'type': 'human', 'content': row['text']}
                    for row in pending
                    if row['kind'] == 'message'
                ],
            }
        )
    record = store.enqueue(thread, value['id'], value['kind'], value.get('text', ''))
    if not record['applied']:
        from agent.task.control import interrupt

        interrupt(state_root(), thread, record['kind'], record['id'])
    return JSONResponse({'received': True, 'id': record['id'], 'applied': bool(record['applied'])})
