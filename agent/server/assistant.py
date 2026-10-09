"""Single-owner tool journal for general queries and existing repository tasks."""
import json
import time

from langchain_core.messages import ToolMessage, message_to_dict

from agent.execution.assistant_files import AssistantFiles
from agent.execution import existing
from agent.execution.web import WEB_NAMES, WebProvider, WebUnknown, archive_file
from agent.model.tools import ASSISTANT_SCHEMA
from agent.storage.records import save

READ_NAMES = {'list_files', 'read_file', 'search_files'}
ASSISTANT_NAMES = {*READ_NAMES, 'run_existing'}


def validate(name, args):
    schema = ASSISTANT_SCHEMA[name]
    if not isinstance(args, dict):
        raise ValueError(name + ': 参数必须是完整 JSON 对象')
    errors = []
    for key in schema['required']:
        if key not in args:
            errors.append(key + ': 缺少必要参数')
    errors.extend(key + ': 未定义字段' for key in args.keys() - schema['properties'].keys())
    types = {'string': str, 'integer': int, 'array': list}
    for key, value in args.items():
        field = schema['properties'].get(key)
        if not field:
            continue
        if type(value) is not types[field['type']] or (field['type'] == 'array' and any(type(v) is not str for v in value)):
            errors.append(f'{key}: 实际值 {str(value)[:100]!r}；要求 {field["type"]}')
    for key, minimum, maximum in (('offset', 0, None), ('limit', 1, 200 if name == 'list_files' else 50 if name == 'search_files' else 1000), ('timeout', 1, None)):
        if type(args.get(key)) is int and (args[key] < minimum or (maximum and args[key] > maximum)):
            errors.append(f'{key}: 实际值 {args[key]}；要求 {minimum}..{maximum or "任务剩余时间"}')
    if errors:
        raise ValueError(name + ': ' + '; '.join(errors))


def invoke(loop, call, folder, history):
    name, args = call['name'], call['args']
    if name in WEB_NAMES:
        if not loop.runtime_settings.get('online'):
            raise ValueError('Networking is disabled for this task')
        field = 'query' if name == 'web_search' else 'url'
        if not isinstance(args, dict) or set(args) != {field} or not isinstance(args[field], str) or not 1 <= len(args[field]) <= 2000:
            raise ValueError(field + ': 要求一个非空文本参数，最多 2000 字符')
        provider = WebProvider()
        return provider.search(args[field], loop, folder) if name == 'web_search' else provider.fetch(args[field], loop, folder)
    validate(name, args)
    files = AssistantFiles(loop.repo, loop.state.get('read_roots', []))
    if name == 'list_files':
        return files.list(args['path'], args['offset'], args['limit'])
    if name == 'search_files':
        return files.search(args['path'], args['query'], args['offset'], args['limit'])
    if name == 'read_file':
        return files.read(args['path'], args['offset'], args['limit'], history, bool(args['reread_reason'].strip()))
    return existing.execute(loop, args, folder)


def execute_calls(loop, response, history=None):
    pending = loop.state.get('assistant_pending')
    if pending is None:
        data = response['data']
        pending = {'response': response, 'cursor': 0, 'results': [], 'call': loop.budget.used_steps,
                   'calls': [*data.get('tool_calls', []), *data.get('invalid_tool_calls', [])]}
        loop.state['assistant_pending'] = pending
        loop.persist('assistant_prepared')
    calls = pending['calls']
    conflict = (any(c['name'] not in ASSISTANT_NAMES | WEB_NAMES for c in calls)
                or (any(c['name'] == 'run_existing' for c in calls) and len(calls) != 1))
    while pending['cursor'] < len(calls):
        call = calls[pending['cursor']]
        folder = loop.root/'assistant'/('call-%04d-%02d' % (pending['call'], pending['cursor']))
        save(folder/'request.json', call)
        archive_file(loop, folder/'request.json', 'assistant_call')
        loop.state['assistant_running'] = call['id']
        loop.persist('assistant_started', {**call, 'request_call': pending['call'], 'started_at': time.time()})
        started = time.monotonic()
        try:
            if conflict:
                raise ValueError('已有任务运行须单独调用；查询不能混入编码、未知工具或应用启动')
            target = loop.state['history'] if history is None else history
            if history is None:
                from agent.model.compaction import history as request_view
                target = request_view(loop.state, loop.root)
            result = invoke(loop, call, folder, [*target, *pending['results']])
            status = 'failed' if call['name'] == 'run_existing' and (result['exit_code'] != 0 or result['timed_out']) else 'succeeded'
        except WebUnknown:
            loop.state.update(status='execution_unknown', reason='已发出的网络请求结果不明，不自动重试')
            loop.persist('assistant_unknown', {**call, 'request_call': pending['call']})
            return
        except (ValueError, OSError, KeyError) as exc:
            if call['name'] == 'run_existing' and loop.state.get('assistant_process'):
                loop.state.update(status='execution_unknown', reason='已有任务执行结果不明，不自动重试')
                loop.persist('assistant_unknown', {**call, 'request_call': pending['call']})
                return
            result, status = {'error': str(exc)}, 'failed'
        seconds = time.monotonic() - started
        result['seconds'] = seconds
        save(folder/'result.json', {'status': status, 'data': result})
        for filename in ('result.json', 'stdout.log', 'stderr.log', 'child-usage.json'):
            if (folder/filename).is_file():
                archive_file(loop, folder/filename, 'assistant_result')
        pending['results'].append(message_to_dict(ToolMessage(content=json.dumps(result, ensure_ascii=False),
                                  name=call['name'], tool_call_id=call['id'])))
        pending['cursor'] += 1
        loop.state.update(assistant_running=None, assistant_process=None)
        if call['name'] == 'run_existing':
            turn = loop.state.get('conversation_turn')
            loop.state['assistant_run_turn'] = turn['id'] if turn else 'initial'
            loop.state.setdefault('existing_runs', []).append({'record': str(folder/'result.json'),
                'entry': call['args'].get('entry') if isinstance(call['args'], dict) else None,
                'status': status, 'seconds': seconds, 'child_usage': result.get('child_usage')})
        loop.persist('assistant_finished', {**call, 'request_call': pending['call'], 'status': status, 'seconds': seconds,
                                           'exit_code': result.get('exit_code'), 'child_usage': result.get('child_usage')})
        if loop.yield_boundary and pending['cursor'] < len(calls):
            return
    target = loop.state['history'] if history is None else history
    target.extend([response, *pending['results']])
    loop.state.update(assistant_pending=None, response_pending=None, status='needs_plan')
    loop.persist('assistant_completed')
