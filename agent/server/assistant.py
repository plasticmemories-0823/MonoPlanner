"""为一般查询和仓库已有任务提供单一执行者管理的工具日志。"""

from agent.execution.assistant_files import AssistantFiles
from agent.execution import existing
from agent.execution.web import WEB_NAMES, WebProvider
from agent.model.tools import ASSISTANT_SCHEMA

OPEN_NAMES = {'open_file', 'launch_application', 'request_file_operation'}


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
        if type(value) is not types[field['type']] or (
            field['type'] == 'array' and any(type(v) is not str for v in value)
        ):
            errors.append(f'{key}: 实际值 {str(value)[:100]!r}；要求 {field["type"]}')
        if field.get('enum') and value not in field['enum']:
            errors.append(f'{key}: 实际值 {value!r}；要求 {field["enum"]}')
    for key, minimum, maximum in (
        ('offset', 0, None),
        ('limit', 1, 200 if name == 'list_files' else 50 if name == 'search_files' else 1000),
        ('timeout', 1, None),
    ):
        if type(args.get(key)) is int and (
            args[key] < minimum or (maximum and args[key] > maximum)
        ):
            errors.append(f'{key}: 实际值 {args[key]}；要求 {minimum}..{maximum or "任务剩余时间"}')
    if errors:
        raise ValueError(name + ': ' + '; '.join(errors))


def invoke(loop, call, folder, history):
    name, args = call['name'], call['args']
    if name == 'retrieve_evidence':
        # 此入口没有当前任务的版本及请求身份，不能把额外授权目录冒充 RAG 资料源。
        raise ValueError('retrieve_evidence requires the bound project retrieval service.')
    if name in WEB_NAMES:
        if not loop.runtime_settings.get('online'):
            raise ValueError('Networking is disabled for this task')
        field = 'query' if name == 'web_search' else 'url'
        if (
            not isinstance(args, dict)
            or set(args) != {field}
            or not isinstance(args[field], str)
            or not 1 <= len(args[field]) <= 2000
        ):
            raise ValueError(field + ': 要求一个非空文本参数，最多 2000 字符')
        provider = WebProvider()
        return (
            provider.search(args[field], loop, folder)
            if name == 'web_search'
            else provider.fetch(args[field], loop, folder)
        )
    validate(name, args)
    if name in OPEN_NAMES or name == 'list_applications':
        from agent.execution.desktop import invoke as desktop_invoke

        return desktop_invoke(loop, name, args)
    files = AssistantFiles(loop.repo, read_sources=loop.state.get('read_sources', []))
    if name == 'list_files':
        return files.list(args['path'], args['offset'], args['limit'])
    if name == 'search_files':
        return files.search(args['path'], args['query'], args['offset'], args['limit'])
    if name == 'read_file':
        return files.read(
            args['path'],
            args['offset'],
            args['limit'],
            history,
            bool(args['reread_reason'].strip()),
        )
    return existing.execute(loop, args, folder)
