"""已有程序的额外运行需要明确确认，不能由进度问答偷偷创建任务。"""

from copy import deepcopy
import json

from agent.application.project_location import resolve_location
from agent.application.requirements import load_proposal, messages, source_messages
from agent.application.store import TERMINAL
from agent.server.assistant import validate
from agent.storage.records import digest, save


def prepare(store, conversation, command, arguments):
    validate('run_existing', arguments)
    value = store.get(conversation)
    current = store.work(value['work_id'])
    if current and current['status'] not in TERMINAL:
        raise ValueError('Work is still active. Read its saved status; do not start a separate run.')
    location = resolve_location(value['repo'], current['project_path'] if current else '.', False)
    original = source_messages(store, conversation, [command['id']])['sources'][0]
    proposal = {
        'kind': 'existing', 'conversation': conversation,
        'project_path': location['path'], 'repo': location['repo'], 'create': False,
        'parent_task': value['work_id'], 'prepared_after': command['id'],
        'source': original, 'arguments': deepcopy(arguments),
    }
    identifier = digest(proposal)
    path = store.root / 'work-proposals' / conversation / (identifier + '.json')
    if path.exists():
        if json.loads(path.read_text('utf-8')) != proposal:
            raise ValueError('Saved execution proposal changed')
    else:
        save(path, proposal)
    content = '\n'.join([
        '待确认的额外运行', '', '项目：' + location['repo'],
        '入口：' + arguments['entry'],
        '参数：' + json.dumps(arguments['args'], ensure_ascii=False),
        '', '这会另行执行已有程序；不会替代原编码任务的验收结果。',
        '只有点击确认后才运行。只查询进度或文件位置时无需确认此运行。',
    ])
    store.publish(conversation, 'work-proposal:' + identifier, content, source='work_proposal')
    return {'status': 'awaiting_confirmation', 'proposal_id': identifier,
            'project_path': location['path'], 'actions_executed': False}


def accept(store, conversation, command, proposal_id):
    proposal = load_proposal(store, conversation, proposal_id)
    if proposal.get('kind') != 'existing':
        raise ValueError('Expected an existing-command proposal')
    rows = messages(store, conversation)
    positions = {row['id']: index for index, row in enumerate(rows)}
    human = next((row for row in rows if row['id'] == command['id']), None)
    if (command.get('kind') != 'confirm_work' or command.get('proposal_id') != proposal_id
            or not human or human['source'] != 'user' or human['type'] != 'human'
            or positions[command['id']] <= positions.get('work-proposal:' + proposal_id, len(rows))):
        raise ValueError('Running requires a later saved human confirmation of this proposal')
    value = store.get(conversation)
    if value['work_id'] != proposal['parent_task']:
        raise ValueError('The selected task changed after this execution proposal')
    location = resolve_location(value['repo'], proposal['project_path'], False)
    if location['repo'] != proposal['repo']:
        raise ValueError('The proposed execution location changed')
    original = source_messages(store, conversation, [proposal['source']['id']])['sources'][0]
    if original != proposal['source']:
        raise ValueError('The original execution request changed')
    validate('run_existing', proposal['arguments'])
    return deepcopy(proposal['arguments'])
