"""检索的真实入模、原件边界和请求级身份；只使用临时资料及假模型。"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict
import pytest

from agent.application.chat import Chatbot
from agent.application.store import ApplicationStore
from agent.execution.local import LocalEnvironment
from agent.rag import integration
from agent.rag.types import RagConfig
from agent.task.budget import StepBudget
from agent.task.runner import TaskRunner
from tests.fixtures.models import ScriptedModel, text_reply


def arguments(**changes):
    return {'query': 'answer', 'path': 'main.py', 'intent': 'locate',
            'required_entities': [], 'required_kinds': [], **changes}


def retrieval_reply(**changes):
    return AIMessage(content='', additional_kwargs={'reasoning_content': 'saved original thought'},
                     tool_calls=[{'name': 'retrieve_evidence', 'id': 'read-evidence',
                                  'args': arguments(**changes)}],
                     usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5})


def search_reply(*, read=False, **changes):
    step = {'id': 'find', 'op': 'search', **arguments(**changes)}
    steps = [step]
    if read:
        steps.append({'id': 'read', 'op': 'read', 'from_step': 'find', 'limit': 20})
    return AIMessage(content='', additional_kwargs={'reasoning_content': 'saved original thought'},
                     tool_calls=[{'name': 'submit_plan', 'id': 'read-evidence',
                                  'args': {'kind': 'inspect', 'note': 'Find original evidence',
                                           'steps': steps}}],
                     usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5})


def blocked():
    return AIMessage(content='', tool_calls=[{
        'name': 'submit_plan', 'id': 'blocked',
        'args': {'kind': 'blocked', 'note': 'Fixture ends without changing acceptance.'},
    }], usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5})


@pytest.fixture
def project(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'main.py').write_text('def answer():\n    return 42\n', encoding='utf-8')
    (repo / '.env').write_text('API_KEY=DO_NOT_INDEX', encoding='utf-8')
    monkeypatch.setattr(integration, '_models', lambda _: SimpleNamespace(vector=None, reranker=None))
    return repo


@pytest.mark.parametrize('intent', ['locate', 'diagnostic', 'audit'])
def test_tokenizer_failure_keeps_semantic_unavailable_without_paid_rewrite(
    project, tmp_path, monkeypatch, intent,
):
    from agent.rag.models import ModelUnavailable

    attempted = []
    def failed_index(*_):
        attempted.append('count')
        raise ModelUnavailable('Local tokenizer cannot fit the memory reserve')

    class ForbiddenRoute:
        def search(self, *_):
            pytest.fail('Unavailable tokenizer must not dispatch a vector request')
        def score(self, *_):
            pytest.fail('Unavailable tokenizer must not dispatch reranking')

    model = SimpleNamespace(manifest={'models': {'embedding': {'configured': True}}},
                            index_documents=failed_index, vector=ForbiddenRoute(),
                            reranker=ForbiddenRoute())
    monkeypatch.setattr(integration, '_models', lambda _: model)
    text = (project / 'main.py').read_text(encoding='utf-8')
    version = hashlib.sha256(text.encode()).hexdigest()
    sources = integration.Sources()
    sources.add('source:main.py', version, len(text), lambda: text, kind='code', path='main.py')
    result = integration._retrieve(
        SimpleNamespace(), sources,
        arguments(mode='semantic', intent=intent, required_entities=['answer'], required_kinds=['code', 'log']),
        root=tmp_path, cache_root=tmp_path / 'tasks/task', repo=project, task_id='task',
        request_id='one-request', config=integration.retrieval_configuration(),
        rewrite=lambda *_: pytest.fail('Unavailable retrieval must not consume a rewrite'),
    )
    assert attempted == ['count']
    assert result['results'] == [] and result['hits'] == []
    assert result['retrieval_mode'] == 'semantic' and result['retrieval_incomplete']
    assert result['rewrite'] == {'status': 'not_triggered', 'triggered': []}


def test_work_tool_reaches_next_request_after_original_and_preserves_pairing(project, tmp_path):
    model = ScriptedModel([search_reply(read=True), blocked()])
    loop = TaskRunner(model, LocalEnvironment(project, artifact_root=tmp_path / 'runtime'),
                      state_dir=tmp_path / 'tasks' / 'current', overlap_preparation=False,
                      user_request_id='one-user-request')
    result = loop.run([], 'Find the answer function', StepBudget(max_steps=5))
    assert result['status'] == 'task_blocked' and len(model.calls) == 2
    tool = next(item for item in model.calls[1] if isinstance(item, ToolMessage))
    assert tool.name == 'submit_plan' and tool.tool_call_id == 'read-evidence'
    payload = json.loads(tool.content)
    assert 'return 42' in tool.content and 'DO_NOT_INDEX' not in tool.content
    search = payload['steps'][0]
    assert search['data']['mode'] == 'exact' and search['data']['hits']
    original = loop.root / search['record']
    saved = json.loads(original.read_text('utf-8'))
    hit = search['data']['hits'][0]
    excerpt = search['data']['results'][hit['result']]
    source = search['data']['sources'][excerpt['source']]
    assert source['path'] == saved['data']['hits'][0]['path']
    assert source['sha256'] == saved['data']['hits'][0]['source_sha256']
    assert hit['match_start'] == saved['data']['hits'][0]['match_start']
    assert excerpt['text'] == saved['data']['results'][0]['text']
    assert 'scope' in saved['data'] and 'scope' not in search['data']
    assert loop.state['rag_request_id'] == 'one-user-request'
    assert not loop.state.get('rag_pending') and not loop.state.get('rag_running')
    assert not (loop.root / 'rag/operations').exists()
    assert not loop.state.get('verification_spec')
    assert loop.state['history'][:1] == [message_to_dict(model.calls[0][1])]


def test_work_original_write_failure_never_dispatches_next_model(project, tmp_path, monkeypatch):
    from agent.task import operations

    save = operations.save
    def fail(path, value):
        if Path(path).name == 'result.json':
            raise OSError('fixture disk failure')
        return save(path, value)

    monkeypatch.setattr(operations, 'save', fail)
    model = ScriptedModel([search_reply(), blocked()])
    loop = TaskRunner(model, LocalEnvironment(project, artifact_root=tmp_path / 'runtime'),
                      state_dir=tmp_path / 'tasks' / 'failed', overlap_preparation=False)
    result = loop.run([], 'Find answer', StepBudget(max_steps=5))
    assert result['status'] == 'runtime_error' and len(model.calls) == 1
    assert loop.state['running_step']['id'] == 'find'
    assert all(raw['type'] != 'tool' for raw in loop.state['history'])
    assert loop.state['pending']['rows'] == []


def test_retired_work_tool_is_rejected_before_any_search(project, tmp_path, monkeypatch):
    monkeypatch.setattr(integration, 'retrieve_work', lambda *_: pytest.fail('Retired lane executed'))
    loop = TaskRunner(ScriptedModel([]), LocalEnvironment(project, artifact_root=tmp_path / 'runtime'))
    loop.state = {}
    with pytest.raises(ValueError):
        loop.decode_plan(message_to_dict(retrieval_reply()))


def test_rewrite_uses_separate_request_state_and_keeps_work_response(project, tmp_path):
    model = ScriptedModel([text_reply('{"query":"answer implementation"}')])
    loop = TaskRunner(model, LocalEnvironment(project, artifact_root=tmp_path / 'runtime'))
    loop.root = tmp_path / 'task'
    loop.root.mkdir()
    loop.budget = StepBudget(max_steps=5)
    current = message_to_dict(search_reply())
    loop.state = {'history': [], 'response_pending': current, 'context_usage': {'used_tokens': 23}}
    loop.persist = lambda *_: None
    result = loop.request_model([HumanMessage(content='rewrite answer')], 'rag_rewrite')
    assert integration._rewrite_text(result) == 'answer implementation'
    assert loop.state['response_pending'] == current
    assert loop.state['context_usage'] == {'used_tokens': 23}
    assert loop.state['rag_request_pending'] is None
    assert loop.state['model_usage'][0]['purpose'] == 'rag_rewrite'
    assert loop.budget.used_steps == 1 and model.tools == ()
    assert (loop.root / loop.state['rag_rewrite_response']['record']).is_file()


def test_unknown_rewrite_stops_and_restart_does_not_repeat(project, tmp_path, monkeypatch):
    from agent.rag.rewrite import RewriteLedger

    attempts = []

    def retrieve(loop, _arguments):
        def unknown(_query, _reasons):
            attempts.append('sent')
            raise RuntimeError('connection ended without a confirmed result')

        return RewriteLedger(tmp_path / 'rewrites.sqlite3').attempt(
            loop.state['rag_request_id'], 'answer', ['required_evidence_missing'], unknown,
        )

    monkeypatch.setattr(integration, 'retrieve_work', retrieve)
    environment = LocalEnvironment(project, artifact_root=tmp_path / 'runtime')
    root = tmp_path / 'tasks' / 'unknown'
    first_model = ScriptedModel([search_reply()])
    first = TaskRunner(first_model, environment, state_dir=root, overlap_preparation=False,
                       user_request_id='shared-request')
    first.run([], 'Find answer', StepBudget(max_steps=5))
    assert attempts == ['sent'] and len(first_model.calls) == 1
    assert first.state['running_step']['id'] == 'find'
    model = ScriptedModel([])
    resumed = TaskRunner(model, environment, state_dir=root, overlap_preparation=False,
                         user_request_id='shared-request')
    result = resumed.run([], 'Find answer', StepBudget(max_steps=5))
    assert result['status'] == 'execution_unknown'
    assert attempts == ['sent'] and model.calls == []


def test_resume_saved_result_preserves_prefix_and_original_tool_id(project, tmp_path, monkeypatch):
    environment = LocalEnvironment(project, artifact_root=tmp_path / 'runtime')
    root = tmp_path / 'tasks' / 'saved'
    first_model = ScriptedModel([search_reply()])
    first = TaskRunner(first_model, environment, state_dir=root, overlap_preparation=False,
                       pause_after_steps=1)
    result = first.run([], 'Find answer', StepBudget(max_steps=5))
    assert result['status'] == 'paused'
    prefix = deepcopy(first.state['history'])
    monkeypatch.setattr(integration, 'retrieve_work',
                        lambda *_: pytest.fail('A saved search must not be replayed'))
    model = ScriptedModel([blocked()])
    resumed = TaskRunner(model, environment, state_dir=root, overlap_preparation=False)
    result = resumed.run([], 'Find answer', StepBudget(max_steps=5))
    assert result['status'] == 'task_blocked' and len(model.calls) == 1
    assert resumed.state['history'][:len(prefix)] == prefix
    tool = next(item for item in model.calls[0] if isinstance(item, ToolMessage))
    assert tool.tool_call_id == 'read-evidence' and 'answer' in tool.content


def test_records_require_current_task_commit_and_skip_model_and_acceptance(tmp_path):
    root = tmp_path / 'task'
    root.mkdir()
    records = {
        'plans/p1/p1-s1/stdout.log': ('shell', 'confirmed failure'),
        'plans/p1/p1-s1/result.json': ('shell', '{"status":"failed","data":{"error":"boom"}}'),
        'requests/call-0001/response.json': ('model_record', 'private thought'),
        'acceptance/t1/result.json': ('shell', 'private acceptance'),
        'plans/p3/p3-s1/result.json': ('read', json.dumps({
            'op': 'read', 'data': {'path': '$record/requests/call-0001/response.json',
                                   'text': 'private thought copied by a read'},
        })),
    }
    with sqlite3.connect(root / 'history.sqlite3') as db:
        db.execute('CREATE TABLE artifacts(path TEXT,sha256 TEXT,kind TEXT)')
        for name, (kind, text) in records.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = text.encode()
            path.write_bytes(raw)
            db.execute('INSERT INTO artifacts VALUES(?,?,?)', (name, hashlib.sha256(raw).hexdigest(), kind))
    uncommitted = root / 'plans/p2/p2-s1/stdout.log'
    uncommitted.parent.mkdir(parents=True)
    uncommitted.write_text('not committed')
    sources = integration.Sources()
    integration._records(sources, root)
    docs = list(sources.documents())
    assert len(docs) == 2
    assert all(doc.source_id.startswith('record:plans/p1/') for doc in docs)
    assert 'confirmed failure' in '\n'.join(doc.text for doc in docs)
    assert 'private' not in '\n'.join(doc.text for doc in docs)
    (root / 'plans/p1/p1-s1/stdout.log').write_text('changed')
    assert len(list(sources.documents())) == 1
    assert sources.reasons and set(sources.reasons) == {'source_unavailable:ValueError'}


def test_chat_uses_bound_project_not_parent_or_additional_sources(project, tmp_path):
    nested = project / 'another'
    nested.mkdir()
    (nested / '.git').mkdir()
    (nested / 'secret.py').write_text('OTHER_PROJECT')
    parent = tmp_path / 'unrelated.txt'
    parent.write_text('PARENT_MATERIAL')
    work = {'repo': str(project), 'task_id': 'current'}
    sources = integration.chat_sources(SimpleNamespace(root=tmp_path / 'state'), work, '.')
    docs = list(sources.documents())
    assert {doc.path for doc in docs} == {'main.py'}
    (project / 'main.py').write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        sources.read(docs[0].source_id, docs[0].version)


def test_chat_without_bound_project_reports_gap_without_model_rewrite(project, tmp_path):
    store = ApplicationStore(tmp_path / 'state')
    conversation = store.create(repo=str(project))['thread_id']
    model = ScriptedModel([retrieval_reply(mode='semantic'), text_reply()])
    bot = Chatbot(store, lambda *_: pytest.fail('Unexpected handoff'), factory=lambda _: model)
    command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Retrieve evidence'}
    store.submit(conversation, command)
    bot.run(conversation, command)
    tool = next(item for item in model.calls[1] if isinstance(item, ToolMessage))
    assert json.loads(tool.content)['reasons'] == ['project_not_bound']
    originals = list((store.root / 'application-records' / conversation).glob('*/tool-result.json'))
    assert len(originals) == 1
    assert json.loads(originals[0].read_text())['tool_call_id'] == 'read-evidence'
    assert len(model.calls) == 2


@pytest.mark.parametrize('changes', [
    {'path': '../other'}, {'path': '/other'}, {'path': 'C:/other'},
    {'path': '$record/requests'}, {'intent': 'anything'}, {'required_kinds': ['web']},
])
def test_retrieval_scope_arguments_reject_unsupported_values(changes):
    with pytest.raises(ValueError):
        integration.validate_arguments(arguments(**changes))


def test_final_tool_envelope_counts_metadata_and_long_identifiers():
    call = {'id': 'x' * 25000, 'name': 'retrieve_evidence'}
    with pytest.raises(ValueError, match='8192'):
        integration.delivery({'results': []}, call)


def test_fixed_configuration_ignores_legacy_database_and_returns_independent_snapshots(tmp_path):
    legacy = tmp_path / 'rag/tuning.sqlite3'
    legacy.parent.mkdir()
    legacy.write_bytes(b'legacy parameter database must not be opened')
    selected = integration.retrieval_configuration()
    assert selected['values'] == asdict(RagConfig())
    for task in ('task-a', 'task-b'):
        loop = SimpleNamespace(root=tmp_path / 'tasks' / task, state={})
        bound = integration.bind_work(loop)
        assert {key: bound[key] for key in selected} == selected
        assert loop.state['rag_config'] == selected
    selected['values']['chunk_target'] = 384
    assert integration.retrieval_configuration()['values'] == asdict(RagConfig())
    assert legacy.read_bytes() == b'legacy parameter database must not be opened'


def test_fixed_configuration_rejects_modified_checkpoint(tmp_path):
    selected = integration.retrieval_configuration()
    loop = SimpleNamespace(root=tmp_path / 'tasks/task-a', state={'rag_config': deepcopy(selected)})
    loop.state['rag_config']['values']['chunk_target'] = 384
    with pytest.raises(ValueError, match='configuration changed'):
        integration.bind_work(loop)


def test_citation_checks_only_delivered_original_ranges_and_never_claims_semantic_proof():
    text = '原文\nanswer = 42\n'
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    original = {'source_id': 'source:main.py', 'version': 'v1', 'source_sha256': sha,
                'start': 0, 'end': 2}
    citations = [original, {**original, 'end': len(text)}, {**original, 'version': 'v2'}]
    reports = integration.verify_citations(citations, [original], lambda *_: text)
    assert [item['status'] for item in reports] == ['source_verified', 'unverified', 'unverified']
    assert all(item['semantic_support'] == 'not_assessed' for item in reports)
    reports = integration.verify_citations([original], [original], lambda *_: 'changed')
    assert reports[0]['status'] == 'unverified'


def test_work_citation_audit_preserves_response_and_does_not_reject_plan(project, tmp_path):
    class CitingModel(ScriptedModel):
        def invoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return search_reply()
            tool = next(item for item in messages if isinstance(item, ToolMessage))
            chunk = json.loads(tool.content)['steps'][0]['data']['results'][0]
            response = blocked()
            response.content = f"Observed source [evidence:{chunk['id']}] [evidence:unknown]."
            self.original = message_to_dict(response)
            return response

    model = CitingModel([])
    loop = TaskRunner(model, LocalEnvironment(project, artifact_root=tmp_path / 'runtime'),
                      state_dir=tmp_path / 'tasks' / 'citing', overlap_preparation=False)
    result = loop.run([], 'Find answer', StepBudget(max_steps=5))
    assert result['status'] == 'task_blocked' and len(model.calls) == 2
    checks = json.loads((loop.root / loop.state['citation_checks']['record']).read_text('utf-8'))
    assert [item['status'] for item in checks['checks']] == ['source_verified', 'unverified']
    assert checks['critical_claim_detection'] == 'not_automated'
    assert json.loads((loop.root / 'requests/call-0002/response.json').read_text('utf-8')) == model.original
    assert loop.state['repair_count'] == 0


def test_citation_marker_cannot_recover_unseen_or_modified_evidence():
    text = 'original source'
    sha = hashlib.sha256(text.encode()).hexdigest()
    chunk = {'id': 'found', 'source': 0, 'start': 0, 'end': len(text)}
    source = {'path': 'main.py', 'version': 'v1', 'sha256': sha}
    tool = ToolMessage(content=json.dumps({'sources': [source], 'results': [chunk]}), name='retrieve_evidence',
                       tool_call_id='lookup')
    raw = message_to_dict(text_reply('[evidence:found]'))
    assert integration._audit_markers(raw, [], lambda *_: text)[0]['status'] == 'unverified'
    assert integration._audit_markers(raw, [tool], lambda *_: 'changed')[0]['status'] == 'unverified'
    assert integration._audit_markers(raw, [tool], lambda *_: text)[0]['status'] == 'source_verified'


def test_public_record_projection_never_reuses_work_private_index(project, tmp_path):
    private = '{"status":"failed","text":"work-only result detail"}'
    public = '{"status":"failed"}'
    version = hashlib.sha256(private.encode()).hexdigest()
    configuration = integration.retrieval_configuration()

    def query(text, reader_scope):
        sources = integration.Sources()
        sources.add('record:plans/p1/p1-s1/result.json', version, len(text), lambda: text,
                    kind='tool_result', path='$record/plans/p1/p1-s1/result.json', run_id='task')
        return integration._retrieve(
            SimpleNamespace(), sources, arguments(query='failed', path='.', intent='diagnostic'),
            root=tmp_path, cache_root=tmp_path / 'tasks/task', repo=project,
            task_id='task', request_id='request',
            config=configuration, rewrite=None, reader_scope=reader_scope,
        )

    assert 'work-only result detail' in json.dumps(query(private, 'work'))
    result = query(public, 'public')
    assert result['reader_scope'] == 'public'
    assert result['results'] and 'work-only result detail' not in json.dumps(result)
    assert len(list((tmp_path / 'tasks/task/rag/indexes').glob('*.sqlite3'))) == 2
    assert not (tmp_path / 'rag/indexes').exists()


def test_index_resource_failure_is_not_presented_as_absent_evidence(project, tmp_path, monkeypatch):
    from agent.rag.index import Index

    def full(*_args, **_kwargs):
        raise sqlite3.OperationalError('database or disk is full')

    monkeypatch.setattr(Index, 'upsert_documents', full)
    sources = integration.Sources()
    sources.add('source:main.py', 'v1', 10, lambda: 'answer', kind='code', path='main.py')
    result = integration._retrieve(
        SimpleNamespace(), sources, arguments(), root=tmp_path,
        cache_root=tmp_path / 'tasks/task', repo=project,
        task_id='task', request_id='request',
        config=integration.retrieval_configuration(),
        rewrite=lambda *_: pytest.fail('Resource failure must not trigger paid rewrite'),
    )
    assert result['retrieval_incomplete'] is True
    assert result['reasons'] == ['retrieval_index_unavailable']


def test_required_gap_uses_one_charged_rewrite_with_isolated_context(project, tmp_path, monkeypatch):
    class Vector:
        def search(self, query, index, limit):
            return [(chunk.id, 1.0) for chunk in index.chunks(query)][:limit]

    class Reranker:
        def score(self, query, chunks):
            return [1.0] * len(chunks)

    monkeypatch.setattr(integration, '_models', lambda _: SimpleNamespace(
        vector=Vector(), reranker=Reranker(),
    ))
    model = ScriptedModel([
        search_reply(mode='semantic', intent='diagnostic', required_entities=['missingSymbol']),
        text_reply('{"query":"answer missingSymbol"}'), blocked(),
    ])
    loop = TaskRunner(model, LocalEnvironment(project, artifact_root=tmp_path / 'runtime'),
                      state_dir=tmp_path / 'tasks' / 'rewrite', overlap_preparation=False,
                      user_request_id='same-request')
    result = loop.run([], 'Find the implementation', StepBudget(max_steps=5))
    assert result['status'] == 'task_blocked' and len(model.calls) == 3
    rewrite_messages = model.calls[1]
    assert len(rewrite_messages) == 2
    assert 'missingSymbol' in rewrite_messages[1].content
    assert 'saved original thought' not in str(rewrite_messages)
    tool = next(item for item in model.calls[2] if isinstance(item, ToolMessage))
    assert tool.tool_call_id == 'read-evidence'
    assert json.loads(tool.content)['steps'][0]['data']['rewrite']['status'] == 'complete'
    assert [entry['purpose'] for entry in loop.state['model_usage']] == ['plan', 'rag_rewrite', 'plan']
    from agent.rag.rewrite import RewriteLedger

    ledger = RewriteLedger(tmp_path / 'rag/rewrites.sqlite3')
    assert ledger.get('same-request')['status'] == 'complete'
    assert ledger.attempt('same-request', 'another', ['required_evidence_missing'],
                          lambda *_: pytest.fail('The rewrite budget is already used'))['status'] == 'already_used'


def test_shared_resource_slot_returns_busy_before_enumeration_or_model(monkeypatch):
    monkeypatch.setattr(integration, 'work_sources',
                        lambda *_: pytest.fail('Busy retrieval must not enumerate sources'))
    with integration._RETRIEVAL_LOCK:
        work = integration.retrieve_work(SimpleNamespace(), arguments())
        chat = integration.retrieve_chat(SimpleNamespace(), 'conversation', 'request', arguments())
    assert work == chat
    assert work['reasons'] == ['retrieval_busy'] and work['retrieval_incomplete']
    assert work['rewrite']['status'] == 'suppressed'


def test_oversized_archived_source_is_rejected_before_reading_its_bytes(tmp_path):
    digest = 'a' * 64
    root = tmp_path / 'source-records'
    (root / 'blobs').mkdir(parents=True)
    with (root / 'blobs' / digest).open('wb') as stream:
        stream.truncate(2_000_001)
    view = SimpleNamespace(root=root, baseline=None, files={'huge.bin': digest},
                           read_bytes=lambda *_: pytest.fail('Oversized source was fully read'))
    loop = SimpleNamespace(evidence=SimpleNamespace(view=lambda: view), root=tmp_path / 'task')
    sources = integration.work_sources(loop, '.')
    assert sources.entries == {} and sources.reasons == ['source_too_large']


def test_model_configuration_is_frozen_per_task_and_invalid_original_keeps_hash(tmp_path, monkeypatch):
    monkeypatch.delenv('MONOPLANER_RAG_MODELS', raising=False)
    assert integration._bind_models(tmp_path, 'old')['hash'] == 'unconfigured'
    manifest = tmp_path / 'models.json'
    raw = b'{"version":1}'
    manifest.write_bytes(raw)
    monkeypatch.setenv('MONOPLANER_RAG_MODELS', str(manifest))
    with pytest.raises(ValueError, match='start a new task'):
        integration._bind_models(tmp_path, 'old')
    current = integration._bind_models(tmp_path, 'new')
    assert current['hash'] == hashlib.sha256(raw).hexdigest()
    assert current['status'] == 'configured'
    manifest.write_bytes(b'{broken')
    invalid = integration._bind_models(tmp_path, 'invalid')
    assert invalid['status'] == 'unavailable'
    assert invalid['hash'] == hashlib.sha256(b'{broken').hexdigest()
    with pytest.raises(ValueError, match='start a new task'):
        integration._bind_models(tmp_path, 'new')


def test_full_source_cache_prunes_stale_versions_before_admitting_replacement(project, tmp_path, monkeypatch):
    from agent.rag.index import Index

    configuration = integration.retrieval_configuration()
    owner = SimpleNamespace()
    def retrieve(version, text):
        sources = integration.Sources()
        sources.add('source:main.py', version, len(text), lambda: text, kind='code', path='main.py')
        return integration._retrieve(
            owner, sources, arguments(query='answer', path='.', intent='diagnostic'),
            root=tmp_path, cache_root=tmp_path / 'tasks/task', repo=project,
            task_id='task', request_id='request', config=configuration, rewrite=None,
        )
    assert retrieve('old', 'answer old')['results']
    original = Index.upsert_documents
    observations = []
    def bounded(index, documents, **kwargs):
        # 确定性的单版本缓存上限：旧版本未释放时拒绝下一次写入。
        with index._connection() as db:
            old = db.execute('SELECT version FROM documents WHERE scope=?',
                             (index.scope.key,)).fetchall()
        observations.append(old)
        if old:
            raise sqlite3.OperationalError('database or disk is full')
        return original(index, documents, **kwargs)
    monkeypatch.setattr(Index, 'upsert_documents', bounded)
    result = retrieve('new', 'answer new')
    assert observations == [[]]
    assert result['results'] and result['results'][0]['version'] == 'new'
    assert not any(reason == 'retrieval_index_unavailable' for reason in result['reasons'])
