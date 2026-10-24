"""模块入口保持原包语义，项目导入路径仅进入实际工作子进程。"""
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.acceptance.execution import ExecutionCheck, validate_execution_result
from agent.task.requirements import DeliveryTarget, checked, confirmed, check_execution
from resources.drivers.execution import execute


def declaration(**changes):
    return {"id": "cli", "requirement_id": "package", "artifact_type": "repository_entry",
            "consumer": "command", "interpreter": "python", "entry_file": "demo/__main__.py",
            "python_module": "demo", "argv": ["literal argument"], **changes}


def target(**changes):
    return {"id": "package", "artifact_type": "repository_entry", "target_runtime": "python",
            "entry": "demo/__main__.py", "python_module": "demo", "argv": ["literal argument"],
            "behavior": "Run the documented package CLI", "input_scope": "Original public inputs",
            "quotes": [{"source_id": "task_request", "text": "Run the documented package CLI"}],
            **changes}


def package(tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'demo').mkdir(parents=True)
    (repo / 'demo/__init__.py').write_text('', encoding='utf-8')
    (repo / 'demo/helper.py').write_text('VALUE = 42\n', encoding='utf-8')
    (repo / 'demo/__main__.py').write_text(
        'import json, sys\nfrom .helper import VALUE\n'
        'print(json.dumps([__package__, VALUE, sys.argv[1:]]))\n', encoding='utf-8')
    return repo


def test_package_cli_uses_module_semantics_and_preserves_host_environment(tmp_path):
    repo = package(tmp_path)
    spec = ExecutionCheck.model_validate(declaration()).model_dump()
    with patch.dict(os.environ, {'PYTHONSAFEPATH': '1', 'PYTHONPATH': str(tmp_path / 'unrelated')}):
        before = dict(os.environ)
        row = execute(spec, repo, tmp_path / 'receipt-cli', lambda: 20)
        assert dict(os.environ) == before
    assert row['status'] == 'passed', row
    assert json.loads(row['stdout']) == ['demo', 42, ['literal argument']]
    assert row['argv'][-3:] == ['-m', 'demo', 'literal argument']
    assert '-I' in row['compile_argv']
    output = tmp_path / 'receipt.json'
    output.write_text(json.dumps({'checks': [row]}), encoding='utf-8')
    validate_execution_result(output, [spec], repository=repo, map_path=lambda p: p)
    row['argv'][-2] = 'other'
    output.write_text(json.dumps({'checks': [row]}), encoding='utf-8')
    with pytest.raises(ValueError):
        validate_execution_result(output, [spec], repository=repo, map_path=lambda p: p)


def test_nested_module_file_runs_once_with_relative_imports(tmp_path):
    repo = package(tmp_path)
    (repo / 'demo/command.py').write_text(
        'from .helper import VALUE\nfrom pathlib import Path\n'
        'p=Path("calls.txt")\np.write_text(p.read_text()+"x" if p.exists() else "x")\n'
        'print(VALUE)\n', encoding='utf-8')
    spec = ExecutionCheck.model_validate(declaration(
        entry_file='demo/command.py', python_module='demo.command', argv=[])).model_dump()
    row = execute(spec, repo, tmp_path / 'run', lambda: 20)
    assert row['status'] == 'passed', row
    assert (repo / 'calls.txt').read_text() == 'x'


@pytest.mark.parametrize('module,entry,interpreter', [
    ('-c', 'demo/__main__.py', 'python'), ('../demo', 'demo/__main__.py', 'python'),
    ('demo;exit', 'demo/__main__.py', 'python'), ('demo', 'other.py', 'python'),
    ('demo', 'demo/__main__.py', 'node'), ('demo..cli', 'demo/cli.py', 'python'),
])
def test_invalid_module_or_substituted_entry_is_rejected(module, entry, interpreter):
    with pytest.raises(ValueError):
        ExecutionCheck.model_validate(declaration(
            python_module=module, entry_file=entry, interpreter=interpreter))
    with pytest.raises(ValueError):
        DeliveryTarget.model_validate(target(python_module=module, entry=entry,
                                              target_runtime=interpreter))


def test_worker_cannot_drop_confirmed_module_mode():
    packet = confirmed('Run the documented package CLI', [target()],
                       confirmation={'id': 'human', 'text': 'Use the documented module'})
    owner = SimpleNamespace(state={'requirements': packet})
    assert check_execution(owner, [declaration()]) == packet['digest']
    with pytest.raises(ValueError, match='python_module'):
        check_execution(owner, [declaration(python_module='')])


def test_legacy_requirement_without_module_field_keeps_its_digest():
    packet = confirmed('Run the documented package CLI', [target(python_module='')],
                       confirmation={'id': 'human', 'text': 'Use the file'})
    packet['targets'][0].pop('python_module', None)
    from agent.storage.records import digest
    packet['digest'] = digest({k: v for k, v in packet.items() if k != 'digest'})
    before = deepcopy(packet)
    assert checked(packet) == before
    assert check_execution(SimpleNamespace(state={'requirements': packet}),
                           [declaration(python_module='')]) == before['digest']


def test_module_package_cannot_select_a_different_compiled_source(tmp_path):
    repo = package(tmp_path)
    (repo / 'demo.py').write_text('print("wrong source")\n', encoding='utf-8')
    spec = ExecutionCheck.model_validate(declaration(entry_file='demo.py', argv=[])).model_dump()
    row = execute(spec, repo, tmp_path / 'run', lambda: 20)
    assert row['status'] != 'passed'
    assert not row['producer_called']


def test_user_module_is_not_imported_by_compile_probe(tmp_path):
    repo = package(tmp_path)
    (repo / 'py_compile.py').write_text('raise RuntimeError("shadowed trusted compiler")\n', encoding='utf-8')
    spec = ExecutionCheck.model_validate(declaration()).model_dump()
    with patch.dict(os.environ, {'PYTHONPATH': str(repo), 'PYTHONSAFEPATH': '1'}):
        row = execute(spec, repo, tmp_path / 'run', lambda: 20)
    assert row['status'] == 'passed', row


def test_missing_module_never_falls_back_to_an_installed_package(tmp_path):
    repo = tmp_path / 'empty'
    repo.mkdir()
    spec = ExecutionCheck.model_validate(declaration(
        python_module='json', entry_file='json/__main__.py', argv=[])).model_dump()
    row = execute(spec, repo, tmp_path / 'run', lambda: 20)
    assert row['status'] == 'failed' and not row['producer_called']


def test_public_tool_schemas_can_express_the_same_module_contract():
    from agent.settings import PROJECT_ROOT

    check = ExecutionCheck.model_validate(declaration()).model_dump()
    plan = json.loads((PROJECT_ROOT / 'resources/plan.schema.json').read_text('utf-8'))
    review = json.loads((PROJECT_ROOT / 'resources/acceptance-review.schema.json').read_text('utf-8'))
    proposal = json.loads((PROJECT_ROOT / 'resources/work-requirements.schema.json').read_text('utf-8'))
    for schema, value in (
        (plan['properties']['execution_checks']['items'], check),
        (review['revise']['properties']['execution_checks']['items'], check),
        (proposal['properties']['targets']['items'], DeliveryTarget.model_validate(target()).model_dump()),
    ):
        assert schema['additionalProperties'] is False
        assert schema['properties']['python_module']['type'] == 'string'
        assert 'python_module' in schema['required']
        assert set(schema['required']) <= value.keys() <= schema['properties'].keys()


def test_confirmation_and_stopped_task_context_show_module_and_arguments(tmp_path):
    import uuid
    from agent.application.store import ApplicationStore
    from agent.application.requirements import prepare, accept
    from agent.application.work_context import context

    repo = package(tmp_path)
    store = ApplicationStore(tmp_path / 'application')
    conversation = store.create(repo=str(repo))['thread_id']
    user = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': 'Run the documented package CLI'}
    store.submit(conversation, user)
    delivery = target()
    delivery['quotes'][0]['source_id'] = user['id']
    proposal = prepare(store, conversation, user, project_path='.', create=False,
                       request_message_ids=[user['id']], targets=[delivery])
    notice = next(m['content'] for m in store.snapshot(conversation)['messages']
                  if m['source'] == 'work_proposal')
    assert 'python -m demo' in notice and 'literal argument' in notice
    action = {'id': str(uuid.uuid4()), 'kind': 'confirm_work', 'proposal_id': proposal['proposal_id']}
    store.submit(conversation, action)
    packet, request = accept(store, conversation, action, proposal['proposal_id'], '.', False)
    work = store.create_work(conversation, action['id'], work_options={
        'confirmed_requirements': packet, 'original_request': request})
    store.work_state(work['task_id'], 'paused')
    targets = context(store, conversation)['task']['confirmed_requirements']['targets']
    assert targets[0]['python_module'] == 'demo' and targets[0]['argv'] == ['literal argument']


def test_file_invocation_also_gets_project_imports_without_mutating_the_host(tmp_path):
    repo = package(tmp_path)
    (repo / 'demo/cli.py').write_text('from demo.helper import VALUE\nprint(VALUE)\n', encoding='utf-8')
    spec = ExecutionCheck.model_validate(declaration(
        python_module='', entry_file='demo/cli.py', argv=[])).model_dump()
    with patch.dict(os.environ, {'PYTHONSAFEPATH': '1'}):
        before = dict(os.environ)
        row = execute(spec, repo, tmp_path / 'run', lambda: 20)
        assert dict(os.environ) == before
    assert row['status'] == 'passed', row
    assert row['stdout'].strip() == '42'
