"""资源回收只影响已确认闲置的自有产物；不需要真实 Docker。"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.storage.resources import (
    ResourceCatalog, RESERVE_BYTES, container_owner, cancel_owned_containers,
    register_active_container, unregister_active_container, release_task_resources,
    preflight_space, delete_owned_docker_resource,
)


def test_pause_keeps_active_stop_receipt_when_persistent_catalog_is_unreadable(tmp_path, monkeypatch):
    import sqlite3
    import agent.storage.resources as resources

    (tmp_path / 'resources.sqlite3').write_bytes(b'fixture')
    stopped = Mock(return_value={'stopped': True, 'receipt': 'saved'})
    token = register_active_container('task:disk-full', 'known-current', stopped, root=tmp_path)
    try:
        monkeypatch.setattr(resources, 'resource_catalog', Mock(
            side_effect=sqlite3.OperationalError('database or disk is full')))
        recovered = Mock()
        monkeypatch.setattr(resources, 'stop_recorded_container', recovered)
        result = cancel_owned_containers('task:disk-full', tmp_path)
    finally:
        unregister_active_container(token)
    stopped.assert_called_once()
    recovered.assert_not_called()
    assert result[0] == {'container': 'known-current', 'stopped': True, 'receipt': 'saved'}
    assert result[1]['stopped'] is False
    assert result[1]['error'] == 'OperationalError'


def test_references_and_leases_protect_shared_resource(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    folder = tmp_path / 'temporary'
    folder.mkdir()
    (folder / 'file').write_text('kept')
    catalog.register('copy', 'temporary', folder, size_bytes=4, owner='task:a')
    catalog.link('task:b', 'copy')
    catalog.unlink('task:a')
    assert catalog.cleanup(['copy'])['retained'] == ['copy']
    catalog.unlink('task:b')
    with catalog.lease('operation', ['copy']):
        assert catalog.cleanup(['copy'])['retained'] == ['copy']
        assert folder.is_dir()
    assert catalog.cleanup(['copy'])['removed'] == ['copy']
    assert not folder.exists()


def test_manual_docker_and_frozen_evidence_are_never_cache_eviction_targets(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    folder = tmp_path / 'original'
    folder.mkdir()
    catalog.register('evidence', 'evidence', folder, size_bytes=100)
    catalog.register('user-image', 'image', 'sha256:' + 'a' * 64, owned=False, size_bytes=100)
    docker = Mock()
    report = catalog.cleanup(['evidence', 'user-image'], docker=docker)
    assert set(report['retained']) == {'evidence', 'user-image'}
    docker.assert_not_called()
    assert folder.is_dir()


def test_cleanup_rejects_directory_outside_private_root(tmp_path):
    catalog = ResourceCatalog(tmp_path / 'state')
    outside = tmp_path / 'project'
    outside.mkdir()
    with pytest.raises(ValueError, match='私有记录'):
        catalog.register('unsafe', 'temporary', outside)
    assert outside.is_dir()


def test_resource_identity_cannot_be_reassigned(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    catalog.register('image', 'image', 'sha256:' + 'a' * 64)
    with pytest.raises(ValueError, match='身份'):
        catalog.register('image', 'image', 'sha256:' + 'b' * 64)


def test_lease_releases_after_python_exception(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    directory = tmp_path / 'copy'
    directory.mkdir()
    catalog.register('copy', 'temporary', directory)
    with pytest.raises(RuntimeError):
        with catalog.lease('task:test', ['copy']):
            raise RuntimeError('operation failed')
    assert catalog.usage()['resources'][0]['leases'] == 0
    assert catalog.cleanup(['copy'])['removed'] == ['copy']


def test_paused_task_keeps_its_resource_references(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    catalog.register('container', 'container', 'owned-container', owner='task:x')
    remove = Mock()
    assert release_task_resources('task:x', root=tmp_path, docker=remove)['preserved']
    assert catalog.usage()['resources'][0]['references'] == 1
    remove.assert_not_called()
    result = release_task_resources('task:x', root=tmp_path, completed=True, docker=remove)
    assert result['removed'] == ['container']
    assert remove.call_args.args[:2] == ('container', 'owned-container')


def test_explicit_environment_cleanup_releases_only_unreferenced_dependency(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    image = 'sha256:' + 'a' * 64
    catalog.register('image', 'image', image)
    catalog.register('volume', 'volume', 'dependency', owner='environment:' + image)
    catalog.link('task:paused', 'volume')
    remove = Mock()
    report = catalog.cleanup(['image'], docker=remove)
    assert report['removed'] == ['image']
    assert report['retained'] == ['volume']
    assert catalog.usage()['resources'][0]['references'] == 1


def test_space_check_reserves_five_gib_before_copy(tmp_path, monkeypatch):
    monkeypatch.setattr('agent.storage.resources.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=RESERVE_BYTES + 10))
    assert preflight_space(tmp_path, 10)['required_bytes'] == 10
    with pytest.raises(OSError, match='磁盘空间不足'):
        preflight_space(tmp_path, 11)


def test_cache_soft_limit_removes_only_enough_idle_resources(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    for name in ('one', 'two', 'three'):
        path = tmp_path / name
        path.mkdir()
        catalog.register(name, 'temporary', path, size_bytes=100)
    report = catalog.cleanup(limit_bytes=200)
    assert report['removed'] == ['one']
    assert catalog.usage()['idle_bytes'] == 200


def test_limits_are_persistent_and_apply_to_copy_budget(tmp_path, monkeypatch):
    catalog = ResourceCatalog(tmp_path)
    catalog.set_limits(10 * 1024**3, 6 * 1024**3)
    assert ResourceCatalog(tmp_path).usage()['cache_limit_bytes'] == 10 * 1024**3
    monkeypatch.setattr('agent.storage.resources.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=5 * 1024**3))
    with container_owner('task:budget', tmp_path), pytest.raises(OSError):
        preflight_space(tmp_path / 'new', 0)


def test_cancel_targets_only_current_owner(tmp_path):
    one, two = Mock(), Mock()
    first = register_active_container('task:a', 'one', one, root=tmp_path)
    second = register_active_container('task:b', 'two', two, root=tmp_path)
    try:
        assert cancel_owned_containers('task:a', tmp_path) == [{'container': 'one', 'stopped': True}]
        one.assert_called_once()
        two.assert_not_called()
    finally:
        unregister_active_container(first)
        unregister_active_container(second)


def test_create_race_kills_actual_container_and_preserves_it(tmp_path, monkeypatch):
    from agent.execution.docker import DockerEnvironment

    repo = tmp_path / 'repo'
    repo.mkdir()
    root = tmp_path / 'state'
    calls = []
    with container_owner('task:race', root):
        environment = DockerEnvironment(repo, 'image', 'owned-race', artifact_root=root / 'logs')

        def docker(args, **kwargs):
            calls.append(args)
            if args[0] == 'run':
                cancel_owned_containers('task:race', root)
                return SimpleNamespace(returncode=0, stdout='a' * 64, stderr='')
            if args[0] == 'inspect':
                return SimpleNamespace(returncode=0, stdout=json.dumps([
                    {'Id': 'a' * 64, 'State': {'Running': False}}]), stderr='')
            return SimpleNamespace(returncode=0, stdout='', stderr='')

        monkeypatch.setattr(environment, '_docker', docker)
        with pytest.raises(InterruptedError):
            environment.start()
        assert ['kill', 'owned-race'] in calls
        assert ['kill', 'a' * 64] in calls
        assert not any(row[0] == 'rm' for row in calls)
        assert environment._execution_unknown
        unregister_active_container(environment._control_key)


def test_docker_cleaner_checks_id_and_never_force_removes_running_container(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stderr='', stdout=json.dumps([
        {'Id': 'a' * 64, 'State': {'Running': True}}
    ])))
    monkeypatch.setattr('subprocess.run', run)
    with pytest.raises(ValueError, match='运行中'):
        delete_owned_docker_resource('container', 'name', {'container_id': 'a' * 64})
    assert run.call_count == 1


def test_snapshot_low_space_leaves_failed_manifest_without_copying(tmp_path, monkeypatch):
    from agent.source.snapshots import SourceSnapshot

    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'source.py').write_text('print(1)')
    monkeypatch.setattr('agent.storage.resources.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=0))
    destination = tmp_path / 'snapshot'
    with pytest.raises(OSError, match='磁盘空间不足'):
        SourceSnapshot.capture(repo, destination)
    assert json.loads((destination / 'manifest.json').read_text())['status'] == 'failed'
    assert not list((destination / 'source').iterdir())
    assert (repo / 'source.py').read_text() == 'print(1)'


def test_environment_recipe_does_not_read_implementation_source(monkeypatch):
    from agent.execution.managed_plan import _recipe

    monkeypatch.setattr(Path, 'read_bytes', Mock(side_effect=AssertionError('no source hashing')))
    assert _recipe() == {'environment_protocol': 3, 'installer': 1, 'download_isolation': 1}


def test_non_git_parent_excludes_multiple_independent_projects_before_copying(tmp_path):
    from agent.source.snapshots import SourceSnapshot

    repo = tmp_path / 'parent'
    for name in ('project-a', 'project-b'):
        folder = repo / name
        (folder / '.git').mkdir(parents=True)
        (folder / 'source.py').write_text('print(1)')
    (repo / 'drawing.svg').write_text('<svg/>')
    destination = tmp_path / 'snapshot'
    snapshot = SourceSnapshot.capture(repo, destination)
    assert set(snapshot.files) == {'drawing.svg'}
    assert {row['path'] for row in json.loads(snapshot.manifest_path.read_text())['excluded']
            if row['reason'] == 'independent_project'} == {'project-a', 'project-b'}
    assert all((repo / name / 'source.py').read_text() == 'print(1)'
               for name in ('project-a', 'project-b'))


def test_non_git_source_excludes_generated_record_signatures_not_generic_names(tmp_path):
    from agent.source.snapshots import SourceSnapshot

    repo = tmp_path / 'project'
    archive = repo / 'renamed-archive'
    archive.mkdir(parents=True)
    (archive / 'history.sqlite3').write_bytes(b'archive')
    (archive / 'state.json').write_text(json.dumps({
        'format': 'task-history-reference', 'database': 'history.sqlite3'}))
    (archive / 'private-copy.py').write_text('history material')
    ordinary = repo / 'validation'
    ordinary.mkdir()
    (ordinary / 'check.py').write_text('print(1)')
    snapshot = SourceSnapshot.capture(repo, tmp_path / 'snapshot')
    assert 'validation/check.py' in snapshot.files
    assert not any(name.startswith('renamed-archive/') for name in snapshot.files)


def test_git_root_keeps_explicitly_tracked_fixture_records(tmp_path):
    import subprocess
    from agent.source.snapshots import SourceSnapshot

    repo = tmp_path / 'project'
    fixture = repo / 'fixtures'
    fixture.mkdir(parents=True)
    (fixture / 'history.sqlite3').write_bytes(b'fixture')
    (fixture / 'state.json').write_text('{}')
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, capture_output=True)
    snapshot = SourceSnapshot.capture(repo, tmp_path / 'snapshot')
    assert 'fixtures/history.sqlite3' in snapshot.files
    assert 'fixtures/state.json' in snapshot.files


def test_git_root_does_not_copy_untracked_product_archive(tmp_path):
    import subprocess
    from agent.source.snapshots import SourceSnapshot

    repo = tmp_path / 'project'
    repo.mkdir()
    (repo / 'main.py').write_text('print(1)')
    subprocess.run(['git', 'init', str(repo)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True, capture_output=True)
    archive = repo / 'saved'
    archive.mkdir()
    (archive / 'history.sqlite3').write_bytes(b'archive')
    (archive / 'state.json').write_text(json.dumps({
        'format': 'task-history-reference', 'database': 'history.sqlite3'}))
    snapshot = SourceSnapshot.capture(repo, tmp_path / 'snapshot')
    assert set(snapshot.files) == {'main.py'}


def test_idle_group_gc_does_not_unpin_unselected_environment_dependencies(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    for name in ('one', 'two'):
        catalog.register(name, 'image', name, size_bytes=100)
        catalog.register(name + '-volume', 'volume', name + '-volume', size_bytes=100,
                         owner='environment:' + name)
    remove = Mock()
    report = catalog.cleanup(limit_bytes=200, docker=remove)
    assert report['removed'] == ['one', 'one-volume']
    rows = {row['id']: row for row in catalog.usage()['resources']}
    assert rows['two-volume']['references'] == 1
    assert set(rows) == {'two', 'two-volume'}


def test_failed_image_delete_keeps_whole_environment_group(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    catalog.register('image', 'image', 'image', size_bytes=100)
    catalog.register('volume', 'volume', 'volume', owner='environment:image', size_bytes=100)
    remove = Mock(side_effect=OSError('in use'))
    report = catalog.cleanup(['image'], docker=remove)
    assert not report['removed']
    assert remove.call_count == 1
    assert {row['id']: row for row in catalog.usage()['resources']}['volume']['references'] == 1


def test_idle_environment_includes_unknown_dependency_in_budget(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    catalog.register('image', 'image', 'image', size_bytes=100)
    catalog.register('volume', 'volume', 'volume', owner='environment:image')
    assert catalog.usage()['idle_unknown_count'] == 1
    remove = Mock()
    assert catalog.cleanup(docker=remove, limit_bytes=1000)['removed'] == ['image', 'volume']


def test_unknown_size_is_explicit_and_paused_task_blocks_environment_cleanup(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    catalog.register('image', 'image', 'image', owner='task:paused')
    usage = catalog.usage()
    assert usage['unknown_size'] and usage['unknown_size_count'] == 1
    assert usage['used_bytes'] == 0
    assert usage['environments'][0]['size_bytes'] is None
    assert usage['environments'][0]['active_task_refs'] == 1
    assert usage['environments'][0]['in_use']
    assert catalog.cleanup(['image'], docker=Mock())['retained'] == ['image']


def test_persistent_cancel_verifies_id_owner_and_image_before_kill(tmp_path, monkeypatch):
    from agent.storage.resources import ownership_labels

    owner, identity = 'task:recovered', 'a' * 64
    labels = ownership_labels(owner, tmp_path)
    catalog = ResourceCatalog(tmp_path)
    catalog.register('container:owned', 'container', 'owned', owner=owner, metadata={
        'container_id': identity, 'image': 'sha256:' + 'b' * 64, 'labels': labels,
    })
    actual = {'Id': identity, 'Name': '/owned', 'Image': 'sha256:' + 'b' * 64,
              'Config': {'Labels': labels}, 'State': {'Running': True}}
    run = Mock(side_effect=[SimpleNamespace(returncode=0, stdout=json.dumps([actual]), stderr=''),
                           SimpleNamespace(returncode=0, stdout='', stderr=''),
                           SimpleNamespace(returncode=0, stdout=json.dumps([
                               {**actual, 'State': {'Running': False}}]), stderr='')])
    monkeypatch.setattr('subprocess.run', run)
    assert cancel_owned_containers(owner, tmp_path)[0]['stopped']
    assert run.call_args_list[-2].args[0][-2:] == ['kill', identity]
    assert run.call_args_list[-1].args[0][-3:] == ['container', 'inspect', identity]
    assert all(call.kwargs['timeout'] == 5 for call in run.call_args_list)


def test_persistent_cancel_rejects_label_mismatch_without_kill(tmp_path, monkeypatch):
    from agent.storage.resources import ownership_labels

    labels = ownership_labels('task:old', tmp_path)
    catalog = ResourceCatalog(tmp_path)
    catalog.register('container:owned', 'container', 'owned', owner='task:new', metadata={
        'container_id': 'a' * 64, 'labels': labels})
    run = Mock(side_effect=AssertionError('must not inspect or kill by guessed identity'))
    monkeypatch.setattr('subprocess.run', run)
    assert not cancel_owned_containers('task:new', tmp_path)[0]['stopped']
    run.assert_not_called()


@pytest.mark.parametrize('recorded_id', ['a' * 64, None])
def test_missing_preparation_container_is_confirmed_and_ledger_is_reconciled(
    tmp_path, monkeypatch, recorded_id
):
    from agent.storage.resources import ownership_labels

    owner, name = 'prepare:missing', 'monoplanner-missing'
    catalog = ResourceCatalog(tmp_path)
    catalog.register('container:' + name, 'container', name, owner=owner, metadata={
        'container_id': recorded_id, 'image': 'sha256:' + 'b' * 64,
        'labels': ownership_labels(owner, tmp_path)})
    query = Mock(return_value=SimpleNamespace(
        returncode=1, stdout='', stderr='Error response from daemon: No such container'))
    monkeypatch.setattr('subprocess.run', query)
    result = cancel_owned_containers(owner, tmp_path)
    assert result[0]['stopped'] and result[0]['resource_state'] == 'missing'
    assert query.call_args.args[0][-3:] == ['container', 'inspect', recorded_id or name]
    assert not catalog.usage()['resources']
    # 同一确认不会在下一次明确重试时再次阻塞或重复查不存在的容器。
    assert cancel_owned_containers(owner, tmp_path) == []
    assert query.call_count == 1


def test_created_container_without_returned_id_is_verified_before_stop(tmp_path, monkeypatch):
    from agent.storage.resources import ownership_labels

    owner, name, identity = 'prepare:created', 'monoplanner-created', 'a' * 64
    labels = ownership_labels(owner, tmp_path)
    catalog = ResourceCatalog(tmp_path)
    catalog.register('container:' + name, 'container', name, owner=owner, metadata={
        'container_id': None, 'image': 'sha256:' + 'b' * 64, 'labels': labels})
    actual = {'Id': identity, 'Name': '/' + name, 'Image': 'sha256:' + 'b' * 64,
              'Config': {'Labels': labels}, 'State': {'Running': True}}
    query = Mock(side_effect=[
        SimpleNamespace(returncode=0, stdout=json.dumps([actual]), stderr=''),
        SimpleNamespace(returncode=0, stdout='', stderr=''),
        SimpleNamespace(returncode=0, stdout=json.dumps([
            {**actual, 'State': {'Running': False}}]), stderr=''),
    ])
    monkeypatch.setattr('subprocess.run', query)
    assert cancel_owned_containers(owner, tmp_path)[0]['stopped']
    assert [call.args[0][1:] for call in query.call_args_list] == [
        ['container', 'inspect', name], ['kill', identity], ['container', 'inspect', identity]]
    saved = catalog.usage()['resources'][0]['metadata']
    assert saved['container_id'] == identity and saved['resource_state'] == 'stopped'


@pytest.mark.parametrize('failure', ['engine', 'timeout', 'running', 'labels', 'image', 'state'])
def test_stop_keeps_unknown_or_changed_container_protected(tmp_path, monkeypatch, failure):
    import subprocess
    from agent.storage.resources import ownership_labels

    owner, name, identity = 'prepare:unknown', 'monoplanner-unknown', 'a' * 64
    labels = ownership_labels(owner, tmp_path)
    catalog = ResourceCatalog(tmp_path)
    catalog.register('container:' + name, 'container', name, owner=owner, metadata={
        'container_id': None, 'image': 'sha256:' + 'b' * 64, 'labels': labels})
    actual = {'Id': identity, 'Name': '/' + name, 'Image': 'sha256:' + 'b' * 64,
              'Config': {'Labels': labels}, 'State': {'Running': False}}
    if failure == 'labels':
        actual['Config']['Labels'] = {'com.monoplanner.owner': 'unrelated'}
    elif failure == 'image':
        actual['Image'] = 'sha256:' + 'c' * 64
    elif failure == 'state':
        actual['State'] = {}
    elif failure == 'running':
        actual['State']['Running'] = True
    if failure == 'timeout':
        query = Mock(side_effect=subprocess.TimeoutExpired('docker', 5))
    elif failure == 'engine':
        query = Mock(return_value=SimpleNamespace(returncode=1, stdout='',
                                                  stderr='Cannot connect to Docker daemon'))
    else:
        query = Mock(return_value=SimpleNamespace(returncode=0,
                                                  stdout=json.dumps([actual]), stderr=''))
    monkeypatch.setattr('subprocess.run', query)
    result = cancel_owned_containers(owner, tmp_path)
    assert not result[0]['stopped']
    saved = catalog.usage()['resources'][0]
    assert saved['references'] == 1 and saved['metadata']['container_id'] is None
    assert saved['status'] == 'ready'
    assert query.call_count == (3 if failure == 'running' else 1)


def test_preparation_pause_does_not_need_disk_to_stop_known_container(tmp_path, monkeypatch):
    from agent.execution.managed import ManagedEnvironments
    from agent.storage.resources import ownership_labels

    owner, name, identity = 'prepare:disk-full', 'monoplanner-disk-full', 'a' * 64
    labels = ownership_labels(owner, tmp_path)
    metadata = {'container_id': identity, 'labels': labels, 'image': 'sha256:' + 'b' * 64}
    actual = {'Id': identity, 'Name': '/' + name, 'Image': metadata['image'],
              'Config': {'Labels': labels}, 'State': {'Running': True}}
    query = Mock(side_effect=[
        SimpleNamespace(returncode=0, stdout=json.dumps([actual]), stderr=''),
        SimpleNamespace(returncode=0, stdout='', stderr=''),
        SimpleNamespace(returncode=0, stdout=json.dumps([
            {**actual, 'State': {'Running': False}}]), stderr=''),
    ])
    monkeypatch.setattr('subprocess.run', query)
    monkeypatch.setattr('agent.storage.resources.resource_catalog',
                        Mock(side_effect=OSError('disk is full')))
    receipt = ManagedEnvironments(tmp_path)._stop_owned_container(name, metadata, owner)
    assert receipt['stopped'] and receipt['resource_state'] == 'stopped'


@pytest.mark.parametrize('state', ['missing', 'unknown', 'ready', 'wrong_id'])
def test_explicit_environment_verification_only_inspects_and_reconciles_confirmed_missing_image(
    tmp_path, monkeypatch, state
):
    from agent.execution.managed import ManagedEnvironments

    image = 'sha256:' + 'a' * 64
    catalog = ResourceCatalog(tmp_path)
    catalog.register('image:' + image, 'image', image, owner='project:one')
    if state in {'missing', 'unknown'}:
        response = SimpleNamespace(returncode=1, stdout='', stderr=(
            'No such image: original' if state == 'missing' else 'Cannot connect to daemon'))
    else:
        response = SimpleNamespace(returncode=0, stdout=json.dumps([{
            'Id': image if state == 'ready' else 'sha256:' + 'b' * 64,
            'Os': 'linux', 'Architecture': 'amd64'}]), stderr='')
    query = Mock(return_value=response)
    monkeypatch.setattr('subprocess.run', query)
    result = ManagedEnvironments(tmp_path).verify({'image': image})
    assert result['state'] == ('unknown' if state == 'wrong_id' else state)
    assert query.call_args.args[0][-3:] == ['image', 'inspect', image]
    assert 0 < query.call_args.kwargs['timeout'] <= 5
    assert bool(catalog.usage()['resources']) is (state != 'missing')
    assert query.call_count == 1
    if state == 'missing':
        query.return_value = SimpleNamespace(returncode=0, stdout=json.dumps([{
            'Id': image, 'Os': 'linux', 'Architecture': 'amd64'}]), stderr='')
        assert ManagedEnvironments(tmp_path).verify({'image': image})['state'] == 'ready'
        assert catalog.usage()['resources'][0]['status'] == 'ready'


def test_missing_existing_dependency_volume_is_not_recreated_during_verification(tmp_path, monkeypatch):
    from agent.execution.managed import ManagedEnvironments
    from agent.execution.docker import managed_dependencies

    image = 'sha256:' + 'a' * 64
    managed = {'node_modules': [{'relative': '.',
                                'image_path': '/opt/monoplanner/node/node_modules'}]}
    item = managed_dependencies(image, managed, root=tmp_path)[0]
    folder = tmp_path / 'managed-volumes' / item['identity']
    folder.mkdir(parents=True)
    original = json.dumps({'status': 'completed', 'image': image, **item})
    (folder / 'completion.json').write_text(original)
    query = Mock(side_effect=[
        SimpleNamespace(returncode=0, stdout=json.dumps([{
            'Id': image, 'Os': 'linux', 'Architecture': 'amd64'}]), stderr=''),
        SimpleNamespace(returncode=1, stdout='', stderr='No such volume: original'),
    ])
    monkeypatch.setattr('subprocess.run', query)
    result = ManagedEnvironments(tmp_path).verify({'image': image, 'managed': managed})
    assert result['state'] == 'missing'
    assert [call.args[0][1:3] for call in query.call_args_list] == [
        ['image', 'inspect'], ['volume', 'inspect']]
    assert (folder / 'completion.json').read_text() == original


def test_builder_cleanup_verifies_volume_labels_and_removes_exact_resources(tmp_path, monkeypatch):
    from agent.storage.resources import ownership_labels

    builder, identity = 'monoplanner-build-owned', 'a' * 64
    name, labels = 'buildx_buildkit_' + builder + '0', ownership_labels('task:x', tmp_path)
    volume = name + '_state'
    metadata = {'container_id': identity, 'labels': labels, 'builder': builder,
                'builder_node': builder + '0', 'builder_volume': volume}
    responses = [
        [{'Id': identity, 'Config': {'Labels': labels}, 'State': {'Running': False}}],
        [{'Labels': {**labels, 'com.monoplanner.builder': builder}}],
    ]
    run = Mock(side_effect=[SimpleNamespace(returncode=0, stdout=json.dumps(value), stderr='')
                           for value in responses] +
               [SimpleNamespace(returncode=0, stdout='', stderr='')] * 3)
    monkeypatch.setattr('subprocess.run', run)
    delete_owned_docker_resource('container', name, metadata)
    calls = [call.args[0][1:] for call in run.call_args_list]
    assert ['buildx', 'rm', '--force', builder] in calls
    assert ['container', 'rm', identity] in calls
    assert ['volume', 'rm', volume] in calls
    assert not any('prune' in command for command in calls)


def test_preparation_services_follow_explicit_resource_owner_root(tmp_path):
    from agent.execution.managed import ManagedEnvironments
    from agent.execution.projects import ProjectEnvironments
    from agent.execution.preparation import jobs, close_jobs

    with container_owner('task:root', tmp_path):
        assert ManagedEnvironments().root == tmp_path / 'managed-environments'
        assert ProjectEnvironments().path.parent == tmp_path
        assert jobs().root == tmp_path
        close_jobs(tmp_path)


def test_single_finish_path_preserves_unknown_and_releases_only_completed_owner(tmp_path):
    catalog = ResourceCatalog(tmp_path)
    catalog.register('container', 'container', 'original', owner='prepare:one')
    remove = Mock()
    result = catalog.finish_resources('prepare:one', ['container'], completed=False, docker=remove)
    assert result['preserved']
    assert catalog.usage()['resources'][0]['references'] == 1
    remove.assert_not_called()
    result = catalog.finish_resources('prepare:one', ['container'], completed=True, docker=remove)
    assert result['removed'] == ['container']
    remove.assert_called_once()


def test_first_dependency_copy_checks_docker_disk_before_creating_volume(tmp_path, monkeypatch):
    from agent.execution.docker import DockerEnvironment

    repo = tmp_path / 'repo'
    repo.mkdir()
    image = 'sha256:' + 'a' * 64
    with container_owner('task:seed-budget', tmp_path / 'state'):
        environment = DockerEnvironment(repo, image, 'seed-budget', artifact_root=tmp_path / 'logs')
    query = Mock(return_value=SimpleNamespace(returncode=0, stderr='', stdout=json.dumps([
        {'Id': image, 'Size': 11}]
    )))
    monkeypatch.setattr(environment, '_docker', query)
    monkeypatch.setattr('agent.storage.disk_space.require_docker_space',
                        lambda _: {'status': 'known', 'location': str(tmp_path)})
    monkeypatch.setattr('agent.storage.resources.shutil.disk_usage',
                        lambda _: SimpleNamespace(free=RESERVE_BYTES + 10))
    with pytest.raises(OSError, match='磁盘空间不足'):
        environment._dependency_space_budget()
    assert [call.args[0] for call in query.call_args_list] == [['image', 'inspect', image]]
