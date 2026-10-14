"""磁盘检测只读取明确的配置路径；测试不会连接真实 Docker。"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent.storage import disk_space


def test_custom_docker_vhd_location_is_used_without_global_scan(tmp_path, monkeypatch):
    roaming = tmp_path / 'roaming'
    settings = roaming / 'Docker/settings-store.json'
    settings.parent.mkdir(parents=True)
    location = tmp_path / 'docker-data'
    (location / 'disk').mkdir(parents=True)
    (location / 'disk/docker_data.vhdx').write_bytes(b'fixture')
    settings.write_text(json.dumps({'CustomWslDistroDir': str(location)}))
    monkeypatch.setenv('APPDATA', str(roaming))
    monkeypatch.setattr(disk_space.shutil, 'disk_usage', lambda path: SimpleNamespace(free=123))
    result = disk_space.docker_storage_status()
    assert result['status'] == 'known'
    assert result['location'] == str(location / 'disk')
    assert result['free_bytes'] == 123


def test_unknown_docker_location_does_not_claim_zero_or_start_prepare(tmp_path, monkeypatch):
    monkeypatch.setenv('APPDATA', str(tmp_path))
    monkeypatch.setattr(disk_space, 'verify_local_docker_context', lambda: None)
    result = disk_space.docker_storage_status()
    assert result['status'] == 'unknown' and result['free_bytes'] is None
    with pytest.raises(OSError, match='尚未确认'):
        disk_space.require_docker_space(100)


def test_docker_host_disk_reserve_is_checked_separately(monkeypatch):
    monkeypatch.setattr(disk_space.shutil, 'disk_usage', lambda _: SimpleNamespace(free=99))
    with pytest.raises(OSError, match='Docker 数据盘空间不足'):
        disk_space.require_docker_space(100, verify_context=False,
            previous={'status': 'known', 'location': 'X:/Docker'})


def test_remote_context_and_host_override_stop_new_prepare(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout='"tcp://remote:2376"'))
    monkeypatch.setattr(disk_space.subprocess, 'run', run)
    with pytest.raises(OSError, match='本机 Docker'):
        disk_space.verify_local_docker_context()
    assert run.call_args.kwargs['timeout'] == 3
    run.return_value.stdout = '"npipe:////./pipe/dockerDesktopLinuxEngine"'
    monkeypatch.setenv('DOCKER_HOST', 'tcp://remote:2376')
    with pytest.raises(OSError, match='本机 Docker'):
        disk_space.verify_local_docker_context()
