"""识别 Docker Desktop 的宿主磁盘；不扫描全盘、不统计用户镜像。"""

import json
import os
from pathlib import Path
import shutil
import subprocess


def docker_storage_status():
    unknown = {'status': 'unknown', 'free_bytes': None, 'location': None,
               'reason': '尚未确认 Docker 数据盘，不能保证准备期间的磁盘余量。'}
    if os.name != 'nt':
        return unknown
    roaming, local = os.getenv('APPDATA'), os.getenv('LOCALAPPDATA')
    settings = None
    for filename in ('settings-store.json', 'settings.json'):
        if not roaming:
            break
        path = Path(roaming) / 'Docker' / filename
        try:
            if path.is_file():
                if path.stat().st_size > 1024 * 1024:
                    return unknown
                settings = json.loads(path.read_text(encoding='utf-8-sig'))
                break
        except (OSError, ValueError):
            return unknown
    if not isinstance(settings, dict):
        return unknown
    configured = next((settings[key] for key in (
        'CustomWslDistroDir', 'dataFolder', 'DataFolder', 'diskImageLocation',
    ) if isinstance(settings.get(key), str) and settings[key]), None)
    if not configured:
        if not local:
            return unknown
        configured = str(Path(local) / 'Docker' / 'wsl')
    if configured.startswith('\\\\'):
        return unknown
    directory = Path(configured).expanduser()
    candidates = ([directory] if directory.suffix.lower() == '.vhdx' else [
        directory / name for name in ('docker_data.vhdx', 'ext4.vhdx', 'DockerDesktop.vhdx',
                                     'disk/docker_data.vhdx', 'data/docker_data.vhdx',
                                     'data/ext4.vhdx')
    ])
    try:
        files = [path for path in candidates if path.is_file() and not path.is_symlink()]
        if len(files) != 1:
            return unknown
        disk = files[0].resolve()
        return {'status': 'known', 'location': str(disk.parent),
                'free_bytes': shutil.disk_usage(disk.parent).free,
                'scope': 'Docker 虚拟磁盘所在的 Windows 文件系统；不是虚拟磁盘内剩余容量'}
    except OSError:
        return unknown


def verify_local_docker_context():
    """新准备前有界核验当前 Docker 指向本机，不把远端存储当作本机空间。"""
    from agent.execution.docker import docker_executable

    try:
        result = subprocess.run(
            [docker_executable(), 'context', 'inspect', '--format', '{{json .Endpoints.docker.Host}}'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3,
            **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}),
        )
        endpoint = json.loads(result.stdout) if result.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        endpoint = None
    allowed = {'npipe:////./pipe/dockerdesktoplinuxengine', 'npipe:////./pipe/docker_engine',
               'npipe://./pipe/dockerdesktoplinuxengine', 'npipe://./pipe/docker_engine'}
    override = os.getenv('DOCKER_HOST')
    if (not isinstance(endpoint, str) or endpoint.lower().replace('\\', '/') not in allowed
            or (override and override.lower().replace('\\', '/') not in allowed)):
        raise OSError('无法确认本机 Docker 数据盘；环境准备尚未开始，请检查 Docker 连接。')


def require_docker_space(reserve_bytes, *, previous=None, verify_context=True):
    if verify_context:
        verify_local_docker_context()
    value = previous or docker_storage_status()
    if value['status'] != 'known':
        raise OSError(value['reason'])
    free = shutil.disk_usage(value['location']).free
    if free < reserve_bytes:
        raise OSError(f'Docker 数据盘空间不足：需保留 {reserve_bytes:,} 字节，当前可用 {free:,} 字节。')
    return {**value, 'free_bytes': free, 'reserve_bytes': reserve_bytes}
