"""保存项目偏好，并为任务冻结不可变的执行环境绑定。

查询项目只读取配置。自动准备只在首次实际执行，或用户在项目窗口
明确点击准备时启动。
"""

import json
import os
from pathlib import Path
import re
import subprocess
from threading import RLock

from agent.execution.docker import runtime_profile, docker_executable
from agent.storage.preferences import atomic_write, state_root


class EnvironmentRequired(ValueError):
    pass


_preferences_lock = RLock()


def docker_query(arguments):
    try:
        result = subprocess.run(
            [docker_executable(), *arguments],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=20,
            **({'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvironmentRequired(
            'Docker 不可用，请安装并启动 Docker 后重试；不会使用本机执行。'
        ) from exc
    if result.returncode:
        raise EnvironmentRequired('Docker 查询失败：' + result.stderr.strip()[:500])
    return result.stdout


def local_images():
    docker_query(['info', '--format', '{{.OSType}}'])
    images = [
        json.loads(line)
        for line in docker_query(
            ['image', 'ls', '--no-trunc', '--format', '{{json .}}']
        ).splitlines()
        if line.strip()
    ]
    grouped = {}
    for row in images:
        image = grouped.setdefault(row['ID'], {'id': row['ID'], 'names': [], 'size': row['Size']})
        if row['Repository'] != '<none>':
            image['names'].append(row['Repository'] + ':' + row['Tag'])
    return {'images': list(grouped.values())}


def workspace_path(repo):
    path = Path(repo).expanduser()
    if not repo or not path.is_absolute() or not path.is_dir():
        raise ValueError('请选择已有工作目录')
    return str(path.resolve())


class ProjectEnvironments:
    def __init__(self, root=None):
        from agent.storage.resources import current_root

        self.path = current_root(root) / 'project-environments.json'

    def get(self, repo):
        if not repo:
            return {'backend': 'docker'}
        values = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
        return values.get(workspace_path(repo), {'backend': 'docker'})

    def save(self, value):
        if not isinstance(value, dict) or set(value) != {'repo', 'image', 'runtime'}:
            raise ValueError('配置须包含 repo、image 和 runtime')
        repo = workspace_path(value['repo'])
        image = value['image']
        if not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('请选择本机已有镜像的完整 ID')
        runtime = runtime_profile(value['runtime'])
        inspected = json.loads(docker_query(['image', 'inspect', image]))[0]
        if (
            inspected['Id'] != image
            or inspected['Os'] != 'linux'
            or inspected['Architecture'] != 'amd64'
        ):
            raise ValueError('当前执行后端需要本机已有的 Linux amd64 镜像')
        with _preferences_lock:
            values = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
            values[repo] = {'backend': 'docker', 'image': image, 'runtime': runtime}
            atomic_write(
                self.path, json.dumps(values, ensure_ascii=False, indent=2).encode('utf-8')
            )
        return values[repo]

    def automatic(self, repo):
        """只重置后续执行的配置；已有任务检查点保留自己的镜像。"""
        repo = workspace_path(repo)
        with _preferences_lock:
            values = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
            values.pop(repo, None)
            atomic_write(
                self.path, json.dumps(values, ensure_ascii=False, indent=2).encode('utf-8')
            )
        return {'backend': 'docker'}


def workspace_binding(repo, *, root=None):
    bindings = json.loads(os.environ.get('MONOPLANER_EVALUATION_WORKSPACES', '{}'))
    if not isinstance(bindings, dict):
        raise ValueError('Evaluation workspace bindings must be a path/image mapping')
    for path, image in bindings.items():
        if not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('Evaluation images must be pinned sha256 IDs')
        if repo and Path(path).resolve() == Path(repo).resolve():
            return {'backend': 'docker', 'image': image, 'runtime': runtime_profile()}
    return ProjectEnvironments(root).get(repo)


def require_binding(repo, binding=None):
    binding = binding or workspace_binding(repo)
    if binding.get('backend') != 'docker' or not binding.get('image'):
        raise EnvironmentRequired('请在工作目录窗口配置 Docker 项目镜像和运行环境；操作尚未执行。')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', binding['image']):
        raise EnvironmentRequired('项目镜像 ID 无效，请重新配置；操作尚未执行。')
    return {**binding, 'runtime': runtime_profile(binding.get('runtime'))}


def prepare_binding(repo, *, progress=None, cancelled=None):
    """仅在首次执行前读取最新的项目偏好。"""
    selected = workspace_binding(repo)
    if selected.get('image'):
        return require_binding(repo, selected)
    from agent.execution.preparation import ensure_prepared

    return require_binding(repo, ensure_prepared(repo, progress=progress, cancelled=cancelled))
