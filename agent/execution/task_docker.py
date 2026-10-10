"""An owned Docker container, with frozen runtime and filtered readonly materials."""
import json
from pathlib import Path
import re
import shutil

from agent.execution.docker import DockerEnvironment
from agent.storage.records import save


class TaskDocker(DockerEnvironment):
    def __init__(self, repo, image, task_root, *, runtime=None, read_sources=()):
        task_root = Path(task_root).resolve()
        super().__init__(repo, image, 'monoplanner-' + task_root.name,
                         artifact_root=task_root.parent.parent/'execution'/task_root.name,
                         runtime=runtime)
        from agent.storage.preferences import state_root
        from agent.settings import PROJECT_ROOT
        if state_root().is_relative_to(self.repo_root) or task_root.is_relative_to(self.repo_root):
            raise ValueError('不能将助手私有记录目录挂入执行容器')
        if (PROJECT_ROOT/'.env').is_file() and (PROJECT_ROOT/'.env').is_relative_to(self.repo_root):
            raise ValueError('工作目录包含助手密钥文件，请使用不包含助手密钥的独立源码目录')
        self.binding_path = task_root/'execution-container.json'
        material_record = task_root/'execution-materials.json'
        from agent.execution.assistant_files import authorized_sources
        sources = authorized_sources(list(read_sources))
        if material_record.exists():
            record = json.loads(material_record.read_text(encoding='utf-8'))
            if record['sources'] != sources:
                raise ValueError('执行资料授权已变化，请新建任务；不会扩大现有容器权限。')
            self.material_mounts = record['mounts']
        elif sources:
            from agent.source.snapshots import SourceSnapshot
            mounts = {}
            for index, item in enumerate(sources):
                path = Path(item['path'])
                target = task_root.parent.parent/'materials'/task_root.name/str(index)
                if target.exists():
                    raise ValueError('资料副本已有未确认记录，不自动覆盖')
                if item['kind'] == 'directory':
                    copied = SourceSnapshot.capture(path, target).root
                else:
                    copied = target/path.name
                    copied.parent.mkdir(parents=True)
                    shutil.copy2(path, copied)
                mounts[str(path)] = str(copied)
            self.material_mounts = mounts
            save(material_record, {'sources': sources, 'mounts': mounts})
        if self.binding_path.exists():
            binding = json.loads(self.binding_path.read_text(encoding='utf-8'))
            if binding != self._binding(binding.get('container_id')) or not re.fullmatch(r'[0-9a-f]{64}', binding['container_id'] or ''):
                raise ValueError('Saved task container binding changed')
            self._container_id = binding['container_id']
            result = self._docker(['inspect', self._container_id], capture_output=True,
                                  text=True, encoding='utf-8', errors='replace')
            if result.returncode:
                raise RuntimeError('Saved task container is unavailable; no automatic replacement')
            actual = json.loads(result.stdout)[0]
            host = actual['HostConfig']
            expected = {'/testbed': (self.repo_root, True), '/agent-artifacts': (self.artifact_root, True)}
            expected.update({f'/agent-materials/{i}': (Path(p), False) for i, p in enumerate(self.material_mounts.values())})
            mounts = actual['Mounts']
            if (actual['Image'] != image or actual['Name'] != '/' + self.container_name
                    or host['NetworkMode'] != 'none' or host.get('Privileged')
                    or set(host.get('CapDrop') or []) != {'ALL'}
                    or 'no-new-privileges' not in (host.get('SecurityOpt') or [])
                    or len(mounts) != len(expected)
                    or {m['Destination'] for m in mounts} != set(expected)
                    or any(Path(m['Source']).resolve() != expected[m['Destination']][0]
                           or m.get('RW', True) != expected[m['Destination']][1] for m in mounts)):
                raise ValueError('Task container identity or isolation changed')
            self.running = bool(actual['State']['Running'])
        else:
            self.running = False

    def _binding(self, container_id):
        return {'container_id': container_id, 'image': self.image, 'runtime': self.runtime,
                'repo': str(self.repo_root), 'artifacts': str(self.artifact_root),
                'materials': self.material_mounts}

    def _check_runtime(self):
        command = self._activated_argv([self.python_executable, '--version'], False)
        result = self._docker(['exec', self._container_id, *command], capture_output=True,
                             text=True, encoding='utf-8', errors='replace')
        if result.returncode:
            save(self.binding_path.parent/'execution-runtime-error.json', {'stdout': result.stdout, 'stderr': result.stderr})
            raise ValueError('容器运行环境不可用，请核对 Python 路径和 Conda 环境；不会使用宿主机 Python。')

    def start(self):
        if self._container_id is not None:
            if not self.running:
                raise RuntimeError('Task container stopped; saved operations are not automatically replayed')
            if (self.binding_path.parent/'execution-runtime-error.json').exists():
                raise ValueError('此任务运行环境检查已失败，请修正项目配置后新建任务')
            return self
        super().start()
        self.running = True
        save(self.binding_path, self._binding(self._container_id))
        self._check_runtime()
        return self
