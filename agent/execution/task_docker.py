"""An owned Docker container, with frozen runtime and filtered readonly materials."""
import json
import os
from pathlib import Path
import re
import shutil

from agent.execution.docker import DockerEnvironment
from agent.storage.records import digest, save


class TaskDocker(DockerEnvironment):
    def __init__(self, repo, image, task_root, *, runtime=None, read_sources=()):
        task_root = Path(task_root).resolve()
        super().__init__(repo, image, 'monoplanner-' + task_root.name,
                         artifact_root=task_root.parent.parent/'execution'/task_root.name,
                         runtime=runtime)
        from agent.storage.preferences import state_root, private_path
        from agent.settings import PROJECT_ROOT
        private_roots = [state_root(), task_root]
        if os.getenv('LOCALAPPDATA'):
            private_roots.append((Path(os.environ['LOCALAPPDATA'])/'MonoPlanner').resolve())
        if private_path(self.repo_root) or any(root.is_relative_to(self.repo_root) for root in private_roots):
            raise ValueError('不能将助手私有记录目录挂入执行容器')
        if (PROJECT_ROOT/'.env').is_file() and (PROJECT_ROOT/'.env').is_relative_to(self.repo_root):
            raise ValueError('工作目录包含助手密钥文件，请使用不包含助手密钥的独立源码目录')
        self.binding_path = task_root/'execution-container.json'
        self.runtime_facts_path = task_root/'execution-runtime-facts.json'
        self._runtime_checked = self._probing_runtime = False
        self._runtime_materials(task_root/'execution-runtime-materials.json')
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
                'materials': self.material_mounts, 'runtime_materials_sha256': self.runtime_materials_sha256}

    def _runtime_materials(self, path):
        from agent.execution.runtime_inventory import runtime_inventory
        from agent.source.snapshots import _safe_file
        if path.exists():
            inventory = json.loads(_safe_file(path.parent, Path(path.name)).read_text(encoding='utf-8'))
            if (not isinstance(inventory, dict) or not isinstance(inventory.get('files'), dict)
                    or not isinstance(inventory.get('packages'), list)
                    or inventory.get('sha256') != digest(inventory['files'])):
                raise ValueError('Saved runtime material inventory changed')
        else:
            if self.binding_path.exists():
                raise ValueError('Saved runtime material inventory is missing; no automatic replacement')
            # Reuse the existing package-local inventory. Do not widen this to
            # all ignored files, virtualenvs, build trees or secret material.
            inventory = runtime_inventory(self.repo_root)
            save(path, inventory)
        self.runtime_files = tuple(inventory['files'])
        self.runtime_packages = tuple(inventory['packages'])
        self.runtime_materials_sha256 = digest(inventory)

    def _check_runtime(self, *, first_start=False):
        if self._runtime_checked:
            return
        binding = digest(self._binding(self._container_id))
        if self.runtime_facts_path.exists():
            from agent.source.snapshots import _safe_file
            path = _safe_file(self.runtime_facts_path.parent, Path(self.runtime_facts_path.name))
            record = json.loads(path.read_text(encoding='utf-8'))
            if (record.get('state') != 'completed' or record.get('binding') != binding
                    or record.get('sha256') != digest(record.get('result'))):
                raise ValueError('运行环境检查记录不完整或已变化；不会自动重复探测。')
            self.runtime_facts = record['result']['facts']
            self._runtime_checked = True
            return
        if not first_start:
            raise ValueError('已绑定容器缺少运行环境检查记录；不会自动重复探测。')
        # Reserve before running: an interrupted probe must not silently replay.
        save(self.runtime_facts_path, {'state': 'pending', 'binding': binding})
        from agent.execution.runtime_inventory import probe_runtime
        directory = self.artifact_root/'runtime-probe'
        self._probing_runtime = True
        try:
            report = probe_runtime(self, (), directory, metadata_only=True)
            result = {'facts': self.runtime_facts, 'report': report,
                      'raw': {name: (directory/name).read_text(encoding='utf-8')
                              for name in ('runtime_probe.py', 'stdout.log', 'stderr.log')}}
            save(self.runtime_facts_path, {'state': 'completed' if report['ok'] else 'failed',
                'binding': binding, 'result': result, 'sha256': digest(result)})
        finally:
            self._probing_runtime = False
        if not report['ok']:
            raise ValueError('容器运行环境不可用，请核对 Python 路径和 Conda 环境；不会使用宿主机 Python。')
        self._runtime_checked = True

    def start(self):
        if self._container_id is not None:
            if not self.running:
                raise RuntimeError('Task container stopped; saved operations are not automatically replayed')
            if not self._probing_runtime:
                self._check_runtime()
            return self
        super().start()
        self.running = True
        save(self.binding_path, self._binding(self._container_id))
        self._check_runtime(first_start=True)
        return self
