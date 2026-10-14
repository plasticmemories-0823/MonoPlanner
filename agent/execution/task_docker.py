"""管理任务专用 Docker 容器，冻结运行环境并提供过滤后的只读资料。"""

import json
import os
from pathlib import Path
import re
import shutil

from agent.execution.docker import DockerEnvironment, PROXY_VARIABLES
from agent.storage.records import digest, save


# ============================================================
# Docker 挂载路径的规范化
# ============================================================
def inspected_host_path(source):
    """Docker Desktop 使用虚拟机挂载前缀报告 Windows 绑定路径。"""
    if os.name == 'nt' and isinstance(source, str):
        match = re.fullmatch(r'/(?:run/desktop/mnt/host|host_mnt)/([A-Za-z])(?:/(.*))?', source)
        if match:
            relative = match[2] or ''
            if '\\' in relative or any(part in {'.', '..'} for part in relative.split('/')):
                raise ValueError('Unsafe Docker Desktop mount path')
            source = match[1] + ':/' + relative
    return Path(source).resolve()


# ============================================================
# 任务专属环境与只读资料
# ============================================================
class TaskDocker(DockerEnvironment):
    def __init__(
        self, repo, image, task_root, *, runtime=None, read_sources=(), managed=None,
        project_scope=None,
    ):
        task_root = Path(task_root).resolve()
        super().__init__(
            repo,
            image,
            'monoplanner-' + task_root.name,
            artifact_root=task_root.parent.parent / 'execution' / task_root.name,
            runtime=runtime,
            managed=managed,
            project_scope=project_scope,
        )
        from agent.storage.preferences import state_root, private_path
        from agent.settings import PROJECT_ROOT

        private_roots = [state_root(), task_root]
        if os.getenv('LOCALAPPDATA'):
            private_roots.append((Path(os.environ['LOCALAPPDATA']) / 'MonoPlanner').resolve())
        if private_path(self.repo_root) or any(
            root.is_relative_to(self.repo_root) for root in private_roots
        ):
            raise ValueError('不能将助手私有记录目录挂入执行容器')
        if (PROJECT_ROOT / '.env').is_file() and (PROJECT_ROOT / '.env').is_relative_to(
            self.repo_root
        ):
            raise ValueError('工作目录包含助手密钥文件，请使用不包含助手密钥的独立源码目录')
        self.binding_path = task_root / 'execution-container.json'
        from agent.storage.resources import current_owner

        self.resource_owner = current_owner('task:' + task_root.name)
        self.resource_root = task_root.parent.parent
        from agent.storage.resources import resource_catalog

        catalog = resource_catalog(self.resource_root)
        image_resource = 'image:' + image
        with catalog._db() as db:
            known = db.execute('SELECT 1 FROM resources WHERE id=?', (image_resource,)).fetchone()
        if known:
            catalog.link(self.resource_owner, image_resource)
        else:
            # 手动指定的镜像仅引用，不据此取得清理权。
            catalog.register(image_resource, 'image', image, owned=False, owner=self.resource_owner)
        self.runtime_facts_path = task_root / 'execution-runtime-facts.json'
        self._runtime_checked = self._probing_runtime = False
        self._runtime_materials(task_root / 'execution-runtime-materials.json')
        material_record = task_root / 'execution-materials.json'
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
                target = task_root.parent.parent / 'materials' / task_root.name / str(index)
                if target.exists():
                    raise ValueError('资料副本已有未确认记录，不自动覆盖')
                catalog.register(
                    f'materials:{task_root.name}:{index}', 'temporary', target,
                    owner=self.resource_owner, metadata={'task': task_root.name},
                )
                if item['kind'] == 'directory':
                    copied = SourceSnapshot.capture(path, target).root
                else:
                    from agent.storage.resources import preflight_space

                    preflight_space(target, path.stat().st_size)
                    copied = target / path.name
                    copied.parent.mkdir(parents=True)
                    shutil.copy2(path, copied)
                mounts[str(path)] = str(copied)
            self.material_mounts = mounts
            save(material_record, {'sources': sources, 'mounts': mounts})
        self._prepare_project_masks(allow_create=not self.binding_path.exists(), full_check=True)
        if self.binding_path.exists():
            binding = json.loads(self.binding_path.read_text(encoding='utf-8'))
            if binding != self._binding(binding.get('container_id')) or not re.fullmatch(
                r'[0-9a-f]{64}', binding['container_id'] or ''
            ):
                raise ValueError('Saved task container binding changed')
            self._container_id = binding['container_id']
            result = self._docker(
                ['inspect', self._container_id],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            if result.returncode:
                raise RuntimeError('Saved task container is unavailable; no automatic replacement')
            actual = json.loads(result.stdout)[0]
            host = actual['HostConfig']
            expected = {
                '/testbed': ('bind', self.repo_root, True),
                '/agent-artifacts': ('bind', self.artifact_root, True),
            }
            expected.update(
                {
                    f'/agent-materials/{i}': ('bind', Path(p), False)
                    for i, p in enumerate(self.material_mounts.values())
                }
            )
            expected.update(
                {
                    target: ('volume', volume, False)
                    for target, volume in self.dependency_mounts.items()
                }
            )
            expected.update(
                {target: ('bind', Path(source), False)
                 for target, source in self.project_masks.items()}
            )
            caches = [mount for mount in actual['Mounts'] if mount.get('Type') == 'tmpfs']
            mounts = [mount for mount in actual['Mounts'] if mount.get('Type') != 'tmpfs']
            proxies = [
                value
                for value in actual.get('Config', {}).get('Env', [])
                if value.partition('=')[0] in PROXY_VARIABLES and value.partition('=')[2]
            ]
            if (
                actual['Image'] != image
                or actual['Name'] != '/' + self.container_name
                or proxies
                or host['NetworkMode'] != 'none'
                or host.get('Privileged')
                or set(host.get('CapDrop') or []) != {'ALL'}
                or 'no-new-privileges' not in (host.get('SecurityOpt') or [])
                or (host.get('Tmpfs') or {}) != self.dependency_caches
                or any(
                    mount['Destination'] not in self.dependency_caches or not mount.get('RW', False)
                    for mount in caches
                )
                or len(mounts) != len(expected)
                or {m['Destination'] for m in mounts} != set(expected)
                or any(
                    (
                        m.get('Type', 'bind') != expected[m['Destination']][0]
                        or (
                            m.get('Name')
                            if m.get('Type') == 'volume'
                            else inspected_host_path(m['Source'])
                        )
                        != expected[m['Destination']][1]
                        or m.get('RW', True) != expected[m['Destination']][2]
                    )
                    for m in mounts
                )
            ):
                raise ValueError('Task container identity or isolation changed')
            self.running = bool(actual['State']['Running'])
            self._register_control()
        else:
            self.running = False

    # ============================================================
    # 冻结环境绑定与运行材料
    # ============================================================
    def _binding(self, container_id):
        result = {
            'container_id': container_id,
            'image': self.image,
            'runtime': self.runtime,
            'repo': str(self.repo_root),
            'artifacts': str(self.artifact_root),
            'materials': self.material_mounts,
            'runtime_materials_sha256': self.runtime_materials_sha256,
        }
        if self.dependency_mounts:
            result['dependency_mounts'] = self.dependency_mounts
            result['dependency_caches'] = self.dependency_caches
        result['project_scope'] = self.project_scope.to_record()
        result['project_paths'] = self.project_paths
        result['project_masks'] = self.project_masks
        result['project_mask_identity'] = self.project_mask_identity
        return result

    def _runtime_materials(self, path):
        from agent.execution.runtime_inventory import runtime_inventory
        from agent.source.snapshots import _safe_file

        if path.exists():
            inventory = json.loads(
                _safe_file(path.parent, Path(path.name)).read_text(encoding='utf-8')
            )
            if (
                not isinstance(inventory, dict)
                or not isinstance(inventory.get('files'), dict)
                or not isinstance(inventory.get('packages'), list)
                or inventory.get('sha256') != digest(inventory['files'])
            ):
                raise ValueError('Saved runtime material inventory changed')
        else:
            if self.binding_path.exists():
                raise ValueError(
                    'Saved runtime material inventory is missing; no automatic replacement'
                )
            # 复用已有的包内运行材料清单；不能扩展到全部被忽略文件、
            # 虚拟环境、构建目录或秘密资料。
            inventory = runtime_inventory(self.repo_root, project_scope=self.project_scope)
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
            if (
                record.get('state') != 'completed'
                or record.get('binding') != binding
                or record.get('sha256') != digest(record.get('result'))
            ):
                raise ValueError('运行环境检查记录不完整或已变化；不会自动重复探测。')
            self.runtime_facts = record['result']['facts']
            self._runtime_checked = True
            return
        if not first_start:
            raise ValueError('已绑定容器缺少运行环境检查记录；不会自动重复探测。')
        # 先记录预留再执行；调查中断后不能静默重放。
        save(self.runtime_facts_path, {'state': 'pending', 'binding': binding})
        from agent.execution.runtime_inventory import probe_runtime

        directory = self.artifact_root / 'runtime-probe'
        self._probing_runtime = True
        try:
            report = probe_runtime(self, (), directory, metadata_only=True)
            result = {
                'facts': self.runtime_facts,
                'report': report,
                'raw': {
                    name: (directory / name).read_text(encoding='utf-8')
                    for name in ('runtime_probe.py', 'stdout.log', 'stderr.log')
                },
            }
            save(
                self.runtime_facts_path,
                {
                    'state': 'completed' if report['ok'] else 'failed',
                    'binding': binding,
                    'result': result,
                    'sha256': digest(result),
                },
            )
        finally:
            self._probing_runtime = False
        if not report['ok']:
            raise ValueError(
                '容器运行环境不可用，请核对 Python 路径和 Conda 环境；不会使用宿主机 Python。'
            )
        self._runtime_checked = True

    # ============================================================
    # 启动或核验已有容器
    # ============================================================
    def _continuation_container(self):
        result = self._docker(
            ['inspect', self._container_id],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        if result.returncode:
            raise RuntimeError('原任务容器已丢失，不能替换后继续。')
        actual = json.loads(result.stdout)[0]
        if actual.get('Id') != self._container_id or actual['Image'] != self.image:
            raise ValueError('原任务容器身份已变化，不能继续。')
        return actual

    @staticmethod
    def _continuation_identity(actual):
        return digest(
            {
                key: actual.get(key)
                for key in ('Id', 'Image', 'Name', 'Config', 'HostConfig', 'Mounts')
            }
        )

    def output_limit_stop_receipt(self):
        actual = self._continuation_container()
        state = actual['State']
        if state.get('Running') or state.get('OOMKilled') or state.get('Error'):
            raise ValueError('容器未正常停止，保留现场等待核对。')
        return {
            'container_id': self._container_id,
            'identity': self._continuation_identity(actual),
            'state': {key: state.get(key) for key in ('StartedAt', 'FinishedAt', 'ExitCode')},
        }

    def restart_after_output_limit(self, receipt, *, allow_running=False):
        self._prepare_project_masks(allow_create=False, full_check=True)
        actual = self._continuation_container()
        if receipt.get('container_id') != self._container_id or receipt.get(
            'identity'
        ) != self._continuation_identity(actual):
            raise ValueError('原任务容器配置已变化，不能继续。')
        if actual['State'].get('Running'):
            if not allow_running:
                raise ValueError('原任务容器已由其他操作启动，请先核对现场。')
        else:
            current = {
                key: actual['State'].get(key) for key in ('StartedAt', 'FinishedAt', 'ExitCode')
            }
            if (
                current != receipt['state']
                or actual['State'].get('OOMKilled')
                or actual['State'].get('Error')
            ):
                raise ValueError('原任务容器停止状态已变化，不能继续。')
            result = self._docker(
                ['start', self._container_id],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
            )
            if result.returncode:
                raise RuntimeError('无法启动原任务容器，未执行任何任务操作。')
        self.running = True
        self._closed = False
        self.start()

    def start(self):
        self._check_cancelled()
        self._prepare_project_masks(allow_create=self._container_id is None)
        if self._container_id is not None:
            self._prepare_dependencies(allow_create=False)
            if not self.running:
                raise RuntimeError(
                    'Task container stopped; saved operations are not automatically replayed'
                )
            if not self._probing_runtime:
                self._check_runtime()
            return self
        super().start()
        self.running = True
        save(self.binding_path, self._binding(self._container_id))
        self._check_runtime(first_start=True)
        return self
