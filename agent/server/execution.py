"""Trusted workspace bindings for formal evaluation; normal chats use LPAC."""
import json
import os
from pathlib import Path
import re

from agent.execution.docker import DockerEnvironment
from agent.storage.records import save


def workspace_binding(repo):
    """Only the server operator, never a model argument, selects an image."""
    bindings = json.loads(os.environ.get('MONOPLANER_EVALUATION_WORKSPACES', '{}'))
    if not isinstance(bindings, dict):
        raise ValueError('Evaluation workspace bindings must be a path/image mapping')
    selected = None
    for path, image in bindings.items():
        if not isinstance(image, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image):
            raise ValueError('Evaluation images must be pinned sha256 IDs')
        if repo and Path(path).resolve() == Path(repo).resolve():
            selected = {'backend': 'docker', 'image': image}
    return selected or {'backend': 'native'}


class ConversationDocker(DockerEnvironment):
    """Reconnect the same owned container between graph nodes; never replay."""
    def __init__(self, repo, image, task_root):
        task_root = Path(task_root)
        super().__init__(repo, image, 'monoplanner-' + task_root.name,
                         artifact_root=task_root.parent.parent/'execution'/task_root.name)
        self.binding_path = task_root/'execution-container.json'
        if self.binding_path.exists():
            binding = json.loads(self.binding_path.read_text(encoding='utf-8'))
            expected = self._binding(binding.get('container_id'))
            if binding != expected or not re.fullmatch(r'[0-9a-f]{64}', binding['container_id'] or ''):
                raise ValueError('Saved evaluation container binding changed')
            self._container_id = binding['container_id']
            result = self._docker(['inspect', self._container_id], capture_output=True,
                                  text=True, encoding='utf-8', errors='replace')
            if result.returncode:
                raise RuntimeError('Saved evaluation container is unavailable; no automatic replacement')
            actual = json.loads(result.stdout)[0]
            mounts = actual['Mounts']
            host = actual['HostConfig']
            if (actual['Image'] != image or actual['Name'] != '/' + self.container_name
                    or host['NetworkMode'] != 'none' or host.get('Privileged')
                    or set(host.get('CapDrop') or []) != {'ALL'}
                    or 'no-new-privileges' not in (host.get('SecurityOpt') or [])
                    or len(mounts) != 2
                    or {m['Destination'] for m in mounts} != {'/testbed', '/agent-artifacts'}
                    or any(Path(m['Source']).resolve() != (self.repo_root if m['Destination'] == '/testbed'
                                                         else self.artifact_root) for m in mounts)):
                raise ValueError('Evaluation container identity or isolation changed')
            self.running = bool(actual['State']['Running'])
        else:
            self.running = False

    def _binding(self, container_id):
        return {'container_id': container_id, 'image': self.image,
                'repo': str(self.repo_root), 'artifacts': str(self.artifact_root)}

    def start(self):
        if self._container_id is not None:
            if not self.running:
                raise RuntimeError('Evaluation container stopped; saved operations are not automatically replayed')
            return self
        # A container left by an uncertain launch makes Docker reject the same
        # name. It is never silently adopted or replaced.
        super().start()
        self.running = True
        save(self.binding_path, self._binding(self._container_id))
        return self
