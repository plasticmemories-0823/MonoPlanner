"""识别已加载的本地服务，不读取凭据或任务数据。"""

import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
import sys

if not __package__:
    sys.path.insert(0, str(PROJECT))
from agent.version import VERSION


def service_identity(root=PROJECT):
    root = Path(root).resolve()
    files = [
        *root.joinpath('agent').rglob('*.py'),
        *(p for p in root.joinpath('resources').rglob('*') if p.is_file()),
        root / 'langgraph.json',
        root / 'requirements-web.txt',
    ]
    digest = hashlib.sha256()
    for file in sorted(files):
        if '__pycache__' not in file.parts and file.is_file():
            digest.update(file.relative_to(root).as_posix().encode())
            digest.update(b'\0' + file.read_bytes() + b'\0')
    return {
        'product': 'MonoPlanner',
        'version': VERSION,
        'project_id': hashlib.sha256(str(root).lower().encode()).hexdigest(),
        'source_hash': digest.hexdigest(),
    }


if __name__ == '__main__':
    print(json.dumps(service_identity(), ensure_ascii=False))
