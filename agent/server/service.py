"""Identify the loaded local service without reading credentials or task data."""
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]


def service_identity(root=PROJECT):
    root = Path(root).resolve()
    files = [*root.joinpath('agent').rglob('*.py'),
             *(p for p in root.joinpath('resources').rglob('*') if p.is_file()),
             root / 'langgraph.json', root / 'requirements-web.txt']
    digest = hashlib.sha256()
    for file in sorted(files):
        if '__pycache__' not in file.parts and file.is_file():
            digest.update(file.relative_to(root).as_posix().encode())
            digest.update(b'\0' + file.read_bytes() + b'\0')
    return {'product': 'MonoPlanner', 'project': str(root),
            'source_hash': digest.hexdigest()}


if __name__ == '__main__':
    print(json.dumps(service_identity(), ensure_ascii=False))
