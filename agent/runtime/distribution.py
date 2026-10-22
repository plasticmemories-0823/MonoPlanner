"""便携包与任务内核共用的生产 Python 文件清单。"""

import csv
from email.parser import Parser
from pathlib import Path
import re


def tree_files(root):
    root = Path(root)
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or path.is_junction():
            raise ValueError('Runtime distribution cannot contain links: ' + str(path))
        if (path.is_file() and '__pycache__' not in path.parts
                and path.suffix != '.pyc' and not path.name.startswith('.env')):
            yield path


def python_files(source, lock):
    """从锁定的 dist-info/RECORD 收集生产依赖，不带开发工具和安装器。"""
    source = Path(source).resolve()
    packages = source / 'Lib/site-packages'
    normalize = lambda value: re.sub(r'[-_.]+', '-', value).lower()
    locked = {}
    for line in Path(lock).read_text('utf-8').splitlines():
        match = re.match(r'([\w.-]+)(?:\[[^]]+\])?==([^ ;]+)', line)
        if match:
            locked[normalize(match[1])] = match[2]
    selected = {
        path for path in tree_files(source)
        if not path.is_relative_to(packages) and not path.is_relative_to(source / 'Scripts')
    }
    found = set()
    for metadata in sorted(packages.glob('*.dist-info/METADATA')):
        value = Parser().parsestr(metadata.read_text('utf-8'))
        name = normalize(value['Name'])
        if name not in locked:
            continue
        if value['Version'] != locked[name]:
            raise ValueError('Runtime version differs from lock: ' + name)
        found.add(name)
        with (metadata.parent / 'RECORD').open(encoding='utf-8', newline='') as stream:
            for relative, *_ in csv.reader(stream):
                path = packages / relative
                if (not path.resolve().is_relative_to(packages) or path.suffix == '.pyc'
                        or path.name == 'direct_url.json'):
                    continue
                if not path.is_file() or path.is_symlink():
                    raise ValueError('Production runtime file missing or linked: ' + relative)
                selected.add(path.resolve())
    if not locked or found != set(locked):
        raise ValueError('Production dependencies missing: ' + ', '.join(sorted(set(locked) - found)))
    return sorted(selected), found
