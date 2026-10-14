"""为工作目录选择器列出当前层的本地目录，不读取文件内容。"""

import os
from pathlib import Path


def directory_page(value=''):
    if not value:
        drives = os.listdrives() if hasattr(os, 'listdrives') else [Path.home().anchor]
        roots = [Path.home(), *(Path(d) for d in drives)]
        return {
            'path': '',
            'parent': None,
            'directories': [
                {'name': str(p), 'path': str(p)} for p in dict.fromkeys(roots) if p.is_dir()
            ],
        }
    folder = Path(value).expanduser()
    if not folder.is_absolute():
        raise ValueError('请输入完整的目录路径')
    folder = folder.resolve(strict=True)
    if not folder.is_dir():
        raise ValueError('请选择一个已有目录')
    directories = []
    with os.scandir(folder) as entries:
        for entry in entries:
            try:
                if entry.is_dir():
                    directories.append({'name': entry.name, 'path': str(folder / entry.name)})
            except OSError:
                continue
    return {
        'path': str(folder),
        'parent': str(folder.parent) if folder.parent != folder else '',
        'directories': sorted(directories, key=lambda d: d['name'].casefold()),
    }
