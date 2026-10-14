"""在开工边界确定项目位置；目录选择本身不创建文件或准备环境。"""

from pathlib import Path, PureWindowsPath
import json
import re

from agent.source.artifacts import generated_records
from agent.source.snapshots import _link
from agent.storage.records import save


class ProjectLocationError(ValueError):
    """位置未确认或发生变化，需要用户核对后重新交接。"""


def _identity(path):
    if _link(path) or not path.is_dir():
        raise ProjectLocationError('项目路径不能是链接或普通文件。')
    info = path.stat()
    return [info.st_dev, info.st_ino]


def _target(workspace, relative):
    if not isinstance(relative, str) or not relative or len(relative) > 1024:
        raise ProjectLocationError('请提供相对于所选工作区的项目目录。')
    name = relative.replace('\\', '/')
    if name != '.':
        if PureWindowsPath(name).drive or name.startswith('/'):
            raise ProjectLocationError('项目位置须在已选择的工作区内，不能使用绝对路径。')
        for part in name.split('/'):
            if (not part or part in {'.', '..'} or part.casefold() == '.git' or part != part.rstrip(' .')
                    or any(ord(c) < 32 or c in '<>:"|?*' for c in part)
                    or re.fullmatch(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',
                                    part, re.IGNORECASE)):
                raise ProjectLocationError('项目目录名称无效；不能越出所选工作区。')
    root = Path(workspace)
    if not root.is_absolute() or not root.is_dir():
        raise ProjectLocationError('请先选择已有工作区。')
    _identity(root)
    target = root if name == '.' else root.joinpath(*name.split('/'))
    for parent in reversed(target.parents):
        if parent.is_relative_to(root):
            _identity(parent)
    if target.exists() or target.is_symlink():
        _identity(target)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ProjectLocationError('项目位置越出了已选择的工作区。')
    if target.exists() and generated_records(target):
        raise ProjectLocationError('该目录是程序运行记录，请另选项目目录。')
    return root, target


def resolve_location(workspace, project_path='.', create=False):
    """只核对路径并生成交接记录；这里不创建目录。"""
    if type(create) is not bool:
        raise ProjectLocationError('create 必须是布尔值。')
    root, target = _target(workspace, project_path)
    if create and (target == root or target.exists()):
        raise ProjectLocationError('新项目目录已经存在；请确认复用已有目录或选择另一个名称。')
    if not create and not target.is_dir():
        raise ProjectLocationError('已有项目目录不存在；新项目请明确使用 create=true。')
    return {
        'workspace': str(root), 'path': target.relative_to(root).as_posix(),
        'repo': str(target), 'create': create, 'workspace_identity': _identity(root),
        'parent_identity': _identity(target.parent if target != root else root),
        'target_identity': _identity(target) if target.exists() else None,
    }


def ensure_location(location, task_root):
    """只在已确认工作开始后创建目录；不覆盖重名，不重放结果未知的创建。"""
    root, target = _target(location['workspace'], location['path'])
    if (str(target) != location['repo'] or _identity(root) != location['workspace_identity']
            or _identity(target.parent if target != root else root) != location['parent_identity']):
        raise ProjectLocationError('已确认的项目位置发生变化；未创建或切换工作目录。')
    if not location['create']:
        if not target.exists() or _identity(target) != location['target_identity']:
            raise ProjectLocationError('已确认的项目目录被移动或替换，请重新确认。')
        return target
    intent = Path(task_root) / 'project-create-intent.json'
    receipt = Path(task_root) / 'project-location.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text('utf-8'))
        if saved['location'] != location or _identity(target) != saved['target_identity']:
            raise ProjectLocationError('新建项目的已保存位置不一致，未自动重建。')
        return target
    if intent.exists():
        raise ProjectLocationError('上次创建目录的结果未确认，现场保留；不会重复创建或覆盖。')
    if target.exists():
        raise ProjectLocationError('项目目录已经出现，未覆盖；请确认后再使用。')
    save(intent, {'location': location})
    target.mkdir(exist_ok=False)
    _target(location['workspace'], location['path'])
    save(receipt, {'location': location, 'target_identity': _identity(target)})
    return target
