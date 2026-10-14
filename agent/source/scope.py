"""冻结选定项目的边界；嵌套独立 Git 项目不属于本次工作。"""

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import time


MAX_SCOPE_ENTRIES = 200_000
MAX_SCOPE_SECONDS = 10


class ProjectScopeError(ValueError):
    """项目边界不明确或已变化，不能继续沿用旧范围。"""


def _link(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, 'st_file_attributes', 0) & 0x400
    )


def _root(value):
    path = Path(value).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise ProjectScopeError('项目范围必须绑定已有绝对目录。')
    if any(_link(part) for part in (path, *path.parents)):
        raise ProjectScopeError('项目范围不能经过链接或 Windows 重解析点。')
    return path.resolve()


def _relative(value):
    text = value.as_posix() if isinstance(value, Path) else str(value)
    path = Path(text)
    if (path.is_absolute() or '..' in path.parts or ':' in text
            or '\\' in text):
        raise ProjectScopeError('项目范围须使用根目录内的相对路径。')
    return path


@dataclass(frozen=True)
class ProjectScope:
    root: Path
    excluded_projects: tuple[str, ...] = ()

    @classmethod
    def discover(cls, root, cancel=None):
        root = _root(root)
        excluded, pending, seen = [], [root], 0
        deadline = time.monotonic() + MAX_SCOPE_SECONDS
        while pending:
            from agent.storage.resources import owner_cancelled

            if owner_cancelled() or (cancel is not None and cancel.is_set()):
                raise InterruptedError('项目范围检查已中断。')
            if time.monotonic() > deadline:
                raise ProjectScopeError('项目范围检查超过时限，请选择实际项目或新建独立子目录。')
            parent = pending.pop()
            # 发现独立仓库后立即剪枝，不读取其文件或继续寻找内部项目。
            marker = parent / '.git'
            if parent != root and (marker.exists() or marker.is_symlink()):
                excluded.append(parent.relative_to(root).as_posix())
                continue
            with os.scandir(parent) as entries:
                for entry in entries:
                    seen += 1
                    if seen % 128 == 0:
                        if owner_cancelled() or (cancel is not None and cancel.is_set()):
                            raise InterruptedError('项目范围检查已中断。')
                        if time.monotonic() > deadline:
                            raise ProjectScopeError('项目范围检查超过时限，请选择实际项目或新建独立子目录。')
                    if seen > MAX_SCOPE_ENTRIES:
                        raise ProjectScopeError('目录范围过大，请选择实际项目或新建独立子目录。')
                    if entry.name == '.git' or not entry.is_dir(follow_symlinks=False):
                        continue
                    path = Path(entry.path)
                    # 链接由源码/安装的访问边界拒绝，这里绝不跟随并扩大扫描。
                    if not _link(path):
                        pending.append(path)
        return cls(root, tuple(sorted(excluded)))

    @classmethod
    def from_record(cls, record, root=None):
        if (not isinstance(record, dict)
                or set(record) != {'version', 'root', 'excluded_projects'}
                or type(record['version']) is not int or record['version'] != 1
                or not isinstance(record['root'], str)
                or not isinstance(record['excluded_projects'], list)
                or any(not isinstance(item, str) for item in record['excluded_projects'])):
            raise ProjectScopeError('项目范围记录不完整。')
        origin = _root(record['root'])
        if root is not None and origin != _root(root):
            raise ProjectScopeError('任务项目根目录已变化。')
        names = record['excluded_projects']
        paths = [_relative(item) for item in names]
        if (any(not path.parts for path in paths) or names != sorted(set(names))
                or any(path.as_posix() != name
                for name, path in zip(names, paths))
                or any(left != right and left.is_relative_to(right)
                       for left in paths for right in paths)):
            raise ProjectScopeError('项目排除目录必须唯一且互不包含。')
        return cls(origin, tuple(names))

    def to_record(self):
        return {'version': 1, 'root': str(self.root),
                'excluded_projects': list(self.excluded_projects)}

    def validate(self, cancel=None, *, full_check=False):
        if _root(self.root) != self.root:
            raise ProjectScopeError('任务项目根目录已变化。')
        for name in self.excluded_projects:
            path = self.root
            for part in Path(name).parts:
                path /= part
                if not path.is_dir() or _link(path):
                    raise ProjectScopeError('排除项目目录已变化，已停止；请核对后新建任务。')
            if not ((path / '.git').exists() or (path / '.git').is_symlink()):
                raise ProjectScopeError('排除项目标记已变化，已停止；请核对后新建任务。')
        if full_check and self != self.discover(self.root, cancel=cancel):
            raise ProjectScopeError('独立项目范围已变化，已停止；请核对后新建任务。')

    def excludes(self, relative):
        path = _relative(relative)
        return any(path.is_relative_to(Path(name)) for name in self.excluded_projects)

    def require(self, relative):
        if self.excludes(relative):
            raise ProjectScopeError('引用指向已排除的独立项目：' + str(relative))

    def glob_directories(self, pattern, excluded_patterns=()):
        """匹配声明的目录，不先遍历被排除项目再过滤结果。"""
        from fnmatch import fnmatchcase

        parts = _relative(Path(pattern)).parts
        pending, visited = [(self.root, 0)], set()
        count, deadline = 0, time.monotonic() + MAX_SCOPE_SECONDS
        while pending:
            from agent.storage.resources import owner_cancelled

            if owner_cancelled():
                raise InterruptedError('项目工作区检查已中断。')
            if time.monotonic() > deadline:
                raise ProjectScopeError('工作区模式范围过大，请使用更明确的项目路径。')
            folder, index = pending.pop()
            if (folder, index) in visited:
                continue
            visited.add((folder, index))
            if index == len(parts):
                yield folder
                continue
            part = parts[index]
            if part == '**':
                pending.append((folder, index + 1))
            with os.scandir(folder) as entries:
                for entry in entries:
                    count += 1
                    if count > MAX_SCOPE_ENTRIES or time.monotonic() > deadline:
                        raise ProjectScopeError('工作区模式范围过大，请使用更明确的项目路径。')
                    if entry.name == '.git' or not (part == '**' or fnmatchcase(
                            os.path.normcase(entry.name), os.path.normcase(part))):
                        continue
                    path = Path(entry.path)
                    if entry.is_symlink() or _link(path):
                        raise ProjectScopeError('工作区模式不能经过链接：' + str(path.relative_to(self.root)))
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    relative = path.relative_to(self.root)
                    if any(fnmatchcase(relative.as_posix(), rule) for rule in excluded_patterns):
                        continue
                    self.require(relative)
                    if (path / '.git').exists():
                        raise ProjectScopeError('工作区检查期间出现新的独立项目，请核对后重新准备。')
                    pending.append((path, index if part == '**' else index + 1))


def project_scope(root, scope=None, *, cancel=None):
    """首次发现或核验冻结范围，供源码、安装材料与执行环境共同使用。"""
    if scope is None:
        return ProjectScope.discover(root, cancel=cancel)
    if isinstance(scope, dict):
        scope = ProjectScope.from_record(scope, root=root)
    if not isinstance(scope, ProjectScope) or scope.root != _root(root):
        raise ProjectScopeError('任务项目根目录已变化。')
    scope.validate(cancel=cancel)
    return scope
