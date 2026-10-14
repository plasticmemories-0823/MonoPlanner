"""按安装需要收集资料；工作目录授权不等于整目录安装输入。"""

import hashlib
import os
from pathlib import Path
import subprocess

from pathspec import GitIgnoreSpec

from agent.execution.managed_plan import (
    MAX_BYTES,
    MAX_FILES,
    MAX_MANIFEST,
    PRIVATE_MARKERS,
    SECRET_ASSIGNMENT,
    SKIP,
    PreparationError,
    _link,
    excluded,
    safe_path,
)
from agent.storage.preferences import private_path
from agent.source.artifacts import generated_records, generated_archive
from agent.source.scope import project_scope as resolve_scope


def project_files(root, under=Path('.'), *, project_scope=None):
    """应用每层 .gitignore；明确声明的依赖由调用方单独加入。"""
    under = Path(under)
    scope = resolve_scope(root, project_scope)
    scope.require(under)
    start = safe_path(root, under, file=False)
    tracked, selected = set(), None
    options = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    git = ['git', '-c', 'core.fsmonitor=false', '-C', str(root)]
    try:
        probe = subprocess.run(
            [*git, 'rev-parse', '--show-toplevel'], capture_output=True, timeout=10, **options
        )
        git_root = Path(os.fsdecode(probe.stdout.strip())).resolve() if probe.returncode == 0 else None
        if git_root is not None and root.is_relative_to(git_root):
            listed = subprocess.run(
                [*git, 'ls-files', '--cached', '-z', '--', under.as_posix()],
                capture_output=True,
                timeout=30,
                **options,
            )
            if listed.returncode:
                raise PreparationError('无法读取项目版本控制清单，未开始准备。')
            tracked = {Path(os.fsdecode(name)) for name in listed.stdout.split(b'\0') if name}
            ignored_root = subprocess.run([*git, 'check-ignore', '-q', '--', under.as_posix()],
                                          capture_output=True, timeout=10, **options)
            if ignored_root.returncode not in {0, 1}:
                raise PreparationError('无法核对安装资料的 Git 忽略规则。')
            if ignored_root.returncode == 1:
                listed = subprocess.run(
                    [*git, 'ls-files', '--cached', '--others', '--exclude-standard',
                     '-z', '--', under.as_posix()],
                    capture_output=True, timeout=30, **options)
                if listed.returncode:
                    raise PreparationError('无法读取项目版本控制清单，未开始准备。')
                selected = {Path(os.fsdecode(name)) for name in listed.stdout.split(b'\0') if name}
    except FileNotFoundError:
        pass
    inherited = []
    for folder in [root, *list(start.parents)[::-1], start]:
        if not folder.is_relative_to(root) or not start.is_relative_to(folder):
            continue
        if any(base == folder for base, _ in inherited):
            continue
        ignore = folder / '.gitignore'
        if ignore.is_file():
            safe_path(root, ignore.relative_to(root))
            if ignore.stat().st_size > MAX_MANIFEST:
                raise PreparationError('项目忽略文件过大，无法确定安装范围。')
            inherited.append(
                (folder, GitIgnoreSpec.from_lines(ignore.read_text('utf-8-sig').splitlines()))
            )

    def ignored(path, rules, directory=False):
        relative = path.relative_to(root)
        if relative in tracked or (
            directory and any(name.is_relative_to(relative) for name in tracked)
        ):
            return False
        if selected is not None:
            return (not any(name.is_relative_to(relative) for name in selected)
                    if directory else relative not in selected)
        decision = None
        for base, spec in rules:
            if path.is_relative_to(base):
                # 明确声明的本地包可以位于被父目录忽略的位置；包内规则仍生效。
                if base != start and spec.match_file(start.relative_to(base).as_posix() + '/'):
                    continue
                match = spec.check_file(
                    path.relative_to(base).as_posix() + ('/' if directory else '')
                )
                if match.include is not None:
                    decision = match.include
        return decision is True

    def visit(folder, rules):
        relative_folder = folder.relative_to(root)
        if scope.excludes(relative_folder):
            return
        if folder != root and ((folder / '.git').exists() or (folder / '.git').is_symlink()):
            raise PreparationError('安装期间项目范围变化，已停止；请核对后重新准备。')
        if not any(name.is_relative_to(relative_folder) for name in tracked):
            if generated_records(folder):
                return
        if folder != start:
            ignore = folder / '.gitignore'
            if ignore.is_file():
                safe_path(root, ignore.relative_to(root))
                if ignore.stat().st_size > MAX_MANIFEST:
                    raise PreparationError('项目忽略文件过大，无法确定安装范围。')
                rules = [
                    *rules,
                    (folder, GitIgnoreSpec.from_lines(ignore.read_text('utf-8-sig').splitlines())),
                ]
        for path in sorted(folder.iterdir()):
            relative = path.relative_to(root)
            if scope.excludes(relative) or excluded(relative) or private_path(path):
                continue
            if path.is_dir():
                if path.name.lower() in SKIP or path.name.lower().startswith('.venv'):
                    continue
                if ignored(path, rules, True):
                    continue
                if _link(path):
                    raise PreparationError(f'安装资料包含目录链接：{relative.as_posix()}')
                yield from visit(path, rules)
            elif not ignored(path, rules):
                safe_path(root, relative)
                if relative not in tracked and generated_archive(path):
                    continue
                yield relative

    yield from visit(start, inherited)


def collect(plan, cancelled=None):
    """先按文件元数据核算边界，再读取内容；不会复制到安装目录。"""
    scope = resolve_scope(plan.root, getattr(plan, "project_scope", None))
    selected = {name: '依赖声明或明确引用的文件' for name in plan.files} if plan.projects else {}
    discovered_sources = set()
    roots = {name: '项目明确声明的本地包' for name in plan.source_projects}
    for project in plan.projects:
        if project.get('has_package') or project.get('needs_source'):
            roots[project['path']] = '项目本身安装或安装脚本需要的源码'
    for name in plan.local_dependencies:
        scope.require(name)
        target = safe_path(plan.root, name, file=False)
        if target.is_file():
            selected[name] = '明确引用的本地依赖文件'
        else:
            roots[name] = '项目明确声明的本地包'
    for name, reason in roots.items():
        for relative in project_files(plan.root, Path(name), project_scope=scope):
            if cancelled and cancelled():
                from agent.execution.managed import PreparationCancelled

                raise PreparationCancelled('环境准备已取消，尚未开始复制安装资料。')
            selected.setdefault(relative.as_posix(), reason)
            discovered_sources.add(relative.as_posix())
            if len(selected) > MAX_FILES:
                raise PreparationError(
                    f'安装资料过多：{relative.as_posix()}，超过 {MAX_FILES:,} 个文件的限制。'
                    f'纳入原因：{reason}。'
                )

    result, total = {}, 0
    for name, reason in sorted(selected.items()):
        if cancelled and cancelled():
            from agent.execution.managed import PreparationCancelled

            raise PreparationCancelled('环境准备已取消，尚未开始复制安装资料。')
        scope.require(name)
        path = safe_path(plan.root, name)
        if name not in discovered_sources:
            parents = [path.parent, *path.parent.parents]
            if any(generated_records(parent) for parent in parents
                   if parent.is_relative_to(plan.root)) or generated_archive(path):
                raise PreparationError(f'明确引用的安装资料是程序运行归档，请核对项目配置：{name}')
        if excluded(Path(name)) or private_path(path):
            raise PreparationError(f'明确引用的安装资料属于受保护文件：{name}')
        size = path.stat().st_size
        total += size
        count = len(result) + 1
        if size > 32 * 1024 * 1024 or total > MAX_BYTES or count > MAX_FILES:
            raise PreparationError(
                f'安装资料过大：{name}，文件 {size:,} 字节，累计 {total:,} 字节、{count:,} 个文件；'
                f'限制为单文件 33,554,432 字节、总计 {MAX_BYTES:,} 字节、{MAX_FILES:,} 个文件。'
                f'纳入原因：{reason}。'
            )
        result[name] = {'size': size, 'reason': reason}
    for name in list(result):
        data = safe_path(plan.root, name).read_bytes()
        if len(data) != result[name]['size']:
            raise PreparationError(f'安装资料在检查期间发生变化：{name}')
        if any(marker in data for marker in PRIVATE_MARKERS) or SECRET_ASSIGNMENT.search(data):
            if name in plan.files:
                raise PreparationError(f'明确引用的安装资料含敏感内容：{name}')
            del result[name]
            continue
        expected = plan.files.get(name)
        actual = hashlib.sha256(data).hexdigest()
        if expected and expected != actual:
            raise PreparationError(f'依赖资料在准备期间发生变化：{name}')
        result[name]['sha256'] = actual
    return result


def summary(materials):
    return {'file_count': len(materials), 'total_bytes': sum(v['size'] for v in materials.values())}
