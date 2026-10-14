"""复制到一次性安装容器中的受信任安装驱动。

本模块不在本机执行。项目脚本只在安装容器内运行，不使用本机解释器；
返回结果不包含源码内容。
"""

import json
import os
from pathlib import Path
import shutil
import subprocess

SOURCE = Path('/monoplanner-source')
VENV = '/opt/monoplanner/venv/bin/python'
NODE = Path('/opt/monoplanner/node')
EDITABLE_PROJECTS = set()


# ============================================================
# 安装命令的执行入口
# ============================================================
def run(argv, cwd=SOURCE):
    print('Preparing project dependencies', flush=True)
    subprocess.run(argv, cwd=str(cwd), check=True)


def pip(*args, cwd=SOURCE):
    run([VENV, '-m', 'pip', '--disable-pip-version-check', '--no-input', *args], cwd)


def install_editable(directory, *, dependencies=True):
    """安装元数据和指向实际源码的导入映射，不安装复制的根项目 wheel。"""
    pip('install', *([] if dependencies else ['--no-deps']), '--editable', '.', cwd=directory)
    EDITABLE_PROJECTS.add(directory)


def requirements_without_root(path, root, stack=None):
    """保留相对路径解析，不安装一次性的根项目源码副本。"""
    stack = stack or set()
    if path in stack:
        return path
    stack.add(path)
    import re

    text = path.read_text(encoding='utf-8-sig')
    lines, needs_metadata = [], False
    for line in text.splitlines():
        stripped = line.strip()
        dep = re.sub(r'^(?:-e\s+|--editable(?:=|\s+))', '', stripped)
        if dep in {'.', './'} and path.parent == root:
            needs_metadata = True
            continue
        include = re.match(
            r'^(?P<option>-r\s*|--requirement(?:=|\s+)|-c\s*|--constraint(?:=|\s+))(?P<path>.+)$',
            stripped,
        )
        if include:
            child = (path.parent / include['path'].strip().strip('"\'')).resolve()
            if not child.is_relative_to(SOURCE):
                raise RuntimeError('Dependency file escaped preparation source')
            rewritten = requirements_without_root(child, root, stack)
            line = include['option'] + str(rewritten)
        lines.append(line)
    if needs_metadata:
        install_editable(root)
    target = path.with_name('.monoplanner-' + path.name)
    target.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return target


# ============================================================
# Python 项目依赖与源码映射
# ============================================================
def prepare_python(project):
    directory = SOURCE / project['path']
    manager = project['manager']
    if manager == 'uv':
        pip('install', 'uv==0.8.22')
        exported = directory / '.monoplanner-uv.txt'
        run(
            [
                '/opt/monoplanner/venv/bin/uv',
                'export',
                '--frozen',
                '--no-emit-project',
                '--output-file',
                str(exported),
            ],
            directory,
        )
        pip('install', '-r', str(exported), cwd=directory)
    elif manager == 'poetry':
        pip('install', 'poetry==2.2.1', 'poetry-plugin-export==1.9.0')
        exported = directory / '.monoplanner-poetry.txt'
        run(
            [
                '/opt/monoplanner/venv/bin/poetry',
                'export',
                '--format',
                'requirements.txt',
                '--without-hashes',
                '--output',
                str(exported),
            ],
            directory,
        )
        pip('install', '-r', str(exported), cwd=directory)
    elif manager == 'pipenv':
        lock = json.loads((directory / 'Pipfile.lock').read_text(encoding='utf-8'))
        deps = []
        for group in ('default', 'develop'):
            for name, value in lock.get(group, {}).items():
                if not isinstance(value, dict) or 'version' not in value:
                    raise RuntimeError(
                        'Pipfile.lock contains an unsupported local or private dependency'
                    )
                deps.append(
                    name
                    + value['version']
                    + ('; ' + value['markers'] if value.get('markers') else '')
                )
        if deps:
            pip('install', *deps, cwd=directory)
    elif manager == 'metadata':
        if project['has_package']:
            install_editable(directory)
    else:
        for filename in project['requirements']:
            rewritten = requirements_without_root(directory / filename, directory)
            pip('install', '-r', str(rewritten), cwd=directory)
    if project['has_package'] and directory not in EDITABLE_PROJECTS:
        install_editable(directory, dependencies=False)


def bind_live_python_source(venv_lib=Path('/opt/monoplanner/venv/lib')):
    """将安装器生成的可编辑导入元数据映射到真实任务挂载。

    只修改生成的导入元数据，不修改项目代码或模型提交的代码。
    支持 src 布局和本地可编辑依赖，避免旧的 /monoplanner-source 安装路径
    排在真实任务源码之前。
    """
    for site in venv_lib.glob('python*/site-packages'):
        candidates = [
            *site.glob('*.pth'),
            *site.glob('*.egg-link'),
            *site.glob('__editable__*.py'),
            *site.glob('*.dist-info/direct_url.json'),
        ]
        for path in candidates:
            data = path.read_bytes()
            if b'/monoplanner-source' in data:
                path.write_bytes(data.replace(b'/monoplanner-source', b'/testbed'))


# ============================================================
# Node.js 项目依赖的安装与保留
# ============================================================
def prepare_node(project):
    directory = NODE / project['path']
    manager, selected = project['manager'], project['package_manager']
    berry = bool(manager == 'yarn' and selected and int(selected.split('@')[1].split('.')[0]) >= 2)
    if berry:
        run(['npm', 'install', '--global', '--no-audit', '--no-fund', 'corepack@0.34.1'], directory)
        run(['corepack', 'enable'], directory)
        run(['corepack', 'prepare', selected, '--activate'], directory)
    elif selected:
        run(['npm', 'install', '--global', '--no-audit', '--no-fund', selected], directory)
    elif manager != 'npm':
        run(
            [
                'npm',
                'install',
                '--global',
                '--no-audit',
                '--no-fund',
                'pnpm@10.17.1' if manager == 'pnpm' else 'yarn@1.22.22',
            ],
            directory,
        )
    if manager == 'npm':
        args = ['npm', 'ci' if project['locked'] else 'install', '--no-audit', '--no-fund']
    elif manager == 'pnpm':
        args = [
            'pnpm',
            'install',
            '--frozen-lockfile' if project['locked'] else '--no-frozen-lockfile',
        ]
    else:
        # Yarn Berry 通过 packageManager 选择版本；经典版本使用 --frozen-lockfile。
        args = [
            'yarn',
            'install',
            *(['--immutable' if berry else '--frozen-lockfile'] if project['locked'] else []),
        ]
    run(args, directory)


def retain_node_dependencies():
    roots = []
    for directory, dirs, _ in os.walk(NODE, followlinks=False):
        if 'node_modules' in dirs:
            module = Path(directory) / 'node_modules'
            if module.is_symlink():
                raise RuntimeError('Dependency root cannot be a link')
            for cache_name in ('.cache', '.vite', '.vite-temp'):
                cache = module / cache_name
                if cache.is_symlink():
                    raise RuntimeError('Dependency cache mountpoint cannot be a link')
                cache.mkdir(exist_ok=True)
            roots.append(module)
            dirs.remove('node_modules')
    roots = sorted(roots)
    for root in roots:
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                if path.is_symlink():
                    target = path.resolve()
                    # 指向工作区源码的链接应跟随真实的任务源码；
                    # 指向已安装依赖的链接仍保留相对路径。
                    if target.is_relative_to(NODE) and not any(
                        target.is_relative_to(p) for p in roots
                    ):
                        path.unlink()
                        path.symlink_to(Path('/testbed') / target.relative_to(NODE))
    # 只保留依赖树，不在镜像中留下第二份项目源码。
    for directory, dirs, files in os.walk(NODE, topdown=False, followlinks=False):
        parent = Path(directory)
        if any(parent.is_relative_to(p) for p in roots):
            continue
        for name in files:
            (parent / name).unlink()
        for name in dirs:
            path = parent / name
            if path.is_symlink():
                path.unlink()
            elif path not in roots and not any(p.is_relative_to(path) for p in roots):
                path.rmdir()
    return [
        {'relative': p.parent.relative_to(NODE).as_posix(), 'image_path': str(p)} for p in roots
    ]


# ============================================================
# 容器内安装流程与结果保存
# ============================================================
def main():
    plan = json.loads(Path('/monoplanner-plan.json').read_text(encoding='utf-8'))
    # Windows 文件复制不保留可执行位。运行包生命周期钩子前，
    # 只在一次性副本中恢复带有 shebang 的脚本权限。
    for path in SOURCE.rglob('*'):
        if path.is_file():
            with path.open('rb') as stream:
                if stream.read(2) == b'#!':
                    path.chmod(path.stat().st_mode | 0o111)
    pip('install', 'pytest==8.4.2', 'packaging==25.0')
    node_projects = [p for p in plan['projects'] if p['language'] == 'node']
    if node_projects:
        shutil.copytree(SOURCE, NODE, symlinks=False, dirs_exist_ok=True)
    for project in plan['projects']:
        if project['language'] == 'python':
            prepare_python(project)
        elif not project['workspace_member']:
            prepare_node(project)
    bind_live_python_source()
    pip('check')
    modules = retain_node_dependencies() if node_projects else []
    shutil.rmtree(SOURCE)
    # 可复用镜像中不需要保留安装 URL 或下载的归档文件。
    for path in ('/root/.cache', '/root/.npm', '/tmp'):
        shutil.rmtree(path, ignore_errors=True)
    Path('/tmp').mkdir(mode=0o1777, exist_ok=True)
    Path('/monoplanner-result.json').write_text(
        json.dumps({'node_modules': modules}), encoding='utf-8'
    )
    print('Environment ready', flush=True)


if __name__ == '__main__':
    main()
