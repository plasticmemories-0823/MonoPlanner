"""Trusted installation driver copied into a throwaway setup container.

This module is not executed by the host. Project scripts run inside the setup
container, never with a host interpreter. Its result contains no source content.
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


def run(argv, cwd=SOURCE):
    print('Preparing project dependencies', flush=True)
    subprocess.run(argv, cwd=str(cwd), check=True)


def pip(*args, cwd=SOURCE):
    run([VENV, '-m', 'pip', '--disable-pip-version-check', '--no-input', *args], cwd)


def install_editable(directory, *, dependencies=True):
    """Install metadata and a live-source import map, never a copied root wheel."""
    pip('install', *([] if dependencies else ['--no-deps']), '--editable', '.', cwd=directory)
    EDITABLE_PROJECTS.add(directory)


def requirements_without_root(path, root, stack=None):
    """Keep relative resolution, but don't install a disposable root source copy."""
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
        include = re.match(r'^(?P<option>-r\s*|--requirement(?:=|\s+)|-c\s*|--constraint(?:=|\s+))(?P<path>.+)$', stripped)
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


def prepare_python(project):
    directory = SOURCE / project['path']
    manager = project['manager']
    if manager == 'uv':
        pip('install', 'uv==0.8.22')
        exported = directory / '.monoplanner-uv.txt'
        run(['/opt/monoplanner/venv/bin/uv', 'export', '--frozen', '--no-emit-project',
             '--output-file', str(exported)], directory)
        pip('install', '-r', str(exported), cwd=directory)
    elif manager == 'poetry':
        pip('install', 'poetry==2.2.1', 'poetry-plugin-export==1.9.0')
        exported = directory / '.monoplanner-poetry.txt'
        run(['/opt/monoplanner/venv/bin/poetry', 'export', '--format', 'requirements.txt',
             '--without-hashes', '--output', str(exported)], directory)
        pip('install', '-r', str(exported), cwd=directory)
    elif manager == 'pipenv':
        lock = json.loads((directory / 'Pipfile.lock').read_text(encoding='utf-8'))
        deps = []
        for group in ('default', 'develop'):
            for name, value in lock.get(group, {}).items():
                if not isinstance(value, dict) or 'version' not in value:
                    raise RuntimeError('Pipfile.lock contains an unsupported local or private dependency')
                deps.append(name + value['version'] + ('; ' + value['markers'] if value.get('markers') else ''))
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
    """Translate installer-generated editable metadata to the real task mount.

    This touches generated import metadata only, not project or model code. It
    supports src-layout packages and local editable dependencies without leaving
    a stale /monoplanner-source installation ahead of the actual task checkout.
    """
    for site in venv_lib.glob('python*/site-packages'):
        candidates = [*site.glob('*.pth'), *site.glob('*.egg-link'),
                      *site.glob('__editable__*.py'), *site.glob('*.dist-info/direct_url.json')]
        for path in candidates:
            data = path.read_bytes()
            if b'/monoplanner-source' in data:
                path.write_bytes(data.replace(b'/monoplanner-source', b'/testbed'))


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
        run(['npm', 'install', '--global', '--no-audit', '--no-fund',
             'pnpm@10.17.1' if manager == 'pnpm' else 'yarn@1.22.22'], directory)
    if manager == 'npm':
        args = ['npm', 'ci' if project['locked'] else 'install', '--no-audit', '--no-fund']
    elif manager == 'pnpm':
        args = ['pnpm', 'install', '--frozen-lockfile' if project['locked'] else '--no-frozen-lockfile']
    else:
        # Yarn Berry is selected via packageManager; classic uses --frozen-lockfile.
        args = ['yarn', 'install', *(['--immutable' if berry else '--frozen-lockfile'] if project['locked'] else [])]
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
                    # Links to workspace source should follow the actual task
                    # checkout. Links to installed dependencies remain relative.
                    if target.is_relative_to(NODE) and not any(target.is_relative_to(p) for p in roots):
                        path.unlink()
                        path.symlink_to(Path('/testbed') / target.relative_to(NODE))
    # Preserve dependency trees only; keep no second project source in the image.
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
    return [{'relative': p.parent.relative_to(NODE).as_posix(), 'image_path': str(p)} for p in roots]


def main():
    plan = json.loads(Path('/monoplanner-plan.json').read_text(encoding='utf-8'))
    # Windows file copies do not preserve executable bits. Restore only scripts
    # carrying a shebang, inside the disposable copy, before package lifecycle hooks.
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
    # Installation URLs and downloaded archives are unnecessary in the reusable image.
    for path in ('/root/.cache', '/root/.npm', '/tmp'):
        shutil.rmtree(path, ignore_errors=True)
    Path('/tmp').mkdir(mode=0o1777, exist_ok=True)
    Path('/monoplanner-result.json').write_text(json.dumps({'node_modules': modules}), encoding='utf-8')
    print('Environment ready', flush=True)


if __name__ == '__main__':
    main()
