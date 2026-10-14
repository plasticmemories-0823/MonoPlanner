"""使用已准备的固定版本运行时，构建可离线启动的 Windows 便携包。"""

import argparse
import csv
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 生产文件的复制与联接处理
# ============================================================
def copy_tree(source, destination, *, exclude=()):
    def visit(folder, target, ancestors):
        canonical = folder.resolve()
        if canonical in ancestors:
            return  # 该目录已在上层模块查找路径中实体化。
        for path in folder.iterdir():
            if (
                path.name in exclude
                or path.name == '__pycache__'
                or path.suffix == '.pyc'
                or path.name == '.env'
                or path.name.startswith('.env.')
            ):
                continue
            actual = path
            if path.is_symlink() or path.is_junction():
                resolved = path.resolve()
                modules = ROOT / 'web/node_modules'
                if not resolved.is_relative_to(modules):
                    raise ValueError('Unexpected distribution link: ' + str(path))
                # Next 在 Windows 上的 pnpm 联接指向开发目录。
                # 只复制构建追踪到的生产依赖，不保留联接。
                actual = ROOT / 'web/.next/standalone/node_modules' / resolved.relative_to(modules)
                if not actual.exists():
                    raise ValueError('Production dependency was not traced: ' + str(actual))
            output = target / path.name
            if actual.is_dir():
                visit(actual, output, ancestors | {canonical})
                if actual != path and (actual / 'package.json').is_file():
                    # 联接解析通常让 Node 能够查找 pnpm 中同级的依赖；
                    # 复制后也需要保留这种查找关系。
                    module_root = (
                        actual.parent.parent
                        if actual.parent.name.startswith('@')
                        else actual.parent
                    )
                    visit(
                        module_root,
                        output / 'node_modules',
                        ancestors | {canonical, actual.resolve()},
                    )
            elif actual.is_file():
                output.parent.mkdir(parents=True, exist_ok=True)
                if not output.exists() or (actual.stat().st_size, actual.stat().st_mtime_ns) != (
                    output.stat().st_size,
                    output.stat().st_mtime_ns,
                ):
                    shutil.copy2(actual, output)

    visit(source, destination, set())


def copy_launcher_inputs(destination):
    """恢复任务时校验已安装的应用和文件辅助程序源码。"""
    (destination / 'launcher').mkdir(parents=True, exist_ok=True)
    for name in ('Applications.cs', 'FilePicker.cs'):
        shutil.copy2(ROOT / 'launcher' / name, destination / 'launcher' / name)


# ============================================================
# 固定 Python 运行时与依赖
# ============================================================
def copy_python_runtime(source, destination):
    """只分发固定解释器及锁文件中的生产依赖。"""
    copy_tree(source, destination, exclude={'Lib', 'Scripts'})
    copy_tree(source / 'Lib', destination / 'Lib', exclude={'site-packages'})
    packages = source / 'Lib/site-packages'
    target = destination / 'Lib/site-packages'
    normalized = lambda name: re.sub(r'[-_.]+', '-', name).lower()
    locked = {}
    for line in (ROOT / 'requirements.lock').read_text(encoding='utf-8').splitlines():
        match = re.match(r'([\w.-]+)(?:\[[^]]+\])?==([^ ;]+)', line)
        if match:
            locked[normalized(match[1])] = match[2]
    found = set()
    for metadata in packages.glob('*.dist-info/METADATA'):
        values = Parser().parsestr(metadata.read_text(encoding='utf-8'))
        name = normalized(values['Name'])
        if name not in locked:
            continue
        if values['Version'] != locked[name]:
            raise ValueError('Runtime version differs from lock: ' + name)
        found.add(name)
        with (metadata.parent / 'RECORD').open(encoding='utf-8', newline='') as record:
            for relative, *_ in csv.reader(record):
                path = packages / relative
                if (
                    not path.resolve().is_relative_to(packages.resolve())
                    or path.suffix == '.pyc'
                    or path.name == 'direct_url.json'
                ):
                    continue
                output = target / path.relative_to(packages)
                if not path.is_file():
                    raise ValueError('Production runtime file is missing: ' + relative)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, output)
    if found != set(locked):
        raise ValueError(
            'Production runtime dependencies missing: ' + ', '.join(sorted(set(locked) - found))
        )
    return found


# ============================================================
# 前端构建元数据的重定位
# ============================================================
def sanitize_frontend_metadata(frontend):
    """重定位生成的构建元数据，不修改框架模块。"""

    def relocate(config):
        for key in ('outputFileTracingRoot', 'repoRoot'):
            if key in config:
                config[key] = '.'
        if 'root' in config.get('turbopack', {}):
            config['turbopack']['root'] = '.'
        return config

    entry = frontend / 'server.js'
    content = entry.read_text(encoding='utf-8')
    pattern = r'^const nextConfig = (.+)$'
    match = re.search(pattern, content, re.MULTILINE)
    if not match:
        raise ValueError('Unexpected generated frontend entrypoint')
    configuration = relocate(json.loads(match[1]))
    content = (
        content[: match.start(1)]
        + json.dumps(configuration, separators=(',', ':'))
        + content[match.end(1) :]
    )
    content = content.replace(
        "const hostname = process.env.HOSTNAME || '0.0.0.0'", "const hostname = '127.0.0.1'"
    )
    content = content.replace(
        'process.env.__NEXT_PRIVATE_STANDALONE_CONFIG',
        "nextConfig.outputFileTracingRoot = dir; nextConfig.repoRoot = dir;\n"
        + "if (nextConfig.turbopack) nextConfig.turbopack.root = dir;\nprocess.env.__NEXT_PRIVATE_STANDALONE_CONFIG",
        1,
    )
    entry.write_text(content, encoding='utf-8', newline='\n')
    metadata = frontend / '.next/required-server-files.json'
    value = json.loads(metadata.read_text(encoding='utf-8'))
    original_directory = value.get('appDir')
    if original_directory and original_directory != '.':
        prefixes = {original_directory, original_directory.replace('\\', '/')}
        # 生成的客户端引用 ID 必须在清单和程序包中保持一致。
        for generated in (frontend / '.next/server').rglob('*.js'):
            content = generated.read_text(encoding='utf-8')
            relocated = content
            for prefix in prefixes:
                escaped = prefix
                for _ in range(3):
                    escaped = json.dumps(escaped)[1:-1]
                    relocated = relocated.replace(escaped, '/monoplanner/web')
            if relocated != content:
                generated.write_text(relocated, encoding='utf-8', newline='\n')
    value['config'] = relocate(value['config'])
    for key in ('appDir', 'relativeAppDir'):
        if key in value:
            value[key] = '.'
    metadata.write_text(json.dumps(value, separators=(',', ':')), encoding='utf-8', newline='\n')


# ============================================================
# 便携包组装、许可与完整性清单
# ============================================================
def build(destination, make_zip=True):
    destination = destination.resolve()
    if destination.parent != ROOT / 'dist' or not destination.name.startswith('MonoPlanner'):
        raise ValueError('Package destination must be a MonoPlanner folder in dist')
    if destination.exists():
        raise ValueError('Use a new empty package destination; existing outputs are never reused')
    destination.mkdir(parents=True)
    for name in ('agent', 'resources'):
        copy_tree(ROOT / name, destination / name)
    for name in (
        'MonoPlanner.exe',
        'README.md',
        'LICENSE',
        'langgraph.json',
        'requirements.txt',
        'requirements-web.txt',
        'requirements.lock',
    ):
        source = ROOT / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for name in ('third-party.md', 'security.md', 'assets/workspace.png'):
        source = ROOT / 'docs' / name
        target = destination / 'docs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ('start-app.ps1',):
        (destination / 'scripts').mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'scripts' / name, destination / 'scripts' / name)
    runtime = ROOT / 'launcher/runtime'
    required = (
        'python/python.exe',
        'node/node.exe',
        'git/cmd/git.exe',
        'MonoPlanner.FilePicker.exe',
        'MonoPlanner.Applications.exe',
        'WebView2Loader.dll',
    )
    if any(not (runtime / name).is_file() for name in required):
        raise ValueError('Prepare all fixed runtimes before packaging; no first-run installation')
    runtime_entries = {
        'python',
        'node',
        'git',
        'webview',
        'MonoPlanner.FilePicker.exe',
        'MonoPlanner.Applications.exe',
        'WebView2Loader.dll',
        'Microsoft.Web.WebView2.Core.dll',
        'Microsoft.Web.WebView2.WinForms.dll',
        'LICENSE.txt',
        'version.txt',
    }
    copy_tree(
        runtime,
        destination / 'launcher/runtime',
        exclude={'python', *{p.name for p in runtime.iterdir() if p.name not in runtime_entries}},
    )
    copy_python_runtime(runtime / 'python', destination / 'launcher/runtime/python')
    copy_launcher_inputs(destination)
    standalone = ROOT / 'web/.next/standalone'
    if not (standalone / 'server.js').is_file():
        raise ValueError('Build the production standalone frontend first')
    copy_tree(standalone, destination / 'web')
    copy_tree(ROOT / 'web/.next/static', destination / 'web/.next/static')
    copy_tree(ROOT / 'web/public', destination / 'web/public')
    sanitize_frontend_metadata(destination / 'web')
    for name in ('UPSTREAM.md', 'LICENSE'):
        if (ROOT / 'web' / name).exists():
            shutil.copy2(ROOT / 'web' / name, destination / 'web' / name)
    from agent.version import VERSION

    metadata = {
        'product': 'MonoPlanner',
        'version': VERSION,
        'platform': 'Windows 11 x64',
        'python': '3.13.15',
        'node': '24.14.0',
        'git': '2.53.0.2',
        'webview2_sdk': '1.0.4258.31',
        'webview2_fixed': '155.0.4283.45',
        'startup_installs_dependencies': False,
        'chat_requires_docker': False,
        'coding_requires_docker': True,
        'existing_tasks_require_docker': True,
    }
    metadata['automatic_project_environments'] = True
    metadata['project_preparation_downloads_dependencies'] = True
    licenses = []
    sources = {
        'git-v2.53.0.windows.2.zip': '2642f2ac3b19e5e734376dadfb5aad403ad8f1e53785125292a270bbf8dbf9be',
        'mingw-w64-git-2.52.0.1-1.src.tar.gz': '3c70256acbc955ce46b2601f69adedadda5999944a3a6c774f3166c1ca4b7bf6',
    }
    for name, expected in sources.items():
        source = runtime / 'sources' / name
        with source.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Third-party source archive differs: ' + name)
        target = destination / 'licenses/sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shipped_runtime = destination / 'launcher/runtime'
    for info in (shipped_runtime / 'python/Lib/site-packages').glob('*.dist-info/METADATA'):
        values = Parser().parsestr(info.read_text(encoding='utf-8'))
        licenses.append(
            {
                'name': values['Name'],
                'version': values['Version'],
                'license': values['License-Expression'] or values['License'] or '',
                'license_files': [
                    str(p.relative_to(shipped_runtime)).replace('\\', '/')
                    for p in info.parent.rglob('*')
                    if p.is_file()
                    and any(v in p.name.lower() for v in ('license', 'copying', 'notice'))
                ],
            }
        )
    npm = {}
    packages = ROOT / 'web/node_modules/.pnpm'
    installed = {}
    for info in [
        *packages.glob('*/node_modules/*/package.json'),
        *packages.glob('*/node_modules/@*/*/package.json'),
    ]:
        values = json.loads(info.read_text(encoding='utf-8'))
        installed[(values.get('name', ''), values.get('version', ''))] = info
    pnpm = shutil.which('pnpm')
    required_pnpm = json.loads((ROOT / 'web/package.json').read_text(encoding='utf-8'))[
        'packageManager'
    ].split('@')[1]
    if not pnpm or subprocess.check_output([pnpm, '--version'], text=True).strip() != required_pnpm:
        raise ValueError('Use the pinned pnpm version to collect frontend licenses')
    graph = json.loads(
        subprocess.check_output(
            [pnpm, '--dir', str(ROOT / 'web'), 'list', '--prod', '--depth', 'Infinity', '--json'],
            text=True,
            encoding='utf-8',
        )
    )
    frontend_inputs = []

    def collect(entries):
        for dependency in entries.values():
            info = Path(dependency['path']) / 'package.json'
            if not info.resolve().is_relative_to((ROOT / 'web/node_modules').resolve()):
                raise ValueError('Unexpected production license source')
            # pnpm 还会记录其他操作系统可选的原生依赖。
            if info.is_file():
                frontend_inputs.append(info)
            collect(dependency.get('dependencies', {}))

    for project in graph:
        collect(project.get('dependencies', {}))
    frontend_inputs.extend((destination / 'web/node_modules').rglob('package.json'))
    for info in frontend_inputs:
        values = json.loads(info.read_text(encoding='utf-8'))
        key = (values.get('name', ''), values.get('version', ''))
        if key in npm:
            continue
        npm[key] = {
            'name': key[0],
            'version': key[1],
            'license': values.get('license', 'See original license text'),
        }
        original = info
        if info.is_relative_to(destination / 'web/node_modules'):
            original = (
                ROOT / 'web/node_modules' / info.relative_to(destination / 'web/node_modules')
            )
        info = installed.get(key, original if original.is_file() else info)
        for license in info.parent.iterdir():
            if license.is_file() and any(
                v in license.name.lower() for v in ('license', 'copying', 'notice')
            ):
                target = (
                    destination
                    / 'licenses/npm'
                    / key[0].replace('/', '--')
                    / (key[1] or 'bundled')
                    / license.name
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(license, target)
    (destination / 'licenses').mkdir(exist_ok=True)
    (destination / 'licenses/dependencies.json').write_text(
        json.dumps(
            {'python': licenses, 'npm': list(npm.values()), 'source_archives': sources},
            ensure_ascii=False,
            indent=2,
        ),
        encoding='utf-8',
    )
    shutil.copy2(ROOT / 'docs/third-party.md', destination / 'licenses/README.md')
    from agent.storage.recovery import code_key

    metadata['source_fingerprint'] = code_key(destination)
    if metadata['source_fingerprint'] != code_key():
        raise ValueError(
            'Packaged recovery inputs differ from the source; distribution not created'
        )
    (destination / 'portable.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    def shipped(path):
        name = path.relative_to(destination).as_posix()
        runtime_entries = {
            'python',
            'node',
            'git',
            'webview',
            'MonoPlanner.FilePicker.exe',
            'MonoPlanner.Applications.exe',
            'WebView2Loader.dll',
            'Microsoft.Web.WebView2.Core.dll',
            'Microsoft.Web.WebView2.WinForms.dll',
            'LICENSE.txt',
            'version.txt',
        }
        if (
            name.startswith('launcher/runtime/')
            and path.relative_to(destination).parts[2] not in runtime_entries
        ):
            return False
        return (
            path.is_file()
            and path.name != 'checksums.json'
            and not name.startswith(('web/.next/server/route-cache/', 'web/.next/cache/'))
            and '__pycache__' not in path.parts
            and path.suffix != '.pyc'
        )

    files = {
        p.relative_to(destination).as_posix(): {
            'bytes': p.stat().st_size,
            'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest(),
        }
        for p in destination.rglob('*')
        if shipped(p)
    }
    (destination / 'checksums.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    print(
        json.dumps(
            {
                'folder': str(destination),
                'files': len(files),
                'bytes': sum(p['bytes'] for p in files.values()),
                **metadata,
            }
        ),
        flush=True,
    )
    if make_zip:
        target = destination.parent / f'MonoPlanner-{VERSION}-Windows11-x64.zip'
        if target.exists():
            raise ValueError('Versioned release ZIP already exists; preserve previous artifacts')
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in [*files, 'checksums.json']:
                archive.write(destination / name, 'MonoPlanner/' + name)
        print(
            json.dumps(
                {
                    'zip': str(target),
                    'bytes': target.stat().st_size,
                    'sha256': hashlib.file_digest(target.open('rb'), 'sha256').hexdigest(),
                }
            ),
            flush=True,
        )


if __name__ == '__main__':
    import sys

    sys.path.insert(0, str(ROOT))
    from agent.version import VERSION

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-zip', action='store_true')
    parser.add_argument('--destination', default=str(ROOT / f'dist/MonoPlanner-{VERSION}'))
    args = parser.parse_args()
    build(Path(args.destination), not args.no_zip)
