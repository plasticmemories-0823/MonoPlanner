"""Build a pinned, offline-starting Windows package from prepared runtimes."""
import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def copy_tree(source, destination):
    def visit(folder, target, ancestors):
        canonical = folder.resolve()
        if canonical in ancestors:
            return  # Already materialized in an ancestor module lookup path.
        for path in folder.iterdir():
            if path.name == '__pycache__' or path.suffix == '.pyc' or path.name == '.env' or path.name.startswith('.env.'):
                continue
            actual = path
            if path.is_symlink() or path.is_junction():
                resolved = path.resolve()
                modules = ROOT/'web/node_modules'
                if not resolved.is_relative_to(modules):
                    raise ValueError('Unexpected distribution link: '+str(path))
                # Next's Windows pnpm junctions point to the development tree.
                # Materialize only its traced production counterpart, no links.
                actual = ROOT/'web/.next/standalone/node_modules'/resolved.relative_to(modules)
                if not actual.exists():
                    raise ValueError('Production dependency was not traced: '+str(actual))
            output = target/path.name
            if actual.is_dir():
                visit(actual, output, ancestors | {canonical})
                if actual != path and (actual/'package.json').is_file():
                    # Junction resolution normally gives Node the package's pnpm
                    # sibling dependencies. Preserve that lookup after copying.
                    module_root = actual.parent.parent if actual.parent.name.startswith('@') else actual.parent
                    visit(module_root, output/'node_modules', ancestors | {canonical, actual.resolve()})
            elif actual.is_file():
                output.parent.mkdir(parents=True, exist_ok=True)
                if not output.exists() or (actual.stat().st_size, actual.stat().st_mtime_ns) != (output.stat().st_size, output.stat().st_mtime_ns):
                    shutil.copy2(actual, output)
    visit(source, destination, set())


def build(destination, make_zip=True):
    destination = destination.resolve()
    if destination.parent != ROOT/'dist' or not destination.name.startswith('MonoPlanner'):
        raise ValueError('Package destination must be a MonoPlanner folder in dist')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('agent', 'resources', 'docs'):
        copy_tree(ROOT/name, destination/name)
    for name in ('MonoPlanner.exe', 'README.md', 'langgraph.json', 'requirements.txt', 'requirements-web.txt'):
        source = ROOT/name
        if source.is_file():
            shutil.copy2(source, destination/name)
    for name in ('start-app.ps1',):
        (destination/'scripts').mkdir(exist_ok=True)
        shutil.copy2(ROOT/'scripts'/name, destination/'scripts'/name)
    runtime = ROOT/'launcher/runtime'
    shutil.copy2(ROOT/'resources/drivers/sitecustomize.py', runtime/'python/Lib/site-packages/sitecustomize.py')
    required = ('python/python.exe', 'node/node.exe', 'git/cmd/git.exe', 'MonoPlanner.Sandbox.exe',
                'MonoPlanner.FilePicker.exe', 'WebView2Loader.dll')
    if any(not (runtime/name).is_file() for name in required):
        raise ValueError('Prepare all fixed runtimes before packaging; no first-run installation')
    copy_tree(runtime, destination/'launcher/runtime')
    standalone = ROOT/'web/.next/standalone'
    if not (standalone/'server.js').is_file():
        raise ValueError('Build the production standalone frontend first')
    copy_tree(standalone, destination/'web')
    copy_tree(ROOT/'web/.next/static', destination/'web/.next/static')
    copy_tree(ROOT/'web/public', destination/'web/public')
    for name in ('UPSTREAM.md', 'LICENSE'):
        if (ROOT/'web'/name).exists():
            shutil.copy2(ROOT/'web'/name, destination/'web'/name)
    metadata = {'product': 'MonoPlanner', 'platform': 'Windows 11 x64',
                'python': '3.13.15', 'node': '24.14.0', 'git': '2.53.0.2',
                'webview2_sdk': '1.0.4258.31', 'webview2_fixed': '155.0.4283.45',
                'startup_installs_dependencies': False, 'daily_requires_docker': False}
    licenses = []
    for info in (runtime/'python/Lib/site-packages').glob('*.dist-info/METADATA'):
        values = Parser().parsestr(info.read_text(encoding='utf-8'))
        licenses.append({'name': values['Name'], 'version': values['Version'],
                         'license': values['License-Expression'] or values['License'] or '',
                         'license_files': [str(p.relative_to(runtime)).replace('\\', '/') for p in info.parent.rglob('*')
                                           if p.is_file() and any(v in p.name.lower() for v in ('license', 'copying', 'notice'))]})
    npm = {}
    packages = ROOT/'web/node_modules/.pnpm'
    for info in [*packages.glob('*/node_modules/*/package.json'), *packages.glob('*/node_modules/@*/*/package.json')]:
        values = json.loads(info.read_text(encoding='utf-8'))
        key = (values.get('name', ''), values.get('version', ''))
        npm[key] = {'name': key[0], 'version': key[1], 'license': values.get('license', '')}
        for license in info.parent.iterdir():
            if license.is_file() and any(v in license.name.lower() for v in ('license', 'copying', 'notice')):
                target = destination/'licenses/npm'/info.parent.name/key[1]/license.name
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(license, target)
    (destination/'licenses').mkdir(exist_ok=True)
    (destination/'licenses/dependencies.json').write_text(json.dumps({'python': licenses, 'npm': list(npm.values())}, ensure_ascii=False, indent=2), encoding='utf-8')
    shutil.copy2(ROOT/'docs/third-party.md', destination/'licenses/README.md')
    from agent.storage.recovery import code_key
    metadata['source_fingerprint'] = code_key()
    (destination/'portable.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    files = {str(p.relative_to(destination)).replace('\\', '/'): {'bytes': p.stat().st_size, 'sha256': hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()}
             for p in destination.rglob('*') if p.is_file() and p.name != 'checksums.json'}
    (destination/'checksums.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
    print(json.dumps({'folder': str(destination), 'files': len(files), 'bytes': sum(p['bytes'] for p in files.values()), **metadata}), flush=True)
    if make_zip:
        target = destination.parent/'MonoPlanner-Windows11-x64.zip'
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in destination.rglob('*'):
                if file.is_file():
                    archive.write(file, 'MonoPlanner/'+file.relative_to(destination).as_posix())
        print(json.dumps({'zip': str(target), 'bytes': target.stat().st_size, 'sha256': hashlib.file_digest(target.open('rb'), 'sha256').hexdigest()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-zip', action='store_true')
    parser.add_argument('--destination', default=str(ROOT/'dist/MonoPlanner'))
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(ROOT))
    build(Path(args.destination), not args.no_zip)
