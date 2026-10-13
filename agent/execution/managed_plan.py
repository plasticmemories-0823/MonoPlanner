"""Read project-owned dependency declarations without executing project code."""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tomllib

from agent.storage.preferences import private_path, state_root

REVISION = 1
MAX_MANIFEST = 4 * 1024 * 1024
MAX_FILES = 20000
MAX_BYTES = 256 * 1024 * 1024
MANIFESTS = {'pyproject.toml', 'setup.py', 'setup.cfg', 'uv.lock', 'poetry.lock',
             'Pipfile', 'Pipfile.lock', 'package.json', 'package-lock.json',
             'npm-shrinkwrap.json', 'pnpm-lock.yaml', 'yarn.lock', '.python-version',
             '.node-version', '.nvmrc', '.tool-versions', 'pnpm-workspace.yaml'}
SKIP = {'.git', '.hg', '.svn', '.venv', 'venv', 'env', 'node_modules', '__pycache__',
        '.tox', '.nox', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.next',
        '.idea', '.vs', '.vscode', 'build', 'dist', 'coverage', 'validation',
        '.langgraph_api', '.coding_agent', '.monoplanner', '.ssh', '.aws', '.azure',
        '.docker', '.kube', '.npm', '.cache', '.yarn'}
SENSITIVE = {'.npmrc', '.yarnrc.yml', '.pypirc', 'pip.conf', 'pip.ini', '.netrc', '_netrc',
             'credentials', 'credentials.json', 'secrets.json', 'secrets.yaml',
             'secrets.yml', 'service-account.json'}
PRIVATE_MARKERS = (b'-----BEGIN PRIVATE KEY-----', b'-----BEGIN RSA PRIVATE KEY-----',
                   b'-----BEGIN OPENSSH PRIVATE KEY-----', b'-----BEGIN EC PRIVATE KEY-----')
SECRET_ASSIGNMENT = re.compile(rb'''(?i)["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret[_-]?key)["']?\s*[:=]\s*["'][^"'\r\n]{12,}["']''')
CHILDREN = {'frontend', 'backend', 'client', 'server', 'web', 'api', 'app',
            'apps', 'packages', 'services', 'src'}


class PreparationError(ValueError):
    """A project needs a concrete fix; host execution is never a fallback."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()


def _recipe():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('managed.py', 'managed_plan.py', 'managed_setup.py', 'managed_proxy.py')}


def _link(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


def project_root(repo):
    path = Path(repo).expanduser()
    if not path.is_absolute() or not path.is_dir():
        raise PreparationError('请选择已有的项目文件夹。')
    if _link(path):
        raise PreparationError('项目文件夹不能是链接或重定向目录。')
    path = path.resolve()
    protected = [state_root()]
    if os.getenv('LOCALAPPDATA'):
        protected.append((Path(os.environ['LOCALAPPDATA']) / 'MonoPlanner').resolve())
    if private_path(path) or any(p.is_relative_to(path) for p in protected):
        raise PreparationError('项目文件夹包含 MonoPlanner 私有记录，不能作为环境安装来源。')
    return path


def safe_path(root, relative, *, file=True):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or ':' in str(relative):
        raise PreparationError('项目依赖引用了项目外路径；请将依赖放入项目或使用公开软件包。')
    path = root
    for part in relative.parts:
        path /= part
        if path.exists() or path.is_symlink():
            if _link(path):
                raise PreparationError(f'项目依赖不能通过链接访问其他目录：{relative.as_posix()}')
    if not path.resolve().is_relative_to(root):
        raise PreparationError('项目依赖路径超出工作目录。')
    if file and not path.is_file():
        raise PreparationError(f'项目声明的依赖文件不存在：{relative.as_posix()}')
    return path


def excluded(relative):
    parts = [p.lower() for p in Path(relative).parts]
    name = parts[-1]
    return (any(p in SKIP or p.startswith('.venv') for p in parts[:-1])
            or name in SENSITIVE or name == '.env' or name.startswith('.env.')
            or name.startswith(('id_rsa', 'id_ed25519'))
            or Path(name).suffix in {'.pem', '.key', '.p12', '.pfx', '.dpapi', '.log', '.pyc', '.pyo'})


def _manifest(path):
    return path.name in MANIFESTS or (path.name.startswith('requirements') and path.suffix == '.txt')


@dataclass
class ProjectPlan:
    root: Path
    python: str = '3.12'
    node: str = '22'
    projects: list = field(default_factory=list)
    files: dict = field(default_factory=dict)
    local_dependencies: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    recipe: dict = field(default_factory=_recipe)

    @property
    def fingerprint(self):
        return _digest({'revision': REVISION, 'repo': str(self.root), 'python': self.python,
                        'node': self.node, 'projects': self.projects, 'files': self.files, 'recipe': self.recipe})

    def public(self):
        return {'mode': 'automatic', 'fingerprint': self.fingerprint,
                'python': self.python, 'node': self.node,
                'projects': [{'path': p['path'], 'language': p['language'],
                              'manager': p['manager']} for p in self.projects],
                'manifest_count': len(self.files), 'warnings': self.warnings}


def _read(plan, relative):
    path = safe_path(plan.root, relative)
    if excluded(relative) or path.stat().st_size > MAX_MANIFEST:
        raise PreparationError(f'依赖文件不可用于自动准备：{Path(relative).as_posix()}')
    data = path.read_bytes()
    if any(marker in data for marker in PRIVATE_MARKERS):
        raise PreparationError('依赖文件含私钥内容，已停止准备。请先移除其中的凭据。')
    # Credentials and environment substitutions are not forwarded to installers.
    text = data.decode('utf-8-sig')
    if re.search(r'https?://[^\s/"\']+@', text) or re.search(r'https?://[^\s"\']*\$\{', text):
        raise PreparationError('项目依赖包含账号信息或环境密钥引用；自动准备不会传递这些凭据。')
    plan.files[Path(relative).as_posix()] = hashlib.sha256(data).hexdigest()
    return text


def _json(plan, relative):
    try:
        return json.loads(_read(plan, relative))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f'项目配置无法解析：{Path(relative).as_posix()}') from exc


def _private_registry_check(plan, folder, names):
    # Never silently discard a private registry setting and fall back to the
    # public registry: doing so could install a same-named unrelated package.
    for name in names & {'.npmrc', '.yarnrc.yml', '.pypirc', 'pip.conf', 'pip.ini', '.netrc', '_netrc'}:
        path = safe_path(plan.root, folder / name)
        if path.stat().st_size > MAX_MANIFEST:
            raise PreparationError('项目包管理器配置过大，无法安全自动准备。')
        text = path.read_text(encoding='utf-8-sig')
        settings = [line.strip() for line in text.splitlines()
                    if line.strip() and not line.lstrip().startswith(('#', ';'))]
        harmless = {'registry=https://registry.npmjs.org/', 'registry=https://registry.npmjs.org',
                    'nodeLinker: node-modules', 'fund=false', 'audit=false', 'save-exact=true'}
        if any(line not in harmless for line in settings):
            raise PreparationError('项目使用了自定义包源或包管理器配置，尚不能安全自动复用；请使用高级设置选择已准备好的项目镜像。不会改用公开包源代替。')


def _local(plan, parent, value, *, editable=False):
    value = value.split('#', 1)[0].split('[', 1)[0].strip()
    if value.startswith('file:'):
        value = value[5:]
    if '\\' in value or value.startswith('/') or ':' in value:
        raise PreparationError('本地依赖须使用项目内的相对路径。')
    combined = parent / value
    # Normalize internal ../ while rejecting escapes and links before resolving.
    parts = []
    for part in combined.parts:
        if part == '..':
            if not parts:
                raise PreparationError('本地依赖超出项目目录。')
            parts.pop()
        elif part != '.':
            parts.append(part)
    relative = Path(*parts)
    target = safe_path(plan.root, relative, file=False)
    if not target.exists():
        raise PreparationError(f'本地依赖不存在：{relative.as_posix()}')
    if editable or relative.as_posix() == parent.as_posix():
        # Editable imports follow /testbed, so source edits do not require a new
        # dependency image. Keep all package metadata in the cache identity.
        if target.is_dir():
            for filename in ('pyproject.toml', 'setup.cfg', 'setup.py', 'package.json'):
                if (target / filename).is_file():
                    _read(plan, relative / filename)
        return
    if relative.as_posix() not in plan.local_dependencies:
        plan.local_dependencies.append(relative.as_posix())


def _requirements(plan, relative, visited=None, base=None):
    visited = visited if visited is not None else set()
    base = relative.parent if base is None else base
    name = relative.as_posix()
    if name in visited:
        return
    if len(visited) >= 64:
        raise PreparationError('依赖文件的引用层数过多，请检查循环或异常配置。')
    visited.add(name)
    text = _read(plan, relative)
    text = text.replace('\\\n', '')
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        include = re.match(r'^(?:-r\s*|--requirement(?:=|\s+)|-c\s*|--constraint(?:=|\s+))(.+)$', line)
        if include:
            target = include[1].strip().split(' #', 1)[0].strip('"\'')
            if re.match(r'\w+://', target):
                raise PreparationError('嵌套依赖清单必须位于项目内，不能从远程加载额外清单。')
            # Resolve only after explicitly inspecting each path component.
            combined = relative.parent / target
            parts = []
            for part in combined.parts:
                if part == '..':
                    if not parts:
                        raise PreparationError('依赖清单引用超出项目目录。')
                    parts.pop()
                elif part != '.':
                    parts.append(part)
            _requirements(plan, Path(*parts), visited, base)
        else:
            dependency = re.sub(r'^(?:-e\s+|--editable(?:=|\s+))', '', line).split(' ;', 1)[0]
            editable = dependency != line
            if dependency.startswith(('.', 'file:')) or (editable and not re.match(r'\w+(?:\+\w+)?://', dependency)):
                # pip resolves package paths from its working directory, whereas
                # -r/-c references resolve from the containing requirements file.
                _local(plan, base, dependency, editable=editable)
            elif dependency.startswith('/') or re.match(r'^[A-Za-z]:', dependency):
                raise PreparationError('依赖声明使用本机绝对路径，不能用于隔离环境。')


def _version(constraints, language):
    # packaging is a mandatory dependency of langchain-core, already in the product lock.
    from packaging.specifiers import SpecifierSet, InvalidSpecifier
    from packaging.version import Version
    if not constraints:
        return '3.12' if language == 'python' else '22'
    candidates = ['3.12', '3.13', '3.11', '3.10', '3.9', '3.14', '3.8', '3.7', '3.6'] if language == 'python' else ['22', '24', '20', '18', '16']
    normalized = []
    for raw in constraints:
        raw = str(raw).strip().lstrip('v')
        if not raw or raw in {'*', 'latest', 'node', 'lts/*'}:
            continue
        if language == 'node' and raw.startswith('lts/'):
            known = {'lts/jod': '22', 'lts/iron': '20', 'lts/hydrogen': '18', 'lts/krypton': '24'}
            raw = known.get(raw, raw)
        if re.fullmatch(r'\d+(?:\.\d+){0,2}', raw):
            if raw not in candidates:
                candidates.insert(0, raw)
            raw = '==' + raw + ('.*' if raw.count('.') < 2 else '')
        if language == 'python':
            # Poetry commonly uses caret version constraints.
            if raw.startswith('^'):
                lower = raw[1:]
                raw = '>=' + lower + ',<' + str(int(lower.split('.')[0]) + 1)
            try:
                normalized.append([SpecifierSet(raw)])
            except InvalidSpecifier as exc:
                raise PreparationError(f'项目的 Python 版本要求无法自动解析：{raw}') from exc
        else:
            alternatives = []
            for term in raw.split('||'):
                term = term.strip()
                if term.startswith(('^', '~')):
                    prefix, lower = term[0], term[1:]
                    nums = lower.split('.')
                    upper = str(int(nums[0]) + 1) if prefix == '^' or len(nums) == 1 else nums[0] + '.' + str(int(nums[1]) + 1)
                    term = '>=' + lower + ',<' + upper
                if re.fullmatch(r'\d+(?:\.\d+)*\.[xX*]', term):
                    term = '==' + term[:-1] + '*'
                term = re.sub(r'(?<![<>=!~\d.])(\d+(?:\.\d+)*)$', r'==\1.*', term) if not re.search(r'[<>=!~]', term) else term
                term = re.sub(r'\s+(?=[<>=!~])', ',', term)
                try:
                    alternatives.append(SpecifierSet(term.replace(' ', '')))
                except InvalidSpecifier as exc:
                    raise PreparationError(f'项目的 Node.js 版本要求无法自动解析：{raw}') from exc
            normalized.append(alternatives)
    for candidate in candidates:
        # Images track the chosen minor/major's newest patch; use a representative
        # patch only for ranges. Exact three-part declarations use their exact tag.
        representative = candidate if candidate.count('.') == 2 else candidate + ('.999' if language == 'python' else '.999.999')
        if all(any(Version(representative) in group for group in alternatives) for alternatives in normalized):
            if language == 'python' and Version(candidate) < Version('3.9'):
                raise PreparationError('这个项目要求较早的 Python 版本，自动准备目前支持 Python 3.9 及以上。已有兼容镜像仍可通过高级设置使用，不会自动升级项目。')
            return candidate
    raise PreparationError(f'项目内的 {"Python" if language == "python" else "Node.js"} 版本要求互不兼容，无法放在同一环境。')


def inspect_project(repo):
    root = project_root(repo)
    plan = ProjectPlan(root)
    folders = [Path('.')]
    for child in sorted(root.iterdir()):
        if child.name.lower() in CHILDREN and child.is_dir() and not _link(child):
            folders.append(Path(child.name))
            if child.name.lower() in {'apps', 'packages', 'services'}:
                folders.extend(Path(child.name) / p.name for p in sorted(child.iterdir())
                               if p.is_dir() and not _link(p) and p.name.lower() not in SKIP)
    if len(folders) > 64:
        raise PreparationError('项目包含过多独立子项目，请选择实际要处理的项目目录。')
    python_versions, node_versions = [], []
    for folder in folders:
        directory = root / folder
        names = {p.name for p in directory.iterdir() if p.is_file()}
        _private_registry_check(plan, folder, names)
        for name in names:
            if not _manifest(Path(name)):
                continue
            _read(plan, folder / name)
        if '.python-version' in names:
            python_versions.append(_read(plan, folder / '.python-version').strip())
        if '.tool-versions' in names:
            for line in _read(plan, folder / '.tool-versions').splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] in {'python', 'nodejs', 'node'}:
                    (python_versions if parts[0] == 'python' else node_versions).append(parts[1])
        for name in ('.node-version', '.nvmrc'):
            if name in names:
                node_versions.append(_read(plan, folder / name).strip())
        pyproject = {}
        if 'setup.cfg' in names:
            import configparser
            config = configparser.ConfigParser(interpolation=None)
            try:
                config.read_string(_read(plan, folder / 'setup.cfg'))
                if config.has_option('options', 'python_requires'):
                    python_versions.append(config['options']['python_requires'])
            except configparser.Error as exc:
                raise PreparationError(f'无法解析 {folder / "setup.cfg"}。') from exc
        if 'Pipfile.lock' in names:
            lock = _json(plan, folder / 'Pipfile.lock')
            if any(source.get('url', '').rstrip('/') != 'https://pypi.org/simple'
                   for source in lock.get('_meta', {}).get('sources', [])):
                raise PreparationError('项目使用自定义 Python 包源，自动准备尚不支持；不会改用公开包源代替。')
            requires = lock.get('_meta', {}).get('requires', {})
            if requires.get('python_full_version') or requires.get('python_version'):
                python_versions.append(requires.get('python_full_version') or requires['python_version'])
        if 'pyproject.toml' in names:
            try:
                pyproject = tomllib.loads(_read(plan, folder / 'pyproject.toml'))
            except tomllib.TOMLDecodeError as exc:
                raise PreparationError(f'无法解析 {folder / "pyproject.toml"}。') from exc
            version = pyproject.get('project', {}).get('requires-python')
            if version:
                python_versions.append(version)
            poetry = pyproject.get('tool', {}).get('poetry', {})
            uv = pyproject.get('tool', {}).get('uv', {})
            if poetry.get('source') or any(uv.get(key) for key in ('index', 'index-url', 'extra-index-url', 'find-links')):
                raise PreparationError('项目使用自定义 Python 包源，自动准备尚不支持；不会改用公开包源代替。')
            if isinstance(poetry.get('dependencies', {}).get('python'), str):
                python_versions.append(poetry['dependencies']['python'])
            for dep in poetry.get('dependencies', {}).values():
                if isinstance(dep, dict) and dep.get('path'):
                    _local(plan, folder, dep['path'])
            for dep in pyproject.get('tool', {}).get('uv', {}).get('sources', {}).values():
                if isinstance(dep, dict) and dep.get('path'):
                    _local(plan, folder, dep['path'])
        requirements = []
        for name in ('requirements.txt', 'requirements-dev.txt', 'requirements-test.txt', 'requirements-tests.txt'):
            if name in names:
                _requirements(plan, folder / name)
                requirements.append(name)
        if 'uv.lock' in names and 'pyproject.toml' in names:
            manager = 'uv'
        elif 'poetry.lock' in names and 'pyproject.toml' in names:
            manager = 'poetry'
        elif 'Pipfile.lock' in names:
            manager = 'pipenv'
        elif requirements:
            manager = 'requirements'
        elif 'pyproject.toml' in names or 'setup.py' in names or 'setup.cfg' in names:
            manager = 'metadata'
        else:
            manager = None
        if manager:
            plan.projects.append({'path': folder.as_posix(), 'language': 'python',
                                  'manager': manager, 'requirements': requirements,
                                  'has_package': ('setup.py' in names or 'setup.cfg' in names
                                      or bool(pyproject.get('project', {}).get('name'))
                                      or bool(pyproject.get('build-system'))
                                      or bool(pyproject.get('tool', {}).get('poetry', {}).get('name')))})
        if 'package.json' in names:
            package = _json(plan, folder / 'package.json')
            if not isinstance(package, dict):
                raise PreparationError('package.json 须是一个对象。')
            if package.get('engines', {}).get('node'):
                node_versions.append(package['engines']['node'])
            for category in ('dependencies', 'devDependencies', 'optionalDependencies'):
                for dep in package.get(category, {}).values():
                    if isinstance(dep, str) and dep.startswith(('file:', 'link:')):
                        _local(plan, folder, dep[5:] if dep.startswith('link:') else dep)
            manager_value = package.get('packageManager', '')
            if manager_value and not re.fullmatch(r'(npm|pnpm|yarn)@\d+\.\d+\.\d+(?:\+[A-Za-z0-9.]+)?', manager_value):
                raise PreparationError('packageManager 须指定 npm、pnpm 或 yarn 的确切版本。')
            manager = (manager_value.split('@')[0] if manager_value else
                       'pnpm' if 'pnpm-lock.yaml' in names else 'yarn' if 'yarn.lock' in names else 'npm')
            plan.projects.append({'path': folder.as_posix(), 'language': 'node', 'manager': manager,
                                  'package_manager': manager_value.split('+')[0],
                                  'locked': bool(names & {'package-lock.json', 'npm-shrinkwrap.json', 'pnpm-lock.yaml', 'yarn.lock'}),
                                  'workspaces': bool(package.get('workspaces')) or 'pnpm-workspace.yaml' in names})
    # npm/pnpm/yarn workspaces are installed by their declaring root. Child
    # manifests remain in the fingerprint but must not be independently resolved.
    workspace_roots = [Path(p['path']) for p in plan.projects if p['language'] == 'node' and p['workspaces']]
    for p in plan.projects:
        if p['language'] == 'node':
            p['workspace_member'] = any(Path(p['path']) != parent and Path(p['path']).is_relative_to(parent)
                                        for parent in workspace_roots)
    for relative in plan.local_dependencies:
        target = safe_path(root, relative, file=False)
        if target.is_file():
            _read(plan, Path(relative))
        else:
            for rel, data in source_files(root, under=Path(relative)):
                plan.files[rel.as_posix()] = hashlib.sha256(data).hexdigest()
    plan.python, plan.node = _version(python_versions, 'python'), _version(node_versions, 'node')
    return plan


def source_files(root, *, under=Path('.')):
    """A bounded, private-data-filtered copy; no Git hooks or project code run."""
    count = total = 0
    for directory, dirs, files in os.walk(safe_path(root, under, file=False), followlinks=False):
        parent = Path(directory)
        for name in list(dirs):
            path = parent / name
            if name.lower() in SKIP or name.lower().startswith('.venv') or private_path(path):
                dirs.remove(name)
            elif _link(path):
                raise PreparationError(f'安装资料包含目录链接：{path.relative_to(root).as_posix()}')
        for name in sorted(files):
            relative = (parent / name).relative_to(root)
            if excluded(relative):
                continue
            path = safe_path(root, relative)
            size = path.stat().st_size
            count += 1
            total += size
            if count > MAX_FILES or total > MAX_BYTES or size > 32 * 1024 * 1024:
                raise PreparationError('用于准备环境的项目资料过大；请移出数据集、缓存或历史运行记录后重试。')
            data = path.read_bytes()
            if len(data) != size:
                raise PreparationError('准备期间项目文件正在变化，请保存文件后重试。')
            if any(marker in data for marker in PRIVATE_MARKERS) or SECRET_ASSIGNMENT.search(data):
                continue
            yield relative, data
