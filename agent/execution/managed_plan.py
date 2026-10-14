"""读取项目自己的依赖声明，不执行项目代码。"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from agent.source.scope import ProjectScope, project_scope as resolve_scope
import re
import stat
import tomllib

from agent.storage.preferences import private_path, state_root

REVISION = 3
MAX_MANIFEST = 4 * 1024 * 1024
MAX_FILES = 20000
MAX_BYTES = 256 * 1024 * 1024
MANIFESTS = {
    'pyproject.toml',
    'setup.py',
    'setup.cfg',
    'uv.lock',
    'poetry.lock',
    'Pipfile',
    'Pipfile.lock',
    'package.json',
    'package-lock.json',
    'npm-shrinkwrap.json',
    'pnpm-lock.yaml',
    'yarn.lock',
    '.python-version',
    '.node-version',
    '.nvmrc',
    '.tool-versions',
    'pnpm-workspace.yaml',
}
SKIP = {
    '.git',
    '.hg',
    '.svn',
    '.venv',
    'venv',
    'env',
    'node_modules',
    '__pycache__',
    '.tox',
    '.nox',
    '.pytest_cache',
    '.mypy_cache',
    '.ruff_cache',
    '.next',
    '.idea',
    '.vs',
    '.vscode',
    'build',
    'dist',
    'coverage',
    'validation',
    '.langgraph_api',
    '.coding_agent',
    '.monoplanner',
    '.ssh',
    '.aws',
    '.azure',
    '.docker',
    '.kube',
    '.npm',
    '.cache',
    '.yarn',
}
SENSITIVE = {
    '.npmrc',
    '.yarnrc.yml',
    '.pypirc',
    'pip.conf',
    'pip.ini',
    '.netrc',
    '_netrc',
    'credentials',
    'credentials.json',
    'secrets.json',
    'secrets.yaml',
    'secrets.yml',
    'service-account.json',
}
PRIVATE_MARKERS = (
    b'-----BEGIN PRIVATE KEY-----',
    b'-----BEGIN RSA PRIVATE KEY-----',
    b'-----BEGIN OPENSSH PRIVATE KEY-----',
    b'-----BEGIN EC PRIVATE KEY-----',
)
SECRET_ASSIGNMENT = re.compile(
    rb'''(?i)["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret[_-]?key)["']?\s*[:=]\s*["'][^"'\r\n]{12,}["']'''
)
CHILDREN = {
    'frontend',
    'backend',
    'client',
    'server',
    'web',
    'api',
    'app',
    'apps',
    'packages',
    'services',
    'src',
}


# ============================================================
# 准备错误与身份计算
# ============================================================
class PreparationError(ValueError):
    """项目需要明确修复；不能回退到本机执行。"""


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    ).hexdigest()


def _recipe():
    # 只有安装协议、工具版本或安全边界变化才升级；排版和注释不使用户环境失效。
    return {'environment_protocol': REVISION, 'installer': 1, 'download_isolation': 1}


def _link(path):
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, 'st_file_attributes', 0) & 0x400)


# ============================================================
# 项目路径和读取边界
# ============================================================
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
    return (
        any(p in SKIP or p.startswith('.venv') for p in parts[:-1])
        or name in SENSITIVE
        or name == '.env'
        or name.startswith('.env.')
        or name.startswith(('id_rsa', 'id_ed25519'))
        or Path(name).suffix in {'.pem', '.key', '.p12', '.pfx', '.dpapi', '.log', '.pyc', '.pyo'}
    )


def _manifest(path):
    return path.name in MANIFESTS or (
        path.name.startswith('requirements') and path.suffix == '.txt'
    )


# ============================================================
# 环境准备方案及公开摘要
# ============================================================
@dataclass
class ProjectPlan:
    root: Path
    python: str = '3.12'
    node: str = '22'
    projects: list = field(default_factory=list)
    files: dict = field(default_factory=dict)
    local_dependencies: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    source_projects: list = field(default_factory=list)
    materials: dict | None = None
    recipe: dict = field(default_factory=_recipe)
    project_scope: ProjectScope | None = None

    @property
    def fingerprint(self):
        return _digest(
            {
                'revision': REVISION,
                'repo': str(self.root),
                'python': self.python,
                'node': self.node,
                'projects': self.projects,
                'source_projects': self.source_projects,
                'files': self.files,
                'recipe': self.recipe,
                **({'excluded_projects': list(self.project_scope.excluded_projects)}
                   if self.project_scope and self.project_scope.excluded_projects else {}),
            }
        )

    def public(self):
        return {
            'mode': 'automatic',
            'fingerprint': self.fingerprint,
            'python': self.python,
            'node': self.node,
            'projects': [
                {'path': p['path'], 'language': p['language'], 'manager': p['manager']}
                for p in self.projects
            ],
            'manifest_count': len(self.files),
            'warnings': self.warnings,
        }


# ============================================================
# 依赖声明的读取与来源检查
# ============================================================
def _read(plan, relative):
    # 包说明等元数据可以引用工作目录内部的父级文件，但不能越出授权根目录。
    relative = Path(relative)
    if relative.is_absolute() or ':' in str(relative):
        raise PreparationError('依赖资料必须使用项目内的相对路径。')
    parts = []
    for part in relative.parts:
        if part == '..':
            if not parts:
                raise PreparationError('依赖资料引用超出工作目录。')
            parts.pop()
        elif part != '.':
            parts.append(part)
    relative = Path(*parts)
    if plan.project_scope is not None:
        plan.project_scope.require(relative)
    path = safe_path(plan.root, relative)
    if excluded(relative) or path.stat().st_size > MAX_MANIFEST:
        raise PreparationError(f'依赖文件不可用于自动准备：{Path(relative).as_posix()}')
    data = path.read_bytes()
    if any(marker in data for marker in PRIVATE_MARKERS):
        raise PreparationError('依赖文件含私钥内容，已停止准备。请先移除其中的凭据。')
    # 不向安装器转发凭据或环境变量替换表达式。
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
    # 不能静默丢弃私有仓库配置，再退回公共仓库，
    # 否则可能安装名称相同但来源无关的软件包。
    for name in names & {
        '.npmrc',
        '.yarnrc.yml',
        '.pypirc',
        'pip.conf',
        'pip.ini',
        '.netrc',
        '_netrc',
    }:
        path = safe_path(plan.root, folder / name)
        if path.stat().st_size > MAX_MANIFEST:
            raise PreparationError('项目包管理器配置过大，无法安全自动准备。')
        text = path.read_text(encoding='utf-8-sig')
        settings = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith(('#', ';'))
        ]
        harmless = {
            'registry=https://registry.npmjs.org/',
            'registry=https://registry.npmjs.org',
            'nodeLinker: node-modules',
            'fund=false',
            'audit=false',
            'save-exact=true',
        }
        if any(line not in harmless for line in settings):
            raise PreparationError(
                '项目使用了自定义包源或包管理器配置，尚不能安全自动复用；请使用高级设置选择已准备好的项目镜像。不会改用公开包源代替。'
            )


def _local(plan, parent, value, *, editable=False):
    value = value.split('#', 1)[0].split('[', 1)[0].strip()
    if value.startswith('file:'):
        value = value[5:]
    if '\\' in value or value.startswith('/') or ':' in value:
        raise PreparationError('本地依赖须使用项目内的相对路径。')
    combined = parent / value
    # 允许内部的 ../ 规范化，但解析前必须拒绝越界和链接。
    parts = []
    for part in combined.parts:
        if part == '..':
            if not parts:
                raise PreparationError('本地依赖超出项目目录。')
            parts.pop()
        elif part != '.':
            parts.append(part)
    relative = Path(*parts)
    if plan.project_scope is not None:
        plan.project_scope.require(relative)
    target = safe_path(plan.root, relative, file=False)
    if not target.exists():
        raise PreparationError(f'本地依赖不存在：{relative.as_posix()}')
    if target.is_dir() and relative.as_posix() not in plan.source_projects:
        plan.source_projects.append(relative.as_posix())
    if editable or relative.as_posix() == parent.as_posix():
        # 可编辑导入指向 /testbed，因此修改源码不需要重新生成依赖镜像。
        # 全部包元数据仍应计入缓存身份。
        if target.is_dir():
            for filename in ('pyproject.toml', 'setup.cfg', 'setup.py', 'package.json'):
                if (target / filename).is_file():
                    _read(plan, relative / filename)
        return
    if relative.as_posix() not in plan.local_dependencies:
        plan.local_dependencies.append(relative.as_posix())


# ============================================================
# Python 依赖和版本要求
# ============================================================
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
        include = re.match(
            r'^(?:-r\s*|--requirement(?:=|\s+)|-c\s*|--constraint(?:=|\s+))(.+)$', line
        )
        if include:
            target = include[1].strip().split(' #', 1)[0].strip('"\'')
            if re.match(r'\w+://', target):
                raise PreparationError('嵌套依赖清单必须位于项目内，不能从远程加载额外清单。')
            # 只有逐个检查完路径组成部分后，才能解析最终路径。
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
            if dependency.startswith(('.', 'file:')) or (
                editable and not re.match(r'\w+(?:\+\w+)?://', dependency)
            ):
                # pip 按工作目录解析包路径；
                # -r/-c 引用则相对于所属 requirements 文件解析。
                _local(plan, base, dependency, editable=editable)
            elif dependency.startswith('/') or re.match(r'^[A-Za-z]:', dependency):
                raise PreparationError('依赖声明使用本机绝对路径，不能用于隔离环境。')


def _version(constraints, language):
    # packaging 是 langchain-core 的必需依赖，已包含在产品锁文件中。
    from packaging.specifiers import SpecifierSet, InvalidSpecifier
    from packaging.version import Version

    if not constraints:
        return '3.12' if language == 'python' else '22'
    candidates = (
        ['3.12', '3.13', '3.11', '3.10', '3.9', '3.14', '3.8', '3.7', '3.6']
        if language == 'python'
        else ['22', '24', '20', '18', '16']
    )
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
            # Poetry 通常使用脱字符表示版本约束。
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
                    upper = (
                        str(int(nums[0]) + 1)
                        if prefix == '^' or len(nums) == 1
                        else nums[0] + '.' + str(int(nums[1]) + 1)
                    )
                    term = '>=' + lower + ',<' + upper
                if re.fullmatch(r'\d+(?:\.\d+)*\.[xX*]', term):
                    term = '==' + term[:-1] + '*'
                term = (
                    re.sub(r'(?<![<>=!~\d.])(\d+(?:\.\d+)*)$', r'==\1.*', term)
                    if not re.search(r'[<>=!~]', term)
                    else term
                )
                term = re.sub(r'\s+(?=[<>=!~])', ',', term)
                try:
                    alternatives.append(SpecifierSet(term.replace(' ', '')))
                except InvalidSpecifier as exc:
                    raise PreparationError(f'项目的 Node.js 版本要求无法自动解析：{raw}') from exc
            normalized.append(alternatives)
    for candidate in candidates:
        # 镜像跟随所选主版本或次版本的最新补丁；代表性补丁号只用于范围判断。
        # 明确指定三段版本号时，使用对应的精确标签。
        representative = (
            candidate
            if candidate.count('.') == 2
            else candidate + ('.999' if language == 'python' else '.999.999')
        )
        if all(
            any(Version(representative) in group for group in alternatives)
            for alternatives in normalized
        ):
            if language == 'python' and Version(candidate) < Version('3.9'):
                raise PreparationError(
                    '这个项目要求较早的 Python 版本，自动准备目前支持 Python 3.9 及以上。已有兼容镜像仍可通过高级设置使用，不会自动升级项目。'
                )
            return candidate
    raise PreparationError(
        f'项目内的 {"Python" if language == "python" else "Node.js"} 版本要求互不兼容，无法放在同一环境。'
    )


# ============================================================
# 识别项目的依赖配置
# ============================================================
def inspect_project(repo, project_scope=None):
    root = project_root(repo)
    scope = resolve_scope(root, project_scope)
    plan = ProjectPlan(root, project_scope=scope)
    folders = [Path('.')]
    for child in sorted(root.iterdir()):
        if (child.name.lower() in CHILDREN and child.is_dir() and not _link(child)
                and not scope.excludes(child.relative_to(root))):
            folders.append(Path(child.name))
            if child.name.lower() in {'apps', 'packages', 'services'}:
                folders.extend(
                    Path(child.name) / p.name
                    for p in sorted(child.iterdir())
                    if p.is_dir() and not _link(p) and p.name.lower() not in SKIP
                    and not scope.excludes(p.relative_to(root))
                )
    # 读取用户声明的工作区成员，不把未声明的相邻目录混入安装。
    patterns = []
    if (root / 'package.json').is_file():
        package = _json(plan, Path('package.json'))
        workspaces = package.get('workspaces', [])
        if isinstance(workspaces, dict):
            workspaces = workspaces.get('packages', [])
        if not isinstance(workspaces, list):
            raise PreparationError('package.json 的 workspaces 必须是目录列表。')
        patterns.extend(workspaces)
    if (root / 'pnpm-workspace.yaml').is_file():
        import yaml

        try:
            workspace = yaml.safe_load(_read(plan, Path('pnpm-workspace.yaml'))) or {}
        except yaml.YAMLError as exc:
            raise PreparationError('无法解析 pnpm-workspace.yaml。') from exc
        members = workspace.get('packages', []) if isinstance(workspace, dict) else None
        if not isinstance(members, list):
            raise PreparationError('pnpm 工作区的 packages 必须是目录列表。')
        patterns.extend(members)
    for pattern in patterns:
        value = pattern[1:] if isinstance(pattern, str) and pattern.startswith('!') else pattern
        if (
            not isinstance(value, str)
            or Path(value).is_absolute()
            or '..' in Path(value).parts
            or ':' in value
        ):
            raise PreparationError('项目工作区必须使用项目内路径。')
    from fnmatch import fnmatchcase

    excluded_patterns = [pattern[1:] for pattern in patterns if pattern.startswith('!')]
    for pattern in (p for p in patterns if not p.startswith('!')):
        for path in sorted(scope.glob_directories(pattern, excluded_patterns)):
            relative = path.relative_to(root)
            if scope.excludes(relative):
                raise PreparationError('工作区声明引用已排除的独立项目：' + relative.as_posix())
            safe_path(root, relative, file=False)
            if (
                not path.is_dir()
                or excluded(relative / 'package.json')
                or any(fnmatchcase(relative.as_posix(), pattern) for pattern in excluded_patterns)
            ):
                continue
            if relative not in folders:
                folders.append(relative)
                if len(folders) > 64:
                    raise PreparationError('项目工作区成员过多，请选择实际项目。')
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
            if any(
                source.get('url', '').rstrip('/') != 'https://pypi.org/simple'
                for source in lock.get('_meta', {}).get('sources', [])
            ):
                raise PreparationError(
                    '项目使用自定义 Python 包源，自动准备尚不支持；不会改用公开包源代替。'
                )
            requires = lock.get('_meta', {}).get('requires', {})
            if requires.get('python_full_version') or requires.get('python_version'):
                python_versions.append(
                    requires.get('python_full_version') or requires['python_version']
                )
        if 'pyproject.toml' in names:
            try:
                pyproject = tomllib.loads(_read(plan, folder / 'pyproject.toml'))
            except tomllib.TOMLDecodeError as exc:
                raise PreparationError(f'无法解析 {folder / "pyproject.toml"}。') from exc
            metadata = pyproject.get('project', {})
            for resource in (metadata.get('readme'), metadata.get('license')):
                filename = (
                    resource.get('file')
                    if isinstance(resource, dict)
                    else (resource if resource == metadata.get('readme') else None)
                )
                if isinstance(filename, str):
                    _read(plan, folder / filename)
            for value in (
                pyproject.get('tool', {}).get('setuptools', {}).get('dynamic', {}).values()
            ):
                if isinstance(value, dict) and 'file' in value:
                    declared = value['file']
                    for filename in [declared] if isinstance(declared, str) else declared:
                        _read(plan, folder / filename)
            version = pyproject.get('project', {}).get('requires-python')
            if version:
                python_versions.append(version)
            poetry = pyproject.get('tool', {}).get('poetry', {})
            uv = pyproject.get('tool', {}).get('uv', {})
            if poetry.get('source') or any(
                uv.get(key) for key in ('index', 'index-url', 'extra-index-url', 'find-links')
            ):
                raise PreparationError(
                    '项目使用自定义 Python 包源，自动准备尚不支持；不会改用公开包源代替。'
                )
            if isinstance(poetry.get('dependencies', {}).get('python'), str):
                python_versions.append(poetry['dependencies']['python'])
            for dep in poetry.get('dependencies', {}).values():
                if isinstance(dep, dict) and dep.get('path'):
                    _local(plan, folder, dep['path'])
            for dep in pyproject.get('tool', {}).get('uv', {}).get('sources', {}).values():
                if isinstance(dep, dict) and dep.get('path'):
                    _local(plan, folder, dep['path'])
        requirements = []
        for name in (
            'requirements.txt',
            'requirements-dev.txt',
            'requirements-test.txt',
            'requirements-tests.txt',
        ):
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
            plan.projects.append(
                {
                    'path': folder.as_posix(),
                    'language': 'python',
                    'manager': manager,
                    'requirements': requirements,
                    'has_package': (
                        'setup.py' in names
                        or 'setup.cfg' in names
                        or bool(pyproject.get('project', {}).get('name'))
                        or bool(pyproject.get('build-system'))
                        or bool(pyproject.get('tool', {}).get('poetry', {}).get('name'))
                    ),
                }
            )
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
            if manager_value and not re.fullmatch(
                r'(npm|pnpm|yarn)@\d+\.\d+\.\d+(?:\+[A-Za-z0-9.]+)?', manager_value
            ):
                raise PreparationError('packageManager 须指定 npm、pnpm 或 yarn 的确切版本。')
            manager = (
                manager_value.split('@')[0]
                if manager_value
                else (
                    'pnpm'
                    if 'pnpm-lock.yaml' in names
                    else 'yarn' if 'yarn.lock' in names else 'npm'
                )
            )
            plan.projects.append(
                {
                    'path': folder.as_posix(),
                    'language': 'node',
                    'needs_source': any(
                        key in package.get('scripts', {})
                        for key in ('preinstall', 'install', 'postinstall', 'prepare', 'prepublish')
                    ),
                    'manager': manager,
                    'package_manager': manager_value.split('+')[0],
                    'locked': bool(
                        names
                        & {
                            'package-lock.json',
                            'npm-shrinkwrap.json',
                            'pnpm-lock.yaml',
                            'yarn.lock',
                        }
                    ),
                    'workspaces': bool(package.get('workspaces')) or 'pnpm-workspace.yaml' in names,
                }
            )
    # npm/pnpm/yarn 工作区由声明它们的根项目统一安装。
    # 子项目清单仍计入指纹，但不能单独解析安装。
    workspace_roots = [
        Path(p['path']) for p in plan.projects if p['language'] == 'node' and p['workspaces']
    ]
    for p in plan.projects:
        if p['language'] == 'node':
            p['workspace_member'] = any(
                Path(p['path']) != parent and Path(p['path']).is_relative_to(parent)
                for parent in workspace_roots
            )
    for relative in plan.local_dependencies:
        target = safe_path(root, relative, file=False)
        if target.is_file():
            if excluded(Path(relative)) or target.stat().st_size > 32 * 1024 * 1024:
                raise PreparationError(f'明确引用的本地包不可用于安装：{relative}')
            plan.files[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        else:
            for rel, data in source_files(root, under=Path(relative), project_scope=scope):
                plan.files[rel.as_posix()] = hashlib.sha256(data).hexdigest()
    # 安装生命周期可能将这些源码编入依赖树，变化时必须重新准备。
    for project in plan.projects:
        if project.get('needs_source'):
            for relative, data in source_files(root, under=Path(project['path']), project_scope=scope):
                plan.files[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    plan.python, plan.node = _version(python_versions, 'python'), _version(node_versions, 'node')
    return plan


# ============================================================
# 构造经过过滤的安装副本
# ============================================================
def source_files(root, *, under=Path('.'), project_scope=None):
    """创建有边界且过滤私有数据的副本，不运行 Git 钩子或项目代码。"""
    from agent.execution.installation_materials import project_files

    count = total = 0
    for relative in project_files(root, under, project_scope=project_scope):
        path = safe_path(root, relative)
        size = path.stat().st_size
        count += 1
        total += size
        if count > MAX_FILES or total > MAX_BYTES or size > 32 * 1024 * 1024:
            raise PreparationError(
                f'安装资料过大：{relative.as_posix()}，文件 {size:,} 字节，累计 {total:,} 字节；'
                f'限制为单文件 33,554,432 字节、总计 {MAX_BYTES:,} 字节、{MAX_FILES:,} 个文件。'
                '纳入原因：项目源码或明确声明的本地包。'
            )
        data = path.read_bytes()
        if len(data) != size:
            raise PreparationError(f'准备期间项目文件正在变化：{relative.as_posix()}')
        if any(marker in data for marker in PRIVATE_MARKERS) or SECRET_ASSIGNMENT.search(data):
            continue
        yield relative, data
