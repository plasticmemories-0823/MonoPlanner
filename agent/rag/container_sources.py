"""任务容器的只读资料授权、可续查清单与原件引用，不接收可执行正文。"""

import base64
import hashlib
import json
import time
import uuid
from pathlib import Path, PurePosixPath

from agent.rag.container_backend import ContainerBackend
from agent.rag.types import canonical, guard_source_read, source_read_limit
from agent.source.paging import ReadBudget, merged, unread
from agent.storage.records import save
from agent.storage.resources import _safe_tree, preflight_space
from agent.task import control


PREFIX = '$container/'
NAMESPACES = {'system_include', 'toolchain_src'}
TEMP_ROOTS = ('/usr/src', '/usr/local', '/opt', '/build')
SOURCE_PARTS = {'src', 'source', 'sources', 'include', 'includes', 'headers',
                'patches', 'docs', 'doc'}
DENIED_PARTS = {'.git', '.ssh', '.aws', '.azure', '.config', '.cache', '__pycache__',
                'node_modules', 'cache', 'caches', 'ccache', 'cmakefiles', 'target',
                'dist', 'out', 'output', 'outputs', 'logs', 'tmp', 'temp', 'credentials',
                'secrets', 'certs', 'certificates', 'private', 'tests-private', 'grader',
                'hidden_tests', 'hidden-tests', 'acceptance'}
DENIED_SUFFIXES = {'.key', '.pem', '.p12', '.pfx', '.crt', '.cer', '.der', '.jks',
                   '.o', '.obj', '.a', '.so', '.dll', '.exe', '.pyc', '.class',
                   '.zip', '.tar', '.gz', '.xz', '.db', '.sqlite', '.sqlite3', '.log'}
TEXT_SUFFIXES = {'.h', '.hh', '.hpp', '.hxx', '.c', '.cc', '.cpp', '.cxx', '.tcc',
                 '.inc', '.inl', '.rs', '.go', '.py', '.js', '.ts', '.java', '.rb',
                 '.cbl', '.cob', '.f', '.f90', '.s', '.asm', '.md', '.rst', '.txt',
                 '.patch', '.diff', '.cmake', '.in', '.m4'}
TTL_SECONDS = 600
SCAN_BATCH = 128
SCAN_SECONDS = 5
MAX_SOURCES = 10_000
CACHE_BYTES = 32 * 1024 * 1024
TERMINAL = {'verified_success', 'task_blocked', 'task_cancelled', 'chat_complete',
            'existing_complete', 'stalled', 'budget_exhausted', 'preparation_failed',
            'cancelled', 'completed'}


class ContainerSourceError(ValueError):
    def __init__(self, code, path='', *, authorization=False):
        self.details = {'reason': code, 'path': path,
                        'requires_authorization': authorization}
        super().__init__(canonical(self.details))


def is_container_path(path):
    return isinstance(path, str) and path.startswith(PREFIX)


def _hash(value):
    raw = value if isinstance(value, bytes) else canonical(value).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _within(path, root):
    return path == root or path.startswith(root.rstrip('/') + '/')


def split_path(path):
    if not is_container_path(path) or '\\' in path or '\0' in path or ':' in path:
        raise ContainerSourceError('invalid_container_source_path', str(path))
    parts = path[len(PREFIX):].split('/')
    if len(parts) < 3 or parts[0] not in NAMESPACES or any(p in {'', '.', '..'} for p in parts):
        raise ContainerSourceError('invalid_container_source_path', path)
    return parts[0], '/' + '/'.join(parts[1:])


def _excluded(path):
    parts = PurePosixPath(path).parts
    lowered = [p.lower() for p in parts]
    return (any(p in DENIED_PARTS or p == '.env' or p.startswith('.env.')
                or p.startswith('id_rsa') or p.startswith('id_ed25519') for p in lowered)
            or PurePosixPath(path).suffix.lower() in DENIED_SUFFIXES
            or any(p.endswith(('.so', '.so.6')) for p in lowered)
            or PurePosixPath(path).name.lower() in {'passwd', 'shadow', 'token', 'tokens',
                                                  '.npmrc', '.pypirc', 'credentials.json',
                                                  'credentials.yaml', 'kubeconfig'})


def _text_file(path, namespace):
    name = PurePosixPath(path).name
    suffix = PurePosixPath(path).suffix.lower()
    return (not _excluded(path) and (suffix in TEXT_SUFFIXES
            or (namespace == 'system_include' and not suffix)
            or name in {'Makefile', 'CMakeLists.txt', 'README', 'LICENSE', 'COPYING'}))


def validate_source_configuration(value):
    """仅宿主设置入口接受明确目录；模型正文和仓库配置不授予访问权。"""
    if not isinstance(value, list) or len(value) > 32:
        raise ContainerSourceError('container_sources_require_at_most_32_explicit_roots')
    result = []
    for item in value:
        if (not isinstance(item, dict) or set(item) - {'namespace', 'path', 'allow_mount'}
                or item.get('namespace') not in NAMESPACES
                or not isinstance(item.get('path'), str)):
            raise ContainerSourceError('invalid_trusted_container_source_configuration')
        root = item['path']
        split_path(PREFIX + item['namespace'] + root)
        if (root in {'/', '/usr', '/usr/local', '/usr/src', '/opt', '/build', '/etc', '/var'}
                or any(_within(root, p) for p in ('/proc', '/sys', '/dev', '/run', '/tmp',
                                                 '/root', '/home', '/tests', '/grader',
                                                 '/agent-artifacts')) or _excluded(root)
                or ('allow_mount' in item and type(item['allow_mount']) is not bool)):
            raise ContainerSourceError('container_source_configuration_excluded', root)
        entry = {'namespace': item['namespace'], 'path': root}
        if 'allow_mount' in item:
            entry['allow_mount'] = item['allow_mount']
        if entry not in result:
            result.append(entry)
    return result


def source_guidance(settings):
    roots = [{'namespace': 'system_include', 'path': path}
             for path in ('/usr/include', '/usr/local/include')]
    roots.extend(validate_source_configuration(settings.get('container_read_sources', [])))
    paths = [PREFIX + root['namespace'] + root['path'] for root in roots]
    return ('\nHost-registered container retrieval roots (availability is checked on explicit use): '
            + canonical(paths) + '. Search project/current-run evidence first. Select a container '
            'path only for a concrete external-source need; never expand scope after a zero result '
            'or use Shell as a retrieval fallback. A request requiring authorization must be '
            'reported with the exact directory; only host settings can grant it.')


def _store(root, path, value):
    raw = canonical(value).encode('utf-8')
    _safe_tree(path, root)
    preflight_space(path, len(raw) * 2)
    save(path, value)


def _read_json(root, path):
    return json.loads(_safe_tree(path, root).read_text('utf-8'))


def _put(root, path, raw):
    _safe_tree(path, root)
    if path.exists():
        if _hash(path.read_bytes()) != _hash(raw):
            raise ContainerSourceError('container_original_hash_mismatch')
        return
    preflight_space(path, len(raw))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


class ContainerSources:
    def __init__(self, owner):
        control.check_interrupted()
        self.owner = owner
        self.root = Path(owner.root)
        self.base = self.root / 'rag' / 'container'
        self.backend = ContainerBackend(owner.environment)
        self.binding = self.backend.binding()
        self.binding_key = _hash(self.binding)
        self.now = time.time()
        self._cache_used = 0
        self._collect_cache()
        self.state_path = self.base / 'grants.json'
        self.state = (_read_json(self.root, self.state_path) if self.state_path.exists()
                      else {'version': 1, 'grants': [], 'binding': self.binding_key})
        if self.state.get('binding') != self.binding_key:
            for grant in self.state['grants']:
                grant['stale'] = True
            self.state['binding'] = self.binding_key
            self._save()
        self.closed = (getattr(owner, 'state', {}).get('status') in TERMINAL
                       or self.state.get('terminated', False))
        self.configuration = validate_source_configuration(
            getattr(owner, 'runtime_settings', {}).get('container_read_sources', []))
        policy = _hash(self.configuration)
        if self.state.get('policy_sha256', policy) != policy:
            raise ContainerSourceError('container_source_policy_changed_start_new_task')
        if 'policy_sha256' not in self.state:
            self.state['policy_sha256'] = policy
            self._save()

    def _save(self):
        _store(self.root, self.state_path, self.state)

    def _collect_cache(self):
        """只回收本任务过期派生文件；原件和操作记录始终保留。"""
        for folder in ('scans', 'cache'):
            directory = self.base / folder
            if not directory.exists():
                continue
            for file in directory.iterdir():
                _safe_tree(file, self.root)
                if self.now - file.stat().st_mtime >= TTL_SECONDS:
                    file.unlink()
                elif folder == 'cache':
                    self._cache_used += file.stat().st_size

    def finish_scan(self, identity):
        """已交付结果有独立操作原件；完成的目录清单不再反复留档。"""
        path = self.base / 'scans' / (identity + '.json')
        _safe_tree(path, self.root)
        if path.exists():
            path.unlink()

    def _request(self, operation, path, grant, **extra):
        control.check_interrupted()
        # 后端每次派发核验容器ID、镜像及挂载；不重复查询不变的镜像元数据。
        return self.backend.call(operation, {'path': path, 'allowed_root': grant['root'],
                                 'denied_roots': grant['denied_roots'], **extra})

    def authorize(self, path):
        namespace, actual = split_path(path)
        if self.closed:
            raise ContainerSourceError('container_source_task_ended', path)
        if _excluded(actual):
            raise ContainerSourceError('container_source_excluded', path)
        declared = [{'namespace': 'system_include', 'path': p} for p in
                    ('/usr/include', '/usr/local/include')]
        declared.extend(self.configuration)
        matches = []
        for item in declared:
            root = item.get('path', '')
            if (item.get('namespace') not in NAMESPACES or not root.startswith('/')
                    or '..' in PurePosixPath(root).parts or _excluded(root) or root == '/'):
                raise ContainerSourceError('invalid_trusted_container_source_configuration')
            if item['namespace'] == namespace and _within(actual, root):
                matches.append(item)
        explicit = max(matches, key=lambda item: len(item['path'])) if matches else None
        if explicit:
            root, temporary = explicit['path'], False
        else:
            parts = PurePosixPath(actual).parts
            candidates = [p for p in TEMP_ROOTS if _within(actual, p) and actual != p]
            anchors = [i for i, p in enumerate(parts) if p.lower() in SOURCE_PARTS]
            if not candidates or not anchors:
                raise ContainerSourceError('container_source_authorization_required', path,
                                           authorization=True)
            root = str(PurePosixPath(*parts[:anchors[-1] + 1]))
            if root == '/usr/src' and _within(actual, root) and actual != root:
                root = (str(PurePosixPath(actual).parent)
                        if PurePosixPath(actual).suffix.lower() in TEXT_SUFFIXES else actual)
            if not any(_within(root, p) and root != p for p in candidates):
                raise ContainerSourceError('container_source_scope_too_broad', path)
            temporary = True
        denied = ['/proc', '/sys', '/dev', '/run', '/tmp', '/root', '/home', '/tests',
                  '/grader', '/agent-artifacts']
        for mount in self.binding.get('mounts', []):
            target = mount.get('Destination', mount.get('destination', ''))
            if target and (mount.get('RW', mount.get('rw', True))
                           or not (explicit and explicit.get('allow_mount') is True)):
                denied.append(target)
        if any(_within(actual, p) for p in denied):
            raise ContainerSourceError('container_mount_authorization_required', path,
                                       authorization=True)
        grant = {'namespace': namespace, 'root': root, 'temporary': temporary,
                 'from_build_tree': _within(root, '/build'), 'binding': self.binding_key,
                 'binding_facts': self.binding, 'denied_roots': sorted(set(denied)),
                 'stale': False, 'expires_at_task_end': True,
                 'activation': {'kind': 'explicit_path', 'path': path},
                 'configuration_sha256': _hash(self.configuration)}
        info = self._request('stat', actual, grant)
        if _excluded(info['realpath']) or not _within(info['realpath'], root):
            raise ContainerSourceError('container_source_excluded', path)
        identity = (namespace, root, self.binding_key)
        existing = next((g for g in self.state['grants'] if
                         (g['namespace'], g['root'], g['binding']) == identity), None)
        if existing and existing.get('configuration_sha256') != grant['configuration_sha256']:
            raise ContainerSourceError('container_source_policy_changed_start_new_task', path)
        if not existing:
            self.state['grants'].append(grant)
            self._save()
        return grant, info

    def raw(self, path, grant=None, *, expected_version=None, expected_hash=None, archive=False):
        namespace, actual = split_path(path)
        grant = grant or self.authorize(path)[0]
        if not _text_file(actual, namespace):
            raise ContainerSourceError('container_source_not_permitted_text', path)
        value = self._request('read', actual, grant, max_bytes=source_read_limit())
        if _excluded(value['realpath']) or not _within(value['realpath'], grant['root']):
            raise ContainerSourceError('container_source_excluded', path)
        raw = base64.b64decode(value['data_base64'], validate=True)
        guard_source_read(len(raw), source_id='container:' + path, path=path)
        sha = _hash(raw)
        if sha != value['sha256']:
            raise ContainerSourceError('container_transport_hash_mismatch', path)
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError as error:
            raise ContainerSourceError('container_source_encoding_not_utf8', path) from error
        if '\0' in text or any(marker in text for marker in
                               ('-----BEGIN PRIVATE KEY', '-----BEGIN RSA PRIVATE KEY',
                                '-----BEGIN OPENSSH PRIVATE KEY', '-----BEGIN CERTIFICATE')):
            raise ContainerSourceError('container_source_secret_or_binary', path)
        version = _hash([self.binding_key, path, value['realpath'], sha])
        if ((expected_version is not None and expected_version != version)
                or (expected_hash is not None and expected_hash != sha)):
            raise ContainerSourceError('container_source_version_changed', path)
        meta = {'path': path, 'actual_path': actual, 'realpath': value['realpath'],
                'sha256': sha, 'version': version, 'binding_key': self.binding_key,
                'source_id': 'container:' + path, 'namespace': namespace,
                'origin': 'container/toolchain', 'temporary': grant['temporary'],
                'from_build_tree': grant['from_build_tree'], 'size': len(raw)}
        if archive:
            self.archive(raw, meta)
        return raw, meta

    def archive(self, raw, meta):
        destination = self.base / 'originals' / meta['sha256']
        cached = self.base / 'cache' / meta['sha256']
        _safe_tree(destination, self.root)
        _safe_tree(cached, self.root)
        if cached.exists() and not destination.exists():
            if _hash(cached.read_bytes()) != meta['sha256']:
                raise ContainerSourceError('container_original_hash_mismatch', meta['path'])
            destination.parent.mkdir(parents=True, exist_ok=True)
            cached.replace(destination)
            self._cache_used = max(0, self._cache_used - len(raw))
        else:
            _put(self.root, destination, raw)
            if cached.exists():
                cached.unlink()
                self._cache_used = max(0, self._cache_used - len(raw))
        reference = self.base / 'references' / (meta['version'] + '.json')
        if not reference.exists():
            _store(self.root, reference, meta)

    def _cache(self, raw, meta):
        original = self.base / 'originals' / meta['sha256']
        if original.exists():
            return
        cache = self.base / 'cache'
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / meta['sha256']
        exists = destination.exists()
        if not exists and self._cache_used + len(raw) > CACHE_BYTES:
            raise ContainerSourceError('container_scan_cache_budget', meta['path'])
        _put(self.root, destination, raw)
        if not exists:
            self._cache_used += len(raw)

    def cached(self, meta):
        for folder in ('originals', 'cache'):
            path = self.base / folder / meta['sha256']
            if path.exists():
                raw = _safe_tree(path, self.root).read_bytes()
                if _hash(raw) != meta['sha256']:
                    raise ContainerSourceError('container_original_hash_mismatch', meta['path'])
                return raw
        raise ContainerSourceError('container_scan_expired_restart_query', meta['path'])

    def scan(self, arguments):
        grant, info = self.authorize(arguments['path'])
        request = {k: v for k, v in arguments.items() if k != 'cursor'}
        request_key = _hash([request, self.binding_key, grant])
        cursor = arguments.get('cursor', '')
        if cursor.startswith('container-scan-v1:'):
            parts = cursor.split(':')
            if len(parts) != 3 or not all(c in '0123456789abcdef' for c in parts[1]) or len(parts[1]) != 32:
                raise ContainerSourceError('invalid_container_scan_cursor')
            file = self.base / 'scans' / (parts[1] + '.json')
            if not file.exists():
                raise ContainerSourceError('container_scan_cursor_stale')
            state = _read_json(self.root, file)
            if (state['request'] != request_key or str(state['sequence']) != parts[2]
                    or self.now - state['created'] >= TTL_SECONDS):
                raise ContainerSourceError('container_scan_cursor_stale')
        else:
            # 命中续页也重新核验清单与文件；旧检索游标只允许同一份来源版本。
            identity = uuid.uuid4().hex
            file = self.base / 'scans' / (identity + '.json')
            state = {'id': identity, 'request': request_key, 'sequence': 0,
                     'created': self.now, 'queue': [[info['path'], '']],
                     'entries': [], 'directories': {}, 'excluded': [], 'bytes': 0,
                     'search_cursor': cursor, 'verified': 0}
        for directory, signature in state['directories'].items():
            current = self._request('stat', directory, grant)
            if [current['realpath'], current['mtime_ns']] != signature:
                raise ContainerSourceError('container_directory_changed_restart_scan', directory)
        deadline, visited = time.monotonic() + SCAN_SECONDS, 0
        namespace = grant['namespace']
        while state['queue'] and visited < SCAN_BATCH and time.monotonic() < deadline:
            actual, after = state['queue'][0]
            current = self._request('stat', actual, grant)
            if current.get('target_type', current['type']) == 'file':
                state['queue'].pop(0)
                virtual = PREFIX + namespace + actual
                raw, meta = self.raw(virtual, grant)
                if len(state['entries']) >= MAX_SOURCES or state['bytes'] + len(raw) > CACHE_BYTES:
                    raise ContainerSourceError('container_source_scan_budget_narrow_path', virtual)
                self._cache(raw, meta)
                state['entries'].append(meta)
                state['bytes'] += len(raw)
                visited += 1
                del raw
                continue
            else:
                state['directories'][actual] = [current['realpath'], current['mtime_ns']]
                page = self._request('list', actual, grant, limit=min(32, SCAN_BATCH - visited), after=after)
                entries = page['entries']
                if page['complete']:
                    state['queue'].pop(0)
                else:
                    state['queue'][0][1] = page['next_after']
            for entry in entries:
                control.check_interrupted()
                visited += 1
                child = entry['path']
                kind = entry.get('target_type', entry['type'])
                if (_excluded(child) or _excluded(entry.get('realpath', child))
                        or entry['type'] == 'blocked'):
                    if len(state['excluded']) < 64:
                        state['excluded'].append({'path': child, 'reason': 'source_policy_excluded'})
                    continue
                if kind == 'directory':
                    # 目录链接不递归，避免循环及同一目录多次展开。
                    if entry['type'] != 'symlink':
                        state['queue'].append([child, ''])
                    continue
                if kind != 'file' or not _text_file(child, namespace):
                    continue
                state['queue'].append([child, ''])
        while (not state['queue'] and state['verified'] < len(state['entries'])
               and visited < SCAN_BATCH and time.monotonic() < deadline):
            meta = state['entries'][state['verified']]
            self.raw(meta['path'], grant, expected_version=meta['version'],
                     expected_hash=meta['sha256'])
            state['verified'] += 1
            visited += 1
        state['sequence'] += 1
        state['complete'] = not state['queue'] and state['verified'] == len(state['entries'])
        _store(self.root, file, state)
        if not state['complete']:
            next_cursor = 'container-scan-v1:' + state['id'] + ':' + str(state['sequence'])
            return None, {'query': arguments['query'], 'effective_query': arguments['query'],
                          'mode': arguments['mode'], 'hits': [], 'results': [],
                          'scan_complete': False, 'complete': False,
                          'retrieval_incomplete': True, 'next_cursor': next_cursor,
                          'reasons': ['container_source_enumeration_pending'],
                          'scanned_sources': len(state['entries']),
                          'notice': 'Source enumeration is incomplete. Continue with the same query '
                                    'and scope; no absence of evidence has been established.',
                          'rewrite': {'status': 'not_triggered'}}
        return state['entries'], {'grant': grant, 'excluded_sources': state['excluded'],
                                  'scan_id': state['id'], 'search_cursor': state['search_cursor']}


def container_original(owner, path, *, expected_version=None, expected_hash=None):
    return ContainerSources(owner).raw(path, expected_version=expected_version,
                                       expected_hash=expected_hash, archive=True)


def container_read_cursor(meta, cursor):
    if not cursor:
        return None
    value = {key: meta[key] for key in ('path', 'version', 'sha256', 'binding_key')}
    value['source_cursor'] = cursor
    return 'container-read-v1:' + base64.urlsafe_b64encode(canonical(value).encode()).decode()


def read_container(owner, path, offset, limit, *, budget=None, cursor=None, repeat=False,
                   expected_version=None, expected_hash=None):
    binding_key = None
    if cursor:
        if not cursor.startswith('container-read-v1:') or len(cursor) > 8192:
            raise ContainerSourceError('invalid_container_read_cursor', path)
        try:
            value = json.loads(base64.urlsafe_b64decode(cursor.split(':', 1)[1]))
            if value['path'] != path:
                raise ValueError('scope')
            expected_version, expected_hash = value['version'], value['sha256']
            binding_key, cursor = value['binding_key'], value['source_cursor']
        except (ValueError, KeyError, TypeError) as error:
            raise ContainerSourceError('invalid_container_read_cursor', path) from error
    raw, meta = container_original(owner, path, expected_version=expected_version,
                                   expected_hash=expected_hash)
    if binding_key and binding_key != meta['binding_key']:
        raise ContainerSourceError('container_read_binding_changed', path)
    budget = budget or ReadBudget()
    executor = getattr(owner, 'executor', owner)
    if not hasattr(executor, 'visible_reads'):
        executor.visible_reads = {}
    key = (path, meta['version'])
    seen = executor.visible_reads.get(key, [])
    stop = min(len(raw.decode('utf-8-sig').splitlines()), offset + limit)
    pending = [[offset, stop]] if repeat else unread(offset, stop, seen)
    if cursor or repeat or not seen or offset >= stop:
        result = budget.source(raw, offset, limit, cursor=cursor)
    elif pending:
        pages = []
        for start, end in pending:
            page = budget.source(raw, start, end - start)
            pages.append(page)
            if page['truncated']:
                break
        result = pages[0] if len(pages) == 1 else {
            'reads': pages, 'truncated': pages[-1]['truncated'],
            'next_cursor': pages[-1].get('next_cursor')}
    else:
        result = {'text': '', 'sha256': meta['sha256'], 'visible_ranges': [],
                  'already_visible': True, 'truncated': False, 'next_cursor': None,
                  'next_offset': stop, 'total_lines': len(raw.decode('utf-8-sig').splitlines())}
    pages = result.get('reads', [result])
    shown = [pair for page in pages for pair in page.get('visible_ranges', [])]
    executor.visible_reads[key] = merged([*seen, *shown])
    for page in pages:
        page.update(meta)
        page['next_cursor'] = container_read_cursor(meta, page.get('next_cursor'))
    result.update(meta)
    if 'reads' in result:
        result['next_cursor'] = pages[-1].get('next_cursor')
    result['continuation'] = {'reason': 'container_original', 'requested_offset': offset,
                              'requested_lines': limit, 'next_offset': pages[-1].get('next_offset')}
    return result


def expire_container_sources(owner):
    path = Path(owner.root) / 'rag/container/grants.json'
    if path.exists() and owner.state.get('status') in TERMINAL:
        value = _read_json(owner.root, path)
        value['terminated'] = True
        for grant in value['grants']:
            grant['stale'] = True
        _store(owner.root, path, value)
