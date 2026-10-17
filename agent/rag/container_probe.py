"""容器内独立运行的固定只读协议；不导入项目模块，不执行用户命令。"""

import base64
import hashlib
import heapq
import json
import math
import os
import signal
import stat
import sys


MAX_BYTES = 7_864_320
MAX_REQUEST_BYTES = 16_384
MAX_LIST_BYTES = 1_048_576
MAX_ENTRIES = 1000


class ProbeError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _absolute(value):
    if (not isinstance(value, str) or not value or len(value) > 4096
            or '\0' in value or not os.path.isabs(value)
            or '..' in value.replace('\\', '/').split('/')):
        raise ProbeError('invalid_path', 'An absolute path without traversal is required.')
    return os.path.normpath(value)


def _inside(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


def _resolve(path, allowed_root, denied_roots):
    """链接逐段核验；授权根及实际目标均不能进入拒绝范围。"""
    path, allowed_root = _absolute(path), _absolute(allowed_root)
    if not _inside(path, allowed_root):
        raise ProbeError('outside_root', 'The path is outside the authorized source root.')
    root = os.path.realpath(allowed_root)
    denied = []
    for item in denied_roots:
        item = _absolute(item)
        denied.extend([item, os.path.realpath(item)])
    current = allowed_root
    parts = os.path.relpath(path, allowed_root).split(os.sep)
    for part in [''] + ([] if parts == ['.'] else parts):
        if part:
            current = os.path.join(current, part)
        os.lstat(current)
        resolved = os.path.realpath(current)
        if not _inside(resolved, root):
            raise ProbeError('outside_root', 'A symbolic link escapes the authorized source root.')
        if any(_inside(candidate, deny) for candidate in (current, resolved) for deny in denied):
            raise ProbeError('denied_path', 'The path is excluded from container retrieval.')
    resolved = os.path.realpath(path)
    if os.name == 'posix':
        descriptor = _open_source(resolved)
        try:
            info = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    else:
        info = os.stat(resolved)
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise ProbeError('special_file', 'Only regular files and directories may be retrieved.')
    return path, resolved, info


def _metadata(path, resolved, info):
    result = {
        'path': path, 'realpath': resolved, 'size': info.st_size,
        'mtime_ns': info.st_mtime_ns,
        'type': 'directory' if stat.S_ISDIR(info.st_mode) else 'file',
    }
    if os.path.islink(path):
        result['target_type'] = result['type']
        result['type'] = 'symlink'
    return result


def _fingerprint(info):
    # Windows 的 stat/fstat 对 ctime 的创建时间与变更时间口径可能不同。
    changed = info.st_ctime_ns if os.name == 'posix' else 0
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, changed)


def _open_source(resolved):
    """Linux 用目录句柄和 NOFOLLOW 避免路径核验后祖先被替换。"""
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NONBLOCK', 0)
    if os.name == 'posix' and hasattr(os, 'O_NOFOLLOW') and os.open in os.supports_dir_fd:
        if resolved == '/':
            return os.open('/', flags | os.O_DIRECTORY | os.O_NOFOLLOW)
        parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
        try:
            pieces = resolved.strip('/').split('/')
            for part in pieces[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                dir_fd=parent)
                os.close(parent)
                parent = child
            return os.open(pieces[-1], flags | os.O_NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
    return os.open(resolved, flags | getattr(os, 'O_NOFOLLOW', 0))


def _read(payload, path, resolved, info):
    maximum = payload.get('max_bytes', MAX_BYTES)
    if type(maximum) is not int or not 0 <= maximum <= MAX_BYTES:
        raise ProbeError('invalid_limit', 'max_bytes exceeds the fixed read budget.')
    if not stat.S_ISREG(info.st_mode):
        raise ProbeError('not_file', 'The requested source is not a regular file.')
    if info.st_size > maximum:
        raise ProbeError('source_too_large', 'The source exceeds the allowed read bytes.')
    descriptor = _open_source(resolved)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or _fingerprint(before) != _fingerprint(info):
            raise ProbeError('source_changed', 'The source changed before reading.')
        chunks, size, digest = [], 0, hashlib.sha256()
        while True:
            chunk = os.read(descriptor, min(65536, maximum + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise ProbeError('source_too_large', 'The source grew beyond the read budget.')
            chunks.append(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        _, current, checked = _resolve(path, payload['allowed_root'], payload['denied_roots'])
        if (current != resolved or _fingerprint(before) != _fingerprint(after)
                or _fingerprint(before) != _fingerprint(checked) or size != before.st_size):
            raise ProbeError('source_changed', 'The source changed while reading.')
        return dict(_metadata(path, resolved, after), sha256=digest.hexdigest(),
                    data_base64=base64.b64encode(b''.join(chunks)).decode('ascii'))
    finally:
        os.close(descriptor)


def _list(payload, path, resolved, info):
    if not stat.S_ISDIR(info.st_mode):
        raise ProbeError('not_directory', 'The requested source is not a directory.')
    limit, after = payload.get('limit', 200), payload.get('after', '')
    if type(limit) is not int or not 1 <= limit <= MAX_ENTRIES:
        raise ProbeError('invalid_limit', 'limit must be between 1 and 1000 entries.')
    if (not isinstance(after, str) or len(after) > 4096
            or any(value in after for value in ('/', '\\', '\0')) or after in {'.', '..'}):
        raise ProbeError('invalid_cursor', 'after must be an exact directory entry name.')
    descriptor = _open_source(resolved) if os.name == 'posix' else None
    try:
        if descriptor is not None and _fingerprint(os.fstat(descriptor)) != _fingerprint(info):
            raise ProbeError('source_changed', 'The directory changed before listing.')
        with os.scandir(descriptor if descriptor is not None else resolved) as iterator:
            # 只保存 limit+1 个名字；大目录不会一次性创建无界列表。
            names = heapq.nsmallest(limit + 1,
                                   (entry.name for entry in iterator if entry.name > after))
        if descriptor is not None and _fingerprint(os.fstat(descriptor)) != _fingerprint(info):
            raise ProbeError('source_changed', 'The directory changed while listing.')
    finally:
        if descriptor is not None:
            os.close(descriptor)
    entries, used = [], 0
    for name in names[:limit]:
        child = os.path.join(path, name)
        try:
            actual, target, child_info = _resolve(
                child, payload['allowed_root'], payload['denied_roots'])
            entry = dict(_metadata(actual, target, child_info), name=name)
        except (OSError, ProbeError) as error:
            entry = {'name': name, 'path': child, 'type': 'blocked',
                     'reason': getattr(error, 'code', 'source_unavailable')}
        size = len(json.dumps(entry, ensure_ascii=True).encode('utf-8'))
        if used + size > MAX_LIST_BYTES:
            break
        entries.append(entry)
        used += size
    _, current, checked = _resolve(path, payload['allowed_root'], payload['denied_roots'])
    if current != resolved or _fingerprint(info) != _fingerprint(checked):
        raise ProbeError('source_changed', 'The directory changed while listing.')
    complete = len(entries) == len(names)
    return dict(_metadata(path, resolved, checked), entries=entries, complete=complete,
                next_after='' if complete else entries[-1]['name'])


def dispatch(request):
    if (not isinstance(request, dict) or not {'operation', 'payload'} <= set(request)
            or set(request) - {'operation', 'payload', 'deadline_seconds'}
            or request['operation'] not in {'stat', 'list', 'read'}):
        raise ProbeError('invalid_operation', 'Only stat, list, and read are supported.')
    operation, payload = request['operation'], request['payload']
    common = {'path', 'allowed_root', 'denied_roots'}
    extra = {'stat': set(), 'list': {'limit', 'after'}, 'read': {'max_bytes'}}[operation]
    if (not isinstance(payload, dict) or not common <= set(payload)
            or set(payload) - common - extra or not isinstance(payload['denied_roots'], list)
            or len(payload['denied_roots']) > 64):
        raise ProbeError('invalid_payload', 'Unsupported or missing read protocol fields.')
    path, resolved, info = _resolve(payload['path'], payload['allowed_root'], payload['denied_roots'])
    if operation == 'stat':
        return _metadata(path, resolved, info)
    return (_list if operation == 'list' else _read)(payload, path, resolved, info)


def _arm_deadline(seconds):
    if (type(seconds) not in (int, float) or not math.isfinite(seconds)
            or not 0 < seconds <= 25):
        raise ProbeError('invalid_deadline', 'The read-only deadline must be at most 25 seconds.')
    if hasattr(signal, 'setitimer') and hasattr(signal, 'SIGALRM'):
        def expired(_number, _frame):
            raise ProbeError('read_deadline', 'The read-only container query reached its deadline.')

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, seconds)


def main():
    try:
        if len(sys.argv) != 2 or len(sys.argv[1].encode('utf-8')) > MAX_REQUEST_BYTES:
            raise ProbeError('invalid_payload', 'The read protocol request is too large.')
        request = json.loads(sys.argv[1])
        _arm_deadline(request.get('deadline_seconds', 25) if isinstance(request, dict) else 25)
        result = {'ok': True, 'result': dispatch(request)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {'ok': False, 'error': {
            'code': getattr(error, 'code', 'source_unavailable'),
            'message': str(error)[:512],
        }}
    print(json.dumps(result, ensure_ascii=True, separators=(',', ':')))


if __name__ == '__main__':
    main()
