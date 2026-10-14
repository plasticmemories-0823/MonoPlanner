"""执行目标明确的单文件操作，不为生成代码临时提升权限。"""

from contextlib import ExitStack, contextmanager
import difflib
import hashlib
import os
from pathlib import Path
import shutil

from agent.settings import PROJECT_ROOT
from agent.source.snapshots import _link, _reason, _inspect_file
from agent.storage.preferences import state_root, private_path


# ============================================================
# 目标路径和原件状态检查
# ============================================================
def target_path(value, *, missing=False, readonly=False):
    path = Path(value)
    if (
        not value
        or not path.is_absolute()
        or '..' in path.parts
        or str(path).startswith(('\\\\', '//'))
        or ':' in str(path)[2:]
    ):
        raise ValueError('必须指定明确的本机绝对文件路径，不接受设备、网络或备用数据流路径')
    for part in (path, *path.parents):
        if part.exists() and _link(part):
            raise ValueError('目标不能经过链接或 Windows 重解析点')
    path = path.resolve()
    if private_path(path):
        raise ValueError('不能操作助手私有记录或凭据')
    protected = [state_root()] + ([] if readonly else [PROJECT_ROOT])
    protected.extend(
        Path(os.environ[n])
        for n in ('SystemRoot', 'ProgramFiles', 'ProgramFiles(x86)', 'ProgramData')
        if n in os.environ
    )
    if any(path.is_relative_to(p.resolve()) for p in protected):
        raise ValueError('不能操作助手私有目录或系统程序目录')
    if path.suffix.lower() == '.dpapi' or _reason(Path(path.name)) in {
        'environment_credentials',
        'possible_private_key',
    }:
        raise ValueError('不能操作密钥或环境凭据')
    if not path.parent.is_dir():
        raise ValueError('目标父目录必须已存在；此工具不创建目录')
    if path.exists():
        if not path.is_file() or path.stat().st_nlink != 1:
            raise ValueError('只支持独立普通文件')
        if _inspect_file(path)[1]:
            raise ValueError('不能操作含私钥的文件')
    elif not missing:
        raise ValueError('来源文件不存在')
    return path


def state(path):
    if not path.exists():
        return {'exists': False}
    info = path.stat()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {
        'exists': True,
        'sha256': sha,
        'identity': [info.st_dev, info.st_ino],
        'bytes': info.st_size,
    }


# ============================================================
# 准备可核对的单文件操作
# ============================================================
def prepare(args, repo):
    operation = args['operation']
    if operation not in {'create', 'replace', 'copy', 'move', 'delete'}:
        raise ValueError('operation 必须为 create、replace、copy、move 或 delete')
    if operation in {'copy', 'move'}:
        if not args['destination'] or args['content']:
            raise ValueError('复制或移动需要 destination，content 必须留空')
    elif args['destination'] or (operation == 'delete' and args['content']):
        raise ValueError('此操作的 destination 必须留空；删除的 content 也须留空')
    path = target_path(args['path'], missing=operation == 'create')
    destination = (
        target_path(args['destination'], missing=True) if operation in {'copy', 'move'} else path
    )
    if (operation == 'create' or operation in {'copy', 'move'}) and destination.exists():
        raise ValueError('新目标已存在，不能隐式覆盖；覆盖须使用 replace')
    workspace = Path(repo).resolve() if repo else None
    changed = [destination] + ([path] if operation in {'move', 'delete'} else [])
    if workspace and any(p.is_relative_to(workspace) for p in changed):
        raise ValueError('工作目录中的修改仍须使用编码流程，不能用文件操作绕过验收')
    content = args['content'].encode('utf-8')
    if len(content) > 2_000_000:
        raise ValueError('直接创建或替换的文本最多 2 MB；大文件请使用复制')
    from agent.source.snapshots import _PRIVATE_MARKERS

    if any(marker in content for marker in _PRIVATE_MARKERS):
        raise ValueError('不能写入私钥')
    before = {str(p): state(p) for p in {path, destination}}
    old = ''
    if operation == 'replace' and path.stat().st_size <= 2_000_000:
        old = path.read_bytes().decode('utf-8', errors='replace')
    diff = ''.join(
        difflib.unified_diff(
            old.splitlines(True),
            args['content'].splitlines(True),
            fromfile=str(path),
            tofile=str(destination),
        )
    )[:8000]
    return {
        'kind': 'file_operation',
        **args,
        'path': str(path),
        'destination': str(destination),
        'before': before,
        'content_sha256': hashlib.sha256(content).hexdigest(),
        'diff': diff,
    }


# ============================================================
# 执行期间的文件与目录锁
# ============================================================
@contextmanager
def locked_file(path, write=False, delete=False):
    """检查和修改文件内容期间，禁止其他 Windows 进程写入或重命名该文件。"""
    if os.name != 'nt':
        with path.open('r+b' if write else 'rb') as stream:
            yield stream
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt

    create = ctypes.WinDLL('kernel32', use_last_error=True).CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    create.restype = wintypes.HANDLE
    handle = create(
        str(path),
        0x80000000 | (0x40000000 if write else 0) | (0x10000 if delete else 0),
        1,
        None,
        3,
        0x00200000,
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    fd = msvcrt.open_osfhandle(handle, os.O_BINARY | (os.O_RDWR if write else os.O_RDONLY))
    with os.fdopen(fd, 'r+b' if write else 'rb') as stream:
        yield stream


@contextmanager
def locked_parents(paths):
    if os.name != 'nt':
        yield
        return
    import ctypes
    from ctypes import wintypes

    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    api.CreateFileW.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = []
    try:
        for path in sorted(
            {p for value in paths for p in value.parents}, key=lambda p: len(p.parts)
        ):
            if _link(path):
                raise ValueError('目标父目录已变为链接')
            handle = api.CreateFileW(str(path), 0x80, 3, None, 3, 0x02200000, None)
            if handle == wintypes.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
            if _link(path):
                raise ValueError('目标父目录已变化')
        yield
    finally:
        for handle in reversed(handles):
            api.CloseHandle(handle)


def delete_open_file(stream, path):
    if os.name != 'nt':
        path.unlink()
        return
    import ctypes
    from ctypes import wintypes
    import msvcrt

    api = ctypes.WinDLL('kernel32', use_last_error=True).SetFileInformationByHandle
    api.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    value = ctypes.c_byte(1)
    if not api(msvcrt.get_osfhandle(stream.fileno()), 4, ctypes.byref(value), ctypes.sizeof(value)):
        raise ctypes.WinError(ctypes.get_last_error())


class FileActionError(OSError):
    def __init__(self, error, changed):
        super().__init__(str(error))
        self.changed = changed


# ============================================================
# 再次校验后执行批准的操作
# ============================================================
def execute(data, backup):
    path, destination = Path(data['path']), Path(data['destination'])
    operation, changed = data['operation'], False
    try:
        with locked_parents({path, destination}), ExitStack() as stack:
            for value, expected in data['before'].items():
                checked = target_path(value, missing=not expected['exists'])
                if state(checked) != expected:
                    raise ValueError('文件已变化，请重新请求并批准：' + value)
            stream = None
            if operation != 'create':
                stream = stack.enter_context(
                    locked_file(
                        path, write=operation == 'replace', delete=operation in {'move', 'delete'}
                    )
                )
                info = os.fstat(stream.fileno())
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                expected = data['before'][str(path)]
                if sha != expected['sha256'] or [info.st_dev, info.st_ino] != expected['identity']:
                    raise ValueError('文件已变化，请重新请求并批准')
                stream.seek(0)
                if operation in {'replace', 'move', 'delete'}:
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    with backup.open('xb') as original:
                        shutil.copyfileobj(stream, original)
                        original.flush()
                        os.fsync(original.fileno())
                    stream.seek(0)
            if operation in {'create', 'copy', 'move'}:
                with destination.open('xb') as output:
                    changed = True
                    if operation == 'create':
                        output.write(data['content'].encode('utf-8'))
                    else:
                        shutil.copyfileobj(stream, output)
                    output.flush()
                    os.fsync(output.fileno())
            elif operation == 'replace':
                changed = True
                stream.write(data['content'].encode('utf-8'))
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
            if operation in {'move', 'delete'}:
                delete_open_file(stream, path)
                changed = True
        return {'status': 'completed', 'backup': str(backup) if backup.exists() else None}
    except (OSError, ValueError) as error:
        raise FileActionError(error, changed) from error
