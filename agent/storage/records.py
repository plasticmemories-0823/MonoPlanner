"""原子 JSON 记录与有界摘录；原始内容保留在磁盘。"""

import json
import hashlib
import os
from pathlib import Path
import uuid


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def write_bytes(path, raw):
    """先刷盘，再原子发布；半写临时文件永远不作为可恢复回执。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if os.name == 'nt':
            import ctypes

            move = ctypes.WinDLL('kernel32', use_last_error=True).MoveFileExW
            move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
            move.restype = ctypes.c_int
            if not move(str(temporary), str(path), 0x1 | 0x8):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def immutable(path, raw):
    """调用方持有所属任务租约；重复发布只能是相同原件。"""
    path = Path(path)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError('Immutable record changed: ' + str(path))
    else:
        write_bytes(path, raw)


def save(path, data):
    write_bytes(path, json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))


def clip(text, maximum=6000):
    if len(text) <= maximum:
        return text
    return (
        text[: maximum // 2]
        + "\n[excerpt; complete content retained in record]\n"
        + text[-maximum // 2 :]
    )
