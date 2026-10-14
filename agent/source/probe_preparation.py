"""模型请求等待期间，最多准备一个不可变的调查副本。"""

from concurrent.futures import CancelledError
from contextvars import copy_context
import hashlib
import os
from pathlib import Path
import stat
from threading import Event
import time
import uuid


def prepare_copy(view, directory, cancel):
    started = time.monotonic()
    snapshot = view.export(directory, cancel=cancel)
    return {
        'snapshot': snapshot,
        'version': view.version,
        'started': started,
        'ended': time.monotonic(),
    }


def validate_copy(snapshot):
    """计算完整新目录树的哈希，避免反复解析每个父目录。

    首条命令执行前不得出现任何新增文件，包括生成文件。
    遍历子目录前拒绝链接；命令执行后的常规审计保持不变。"""
    root = snapshot.root
    if not root.is_dir() or root.resolve() != root:
        raise ValueError('Prepared probe source integrity failed')
    found, folders = {}, [(root, Path())]
    while folders:
        folder, relative = folders.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode) or (
                    getattr(info, 'st_file_attributes', 0)
                    & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
                ):
                    raise ValueError('Prepared probe source integrity failed: link')
                name = relative / entry.name
                if stat.S_ISDIR(info.st_mode):
                    folders.append((Path(entry.path), name))
                elif stat.S_ISREG(info.st_mode):
                    with open(entry.path, 'rb') as source:
                        found[name.as_posix()] = hashlib.file_digest(source, 'sha256').hexdigest()
                else:
                    raise ValueError('Prepared probe source integrity failed: special file')
    if found != snapshot.files:
        raise ValueError('Prepared probe source integrity failed')


class ProbePreparation:
    """复用源码工作线程；只有所属执行者使用准备完成的副本。"""

    def __init__(self, pool, directory):
        self.pool, self.directory = pool, Path(directory)
        self.active = None
        self.jobs = []

    def start(self, view):
        if self.active is not None:
            if self.active['version'] == view.version and not self.active['cancel'].is_set():
                return
            self.cancel()
            if not self.active['future'].done():
                return
            self._finish('stale')
        cancel = Event()
        directory = self.directory / ('copy-' + uuid.uuid4().hex)
        self.active = {
            'version': view.version,
            'cancel': cancel,
            'future': self.pool.submit(copy_context().run, prepare_copy, view, directory, cancel),
        }

    def _finish(self, status):
        active, self.active = self.active, None
        try:
            result = active['future'].result()
        except CancelledError:
            self.jobs.append({'version': active['version'], 'status': 'cancelled'})
            return None
        except Exception as exc:
            self.jobs.append(
                {'version': active['version'], 'status': 'error', 'error': type(exc).__name__}
            )
            if status == 'consumed':
                raise
            return None
        self.jobs.append({k: v for k, v in result.items() if k != 'snapshot'} | {'status': status})
        if status in {'unused', 'stale'}:
            from agent.storage.resources import owner_cancelled, remove_private_tree

            if not owner_cancelled():
                remove_private_tree(result['snapshot'].root.parent, self.directory)
                return None
        return result['snapshot']

    def take(self, version):
        if self.active is None:
            return None
        if not self.active['future'].done():
            # 避免重复复制，与同步后备流程争用资源。
            self.cancel()
            return None
        if self.active['version'] != version or self.active['cancel'].is_set():
            self._finish('stale')
            return None
        snapshot = self._finish('consumed')
        if snapshot is not None:
            validate_copy(snapshot)
        return snapshot

    def cancel(self):
        if self.active is not None:
            self.active['cancel'].set()
            self.active['future'].cancel()

    def finish(self):
        if self.active is not None:
            self._finish('unused')
