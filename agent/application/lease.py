"""同一数据目录只有一个应用执行者；展示客户端不持有任务执行租约。"""

import os


class LeaseBusy(RuntimeError):
    """原执行者仍持有租约；这不表示任务故障，也不能启动第二个执行者。"""


class ServiceLease:
    def __init__(self, root):
        self.path = root / '.application.lock'
        self.handle = None

    def acquire(self):
        if self.path.is_symlink():
            raise ValueError('Application lock cannot be a link')
        handle = self.path.open('a+b')
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise LeaseBusy('该数据目录已有运行中的 MonoPlanner 服务；请连接原服务或选择独立数据目录') from exc
        self.handle = handle

    def close(self):
        if self.handle:
            self.handle.seek(0)
            if os.name == 'nt':
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
