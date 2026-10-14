"""宿主源码物化失败属于基础设施错误，不是一次可由模型纠错的命令失败。"""

from functools import wraps


class SourceStorageError(OSError):
    pass


def source_storage_boundary(operation):
    @wraps(operation)
    def guarded(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except (InterruptedError, SourceStorageError):
            raise
        except OSError as exc:
            raise SourceStorageError(exc.errno, str(exc), exc.filename) from exc
    return guarded
