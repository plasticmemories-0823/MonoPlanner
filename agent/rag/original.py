"""受限 UTF-8 原件读取和哈希核验，供证据引用审计使用。"""

import codecs
import hashlib
from pathlib import Path

from agent.task.control import check_interrupted


class SearchError(ValueError):
    def __init__(self, code, message=""):
        self.code = code
        super().__init__(message or code)



class Original:
    """逐块核对原文哈希和字符数；引用审计不分配整份正文。"""

    def __init__(self, path, expected_hash, *, max_bytes):
        self.path = Path(path)
        self.expected_hash = expected_hash
        signature = self._signature()
        self.size = self.path.stat().st_size
        if self.size > max_bytes:
            raise SearchError("source_scan_file_budget")
        self.binary = False
        self.chars = 0
        consumed = 0
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            while block := stream.read(65536):
                check_interrupted(pause_only=True)
                consumed += len(block)
                if consumed > max_bytes:
                    raise SearchError("source_scan_file_budget")
                if b"\0" in block:
                    self.binary = True
                    return
                digest.update(block)
                text = decoder.decode(block)
                self.chars += len(text)
            decoder.decode(b"", final=True)
        if digest.hexdigest() != expected_hash:
            raise SearchError("source_version_changed")
        if self._signature() != signature:
            raise SearchError("source_version_changed")
        self.signature = signature
        self.size = consumed

    def _signature(self):
        value = self.path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino
