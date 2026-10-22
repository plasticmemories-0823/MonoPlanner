"""固定版本 rg 的只读执行与 UTF-8 原件窗口；不继承 Shell 或用户输入。"""

from bisect import bisect_right
import codecs
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import subprocess
import tempfile
from threading import Event, Thread
import time
import unicodedata

from agent.task.control import check_interrupted


ROOT = Path(__file__).resolve().parents[2]
MAX_ROW_BYTES = 4 * 1024 * 1024
MAX_MATCH_BYTES = 32768
WORD_CHARACTER = r"[\p{L}\p{N}\p{M}\p{Pc}\x{200c}\x{200d}]"


class SearchError(ValueError):
    def __init__(self, code, message=""):
        self.code = code
        super().__init__(message or code)


def executable():
    """只接受随产品准备且字节匹配固定清单的运行时。"""
    path = ROOT / "launcher/runtime/ripgrep/rg.exe"
    manifest = ROOT / "resources/ripgrep.json"
    try:
        expected = json.loads(manifest.read_text("utf-8"))["executable_sha256"]
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
    except (OSError, ValueError, KeyError) as error:
        raise SearchError("ripgrep_unavailable") from error
    if actual != expected:
        raise SearchError("ripgrep_integrity_failed")
    return path


def word_character(char):
    """Unicode 字母、数字、组合标记、连接标点与连接控制符组成完整词。"""
    return bool(char) and (
        unicodedata.category(char)[0] in "LNM"
        or unicodedata.category(char) == "Pc"
        or char in "\u200c\u200d"
    )


def literal_pattern(terms, whole_word=True):
    # 只在查询的词字符外缘加边界；内部标点、空白和零宽重叠枚举不变。
    patterns = []
    for term in terms:
        left = "(?<!" + WORD_CHARACTER + ")" if whole_word and word_character(term[0]) else ""
        right = "(?!" + WORD_CHARACTER + ")" if whole_word and word_character(term[-1]) else ""
        body = "".join(r"\x{" + format(ord(char), "x") + "}" for char in term)
        patterns.append(left + "(?=" + body + right + ")")
    return "|".join(patterns)


def literal_boundaries(original, start, end, term):
    """分别复核每个候选词，避免长词命中时重新加入它的短子串。"""
    if word_character(term[0]) and start:
        # 原件已完整验证为 UTF-8；最多四字节的片段只可能在外缘切开字符。
        before = original.bytes_at(max(0, start - 4), min(4, start)).decode("utf-8", "ignore")
        if before and word_character(before[-1]):
            return False
    if word_character(term[-1]) and end < original.size:
        after = original.bytes_at(end, 4).decode("utf-8", "ignore")
        if after and word_character(after[0]):
            return False
    return True


def rows(program, files, query, *, timeout=30):
    """有界读取 JSON 行，主线程负责超时、暂停及所属子进程回收。"""
    terms = query.terms if query.mode == "keywords" else (query.text,)
    pattern = literal_pattern(terms, query.match == "whole_word")
    command = [
        str(program), "--no-config", "--json", "--color", "never", "--threads", "1",
        "--no-ignore", "--hidden", "--no-mmap", "--encoding", "none",
        "--multiline", "--pcre2",
        "-e", pattern, "--", *map(str, files),
    ]
    environment = dict(os.environ)
    environment.pop("RIPGREP_CONFIG_PATH", None)
    queue, done = Queue(maxsize=2), Event()

    def deliver(value):
        while not done.is_set():
            try:
                queue.put(value, timeout=0.1)
                return
            except Full:
                continue

    with tempfile.TemporaryDirectory(prefix="monoplanner-rg-") as pattern_directory, \
            tempfile.TemporaryFile() as errors:
        pattern_file = Path(pattern_directory) / "patterns.txt"
        pattern_file.write_text(pattern + "\n", encoding="utf-8")
        index = command.index("-e")
        command[index:index + 2] = ["-f", str(pattern_file)]
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors,
            env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def read():
            try:
                while not done.is_set():
                    line = process.stdout.readline(MAX_ROW_BYTES + 1)
                    if not line:
                        break
                    if len(line) > MAX_ROW_BYTES:
                        deliver(SearchError("ripgrep_output_line_budget"))
                        return
                    deliver(line)
            except Exception as error:
                deliver(error)
            finally:
                deliver(None)

        worker = Thread(target=read, daemon=True)
        worker.start()
        deadline = time.monotonic() + timeout
        try:
            while True:
                check_interrupted(pause_only=True)
                if time.monotonic() >= deadline:
                    raise SearchError("ripgrep_timeout")
                try:
                    line = queue.get(timeout=0.1)
                except Empty:
                    continue
                if line is None:
                    break
                if isinstance(line, Exception):
                    raise line
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError) as error:
                    raise SearchError("ripgrep_invalid_response") from error
                yield row
            code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
            if code not in (0, 1):
                errors.seek(0)
                detail = errors.read(4096).decode("utf-8", "replace")
                raise SearchError("ripgrep_query_failed", detail)
        finally:
            done.set()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            worker.join(timeout=1)
            process.stdout.close()


class Original:
    """小块建立字节到字符的稀疏坐标；只为交付命中读取正文窗口。"""

    def __init__(self, path, expected_hash, *, max_bytes):
        self.path = Path(path)
        self.expected_hash = expected_hash
        signature = self._signature()
        self.size = self.path.stat().st_size
        if self.size > max_bytes:
            raise SearchError("source_scan_file_budget")
        self.points = [(0, 0, 1, 0)]
        self.binary = False
        self.chars = 0
        self.lines = 1
        last_line = 0
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
                if "\n" in text:
                    last_line = self.chars + text.rfind("\n") + 1
                self.chars += len(text)
                self.lines += text.count("\n")
                self.points.append(
                    (consumed - len(decoder.getstate()[0]), self.chars, self.lines, last_line)
                )
            decoder.decode(b"", final=True)
        if digest.hexdigest() != expected_hash:
            raise SearchError("source_version_changed")
        if self._signature() != signature:
            raise SearchError("source_version_changed")
        self.signature = signature
        self.size = consumed
        self.offsets = [point[0] for point in self.points]

    def _signature(self):
        value = self.path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_ino

    def verify(self):
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            while block := stream.read(65536):
                check_interrupted(pause_only=True)
                digest.update(block)
        if digest.hexdigest() != self.expected_hash:
            raise SearchError("source_version_changed")

    def bytes_at(self, start, length):
        with self.path.open("rb") as stream:
            stream.seek(start)
            return stream.read(length)

    def position(self, offset):
        if not 0 <= offset <= self.size:
            raise SearchError("ripgrep_invalid_range")
        base, chars, line, line_start = self.points[
            bisect_right(self.offsets, offset) - 1
        ]
        text = self.bytes_at(base, offset - base).decode("utf-8")
        if "\n" in text:
            line_start = chars + text.rfind("\n") + 1
        return chars + len(text), line + text.count("\n"), line_start

    def window(self, start, end):
        if self._signature() != self.signature:
            raise SearchError("source_version_changed")
        if end - start > MAX_MATCH_BYTES:
            raise SearchError("match_message_budget")
        first, line, line_start = self.position(start)
        last, line_end, end_line_start = self.position(end)
        base = self.points[bisect_right(self.offsets, max(0, start - 384)) - 1][0]
        raw = self.bytes_at(base, min(self.size, end + 384) - base)
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        text = decoder.decode(raw, final=base + len(raw) == self.size)
        before = raw[:start - base].decode("utf-8")
        through = raw[:end - base].decode("utf-8")
        local_start, local_end = len(before), len(through)
        left = max(before.rfind("\n") + 1, local_start - 96)
        next_line = text.find("\n", local_end)
        right = min(len(text) if next_line < 0 else next_line, local_end + 96)
        # 完整字面匹配必须保留可交付的原文窗口。
        if right == left:
            left = max(0, left - 1)
            right = min(len(text), right + 1)
        if self._signature() != self.signature:
            raise SearchError("source_version_changed")
        return {
            "text": text[left:right], "start": first - (local_start - left),
            "end": last + right - local_end, "match_start": first, "match_end": last,
            "line": line, "line_end": line_end,
            "column_start": first - line_start + 1,
            "column_end": last - end_line_start + 1,
            "truncated": left > before.rfind("\n") + 1 or (
                next_line >= 0 and right < next_line
            ) or (next_line < 0 and base + len(raw) < self.size),
        }
