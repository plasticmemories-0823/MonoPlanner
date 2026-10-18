"""只调用本地固定模型；后台建立真实向量，失败时明确降级。"""

from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

from agent.rag.types import RAG_PARENT_MEMORY_BYTES


GIB = 1024 ** 3
WORKER_BYTES = GIB - RAG_PARENT_MEMORY_BYTES
_INFERENCE = threading.Lock()
_BUILD = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rag-index')
_BUILD_LOCK = threading.Lock()
_JOBS = {}
_CLOSED_ROOTS = set()
_CLOSED_TASKS = set()
_OWNER = ContextVar('rag_background_owner', default=None)


@dataclass
class _BuildJob:
    owner: tuple[str, str]
    signal: object
    future: Future | None = None


def _owner(root, task_id):
    return (str(Path(root).resolve()), str(task_id))


def _scope_open(owner):
    if owner[0] in _CLOSED_ROOTS or owner in _CLOSED_TASKS:
        raise ModelUnavailable('RAG background dispatch is closed for this owner.')


def ensure_scope_open(root, task_id):
    with _BUILD_LOCK:
        _scope_open(_owner(root, task_id))


@contextmanager
def background_scope(root, task_id):
    """绑定真实任务所有者；复制上下文不能延长聊天注册的生命周期。"""
    owner = _owner(root, task_id)
    with _BUILD_LOCK:
        _scope_open(owner)
    token = _OWNER.set(owner)
    try:
        yield
    finally:
        _OWNER.reset(token)


def pending_jobs(root):
    key = str(Path(root).resolve())
    with _BUILD_LOCK:
        return sorted({job.owner[1] for job in _JOBS.values()
                       if job.owner[0] == key and not job.future.done()})


def cancel_jobs(root, task_ids=None, *, close=False, timeout=0, on_complete=None):
    """只停止归属匹配的任务；close 同锁封禁派发，删除前必须确认退出。"""
    key = str(Path(root).resolve())
    tasks = None if task_ids is None else {str(item) for item in task_ids}
    with _BUILD_LOCK:
        if close:
            if tasks is None:
                _CLOSED_ROOTS.add(key)
            else:
                _CLOSED_TASKS.update((key, task) for task in tasks)
        jobs = [job for job in _JOBS.values() if job.owner[0] == key
                and (tasks is None or job.owner[1] in tasks) and not job.future.done()]
    # Future.cancel 可能同步调用完成回调，不能持有登记锁。
    for job in jobs:
        job.signal.interrupt('pause')
        if on_complete:
            job.future.add_done_callback(lambda _, callback=on_complete: callback())
        job.future.cancel()
    if jobs and timeout:
        wait([job.future for job in jobs], timeout=timeout)
    pending = sorted({job.owner[1] for job in jobs if not job.future.done()})
    return {'stopped': not pending, 'pending': pending, 'jobs': len(jobs)}


def _completed_build(key, job, future):
    """失败墓碑只留短诊断；完成和取消均释放语料、旧 Signal 和索引引用。"""
    from agent.model.errors import ModelInterrupted

    error = None if future.cancelled() else future.exception()
    with _BUILD_LOCK:
        if _JOBS.get(key) is not job:
            return
        if future.cancelled() or error is None or isinstance(error, ModelInterrupted):
            _JOBS.pop(key)
        else:
            light = Future()
            light.set_exception(ModelUnavailable(type(error).__name__))
            job.future = light


class ModelUnavailable(RuntimeError):
    """本地能力不可用，不伪装成已完成的向量或重排结果。"""


def available_memory():
    if os.name == 'nt':
        import ctypes

        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total', 'available', 'page', 'available_page', 'virtual', 'available_virtual', 'ex')
            ]

        value = Memory()
        value.length = ctypes.sizeof(value)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
            raise ModelUnavailable('Cannot inspect host memory.')
        return value.available
    try:
        lines = Path('/proc/meminfo').read_text().splitlines()
        return int(next(line.split()[1] for line in lines if line.startswith('MemAvailable:'))) * 1024
    except (OSError, StopIteration, ValueError) as exc:
        raise ModelUnavailable('Cannot inspect host memory.') from exc


def _check_cancel():
    from agent.task.control import check_interrupted

    check_interrupted(pause_only=True)


@contextmanager
def _process_slot():
    """Windows 各进程共享同一模型槽，不并发常驻多份本地模型。"""
    if os.name != 'nt':
        yield
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    suffix = hashlib.sha256(os.environ.get('USERPROFILE', '').encode()).hexdigest()[:16]
    handle = kernel.CreateMutexW(None, False, 'Local\\MonoPlannerRag-' + suffix)
    if not handle:
        raise ModelUnavailable('Cannot acquire the shared local model slot.')
    acquired = False
    try:
        deadline = time.monotonic() + 30
        while not acquired:
            _check_cancel()
            result = kernel.WaitForSingleObject(handle, 100)
            acquired = result in (0, 0x80)
            if not acquired and (result != 0x102 or time.monotonic() >= deadline):
                raise ModelUnavailable('The shared local model slot is unavailable.')
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def _memory(handle):
    if os.name != 'nt':
        return {}
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in
            ('peak_rss', 'rss', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged',
             'commit', 'peak_commit', 'private')
        ]

    value = Counters()
    value.size = ctypes.sizeof(value)
    library = ctypes.WinDLL('psapi', use_last_error=True)
    library.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    if not library.GetProcessMemoryInfo(wintypes.HANDLE(int(handle)), ctypes.byref(value), value.size):
        return {}
    return {'peak_rss': value.peak_rss, 'peak_commit': value.peak_commit}


class LocalModels:
    """清单缺失时不下载；每次推理在可终止的独立进程中完成。"""

    def __init__(self, config_path=None, *, timeout=30):
        value = config_path or os.environ.get('MONOPLANER_RAG_MODELS')
        self.path = Path(value).resolve() if value else None
        self.timeout = timeout
        self.manifest = {}
        if self.path:
            if self.path.stat().st_size > 65536:
                raise ModelUnavailable('Local model manifest exceeds 64 KiB.')
            self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
            if self.manifest.get('version') != 1:
                raise ModelUnavailable('Unsupported local model manifest version.')
        self.vector = LocalVectorRoute(self)
        self.reranker = self

    def _spec(self, role):
        self.identity_for(role)
        spec = self.manifest['models'][role]
        if not isinstance(spec, dict) or not self.path:
            raise ModelUnavailable(f'No local {role} model configured; no download attempted.')
        root = Path(spec.get('root', ''))
        root = (self.path.parent / root).resolve() if not root.is_absolute() else root.resolve()
        hashes = spec.get('sha256')
        if not root.is_dir() or not isinstance(hashes, dict) or not hashes:
            raise ModelUnavailable('Local model requires an existing root and pinned file hashes.')
        for name, expected in hashes.items():
            path = root / name
            if (not path.resolve().is_relative_to(root) or path.is_symlink()
                    or any(part.is_junction() for part in (path, *path.parents))):
                raise ModelUnavailable('Model files must remain inside the pinned local root.')
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    _check_cancel()
                    digest.update(block)
            if digest.hexdigest() != expected:
                raise ModelUnavailable('Local model file hash changed.')
        for key in ('tokenizer', 'model'):
            if spec.get(key) not in hashes:
                raise ModelUnavailable(f'Pin the {key} file in the local model manifest.')
        if type(spec.get('max_tokens')) is not int or not 16 <= spec['max_tokens'] <= 8192:
            raise ModelUnavailable('Set a verified input window of 16..8192 tokens.')
        return {**spec, 'root': str(root)}

    def identity_for(self, role):
        if role not in {'embedding', 'reranker'}:
            raise ModelUnavailable('Only embedding and reranker models are supported.')
        spec = self.manifest.get('models', {}).get(role)
        if not spec:
            raise ModelUnavailable(f'No local {role} model configured.')
        expected = spec.get('runtime_versions', {})
        required = {'onnxruntime', 'tokenizers', 'numpy'}
        if not required.issubset(expected):
            raise ModelUnavailable('Pin the actual inference runtime package versions.')
        try:
            dependencies = self.manifest.get('dependency_path')
            versions = None
            if dependencies:
                directory = (self.path.parent / dependencies).resolve()
                versions = {item.metadata['Name'].lower().replace('_', '-'): item.version
                            for item in importlib.metadata.distributions(path=[str(directory)])}
            actual = lambda name: (versions.get(name.lower().replace('_', '-')) if versions is not None
                                   else importlib.metadata.version(name))
            if any(actual(name) != version for name, version in expected.items()):
                raise ModelUnavailable('Inference runtime changed; revalidate model qualification.')
        except importlib.metadata.PackageNotFoundError as exc:
            raise ModelUnavailable('Optional local inference dependencies are not installed.') from exc
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()

    def _call(self, role, payload, *, consume=None):
        _check_cancel()
        spec = self._spec(role)
        # 不把模型文件大小当作运行内存；实际进程受 768 MiB 硬限约束，
        # 为索引缓存、分词和结果缓冲保留其余 256 MiB。
        if available_memory() < WORKER_BYTES + 3 * GIB // 2:
            raise ModelUnavailable('Host memory reserve is insufficient for local inference.')
        raw = json.dumps({'role': role, 'spec': spec, 'payload': payload},
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
        if len(raw) > 1024 * 1024:
            raise ModelUnavailable('Local inference input exceeds 1 MiB.')
        while not _INFERENCE.acquire(timeout=0.1):
            _check_cancel()
        try:
            _check_cancel()
            if available_memory() < WORKER_BYTES + 3 * GIB // 2:
                raise ModelUnavailable('Host memory reserve changed while waiting for inference.')
            with _process_slot():
                return self._run(raw, consume=consume)
        finally:
            _INFERENCE.release()

    def _run(self, raw, *, consume=None):
        env = {**os.environ, 'HF_HUB_OFFLINE': '1',
               'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'PYTHONUTF8': '1',
               'PYTHONDONTWRITEBYTECODE': '1', 'MONOPLANER_RAG_WORKER_BYTES': str(WORKER_BYTES),
               'MONOPLANER_RAG_PARENT_LIMIT': '1' if os.name == 'nt' else '0'}
        dependencies = self.manifest.get('dependency_path')
        if dependencies:
            env['PYTHONPATH'] = str((self.path.parent / dependencies).resolve())
        # 临时输入和输出均限量，进程退出即删除，不成为重复的历史前缀。
        with tempfile.TemporaryDirectory() as scratch, (
            Path(scratch) / 'output.jsonl'
        ).open('w+b') as output, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                # Windows venv 的 python.exe 可能只是再派生解释器的重定向器。
                # 直接启动同版本基础解释器，Job 与内存测量才对应真实推理进程。
                [getattr(sys, '_base_executable', sys.executable),
                 '-B', '-X', 'utf8', '-m', 'agent.rag.model_worker',
                 *(['--count'] if consume is not None else [])],
                stdin=subprocess.PIPE, stdout=output, stderr=errors, env=env,
                cwd=Path(__file__).resolve().parents[2],
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            job = None
            started, peak = time.monotonic(), {'peak_rss': 0, 'peak_commit': 0}
            try:
                if os.name == 'nt':
                    from agent.rag.model_worker import memory_limit

                    job = memory_limit(process=process, limit=WORKER_BYTES)
                process.stdin.write(raw + (b'\n' if consume is not None else b''))
                process.stdin.flush()
                deadline = time.monotonic() + self.timeout

                def guard():
                    nonlocal peak
                    _check_cancel()
                    memory = _memory(process._handle) if os.name == 'nt' else {}
                    peak = {name: max(value, memory.get(name, 0)) for name, value in peak.items()}
                    if memory.get('peak_rss', 0) > WORKER_BYTES:
                        raise ModelUnavailable('Local inference exceeded the resident memory budget.')
                    if time.monotonic() >= deadline:
                        raise ModelUnavailable('Local inference timed out.')
                    if (os.fstat(output.fileno()).st_size > 2 * 1024 * 1024
                            or os.fstat(errors.fileno()).st_size > 65536):
                        raise ModelUnavailable('Local inference output exceeded its bound.')
                    if available_memory() < 3 * GIB // 2:
                        raise ModelUnavailable('Host memory reserve was exhausted.')

                if consume is not None:
                    # 独立读取句柄不移动子进程的写入位置；每次只交换一个有界计数。
                    with (Path(scratch) / 'output.jsonl').open('rb') as reader:
                        def count(text):
                            guard()
                            if len(text) > 65536:
                                return 10 ** 9  # 超长完整结构继续二分，不发送巨型计数请求。
                            encoded = json.dumps(text, ensure_ascii=False).encode('utf-8')
                            process.stdin.write(encoded + b'\n')
                            process.stdin.flush()
                            line = b''
                            while not line.endswith(b'\n'):
                                guard()
                                line += reader.readline(1024 - len(line))
                                if len(line) >= 1024:
                                    raise ModelUnavailable('Tokenizer response exceeded its bound.')
                                if not line.endswith(b'\n'):
                                    if process.poll() is not None:
                                        raise ModelUnavailable('Tokenizer stopped before returning a count.')
                                    time.sleep(0.002)
                            result = json.loads(line)
                            if type(result.get('tokens')) is not int:
                                raise ModelUnavailable('Tokenizer did not return a valid count.')
                            return result['tokens']

                        consumed = consume(count)
                process.stdin.close()
                while process.poll() is None:
                    guard()
                    time.sleep(0.05)
                if consume is not None:
                    if process.returncode:
                        raise ModelUnavailable('Tokenizer worker did not exit normally.')
                    return consumed
                output.seek(0)
                result = json.loads(output.read(2 * 1024 * 1024).decode('utf-8'))
                if process.returncode or result.get('error'):
                    raise ModelUnavailable(result.get('error', 'Local inference worker failed.'))
                return result['result']
            except (OSError, ValueError, KeyError) as exc:
                raise ModelUnavailable('Local inference did not produce a complete result.') from exc
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                self.last_profile = {**peak, 'seconds': time.monotonic() - started,
                                     'worker_limit_bytes': WORKER_BYTES,
                                     'parent_reserve_bytes': GIB - WORKER_BYTES,
                                     'exit_code': process.returncode}
                if job:
                    import ctypes
                    from ctypes import wintypes

                    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
                    kernel.CloseHandle(job)

    def index_documents(self, index, documents):
        """分片与固定 Embedding tokenizer 对齐，完成后立即释放分词进程。"""
        def consume(counter):
            previous = index.token_count
            index.token_count = counter
            try:
                return index.upsert_documents(documents)
            finally:
                index.token_count = previous

        return self._call('embedding', {}, consume=consume)

    def embed(self, texts, *, query=False):
        if not 1 <= len(texts) <= 16:
            raise ModelUnavailable('Embedding batches must contain 1..16 fragments.')
        role = self.manifest.get('models', {}).get('embedding', {})
        prefix = role.get('query_prefix' if query else 'document_prefix', '')
        return self._call('embedding', {'texts': [prefix + text for text in texts]})

    def score(self, query, chunks):
        if not chunks:
            return []
        scores = []
        for start in range(0, len(chunks), 16):
            scores.extend(self._call('reranker', {
                'query': query, 'texts': [c.embedding_text() for c in chunks[start:start + 16]],
            }))
        return scores

class LocalVectorRoute:
    """未完成的向量索引只报告不完整；相同输入的真实向量按哈希复用。"""

    def __init__(self, models):
        self.models = models

    def search(self, query, index, k):
        identity = self.models.identity_for('embedding')
        chunks = index.chunks(query)
        vectors = index.vector_cache(identity, chunks)
        missing = [chunk for chunk in chunks if chunk.id not in vectors]
        if missing:
            batch = hashlib.sha256(''.join(chunk.id for chunk in missing).encode()).hexdigest()
            key = (str(index.path), identity, batch)
            from agent.task.control import Signal, current

            owner = _OWNER.get()
            if owner is None:
                raise ModelUnavailable('A task owner is required for background vector indexing.')
            with _BUILD_LOCK:
                _scope_open(owner)
                job = _JOBS.get(key)
                if job is not None and job.future.done():
                    # 失败不在每次查询中悄悄重复调用模型。
                    if not job.future.cancelled() and job.future.exception():
                        raise ModelUnavailable('Background vector build failed; inspect diagnostics.')
                    _JOBS.pop(key)
                if key not in _JOBS:
                    if any(not item.future.done() for item in _JOBS.values()):
                        raise ModelUnavailable('Another bounded vector build is in progress.')
                    context = copy_context()
                    job = _BuildJob(owner, Signal())
                    future = _BUILD.submit(context.run, self._build, index, identity,
                                           missing, job, current())
                    job.future = future
                    _JOBS[key] = job
            # callback 可能立即执行，不能在持有同一个锁时注册它。
            if 'future' in locals():
                future.add_done_callback(lambda done: _completed_build(key, job, done))
            raise ModelUnavailable('Vector index is building; lexical results remain available.')
        if not chunks:
            return []
        needle = self.models.embed([query.text], query=True)[0]
        scored = []
        for chunk in chunks:
            value = vectors[chunk.id]
            if len(needle) != len(value):
                raise ModelUnavailable('Embedding dimensions changed.')
            denominator = math.sqrt(sum(x * x for x in needle) * sum(x * x for x in value))
            score = sum(a * b for a, b in zip(needle, value)) / denominator if denominator else 0.0
            if not math.isfinite(score):
                raise ModelUnavailable('Embedding output is not finite.')
            scored.append((chunk.id, score))
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:k]

    def _build(self, index, identity, chunks, job, parent):
        from contextlib import nullcontext
        from agent.rag.index import Index
        from agent.task.control import using

        def forward_pause():
            if parent is not None and parent.kind == 'pause':
                job.signal.interrupt('pause')

        def check():
            with _BUILD_LOCK:
                _scope_open(job.owner)
            job.signal.check(pause_only=True)

        # 独立 Signal 持续登记到后台退出；聊天返回后仍可由暂停/关闭/删除取消。
        subscription = parent.subscribe(forward_pause) if parent else nullcontext()
        with subscription, using(job.signal):
            check()
            with Index(index.path, index.scope, index.config, index.token_count) as owned:
                for start in range(0, len(chunks), 16):
                    check()
                    group = chunks[start:start + 16]
                    values = self.models.embed([chunk.embedding_text() for chunk in group])
                    check()
                    owned.cache_vectors(identity, group, values)
