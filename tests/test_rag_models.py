"""验证资源准入、线程取消和请求级改写，不调用真实模型。"""

from concurrent.futures import Future
from dataclasses import replace
import hashlib
import json
import os
import subprocess
import sys
import threading

import pytest

from agent.model.errors import ModelInterrupted
from agent.rag.models import LocalModels, LocalVectorRoute, ModelUnavailable
from agent.rag.rewrite import RewriteLedger, RewriteUnknown, assess
from agent.rag.index import Index
from agent.rag.types import Document, Query, RagConfig, Scope
from agent.task import control


def test_rewrite_signals_require_applicability():
    located = assess(intent='locate', exact_current_hit=True, missing_entities=['x'],
                     single_source=True)
    assert not located['triggered']
    assert located['suppressed']['single_source'] == 'exact_current_version_short_circuit'
    absent = assess(intent='audit')
    assert not absent['triggered']
    assert 'dual_disagreement' not in absent['suppressed']
    missing = assess(intent='diagnostic', missing_entities=['x'])
    assert missing['triggered'] == ['required_evidence_missing']


def test_rewrite_budget_survives_restart_and_multiple_queries(tmp_path):
    path = tmp_path / 'rewrite.sqlite3'
    calls = []
    def callback(query, reasons):
        calls.append(query)
        return {'query': 'new query'}
    first = RewriteLedger(path).attempt('user-1', 'old', ['missing'], callback)
    assert first['query'] == 'new query'
    second = RewriteLedger(path).attempt('user-1', 'another', ['missing'], callback)
    assert second['status'] == 'already_used'
    assert calls == ['old']
    assert RewriteLedger(path).attempt('user-2', 'other', ['missing'], callback)['status'] == 'complete'


def test_unknown_rewrite_is_not_replayed(tmp_path):
    ledger = RewriteLedger(tmp_path / 'rewrite.sqlite3')
    calls = []
    def fail(query, reasons):
        calls.append(query)
        raise TimeoutError('response not received')
    with pytest.raises(RewriteUnknown):
        ledger.attempt('request', 'q', ['missing'], fail)
    with pytest.raises(RewriteUnknown):
        ledger.attempt('request', 'q', ['missing'], fail)
    assert calls == ['q']
    assert ledger.get('request')['status'] == 'unknown'


def test_invalid_complete_rewrite_consumes_only_one_attempt(tmp_path):
    ledger = RewriteLedger(tmp_path / 'rewrite.sqlite3')
    assert ledger.attempt('id', 'q', ['missing'], lambda *args: None)['status'] == 'invalid'
    assert ledger.attempt('id', 'q', ['missing'], lambda *args: 'fixed')['status'] == 'already_used'


def test_missing_local_models_never_spawn_or_download(monkeypatch):
    monkeypatch.delenv('MONOPLANER_RAG_MODELS', raising=False)
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('No model may start'))
    with pytest.raises(ModelUnavailable, match='No local'):
        LocalModels().embed(['source'])


@pytest.mark.parametrize('role', ['decision', 'unknown'])
def test_local_models_reject_unsupported_roles_before_loading(tmp_path, role):
    config = tmp_path / 'models.json'
    config.write_text(json.dumps({'version': 1, 'models': {role: {}}}))
    models = LocalModels(config)
    with pytest.raises(ModelUnavailable, match='Only embedding and reranker'):
        models.identity_for(role)
    with pytest.raises(ModelUnavailable, match='Only embedding and reranker'):
        models._spec(role)


def test_model_worker_rejects_unsupported_roles_before_optional_imports():
    from agent.rag.model_worker import infer

    with pytest.raises(ValueError, match='Only embedding and reranker'):
        infer({'spec': {}, 'payload': {}, 'role': 'decision'})


def test_runtime_versions_are_part_of_admission(tmp_path, monkeypatch):
    config = tmp_path / 'models.json'
    config.write_text(json.dumps({'version': 1, 'models': {'embedding': {
        'runtime_versions': {'onnxruntime': '1', 'tokenizers': '2', 'numpy': '3'},
    }}}))
    monkeypatch.setattr('importlib.metadata.version', lambda name: 'changed')
    with pytest.raises(ModelUnavailable, match='runtime changed'):
        LocalModels(config).identity_for('embedding')


def test_indexing_uses_pinned_counter_and_restores_message_estimator(tmp_path, monkeypatch):
    models = LocalModels()
    measured = []

    def invoke(role, payload, *, consume):
        assert role == 'embedding'
        def count(text):
            measured.append(text)
            return len(text) // 3
        return consume(count)

    monkeypatch.setattr(models, '_call', invoke)
    estimator = lambda text: len(text) // 8
    document = Document('source:main.py', 'v1', 'def test():\n' + '    value = 1\n' * 100,
                        kind='code', path='main.py')
    with Index(tmp_path / 'index.sqlite3', Scope('p', 't'), RagConfig(), estimator) as index:
        models.index_documents(index, [document])
        chunks = index.chunks(Query('value', {'source:main.py': 'v1'}))
        assert len(chunks) > 1
        assert all(len(chunk.render()) // 3 <= 512 for chunk in chunks)
        assert index.token_count is estimator
    assert measured


def test_failed_token_count_does_not_commit_partial_document(tmp_path, monkeypatch):
    models = LocalModels()

    def invoke(role, payload, *, consume):
        def failed(_):
            raise ModelUnavailable('Tokenizer stopped')
        return consume(failed)

    monkeypatch.setattr(models, '_call', invoke)
    with Index(tmp_path / 'index.sqlite3', Scope('p', 't'), RagConfig(), len) as index:
        with pytest.raises(ModelUnavailable):
            models.index_documents(index, [Document('doc', 'v1', 'evidence')])
        assert index.stats()['documents'] == 0
        assert index.token_count is len


def test_failed_background_vector_build_is_not_retried(tmp_path, monkeypatch):
    from agent.rag import models

    class Fake:
        def identity_for(self, role): return 'fixed-model'
    with Index(tmp_path / 'index.sqlite3', Scope('p', 't'), RagConfig(), len) as index:
        index.upsert(Document('doc', 'v1', 'def token():\n    return 1\n', path='main.py'))
        future = Future()
        future.set_exception(ValueError('failed'))
        missing = index.chunks(Query('token', {'doc': 'v1'}))
        batch = hashlib.sha256(''.join(chunk.id for chunk in missing).encode()).hexdigest()
        key = (str(index.path), 'fixed-model', batch)
        job = models._BuildJob(models._owner(tmp_path, 't'), control.Signal(), future)
        monkeypatch.setitem(models._JOBS, key, job)
        monkeypatch.setattr(models._BUILD, 'submit', lambda *a: pytest.fail('No hidden retry'))
        with models.background_scope(tmp_path, 't'):
            for _ in range(2):
                with pytest.raises(ModelUnavailable, match='failed'):
                    LocalVectorRoute(Fake()).search(Query('token', {'doc': 'v1'}), index, 10)


def test_pause_prevents_model_admission(monkeypatch):
    monkeypatch.delenv('MONOPLANER_RAG_MODELS', raising=False)
    signal = control.Signal()
    signal.interrupt('pause')
    with control.using(signal), pytest.raises(ModelInterrupted):
        LocalModels().embed(['source'])


@pytest.mark.skipif(os.name != 'nt', reason='Windows owned process job')
def test_parent_job_close_terminates_only_owned_worker():
    import ctypes
    from ctypes import wintypes
    from agent.rag.model_worker import memory_limit

    owned = subprocess.Popen([getattr(sys, '_base_executable', sys.executable),
                              '-c', 'import time; time.sleep(30)'],
                             creationflags=subprocess.CREATE_NO_WINDOW)
    handle = None
    try:
        handle = memory_limit(owned, 256 * 1024 * 1024)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        assert kernel.CloseHandle(handle)
        handle = None
        owned.wait(timeout=5)
        assert owned.returncode is not None
    finally:
        if handle:
            ctypes.windll.kernel32.CloseHandle(wintypes.HANDLE(handle))
        if owned.poll() is None:
            owned.kill()
            owned.wait(timeout=5)


class BlockingEmbedding:
    """可控的模型边界；取消后故意等 release，用来核验停止确认而非发信即成功。"""
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def identity_for(self, role):
        return 'offline-embedding'

    def embed(self, texts, **kwargs):
        self.calls += 1
        self.started.set()
        assert self.release.wait(5), 'Test did not release the offline model'
        control.check_interrupted(pause_only=True)
        return [[1.0] for _ in texts]


def start_background(root, task, embedding):
    from agent.rag.models import background_scope

    path = root / 'tasks' / task / 'rag' / 'indexes' / 'index.sqlite3'
    with Index(path, Scope('project', task), RagConfig(), len) as index:
        index.upsert(Document('source:main.py', 'v1', 'def token(): return 1', path='main.py'))
        with control.active(root, 'chat:' + task), background_scope(root, task):
            with pytest.raises(ModelUnavailable, match='building'):
                LocalVectorRoute(embedding).search(Query('token', {'source:main.py': 'v1'}), index, 10)
    assert embedding.started.wait(1)
    return path


def test_background_cancel_survives_chat_owner_exit_and_is_scoped(tmp_path):
    from agent.rag import models

    model = BlockingEmbedding()
    start_background(tmp_path, 'task', model)
    try:
        assert control.active_count(tmp_path) == 0
        assert models.pending_jobs(tmp_path) == ['task']
        assert models.cancel_jobs(tmp_path, ['other'])['jobs'] == 0
        assert models.cancel_jobs(tmp_path / 'other')['jobs'] == 0
        waiting = models.cancel_jobs(tmp_path, ['task'])
        assert not waiting['stopped'] and waiting['pending'] == ['task']
    finally:
        model.release.set()
        assert models.cancel_jobs(tmp_path, ['task'], timeout=2)['stopped']
    assert not models.pending_jobs(tmp_path)


def test_closed_background_scope_cannot_recreate_deleted_cache(tmp_path):
    from agent.rag import models
    import shutil

    model = BlockingEmbedding()
    path = start_background(tmp_path, 'task', model)
    try:
        assert not models.cancel_jobs(tmp_path, ['task'], close=True)['stopped']
        model.release.set()
        assert models.cancel_jobs(tmp_path, ['task'], close=True, timeout=2)['stopped']
        shutil.rmtree(path.parents[2])
        with pytest.raises(ModelUnavailable, match='closed'):
            with models.background_scope(tmp_path, 'task'):
                pytest.fail('Deleted scope was reopened')
        assert not path.parents[2].exists()
        assert model.calls == 1
    finally:
        model.release.set()
        models.cancel_jobs(tmp_path, ['task'], close=True, timeout=2)


def test_scope_closed_while_queued_never_opens_cache_or_loads_model(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from agent.rag import models
    import shutil

    pool = ThreadPoolExecutor(max_workers=1)
    hold = threading.Event()
    blocker = pool.submit(hold.wait, 5)
    monkeypatch.setattr(models, '_BUILD', pool)
    model = BlockingEmbedding()
    path = tmp_path / 'tasks/task/rag/indexes/index.sqlite3'
    try:
        with Index(path, Scope('p', 'task'), RagConfig(), len) as index:
            index.upsert(Document('source:main.py', 'v1', 'token', path='main.py'))
            with models.background_scope(tmp_path, 'task'), pytest.raises(ModelUnavailable, match='building'):
                LocalVectorRoute(model).search(Query('token', {'source:main.py': 'v1'}), index, 10)
        assert models.cancel_jobs(tmp_path, ['task'], close=True)['stopped']
        shutil.rmtree(path.parents[2])
        hold.set()
        blocker.result(timeout=1)
        assert not path.parents[2].exists() and model.calls == 0
    finally:
        hold.set()
        model.release.set()
        pool.shutdown(wait=True, cancel_futures=True)
