"""记录检索信号和请求级的一次改写机会，不重放未知请求。"""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import time


class RewriteUnknown(RuntimeError):
    """改写调用的结果不明，调用方必须保留现场并停止派发。"""


def assess(*, intent, missing_entities=(), missing_kinds=(), exact_current_hit=False,
           vector_ready=False, dual_disagreement=False, rerank_no_winner=False,
           single_source=False, temporal_mismatch=False):
    """只解释已有检索事实；数量少和探索词未命中不是付费触发条件。"""
    signals = {
        'required_evidence_missing': bool(missing_entities or missing_kinds),
        'dual_disagreement': bool(dual_disagreement),
        'rerank_no_winner': bool(rerank_no_winner),
        'single_source': bool(single_source),
        'temporal_mismatch': bool(temporal_mismatch),
    }
    triggered, suppressed = [], {}
    for name, present in signals.items():
        reason = None
        if not present:
            reason = 'signal_absent'
        elif intent == 'locate' and exact_current_hit:
            reason = 'exact_current_version_short_circuit'
        elif name != 'required_evidence_missing' and intent not in {'diagnostic', 'audit'}:
            reason = 'query_intent_not_applicable'
        elif name == 'dual_disagreement' and not vector_ready:
            reason = 'vector_route_incomplete'
        if reason:
            suppressed[name] = reason
        else:
            triggered.append(name)
    return {'triggered': triggered, 'suppressed': suppressed,
            'missing_entities': list(missing_entities), 'missing_kinds': list(missing_kinds)}


class RewriteLedger:
    """SQLite 唯一键保证多次工具查询共享同一用户请求额度。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS rewrites (
                    request_id TEXT PRIMARY KEY, query_hash TEXT NOT NULL,
                    status TEXT NOT NULL, result TEXT, updated REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assessments (
                    id INTEGER PRIMARY KEY, request_id TEXT NOT NULL,
                    facts TEXT NOT NULL, created REAL NOT NULL
                );
            ''')

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.execute('PRAGMA busy_timeout=5000')
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def record_assessment(self, request_id, facts):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('A stable user request ID is required for query rewriting.')
        raw = json.dumps(facts, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if len(raw.encode('utf-8')) > 16_384:
            raise ValueError('Rewrite assessment exceeds its bounded record size.')
        with self.connect() as connection:
            connection.execute('INSERT INTO assessments(request_id, facts, created) VALUES(?,?,?)',
                               (request_id, raw, time.time()))

    def get(self, request_id):
        with self.connect() as connection:
            row = connection.execute('SELECT status, result FROM rewrites WHERE request_id=?',
                                     (request_id,)).fetchone()
        return {'status': row[0], 'result': json.loads(row[1]) if row[1] else None} if row else None

    def attempt(self, request_id, query, reasons, callback):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError('A stable user request ID is required for query rewriting.')
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise ValueError('Rewrite query must contain 1..4096 characters.')
        if not reasons:
            return {'status': 'not_triggered', 'query': query}
        if callback is None:
            return {'status': 'unavailable', 'query': query, 'reason': 'No rewrite provider configured'}
        query_hash = hashlib.sha256(query.encode('utf-8')).hexdigest()
        with self.connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            old = connection.execute('SELECT status FROM rewrites WHERE request_id=?',
                                     (request_id,)).fetchone()
            if old:
                if old[0] in {'pending', 'unknown'}:
                    raise RewriteUnknown('Previous rewrite result is unknown; no automatic resend.')
                return {'status': 'already_used', 'query': query}
            connection.execute('INSERT INTO rewrites VALUES(?,?,?,?,?)',
                               (request_id, query_hash, 'pending', None, time.time()))
        try:
            result = callback(query, list(reasons))
        except Exception as exc:
            from agent.model.errors import ModelInterrupted, ModelRequestNotSent

            not_sent = isinstance(exc, ModelRequestNotSent) or (
                isinstance(exc, ModelInterrupted) and exc.sent is False
            )
            status = 'not_sent' if not_sent else 'unknown'
            self._finish(request_id, status, {'error_type': type(exc).__name__})
            if status == 'unknown':
                raise RewriteUnknown('Rewrite result is unknown; no automatic resend.') from exc
            raise
        rewritten = result.get('query') if isinstance(result, dict) else result
        if not isinstance(rewritten, str) or not rewritten.strip() or len(rewritten) > 4096:
            self._finish(request_id, 'invalid', {'reason': 'Expected one non-empty bounded query'})
            return {'status': 'invalid', 'query': query}
        value = {'status': 'complete', 'query': rewritten, 'original_query_hash': query_hash}
        self._finish(request_id, 'complete', value)
        return value

    def _finish(self, request_id, status, result):
        with self.connect() as connection:
            connection.execute('UPDATE rewrites SET status=?,result=?,updated=? WHERE request_id=?',
                               (status, json.dumps(result, ensure_ascii=False), time.time(), request_id))
