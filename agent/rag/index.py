"""仅索引调用方已经授权的资料；SQLite 索引不扫描文件系统。"""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from agent.rag.chunks import chunk_document
from agent.rag.types import (
    CANDIDATE_MEMORY_BYTES, VECTOR_MEMORY_BYTES,
    Chunk, Document, Query, RagConfig, Scope, TokenCount, canonical,
)
from agent.task.control import check_interrupted, current



def _bounded_rows(cursor):
    rows = []
    estimated = 0
    for row in cursor:
        check_interrupted(pause_only=True)
        estimated += len((row[0] or "").encode("utf-8")) * 4 + 1024
        if estimated > CANDIDATE_MEMORY_BYTES:
            raise ValueError("Retrieval candidates exceed the reserved memory budget.")
        rows.append(row)
    return rows


def _allowed(chunk, query):
    path = chunk.path.replace("\\", "/")
    return query.source_versions.get(chunk.source_id) == chunk.version and not any(
        path == item.rstrip("/\\") or path.startswith(item.rstrip("/\\") + "/")
        for item in (value.replace("\\", "/") for value in query.excluded_paths)
    )


class Index:
    def __init__(self, path, scope: Scope, config: RagConfig, token_count: TokenCount):
        self.path = Path(path)
        self.scope = scope
        self.config = config
        self.token_count = token_count
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS configurations (
                    scope TEXT PRIMARY KEY, settings TEXT
                );
                CREATE TABLE IF NOT EXISTS documents (
                    scope TEXT, source_id TEXT, version TEXT, sha TEXT,
                    PRIMARY KEY(scope, source_id, version)
                );
                CREATE TABLE IF NOT EXISTS source_headers (
                    scope TEXT, source_id TEXT, version TEXT, payload TEXT,
                    PRIMARY KEY(scope, source_id, version)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    rowid INTEGER PRIMARY KEY, scope TEXT, id TEXT, source_id TEXT,
                    version TEXT, payload TEXT,
                    UNIQUE(scope, id)
                );
                CREATE INDEX IF NOT EXISTS chunks_source
                    ON chunks(scope, source_id, version);
                CREATE TABLE IF NOT EXISTS vectors (
                    model TEXT, hash TEXT, vector TEXT, PRIMARY KEY(model, hash)
                );
                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    scope TEXT, chunk_id TEXT, model TEXT, hash TEXT,
                    PRIMARY KEY(scope, chunk_id, model)
                );
            """)
            settings = canonical(asdict(config))
            old = connection.execute(
                "SELECT settings FROM configurations WHERE scope=?", (self.scope.key,)
            ).fetchone()
            if old and old[0] != settings:
                raise ValueError("Changed retrieval settings require a new configuration version.")
            connection.execute(
                "INSERT OR IGNORE INTO configurations VALUES(?,?)", (self.scope.key, settings)
            )

    @contextmanager
    def _connection(self):
        if self.closed:
            raise RuntimeError("The retrieval index is closed.")
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.set_progress_handler(
                lambda: int(current() is not None and current().kind == "pause"), 1000
            )
            connection.execute("PRAGMA cache_size=-4096")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute("PRAGMA temp_store=FILE")
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            page_limit = max(1, self.config.max_index_bytes // page_size)
            connection.execute(f"PRAGMA max_page_count={page_limit}")
            with connection:
                yield connection
        except sqlite3.OperationalError:
            check_interrupted(pause_only=True)
            raise
        finally:
            connection.close()

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _delete(self, connection, source_id, version):
        sql = "scope=? AND source_id=?"
        args = [self.scope.key, source_id]
        if version is not None:
            sql += " AND version=?"
            args.append(version)
        connection.execute(
            f"DELETE FROM chunk_vectors WHERE scope=? AND chunk_id IN "
            f"(SELECT id FROM chunks WHERE {sql})", [self.scope.key, *args]
        )
        connection.execute(f"DELETE FROM chunks WHERE {sql}", args)
        connection.execute(f"DELETE FROM documents WHERE {sql}", args)
        connection.execute(f"DELETE FROM source_headers WHERE {sql}", args)
        connection.execute(
            "DELETE FROM vectors WHERE NOT EXISTS (SELECT 1 FROM chunk_vectors "
            "WHERE chunk_vectors.model=vectors.model AND chunk_vectors.hash=vectors.hash)"
        )

    def delete(self, source_id, version=None):
        """仅移除派生缓存；调用方持有的原件不受影响。"""
        with self._connection() as connection:
            self._delete(connection, source_id, version)

    def prune_sources(self, authorized_versions):
        """移除当前 scope 失效的源码派生缓存；操作记录及原件始终不动。"""
        removed = 0
        with self._connection() as connection:
            versions = connection.execute(
                "SELECT source_id,version FROM documents WHERE scope=? "
                "AND substr(source_id,1,7)='source:'", (self.scope.key,),
            ).fetchall()
            for source_id, version in versions:
                check_interrupted(pause_only=True)
                if authorized_versions.get(source_id) != version:
                    self._delete(connection, source_id, version)
                    removed += 1
        return removed

    def upsert(self, document: Document, *, exact=False):
        check_interrupted(pause_only=True)
        with self._connection() as connection:
            previous = connection.execute(
                "SELECT sha FROM documents WHERE scope=? AND source_id=? AND version=?",
                (self.scope.key, document.source_id, document.version),
            ).fetchone()
            if previous:
                if previous[0] != document.sha256:
                    raise ValueError("An immutable source version changed its content hash.")
                return False
            chunks = [] if exact else chunk_document(document, self.config, self.token_count)
            connection.execute(
                "INSERT INTO documents VALUES(?,?,?,?)",
                (self.scope.key, document.source_id, document.version, document.sha256),
            )
            header = Chunk(
                id=hashlib.sha256(f"{document.source_id}\0{document.version}".encode()).hexdigest(),
                source_id=document.source_id, version=document.version,
                source_sha256=document.sha256, path=document.path, kind=document.kind,
                start=0, end=0, parent_start=0, parent_end=len(document.text),
                line_start=1, line_end=1, text="", run_id=document.run_id,
                timestamp=document.timestamp,
            )
            connection.execute("INSERT INTO source_headers VALUES(?,?,?,?)",
                               (self.scope.key, document.source_id, document.version, header.render()))
            for chunk in chunks:
                check_interrupted(pause_only=True)
                connection.execute(
                    "INSERT INTO chunks(scope,id,source_id,version,payload) VALUES(?,?,?,?,?)",
                    (self.scope.key, chunk.id, chunk.source_id, chunk.version, chunk.render()),
                )
        return True

    def upsert_documents(self, documents, *, exact=False):
        count = changed = 0
        for document in documents:
            count += 1
            changed += self.upsert(document, exact=exact)
            del document
        return {"documents": count, "changed": changed, "chunks": self.stats()["chunks"]}

    def stats(self):
        with self._connection() as connection:
            result = {
                table: connection.execute(
                    f"SELECT count(*) FROM {table} WHERE scope=?", (self.scope.key,)
                ).fetchone()[0]
                for table in ("documents", "chunks", "chunk_vectors")
            }
        result["bytes"] = self.path.stat().st_size
        return result

    def _rows(self, connection, query, extra="", args=(), limit=20001):
        # JSON 白名单与 scope 在排名/取数前筛选，不能先拿全局 topK 再补做 ACL。
        return _bounded_rows(connection.execute(
            "SELECT chunks.payload FROM chunks WHERE scope=? AND EXISTS "
            "(SELECT 1 FROM json_each(?) a WHERE a.key=chunks.source_id "
            "AND a.value=chunks.version) " + extra + " LIMIT ?",
            (self.scope.key, canonical(dict(query.source_versions)), *args, limit),
        ))

    def chunks(self, query: Query, limit=20000):
        with self._connection() as connection:
            rows = self._rows(connection, query, limit=limit + 1)
        if len(rows) > limit:
            raise ValueError("Authorized vector candidates exceed the bounded index scan.")
        return [chunk for row in rows if _allowed(chunk := Chunk(**json.loads(row[0])), query)]

    def get(self, ids, query: Query):
        ids = tuple(dict.fromkeys(ids))
        if len(ids) > 20000:
            raise ValueError("Too many retrieval candidates.")
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._connection() as connection:
            rows = self._rows(connection, query, f"AND id IN ({placeholders})", ids)
        items = {chunk.id: chunk for row in rows
                 if _allowed(chunk := Chunk(**json.loads(row[0])), query)}
        return [items[item] for item in ids if item in items]

    def missing_sources(self, query: Query):
        with self._connection() as connection:
            return connection.execute(
                "SELECT count(*) FROM json_each(?) a WHERE NOT EXISTS "
                "(SELECT 1 FROM documents d WHERE d.scope=? AND d.source_id=a.key "
                "AND d.version=a.value)",
                (canonical(dict(query.source_versions)), self.scope.key),
            ).fetchone()[0]

    def sources(self, query: Query):
        """每个授权版本只取定位头，不把原文块当成完整检索范围。"""
        with self._connection() as connection:
            rows = _bounded_rows(connection.execute(
                "SELECT coalesce(h.payload,(SELECT c.payload FROM chunks c "
                "WHERE c.scope=d.scope AND c.source_id=d.source_id AND c.version=d.version "
                "ORDER BY c.rowid LIMIT 1)) FROM documents d LEFT JOIN source_headers h "
                "ON h.scope=d.scope AND h.source_id=d.source_id AND h.version=d.version "
                "WHERE d.scope=? AND EXISTS (SELECT 1 FROM json_each(?) a "
                "WHERE a.key=d.source_id AND a.value=d.version) "
                "ORDER BY d.source_id,d.version",
                (self.scope.key, canonical(dict(query.source_versions))),
            ))
        return [chunk for row in rows if row[0] is not None
                and _allowed(chunk := Chunk(**json.loads(row[0])), query)]

    def vector_cache(self, model_id, chunks):
        result = {}
        estimated = 0
        with self._connection() as connection:
            for chunk in chunks:
                check_interrupted(pause_only=True)
                exists = connection.execute(
                    "SELECT 1 FROM chunks WHERE scope=? AND id=?", (self.scope.key, chunk.id)
                ).fetchone()
                if not exists:
                    continue
                digest = hashlib.sha256(chunk.embedding_text().encode("utf-8")).hexdigest()
                row = connection.execute(
                    "SELECT vector FROM vectors WHERE model=? AND hash=?", (model_id, digest)
                ).fetchone()
                if row:
                    # JSON 数字解码成 Python float/list 后明显增大，解码前保守计入。
                    estimated += len((row[0] or "").encode("utf-8")) * 8 + 1024
                    if estimated > VECTOR_MEMORY_BYTES:
                        raise ValueError("Vector cache exceeds the reserved memory budget.")
                    result[chunk.id] = json.loads(row[0])
                    connection.execute(
                        "INSERT OR IGNORE INTO chunk_vectors VALUES(?,?,?,?)",
                        (self.scope.key, chunk.id, model_id, digest),
                    )
        return result

    def cache_vectors(self, model_id, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError("Embedding count does not match input count.")
        with self._connection() as connection:
            for chunk, vector in zip(chunks, vectors):
                check_interrupted(pause_only=True)
                if not vector or len(vector) > 8192 or not all(math.isfinite(x) for x in vector):
                    raise ValueError("Embedding must be a bounded finite numeric vector.")
                exists = connection.execute(
                    "SELECT 1 FROM chunks WHERE scope=? AND id=?", (self.scope.key, chunk.id)
                ).fetchone()
                if not exists:
                    raise ValueError("Cannot cache a vector for an absent scoped chunk.")
                digest = hashlib.sha256(chunk.embedding_text().encode("utf-8")).hexdigest()
                connection.execute(
                    "INSERT OR IGNORE INTO vectors VALUES(?,?,?)",
                    (model_id, digest, canonical(list(vector))),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO chunk_vectors VALUES(?,?,?,?)",
                    (self.scope.key, chunk.id, model_id, digest),
                )
