"""只保存授权来源头；没有正文分块、向量或神经模型缓存。"""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3

from agent.rag.types import CANDIDATE_MEMORY_BYTES
from agent.rag.types import Chunk, canonical
from agent.task.control import check_interrupted, current


class Index:
    def __init__(self, path, scope, config, token_count):
        self.path = Path(path)
        self.scope, self.config, self.token_count = scope, config, token_count
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            old = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'"
            ).fetchone()
            if old:
                raise ValueError("Historical vector indexes require their frozen source version.")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS configurations (scope TEXT PRIMARY KEY, settings TEXT);
                CREATE TABLE IF NOT EXISTS source_headers (
                    scope TEXT, source_id TEXT, version TEXT, payload TEXT,
                    PRIMARY KEY(scope, source_id, version)
                );
            """)
            settings = canonical(asdict(config))
            previous = connection.execute(
                "SELECT settings FROM configurations WHERE scope=?", (scope.key,)
            ).fetchone()
            if previous and previous[0] != settings:
                raise ValueError("Changed retrieval settings require a new configuration version.")
            connection.execute(
                "INSERT OR IGNORE INTO configurations VALUES(?,?)", (scope.key, settings)
            )

    @contextmanager
    def _connection(self):
        if self.closed:
            raise RuntimeError("The retrieval catalog is closed.")
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.set_progress_handler(
                lambda: int(current() is not None and current().kind == "pause"), 1000
            )
            connection.execute("PRAGMA cache_size=-4096")
            connection.execute("PRAGMA mmap_size=0")
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            connection.execute(f"PRAGMA max_page_count={self.config.max_index_bytes // page_size}")
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

    def register(self, source_id, version, sha256, *, path="", kind="text", run_id="",
                 timestamp="", length=0):
        check_interrupted(pause_only=True)
        header = Chunk(
            id=hashlib.sha256(f"{source_id}\0{version}".encode()).hexdigest(),
            source_id=source_id, version=version, source_sha256=sha256,
            path=path, kind=kind, start=0, end=0, parent_start=0, parent_end=length,
            line_start=1, line_end=1, text="", run_id=run_id, timestamp=timestamp,
        )
        with self._connection() as connection:
            previous = connection.execute(
                "SELECT payload FROM source_headers WHERE scope=? AND source_id=? AND version=?",
                (self.scope.key, source_id, version),
            ).fetchone()
            if previous:
                if json.loads(previous[0])["source_sha256"] != sha256:
                    raise ValueError("An immutable source version changed its content hash.")
                return False
            connection.execute(
                "INSERT INTO source_headers VALUES(?,?,?,?)",
                (self.scope.key, source_id, version, header.render()),
            )
        return True

    def upsert(self, document):
        return self.register(
            document.source_id, document.version, document.sha256, path=document.path,
            kind=document.kind, run_id=document.run_id, timestamp=document.timestamp,
            length=len(document.text),
        )

    def prune_sources(self, versions, *, source_prefix=None):
        removed = 0
        with self._connection() as connection:
            for identity, version in connection.execute(
                "SELECT source_id,version FROM source_headers WHERE scope=?", (self.scope.key,)
            ).fetchall():
                if source_prefix is None and not identity.startswith('source:'):
                    continue
                if source_prefix is not None and not (
                    identity == source_prefix or identity.startswith(source_prefix.rstrip('/') + '/')
                ):
                    continue
                if versions.get(identity) != version:
                    connection.execute(
                        "DELETE FROM source_headers WHERE scope=? AND source_id=? AND version=?",
                        (self.scope.key, identity, version),
                    )
                    removed += 1
        return removed

    def sources(self, query):
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT payload FROM source_headers WHERE scope=? AND EXISTS "
                "(SELECT 1 FROM json_each(?) a WHERE a.key=source_id AND a.value=version) "
                "ORDER BY source_id,version",
                (self.scope.key, canonical(dict(query.source_versions))),
            )
            result, memory = [], 0
            for (raw,) in rows:
                check_interrupted(pause_only=True)
                memory += len(raw.encode("utf-8")) * 4 + 1024
                if memory > CANDIDATE_MEMORY_BYTES:
                    raise ValueError("Authorized source catalog exceeds its memory budget.")
                item = Chunk(**json.loads(raw))
                path = item.path.replace("\\", "/")
                if not any(path == p.rstrip("/") or path.startswith(p.rstrip("/") + "/")
                           for p in query.excluded_paths):
                    result.append(item)
            return result

    def missing_sources(self, query):
        return len(query.source_versions) - len(self.sources(query))

    def stats(self):
        with self._connection() as connection:
            count = connection.execute(
                "SELECT count(*) FROM source_headers WHERE scope=?", (self.scope.key,)
            ).fetchone()[0]
        return {"documents": count, "bytes": self.path.stat().st_size}
