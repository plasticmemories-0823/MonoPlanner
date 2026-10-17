"""检索边界的数据对象；授权和版本由调用方显式传入。"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Callable, Mapping, Protocol, Sequence


TokenCount = Callable[[str], int]


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


RAG_PARENT_MEMORY_BYTES = 256 * 1024 * 1024
CANDIDATE_MEMORY_BYTES = 32 * 1024 * 1024
VECTOR_MEMORY_BYTES = 64 * 1024 * 1024
# 来源头、向量缓存、命中页分别保留既有候选级空间；SQLite 与消息包装另留 8 MiB。
SOURCE_FIXED_MEMORY_BYTES = 2 * CANDIDATE_MEMORY_BYTES + VECTOR_MEMORY_BYTES + 8 * 1024 * 1024
# 混合 Unicode 字符串可占原始 UTF-8 的四倍；包含旧正文、读取、校验及临时复制。
SOURCE_TRANSIENT_MULTIPLIER = 16


class SourceReadBudgetExceeded(ValueError):
    """原件尚未完整读取；携带定位信息，不能把资源拒绝解释成没有匹配。"""

    def __init__(self, size_bytes, *, source_id="", path=""):
        self.details = {
            "source_id": str(source_id), "path": str(path),
            "reason": "source_read_memory_budget", "size_bytes": size_bytes,
            "estimated_bytes": SOURCE_FIXED_MEMORY_BYTES + SOURCE_TRANSIENT_MULTIPLIER * size_bytes,
            "parent_budget_bytes": RAG_PARENT_MEMORY_BYTES, "unread_from_byte": 0,
        }
        super().__init__(canonical(self.details))


def source_read_limit():
    """从共享父进程预留推导读入上限，不复用语义模型的输入大小限制。"""
    return max(0, (RAG_PARENT_MEMORY_BYTES - SOURCE_FIXED_MEMORY_BYTES)
               // SOURCE_TRANSIENT_MULTIPLIER)


def guard_source_read(size_bytes, *, source_id="", path=""):
    if type(size_bytes) is not int or size_bytes < 0:
        raise ValueError("Original source size must be a non-negative integer.")
    estimated = SOURCE_FIXED_MEMORY_BYTES + SOURCE_TRANSIENT_MULTIPLIER * size_bytes
    if estimated > RAG_PARENT_MEMORY_BYTES:
        raise SourceReadBudgetExceeded(size_bytes, source_id=source_id, path=path)
    return estimated


@dataclass(frozen=True)
class Scope:
    project_id: str
    task_id: str
    config_version: str = "v1"

    @property
    def key(self) -> str:
        return hashlib.sha256(canonical(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    source_id: str
    version: str
    text: str
    kind: str = "text"
    path: str = ""
    run_id: str = ""
    timestamp: str = ""

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Query:
    text: str
    source_versions: Mapping[str, str]
    intent: str = "diagnostic"
    mode: str = "exact"
    cursor: str = ""
    limit: int = 80
    offset: int = 0
    required_entities: tuple[str, ...] = ()
    required_kinds: tuple[str, ...] = ()
    excluded_paths: tuple[str, ...] = ()
    original_text: str | None = None
    source_priorities: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RagConfig:
    chunk_target: int = 256
    chunk_hard_limit: int = 512
    chunk_overlap: int = 64
    recall_top_k: int = 100
    rerank_k: int = 20
    message_token_limit: int = 8192
    rewrite_threshold: float = 0.15
    max_document_bytes: int = 4 * 1024 * 1024
    max_index_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if not 0 < self.chunk_target <= self.chunk_hard_limit <= 512:
            raise ValueError("Chunk target must fit the hard input limit (at most 512 tokens).")
        if not 0 <= self.chunk_overlap <= 64:
            raise ValueError("Chunk overlap must be between 0 and 64 tokens.")
        if not 1 <= self.rerank_k <= self.recall_top_k <= 200:
            raise ValueError("Require 1 <= rerank_k <= recall_top_k <= 200.")
        if not 1 <= self.message_token_limit <= 8192:
            raise ValueError("The retrieval message limit cannot exceed 8192 tokens.")
        if not 0 <= self.rewrite_threshold <= 1:
            raise ValueError("Rewrite threshold must be in [0, 1].")
        if self.max_document_bytes <= 0 or self.max_index_bytes <= 0:
            raise ValueError("Index resource limits must be positive.")

@dataclass(frozen=True)
class Chunk:
    id: str
    source_id: str
    version: str
    source_sha256: str
    path: str
    kind: str
    start: int
    end: int
    parent_start: int
    parent_end: int
    line_start: int
    line_end: int
    text: str
    symbol: str = ""
    run_id: str = ""
    timestamp: str = ""
    range_unit: str = "unicode_codepoints"

    def payload(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        return canonical(self.payload())

    def embedding_text(self) -> str:
        """语义编码只保留有意义的来源头；版本与哈希在交付消息中完整保留。"""
        header = canonical({"path": self.path, "kind": self.kind, "symbol": self.symbol})
        return header + "\n" + self.text


class VectorProvider(Protocol):
    def search(self, query: Query, index, k: int) -> Sequence[tuple[str, float]]: ...


class RerankProvider(Protocol):
    def score(self, query: str, chunks: Sequence[Chunk]) -> Sequence[float]: ...


@dataclass(frozen=True)
class RetrievalResult:
    message: str
    chunks: tuple[Chunk, ...]
    incomplete: bool
    reasons: tuple[str, ...]
    missing_entities: tuple[str, ...]
    missing_kinds: tuple[str, ...]
    continuation: tuple[dict, ...]
    route_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    exact_hit: bool = False
    rerank_scores: tuple[float, ...] = ()
