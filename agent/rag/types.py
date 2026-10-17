"""检索边界的数据对象；授权和版本由调用方显式传入。"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Callable, Mapping, Protocol, Sequence


TokenCount = Callable[[str], int]


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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
    exact_path: str = ""
    exact_symbol: str = ""
    required_entities: tuple[str, ...] = ()
    required_kinds: tuple[str, ...] = ()
    excluded_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class RagConfig:
    chunk_target: int = 256
    chunk_hard_limit: int = 512
    chunk_overlap: int = 64
    recall_top_k: int = 100
    rerank_k: int = 20
    message_token_limit: int = 8192
    vector_weight: float = 0.5
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
        if not 0 <= self.vector_weight <= 1 or not 0 <= self.rewrite_threshold <= 1:
            raise ValueError("Fusion weight and rewrite threshold must be in [0, 1].")
        if self.max_document_bytes <= 0 or self.max_index_bytes <= 0:
            raise ValueError("Index resource limits must be positive.")

    @property
    def bm25_weight(self):
        return 1 - self.vector_weight


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
