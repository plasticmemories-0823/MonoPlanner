"""不可变参数版本和单变量候选，不允许模型自由生成配置。"""

from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Mapping


KNOBS = frozenset({'recall_top_k', 'vector_weight', 'rewrite_threshold', 'chunk_target'})
CHUNK_LEVELS = (256, 384, 512)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class Config:
    # 存规范化 JSON，避免 frozen 数据类内部的字典仍被外部修改。
    payload: str
    parent: str = ''

    @classmethod
    def create(cls, values: Mapping[str, Any], parent: str = '') -> 'Config':
        copied = json.loads(canonical(dict(values)))
        if not KNOBS.issubset(copied):
            raise ValueError('Missing required RAG tuning parameters')
        top_k = copied['recall_top_k']
        if type(top_k) is not int or not 10 <= top_k <= 200:
            raise ValueError('recall_top_k must be an integer between 10 and 200')
        if 'rerank_k' in copied and not 1 <= copied['rerank_k'] <= top_k:
            raise ValueError('rerank_k must not exceed recall_top_k')
        for name in ('vector_weight', 'rewrite_threshold'):
            if not _number(copied[name]) or not 0 <= copied[name] <= 1:
                raise ValueError(f'{name} must be a finite number between 0 and 1')
        if type(copied['chunk_target']) is not int or copied['chunk_target'] not in CHUNK_LEVELS:
            raise ValueError('chunk_target must be 256, 384, or 512')
        if copied['chunk_target'] > copied.get('chunk_hard_limit', 512):
            raise ValueError('chunk_target must not exceed the fixed chunk hard limit')
        return cls(canonical(copied), parent)

    @property
    def values(self) -> dict[str, Any]:
        return json.loads(self.payload)

    @property
    def version(self) -> str:
        return fingerprint({'schema': 1, 'parent': self.parent, 'values': self.values})


@dataclass(frozen=True)
class Candidate:
    id: str
    knob: str | None
    config: Config


def enumerate_candidates(base: Config) -> tuple[Candidate, ...]:
    values = base.values
    candidates = [Candidate('no_change', None, base)]
    for name in sorted(KNOBS):
        old = values[name]
        if name == 'recall_top_k':
            # 向下取整步长保证整数调整不会超出 10%。
            step = max(1, old // 10)
            choices = [old - step, old + step]
        elif name == 'chunk_target':
            position = CHUNK_LEVELS.index(old)
            choices = [value for index, value in enumerate(CHUNK_LEVELS)
                       if abs(index - position) == 1]
        elif name == 'vector_weight':
            choices = [float(Decimal(str(old)) + delta)
                       for delta in (Decimal('-0.05'), Decimal('0.05'))]
        else:
            choices = [float(Decimal(str(old)) * scale)
                       for scale in (Decimal('0.95'), Decimal('1.05'))]

        for new in choices:
            if new == old:
                continue
            changed = dict(values)
            changed[name] = new
            try:
                config = Config.create(changed, base.version)
            except ValueError:
                continue
            candidates.append(Candidate(config.version, name, config))
    return tuple(candidates)


def validate_candidate(base: Config, proposed: Config) -> str | None:
    if proposed == base:
        return None
    for candidate in enumerate_candidates(base):
        if candidate.config == proposed:
            return candidate.knob
    raise ValueError('Candidate must change exactly one allowed parameter by one legal step')
