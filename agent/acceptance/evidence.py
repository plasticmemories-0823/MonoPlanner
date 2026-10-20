"""按可信原件核验引用并保存逐项修订记录；不作语义正确性证明。"""

import hashlib
import json
from pathlib import PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from agent.acceptance.expectations import rewrite
from agent.model.errors import ModelRequestNotSent
from agent.source.snapshots import _safe_file


RULE_VERSION = "verbatim-expectations-v2"
QUOTE_ERROR = "未找到原文或原文错误 请输入完全一致的原文"


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=4096)
    quote: str = Field(default="", max_length=65536)


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    before: str = Field(min_length=1, max_length=32768)
    after: str = Field(min_length=1, max_length=32768)
    evidence_ids: list[str] = Field(min_length=1, max_length=24)
    reason: str = Field(min_length=1, max_length=1600)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot(loop):
    base = loop.baseline
    try:
        manifest = json.loads(base.manifest_path.read_text(encoding="utf-8"))
        computed = hashlib.sha256(json.dumps(
            {"files": base.files, "excluded_files": base.excluded_files},
            sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        valid = (manifest["files"] == base.files
                 and manifest.get("excluded_files", {}) == base.excluded_files
                 and manifest["digest"] == base.digest == computed
                 and base.verify_integrity()["valid"])
    except (OSError, ValueError, KeyError):
        valid = False
    if not valid:
        raise ModelRequestNotSent("state_conflict", "Original evidence snapshot changed")


def _text(loop, source):
    if source == "task_request":
        return loop.state["request"]
    if source.startswith("user_update:"):
        identifier = source[len("user_update:"):]
        matches = [row["text"] for row in loop.state.get("user_updates", [])
                   if row["id"] == identifier]
        if len(matches) != 1:
            raise ValueError(QUOTE_ERROR)
        return matches[0]
    if source.startswith("original_file:"):
        name = source[len("original_file:"):]
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts or ":" in name
                or "\\" in name or path.as_posix() != name):
            raise ValueError("Evidence requires a canonical original-file relative path")
        if name not in loop.baseline.files:
            raise ValueError(QUOTE_ERROR)
        raw = _safe_file(loop.baseline.root, path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != loop.baseline.files[name]:
            raise ModelRequestNotSent("state_conflict", "Original evidence snapshot changed")
        try:
            return raw.decode("utf-8")  # 不转换 CRLF、BOM 或其它空白。
        except UnicodeError as exc:
            raise ValueError("Evidence original file must contain UTF-8 text") from exc
    raise ValueError(
        "Evidence source must be task_request, user_update:<recorded-id>, "
        "or original_file:<frozen-relative-path>; current outputs are not evidence"
    )


def verify_sources(loop, items):
    if not items:
        raise ValueError(QUOTE_ERROR)
    _snapshot(loop)
    rows, seen = [], set()
    for item in items:
        if not item.quote:
            raise ValueError(QUOTE_ERROR)
        if item.id in seen:
            raise ValueError("Evidence ids must be unique")
        seen.add(item.id)
        original = _text(loop, item.source)
        offset = original.find(item.quote)
        if offset < 0:
            raise ValueError(QUOTE_ERROR)
        rows.append({**item.model_dump(), "source_sha256": sha(original),
                     "quote_sha256": sha(item.quote), "match_offset": offset})
    return rows


def prepare(loop, base_ref, old_code, new_code, items, corrections):
    sources = verify_sources(loop, items)
    ids = {item.id for item in items}
    used = {ref for item in corrections for ref in item.evidence_ids}
    if not corrections or used != ids or any(not item.reason.strip() for item in corrections):
        raise ValueError("Every correction requires cited evidence and a reason; no unused references")
    revised = rewrite(old_code, corrections)
    if revised != new_code:
        raise ValueError("Undeclared acceptance changes; test_code must exactly match the listed corrections")
    return {
        "rule_version": RULE_VERSION,
        "base_version": base_ref,
        "original_source": loop.baseline.digest,
        "before_code_sha256": sha(old_code),
        "after_code_sha256": sha(new_code),
        "evidence": sources,
        "corrections": [item.model_dump() for item in corrections],
        "semantic_correctness_proven": False,
    }
