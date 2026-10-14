"""按内容保存请求中的消息；完整请求可以校验后重建，不重复写入历史前缀。"""

import gzip
import hashlib
import json
from pathlib import Path

from agent.source.snapshots import _link, _safe_file
from agent.storage.records import save


FORMAT = "monoplanner-request-refs-v1"


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def save_request(root, folder, messages, purpose, archive=None):
    root, folder = Path(root).resolve(), Path(folder).resolve()
    folder.relative_to(root)
    references = []
    for message in messages:
        raw = _encode(message)
        digest = hashlib.sha256(raw).hexdigest()
        relative = Path("request-objects") / (digest + ".json.gz")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if _link(target.parent) or ((target.exists() or target.is_symlink()) and _link(target)):
            raise ValueError("Request archive cannot contain links")
        if target.exists():
            existing = gzip.decompress(_safe_file(root, relative).read_bytes())
            if existing != raw:
                raise ValueError("Archived request message changed")
        else:
            # 原件写失败须阻止请求发送；不同请求只共享不可变原件。
            with target.open("xb") as stream:
                stream.write(gzip.compress(raw, mtime=0))
                stream.flush()
                import os
                os.fsync(stream.fileno())
        if archive is not None:
            archive._register(archive._reference(target), "request_message")
        references.append({"path": relative.as_posix(), "sha256": digest, "bytes": len(raw)})
    original = {"messages": messages, "purpose": purpose}
    save(folder / "request.json", {
        "format": FORMAT, "purpose": purpose, "messages": references,
        "sha256": hashlib.sha256(_encode(original)).hexdigest(),
    })


def read_request(path):
    """兼容旧原件；新原件逐项核验后恢复原始消息值与工具参数字符串。"""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != FORMAT:
        return data
    # 本格式只用于任务 requests/call-NNNN，避免原件内路径控制读取位置。
    if path.parent.parent.name != "requests":
        raise ValueError("Invalid request archive location")
    root = path.parent.parent.parent
    messages = []
    for item in data["messages"]:
        digest = item["sha256"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid request object identity")
        relative = Path("request-objects") / (digest + ".json.gz")
        if item["path"] != relative.as_posix():
            raise ValueError("Request object escaped its archive")
        raw = gzip.decompress(_safe_file(root, relative).read_bytes())
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Archived request checksum mismatch")
        messages.append(json.loads(raw))
    result = {"messages": messages, "purpose": data["purpose"]}
    if hashlib.sha256(_encode(result)).hexdigest() != data["sha256"]:
        raise ValueError("Archived request order or metadata changed")
    return result
