"""为仓库导航渲染有大小限制的源码摘录。"""

import hashlib
from pathlib import Path
from agent.source.snapshots import _safe_file

MAX_FILE_BYTES = 512 * 1024


def _code_excerpt(self, name, line, count, max_chars):
    entry = self.entries[name]
    if getattr(self, '_frozen_view', None) is not None:
        raw = self._frozen_view.read_bytes(name)
    else:
        path = _safe_file(self.root, Path(name))
        with path.open('rb') as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
    if hashlib.sha256(raw).hexdigest() != entry["digest"]:
        return {"path": name, "error": "源码已变化，请重新定位。"}
    lines = raw.decode("utf-8-sig", errors="replace").splitlines()
    enclosing = [s for s in entry["symbols"] if s["line"] <= line <= s["end_line"]]
    symbol = min(enclosing, key=lambda s: s["end_line"] - s["line"]) if enclosing else None
    start = max(1, symbol["line"] if symbol else line - 4)
    end = min(len(lines), start + count - 1, symbol["end_line"] if symbol else len(lines))
    rows, size = [], 0
    for number in range(start, end + 1):
        row = f"{number}: {lines[number-1]}"
        if size + len(row) + 1 > max_chars:
            break
        rows.append(row)
        size += len(row) + 1
    actual_end = start + len(rows) - 1
    return {
        "path": name,
        "start_line": start,
        "end_line": actual_end,
        "source_sha256": entry["digest"],
        "text": "\n".join(rows),
        "truncated": actual_end < (symbol["end_line"] if symbol else len(lines)),
        "read_hint": {
            "mode": "read",
            "path": name,
            "offset": max(start - 1, actual_end),
            "limit": count,
        },
    }
