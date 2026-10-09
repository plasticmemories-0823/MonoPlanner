"""Atomic JSON records and bounded excerpts; originals stay on disk."""
import json
import hashlib
from pathlib import Path


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def clip(text, maximum=6000):
    if len(text) <= maximum:
        return text
    return text[:maximum // 2] + "\n[excerpt; complete content retained in record]\n" + text[-maximum // 2:]
