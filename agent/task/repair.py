"""Keep rejected drafts local; apply explicit, source-bound edits atomically."""
from copy import deepcopy
import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from agent.model.errors import ModelRequestNotSent
from agent.storage.records import save
from agent.source.snapshots import _safe_file


class Edit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    operation: Literal["set", "remove", "edit_text"]
    before: str
    after: str


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    draft_id: str
    edits: list[Edit] = Field(min_length=1, max_length=24)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def loads(text):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field: " + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=object_pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Invalid JSON number: " + v)))


def location(value, path):
    if not path.startswith("/") or re.search(r"~(?![01])", path):
        raise ValueError("Expected a JSON pointer to a field: " + path)
    parts = [p.replace("~1", "/").replace("~0", "~") for p in path[1:].split("/")]
    parent = value
    for part in parts[:-1]:
        try:
            parent = parent[int(part)] if isinstance(parent, list) and part.isdecimal() else parent[part]
        except (KeyError, IndexError, TypeError):
            raise ValueError("Correction parent does not exist: " + path) from None
    if not isinstance(parent, (dict, list)):
        raise ValueError("Correction parent is not an object or list: " + path)
    key = int(parts[-1]) if isinstance(parent, list) and parts[-1].isdecimal() else parts[-1]
    if isinstance(parent, list) and not isinstance(key, int):
        raise ValueError("Expected a list position: " + path)
    return parent, key


def apply(value, edits, allowed):
    result = deepcopy(value)
    for edit in edits:
        if not any(edit.path == p or (p != "$raw" and edit.path.startswith(p + "/")) for p in allowed):
            raise ValueError("Correction outside reported fields: " + edit.path)
        if edit.path == "$raw":
            if edit.operation != "edit_text":
                raise ValueError("Raw JSON correction requires edit_text")
            parent, key = {"raw": result}, "raw"
        else:
            parent, key = location(result, edit.path)
        present = key in parent if isinstance(parent, dict) else isinstance(key, int) and 0 <= key < len(parent)
        current = parent[key] if present else None
        if edit.operation == "edit_text":
            if not isinstance(current, str) or not edit.before or current.count(edit.before) != 1:
                raise ValueError("Text correction requires one exact nonempty match: " + edit.path)
            parent[key] = current.replace(edit.before, edit.after, 1)
        else:
            if (not present and edit.before != "") or (present and encode(current) != encode(loads(edit.before))):
                raise ValueError("Original value does not match: " + edit.path)
            if edit.operation == "remove":
                if not present or edit.after:
                    raise ValueError("remove requires an existing field and empty after: " + edit.path)
                del parent[key]
            else:
                if isinstance(parent, list) and not present:
                    raise ValueError("List position does not exist: " + edit.path)
                parent[key] = loads(edit.after)
        if edit.path == "$raw":
            result = parent[key]
    return loads(result) if isinstance(result, str) else result


def fields(error, value):
    if isinstance(error, ValidationErrors):
        return error.paths
    if isinstance(value, str):
        return ["$raw"]
    if hasattr(error, "errors"):
        paths = ["/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in e["loc"])
                 for e in error.errors() if e["loc"]]
        if paths:
            return sorted(set(paths))
    message = str(error)
    paths = []
    if "test_driver" in message:
        paths.append("/test_driver")
    if "regression command" in message:
        paths.append("/regression_command")
    if any(word in message for word in ("验收", "产物", "预期", "Fixed task tests", "test_code")):
        paths.append("/test_code")
    if paths:
        return paths
    if any(word in message for word in ("Step", "from_step", "inspect", "blocked")):
        return ["/steps"]
    return []


class ValidationErrors(ValueError):
    """Independent checks report together, without running rejected work."""
    def __init__(self, errors):
        self.paths = sorted({path for paths, _ in errors for path in paths})
        self.issues = []
        for paths, error in errors:
            if hasattr(error, "errors"):
                for issue in error.errors():
                    location = "/" + "/".join(map(str, issue["loc"]))
                    self.issues.append(f"{location}: {issue['msg']}; actual={encode(issue.get('input'))[:100]}")
            else:
                self.issues.append(", ".join(paths) + ": " + str(error))
        super().__init__("\n".join(self.issues))


def preview(value, paths, error):
    snippets = {}
    for path in paths:
        if path == "$raw":
            text = value
            start = max(0, getattr(error, "pos", 0) - 100)
            snippets[path] = text[start:start + 400]
            continue
        try:
            parent, key = location(value, path)
            text = encode(parent[key])
            if path == "/test_code" and isinstance(parent[key], str):
                rows = parent[key].splitlines(keepends=True)
                cause = error.__cause__
                match = re.search(r":(\d+)\b", str(error))
                line = getattr(cause, "lineno", None) or (int(match[1]) if match else None)
                indices = ([max(0, line - 2)] if line else
                           [i for i, row in enumerate(rows) if any(word in row and word in str(error)
                            for word in ("imports", "ARTIFACT_CHECKS"))])
                text = "".join("".join(rows[i:i + 3]) for i in indices[:3]) if indices else parent[key][:600]
            snippets[path] = text[:1200]
        except (KeyError, IndexError, TypeError, ValueError):
            snippets[path] = "<missing>"
    return snippets


def reject(owner, error):
    value = getattr(owner, "rejected_input", None)
    if value is not None:
        paths = fields(error, value)
        record = {"value": value, "source": owner.state["source"], "paths": paths,
                  "error": str(error), "snippets": preview(value, paths, error)}
        if isinstance(error, ValidationErrors):
            record["issues"] = error.issues
        raw = encode(record).encode()
        identifier = hashlib.sha256(raw).hexdigest()
        relative = "drafts/" + identifier + ".json"
        path = owner.root / relative
        if not path.exists():
            save(path, record)
        ref = {"id": identifier, "record": relative,
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source": record["source"]}
        read(owner, ref)
        owner.state["repair_draft"] = ref
    ref = owner.state.get("repair_draft")
    if not ref:
        return "[Plan rejected; no actions executed] " + str(error)[:1800] + "\nSubmit a valid work request."
    record = read(owner, ref)
    notice = {"draft_id": ref["id"], "error": str(error)[:1800], "allowed_paths": record["paths"],
              "original_fragments": record["snippets"], "original_record": "$record/" + ref["record"],
              "origin": "Your rejected tool arguments; unchanged work is retained locally, not generated by the host."}
    if record.get("issues"):
        notice["issues"] = record["issues"]
    if str(error) != record["error"]:
        notice["original_error"] = record["error"][:1800]
    instruction = ("Use repair_work for these fields; preserve all other work." if record["paths"] else
                   "No field correction is available; submit a valid work request.")
    return "[Plan rejected; no actions executed] " + encode(notice) + "\n" + instruction


def read(owner, ref):
    try:
        raw = _safe_file(owner.root, Path(ref["record"])).read_bytes()
        record = loads(raw.decode())
    except (OSError, ValueError):
        raise ModelRequestNotSent("state_conflict", "Rejected draft unavailable; no correction applied.") from None
    if (not isinstance(record, dict) or hashlib.sha256(raw).hexdigest() != ref["sha256"] or ref["source"] != owner.state["source"]
            or record.get("source") != ref["source"] or hashlib.sha256(encode(record).encode()).hexdigest() != ref["id"]):
        raise ModelRequestNotSent("state_conflict", "Rejected draft changed; no correction or operation applied.")
    return record


def merge(owner, arguments):
    correction = Correction.model_validate(arguments)
    ref = owner.state.get("repair_draft")
    if not ref or correction.draft_id != ref["id"]:
        raise ValueError("repair_work requires the current draft_id")
    record = read(owner, ref)
    return apply(record["value"], correction.edits, record["paths"])
