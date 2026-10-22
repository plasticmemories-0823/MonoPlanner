"""按需读取本地边界提醒，并保存任务私有的读取凭据。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.model.errors import ModelRequestNotSent
from agent.settings import PROJECT_ROOT
from agent.source.snapshots import _safe_file
from agent.storage.records import digest, save
from agent.task import control


CATALOG_PATH = PROJECT_ROOT / "resources/boundary-hints.json"
BoundaryKey = Literal[
    "general", "math", "numeric", "collections", "text", "state", "time",
    "permissions", "concurrency", "external_io", "generation", "interaction",
]
FORMAT = "boundary-hints-v1"


class Lookup(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    keys: list[BoundaryKey] = Field(min_length=1, max_length=12)

    @field_validator("keys")
    @classmethod
    def distinct(cls, keys):
        if len(keys) != len(set(keys)):
            raise ValueError("Boundary keys must be distinct; batch each category once")
        return keys


def load_catalog():
    if CATALOG_PATH.stat().st_size > 32 * 1024:
        raise ValueError("Boundary catalog exceeds its size limit")
    raw = CATALOG_PATH.read_bytes()
    data = json.loads(raw)
    if data.get("version") != 1 or set(data.get("entries", {})) != set(get_args(BoundaryKey)):
        raise ValueError("Boundary catalog has an unsupported version or category set")
    for entry in data["entries"].values():
        if (set(entry) != {"summary", "hints"}
                or not isinstance(entry["summary"], str)
                or not 1 <= len(entry["summary"]) <= 120
                or not entry["summary"].isascii()
                or not isinstance(entry["hints"], list)
                or not 4 <= len(entry["hints"]) <= 6
                or any(not isinstance(hint, str) or not 1 <= len(hint) <= 360
                       or not hint.isascii() for hint in entry["hints"])):
            raise ValueError("Boundary catalog contains an invalid reminder card")
    return data, hashlib.sha256(raw).hexdigest()


CATALOG, CATALOG_SHA256 = load_catalog()


def lookup(keys):
    request = Lookup(keys=keys)
    try:
        _, current_hash = load_catalog()
        if current_hash != CATALOG_SHA256:
            raise ValueError("catalog changed after loading")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ModelRequestNotSent("state_conflict", "Boundary catalog unavailable or changed") from exc
    return result_for(request.keys)


def result_for(keys):
    return {
        "outcome": "boundary_hints",
        "catalog_sha256": CATALOG_SHA256,
        "guidance": "Reminders only, not task requirements or expected answers. Apply relevant checks to the actual deliverable; record gaps without inventing requirements.",
        "entries": {key: deepcopy(CATALOG["entries"][key]) for key in keys},
    }


def read_receipt(loop, ref):
    try:
        relative = Path(ref["record"])
        if (len(relative.parts) != 2 or relative.parts[0] != "boundary-hints"
                or relative.suffix != ".json"):
            raise ValueError("invalid receipt path")
        raw = _safe_file(loop.root, relative).read_bytes()
        data = json.loads(raw)
        request = Lookup.model_validate({"keys": data["keys"]})
        calls = data["response"]["data"]["tool_calls"]
        if (hashlib.sha256(raw).hexdigest() != ref["sha256"]
                or data["format"] != FORMAT
                or relative.stem != digest(data["response"])
                or len(calls) != 1 or calls[0]["name"] != "get_boundary_hints"
                or calls[0]["args"] != request.model_dump()
                or data["result"] != result_for(request.keys)):
            raise ValueError("receipt content changed")
        return data
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ModelRequestNotSent("state_conflict", "Boundary lookup receipt unavailable or changed") from exc


def require_general(loop):
    ref = loop.state.get("boundary_reads", {}).get("general")
    if not ref:
        raise ValueError(
            'Before drafting generated acceptance, call get_boundary_hints with keys including '
            '"general" and any relevant categories. Read the returned reminders, then submit '
            'acceptance with applicable real checks. No trial or source edit has run.'
        )
    data = read_receipt(loop, ref)
    if "general" not in data["keys"]:
        raise ModelRequestNotSent("state_conflict", "General boundary card was not read")


def execute(loop, request, response):
    from agent.acceptance.revisions import append_result

    control.check_interrupted(pause_only=True)
    relative = "boundary-hints/" + digest(response) + ".json"
    folder = loop.root / "boundary-hints"
    if folder.is_symlink() or folder.is_junction():
        raise ModelRequestNotSent("state_conflict", "Boundary receipt directory is a link")
    folder.mkdir(exist_ok=True)
    path = loop.root / relative
    if path.exists():
        path = _safe_file(loop.root, Path(relative))
        ref = {"record": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        data = read_receipt(loop, ref)
        if data["response"] != response or data["keys"] != request.keys:
            raise ModelRequestNotSent("state_conflict", "Boundary lookup response changed")
    else:
        data = {
            "format": FORMAT, "response": response, "keys": request.keys,
            "result": lookup(request.keys),
        }
        save(path, data)
        ref = {"record": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    append_result(loop, response, data["result"])
    reads = loop.state.setdefault("boundary_reads", {})
    for key in request.keys:
        reads[key] = ref
    loop.persist("boundary_hints_read", {"keys": request.keys, **ref})
