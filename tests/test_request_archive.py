"""完整请求去重归档须保持可重建及可核验，不能改变模型原文。"""

import json

import pytest

from agent.storage.requests import read_request, save_request


def test_shared_prefix_saved_once_and_invalid_tool_json_preserved(tmp_path):
    prefix = {"type": "human", "data": {"content": "已读资料" * 10000}}
    invalid = {"type": "ai", "data": {"reasoning_content": "保留思考", "arguments": "{bad"}}
    messages = [prefix, invalid]
    for i in range(10):
        folder = tmp_path / "requests" / f"call-{i:04d}"
        save_request(tmp_path, folder, messages, "plan")
        assert read_request(folder / "request.json") == {"messages": messages, "purpose": "plan"}
    objects = list((tmp_path / "request-objects").glob("*.gz"))
    assert len(objects) == 2
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 16000


def test_modified_object_and_message_order_cannot_pass(tmp_path):
    folder = tmp_path / "requests" / "call-0001"
    save_request(tmp_path, folder, [{"value": 1}, {"value": 2}], "plan")
    path = folder / "request.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["messages"].reverse()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="order or metadata"):
        read_request(path)


def test_legacy_request_is_read_without_rewriting(tmp_path):
    path = tmp_path / "old.json"
    old = {"messages": [{"content": "原始请求"}], "purpose": "plan"}
    path.write_text(json.dumps(old), encoding="utf-8")
    assert read_request(path) == old
