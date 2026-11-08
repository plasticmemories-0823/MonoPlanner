"""同一物理源码的路径别名可计数，源码身份和测试体边界仍必须成立。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def witness_class():
    path = Path(__file__).resolve().parents[1] / "resources/drivers/consumption.py"
    spec = importlib.util.spec_from_file_location("alias_path_witness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Witness


@pytest.mark.parametrize(
    "relationship,inside_test,changed_body,expected_calls",
    [
        ("declared", True, False, 1),
        ("hardlink", True, False, 1),
        ("copy", True, False, 0),
        ("hardlink", True, True, 0),
        ("hardlink", False, False, 0),
    ],
)
def test_api_path_identity_preserves_source_and_test_boundary(
    tmp_path, relationship, inside_test, changed_body, expected_calls
):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "sample.py"
    original = "def value():\n    return 42\n"
    source.write_text(original, encoding="utf-8")
    runtime = tmp_path / "other-mount"
    runtime.mkdir()
    alias = runtime / "sample.py"
    if relationship == "hardlink":
        alias.hardlink_to(source)
        assert alias.samefile(source)
    elif relationship == "copy":
        alias.write_bytes(source.read_bytes())
        assert not alias.samefile(source)
    else:
        alias = source

    test_path = tmp_path / "test_saved.py"
    test_path.write_text("# Fixed test body location\n", encoding="utf-8")
    check = {
        "id": "sample_api",
        "artifact_type": "python_api",
        "consumer": "python_tests",
        "api_entry": "sample:value",
        "test_ids": ["Check.test_value"],
    }
    witness = witness_class()([check], repo, test_path)
    body = "def value():\n    return 99\n" if changed_body else original
    namespace = {}
    exec(compile(body, str(alias), "exec", dont_inherit=True), namespace)
    value = 99 if changed_body else 42
    if inside_test:
        test_code = f"def test_value():\n    assert value() == {value}\n"
    else:
        assert namespace["value"]() == value
        test_code = "def test_value():\n    assert True\n"
    exec(compile(test_code, str(test_path), "exec", dont_inherit=True), namespace)
    witness.start(SimpleNamespace(id=lambda: "saved.Check.test_value"))
    try:
        namespace["test_value"]()
    finally:
        witness.stop()
    result = witness.finish([{"id": "saved.Check.test_value", "status": "passed"}])
    row = result["checks"][0]
    assert row["calls"] == {"Check.test_value": expected_calls}
    assert row["status"] == ("passed" if expected_calls else "error")
    assert source.read_text(encoding="utf-8") == original
