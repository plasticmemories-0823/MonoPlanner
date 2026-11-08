"""真实 API 见证必须理解继承，保留来源诊断，并拒绝抽象占位实现。"""
import hashlib
import json
from pathlib import Path
import textwrap

import pytest

from agent.acceptance.contracts import VerificationSpec
from agent.acceptance.execution import validate_execution_result
from agent.acceptance.runner import AcceptanceRun
from agent.execution.local import LocalEnvironment


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    return path


def write(repo, name, source):
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


def run(repo, entry, statement, imports="import sample", check_ids=None):
    code = "import unittest\n" + imports + "\nclass Check(unittest.TestCase):\n"
    code += "    def test_value(self):\n" + textwrap.indent(statement, "        ") + "\n"
    check = {
        "id": "library", "artifact_type": "python_api", "consumer": "python_tests",
        "api_entry": entry, "test_ids": check_ids or ["Check.test_value"],
    }
    spec = VerificationSpec(("Exercise the original API",), test_code=code,
                            execution_checks=[check])
    runner = AcceptanceRun(
        repo, spec, environment=LocalEnvironment(repo, artifact_root=repo.parent / "runtime"),
        artifacts_root=repo.parent / "acceptance",
    )
    result = runner.run("candidate")
    return runner, result, result["execution"]["checks"][0]


def test_inherited_method_keeps_the_declared_entry_and_runs_once(repo):
    write(repo, "sample.py", """
        class Base:
            def value(self):
                from pathlib import Path
                path = Path('calls.txt')
                path.write_text(path.read_text() + 'x' if path.exists() else 'x')
                return 42
        class Middle(Base):
            pass
        class Child(Middle):
            pass
    """)
    _, result, row = run(repo, "sample:Child.value", "self.assertEqual(sample.Child().value(), 42)")
    assert result["status"] == "passed" and result["execution_ready"], result
    assert row["api_entry"] == "sample:Child.value"
    assert row["resolved_api_entry"] == "sample:Base.value"
    assert row["calls"] == {"Check.test_value": 1}
    assert (repo / "calls.txt").read_text() == "x"


@pytest.mark.parametrize("base_import", [
    "from .base import Base as Parent\nclass Child(Parent): pass",
    "import package.base as other\nclass Child(other.Base): pass",
    "import package.base\nclass Child(package.base.Base): pass",
    "from . import base\nclass Child(base.Base): pass",
])
def test_cross_module_inheritance_binds_both_original_sources(repo, base_import):
    write(repo, "package/__init__.py", "")
    write(repo, "package/base.py", "class Base:\n    def value(self): return 42\n")
    write(repo, "package/child.py", base_import)
    _, result, row = run(
        repo, "package.child:Child.value", "self.assertEqual(Child().value(), 42)",
        "from package.child import Child",
    )
    assert result["status"] == "passed" and result["execution_ready"], result
    assert row["resolved_api_entry"] == "package.base:Base.value"
    for name in ("package/base.py", "package/child.py"):
        assert row["source_inventory"][name] == hashlib.sha256((repo / name).read_bytes()).hexdigest()


def test_diamond_uses_python_c3_order_instead_of_depth_first_search(repo):
    write(repo, "sample.py", """
        class Root:
            def value(self): return 0
        class Left(Root): pass
        class Right(Root):
            def value(self): return 42
        class Child(Left, Right): pass
    """)
    _, result, row = run(repo, "sample:Child.value", "self.assertEqual(sample.Child().value(), 42)")
    assert result["status"] == "passed", result
    assert row["resolved_api_entry"] == "sample:Right.value"


def test_overridden_method_cannot_be_satisfied_by_calling_the_base(repo):
    write(repo, "sample.py", """
        class Base:
            def value(self): return 42
        class Child(Base):
            def value(self): return 0
    """)
    _, result, row = run(repo, "sample:Child.value", "self.assertEqual(sample.Base().value(), 42)")
    assert not result["execution_ready"] and result["status"] != "passed", result
    assert row["calls"] == {"Check.test_value": 0}


def test_assignment_shadowing_an_inherited_method_is_not_skipped(repo):
    write(repo, "sample.py", """
        class Base:
            def value(self): return 42
        class Child(Base):
            value = None
    """)
    _, result, row = run(repo, "sample:Child.value", "self.assertEqual(sample.Base().value(), 42)")
    assert not result["execution_ready"], result
    assert row["error_kind"] == "declaration"
    assert "not consumed" not in row["reason"]


@pytest.mark.parametrize("entry,source,diagnostic", [
    ("sample:Missing.value", "def value(): return 42", "symbol"),
    ("sample:value", "async def value(): return 42", "synchronous"),
])
def test_source_resolution_error_survives_finish(repo, entry, source, diagnostic):
    write(repo, "sample.py", source)
    _, result, row = run(repo, entry, "self.assertTrue(True)")
    assert not result["execution_ready"], result
    assert row["error_kind"] == "declaration"
    assert diagnostic in row["reason"] and "not consumed" not in row["reason"]


@pytest.mark.parametrize("body", [
    '"Interface only."', "pass", "...", "return None", "raise NotImplementedError",
    "raise NotImplementedError('implement in a subclass')",
])
def test_abstract_placeholder_cannot_supply_consumption_evidence(repo, body):
    write(repo, "sample.py", "from abc import ABC, abstractmethod as contract\n"
          "class Base(ABC):\n    @contract\n    def value(self):\n        " + body + "\n")
    statement = "try:\n    sample.Base.value(None)\nexcept NotImplementedError:\n    pass"
    _, result, row = run(repo, "sample:Base.value", statement)
    assert not result["execution_ready"] and result["status"] != "passed", result
    assert row["error_kind"] == "declaration" and "abstract" in row["reason"].lower()


def test_abstract_method_with_real_implementation_is_still_observable(repo):
    write(repo, "sample.py", """
        import abc
        class Base(abc.ABC):
            @abc.abstractmethod
            def value(self): return 41 + self.extra
        class Child(Base):
            extra = 1
            def value(self): return super().value()
    """)
    _, result, row = run(repo, "sample:Base.value", "self.assertEqual(sample.Child().value(), 42)")
    assert result["status"] == "passed", result
    assert row["calls"] == {"Check.test_value": 1}


def test_concrete_noop_is_not_rejected_as_an_abstract_placeholder(repo):
    write(repo, "sample.py", "class Hook:\n    def value(self): pass\n")
    _, result, _ = run(repo, "sample:Hook.value", "self.assertIsNone(sample.Hook().value())")
    assert result["status"] == "passed", result


def test_inheritance_through_standard_abc_base_is_observable(repo):
    write(repo, "sample.py", """
        from abc import ABC
        class Base(ABC):
            def value(self): return 42
        class Child(Base): pass
    """)
    _, result, _ = run(repo, "sample:Child.value", "self.assertEqual(sample.Child().value(), 42)")
    assert result["status"] == "passed", result


def test_modified_inheritance_source_cannot_keep_a_valid_receipt(repo):
    write(repo, "base.py", "class Base:\n    def value(self): return 42\n")
    write(repo, "sample.py", "from base import Base\nclass Child(Base): pass\n")
    statement = "self.assertEqual(sample.Child().value(), 42)\n"
    statement += "from pathlib import Path\nPath('sample.py').write_text('class Child: pass')"
    _, result, row = run(repo, "sample:Child.value", statement)
    assert not result["execution_ready"], result
    assert row["error_kind"] == "integrity"


def test_forged_inherited_code_cannot_replace_original_implementation(repo):
    write(repo, "sample.py", "class Base:\n    def value(self): return 0\nclass Child(Base): pass\n")
    statement = "replacement = {}\n"
    statement += "exec(compile('class Base:\\n    def value(self): return 42\\n', sample.__file__, 'exec'), replacement)\n"
    statement += "sample.Child.value = replacement['Base'].value\nself.assertEqual(sample.Child().value(), 42)"
    _, result, row = run(repo, "sample:Child.value", statement)
    assert not result["execution_ready"], result
    assert row["calls"] == {"Check.test_value": 0}


@pytest.mark.parametrize("field,value", [
    ("resolved_api_entry", "sample:Other.value"), ("source_inventory", {}),
    ("source_line", 999),
])
def test_host_rechecks_resolution_instead_of_trusting_receipt(repo, field, value):
    write(repo, "sample.py", "def value(): return 42\n")
    runner, result, row = run(repo, "sample:value", "self.assertEqual(sample.value(), 42)")
    assert result["status"] == "passed", result
    receipt_path = runner.artifact_dir / "candidate.execution.json"
    receipt = json.loads(receipt_path.read_text("utf-8"))
    report_path = runner.artifact_dir / "candidate.tests.json"
    report = json.loads(report_path.read_text("utf-8"))
    row[field] = value
    receipt["checks"] = [row]
    report["consumption"]["checks"] = [row]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    receipt["test_report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="API source"):
        validate_execution_result(
            receipt_path, runner.spec.execution_checks, repository=repo, map_path=str,
        )


def test_api_resolver_is_part_of_the_recovery_identity(monkeypatch):
    from agent.settings import PROJECT_ROOT
    from agent.storage.recovery import code_key

    before = code_key()
    target = PROJECT_ROOT / "resources/drivers/api_source.py"
    original = Path.read_bytes

    def changed(path):
        raw = original(path)
        return raw + b"\n" if path == target else raw

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert code_key() != before


def test_resolution_does_not_import_or_execute_repository_code(repo):
    from resources.drivers.api_source import resolve_api

    write(repo, "sample.py", """
        from pathlib import Path
        Path(__file__).with_name('must-not-run.txt').write_text('unexpected execution')
        class Base:
            def value(self): return 42
        class Child(Base): pass
        raise RuntimeError('repository imported during source inspection')
    """)
    evidence, _ = resolve_api(repo, "sample:Child.value")
    assert evidence["resolved_api_entry"] == "sample:Base.value"
    assert not (repo / "must-not-run.txt").exists()


@pytest.mark.parametrize("bases", [
    "factory()", "Base[int]", "External",
])
def test_dynamic_or_external_inheritance_has_explicit_diagnostic(repo, bases):
    from resources.drivers.api_source import resolve_api

    write(repo, "sample.py", "from collections import UserDict as External\n"
          "class Base:\n    def value(self): return 42\n"
          "def factory(): return Base\nclass Child(" + bases + "): pass\n")
    with pytest.raises(ValueError, match="static|repository"):
        resolve_api(repo, "sample:Child.value")


def test_classmethod_and_staticmethod_keep_inherited_calls(repo):
    write(repo, "sample.py", """
        class Base:
            @classmethod
            def value(cls): return cls.number()
            @staticmethod
            def number(): return 42
        class Child(Base): pass
    """)
    _, result, row = run(repo, "sample:Child.value", "self.assertEqual(sample.Child.value(), 42)")
    assert result["status"] == "passed", result
    assert row["calls"] == {"Check.test_value": 1}


def test_fixed_source_resolver_cannot_be_shadowed_by_project(repo, monkeypatch):
    monkeypatch.setenv("PYTHONSAFEPATH", "1")
    write(repo, "sample.py", "def value(): return 42\n")
    write(repo, "api_source.py", "raise AssertionError('untrusted helper loaded')\n")
    _, result, _ = run(repo, "sample:value", "self.assertEqual(sample.value(), 42)")
    assert result["status"] == "passed", result


def test_inherited_api_still_requires_every_bound_test_to_call_it(repo):
    write(repo, "sample.py", "class Base:\n    def value(self): return 42\nclass Child(Base): pass\n")
    _, result, row = run(repo, "sample:Child.value", "self.assertEqual(sample.Child().value(), 42)",
                         check_ids=["Check.test_value", "Check.test_missing"])
    assert not result["execution_ready"], result
    assert row["calls"] == {"Check.test_value": 1, "Check.test_missing": 0}


def test_invalid_syntax_remains_a_source_failure_not_a_binding_error(repo):
    write(repo, "sample.py", "def value(:\n    return 42\n")
    _, result, row = run(repo, "sample:value", "self.assertTrue(True)", imports="")
    assert result["status"] == "failed", result
    assert row["error_kind"] == "source" and "syntax" in row["reason"]
    assert row["source_sha256"] == hashlib.sha256((repo / "sample.py").read_bytes()).hexdigest()
