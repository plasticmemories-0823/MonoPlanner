"""通用库接口和文件产物必须留下与冻结测试关联的实际消费证据。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.acceptance.contracts import VerificationSpec
from agent.acceptance.runner import AcceptanceRun
from agent.execution.local import LocalEnvironment
from agent.model.errors import ModelRequestNotSent
from agent.task.requirements import checked, confirmed, check_execution


def api_check(**changes):
    return {"id": "library", "artifact_type": "python_api", "consumer": "python_tests",
            "interpreter": "python", "api_entry": "sample:value",
            "test_ids": ["Check.test_value"], **changes}


def artifact_check(**changes):
    return {"id": "artifact", "artifact_type": "file_artifact", "consumer": "python_tests",
            "interpreter": "python", "entry_file": "output/result.json",
            "test_ids": ["Check.test_value"], **changes}


API_TEST = """import unittest
from sample import value
class Check(unittest.TestCase):
    def test_value(self):
        self.assertEqual(value(), 42)
"""

ARTIFACT_TEST = """import json, unittest
from acceptance_artifacts import read_artifact
class Check(unittest.TestCase):
    def test_value(self):
        data = json.loads(read_artifact('output/result.json'))
        self.assertEqual(data['value'], 42)
"""


@pytest.fixture
def scene(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "sample.py").write_text("def value():\n    return 42\n", encoding="utf-8")
    return repo


def run(scene, code=API_TEST, checks=None):
    spec = VerificationSpec(("Deliver the original public behavior",), test_code=code,
                            execution_checks=checks or [api_check()])
    env = LocalEnvironment(scene, artifact_root=scene.parent / "runtime")
    runner = AcceptanceRun(scene, spec, environment=env,
                           artifacts_root=scene.parent / "acceptance")
    return runner, runner.run("candidate")


def test_real_api_call_and_assertion_are_used_without_a_second_execution(scene):
    (scene / "sample.py").write_text(
        "from pathlib import Path\ndef value():\n"
        "    p=Path('calls.txt')\n"
        "    p.write_text(p.read_text()+'x' if p.exists() else 'x')\n"
        "    return 42\n", encoding="utf-8")
    runner, result = run(scene)
    assert result["status"] == "passed", result
    assert (scene / "calls.txt").read_text() == "x"
    row = result["execution"]["checks"][0]
    assert row["api_entry"] == "sample:value"
    assert row["test_ids"] == ["Check.test_value"]
    assert row["calls"]["Check.test_value"] == 1
    assert len(row["source_sha256"]) == 64
    with pytest.raises(ModelRequestNotSent):
        runner.run("candidate")
    assert (scene / "calls.txt").read_text() == "x"


def test_fixed_consumption_helper_loads_with_safe_path_and_ignores_project_shadow(scene, monkeypatch):
    monkeypatch.setenv('PYTHONSAFEPATH', '1')
    (scene / 'acceptance_consumption.py').write_text(
        "raise AssertionError('Project module must not replace the trusted helper')\n", encoding='utf-8')
    _, result = run(scene)
    assert result['status'] == 'passed' and result['execution_ready'], result


def test_called_but_wrong_library_result_is_a_real_failure(scene):
    (scene / "sample.py").write_text("def value():\n    return 0\n", encoding="utf-8")
    _, result = run(scene)
    assert result["status"] == "failed" and result["tests_ready"], result


def test_exception_inside_original_api_remains_source_failure(scene):
    (scene / "sample.py").write_text(
        "def value():\n    raise ValueError('source failure')\n", encoding="utf-8")
    _, result = run(scene)
    assert result["status"] == "failed" and result["tests_ready"], result


def test_forged_code_filename_cannot_replace_the_original_implementation(scene):
    (scene / "sample.py").write_text("def value():\n    return 0\n", encoding="utf-8")
    code = """import unittest, sample
class Check(unittest.TestCase):
    def test_value(self):
        replacement = {}
        exec(compile('def value():\\n    return 42\\n', sample.__file__, 'exec'), replacement)
        sample.value = replacement['value']
        self.assertEqual(sample.value(), 42)
"""
    _, result = run(scene, code)
    assert result['status'] != 'passed' and not result['execution_ready'], result


def test_frozen_api_source_cannot_be_rewritten_during_testing(scene):
    code = API_TEST.replace("self.assertEqual(value(), 42)",
        "self.assertEqual(value(), 42)\n"
        "        from pathlib import Path\n"
        "        Path('sample.py').write_text('def value(): return 0')")
    _, result = run(scene, code)
    assert result['status'] != 'passed' and not result['execution_ready'], result


@pytest.mark.parametrize("body", [
    "self.assertTrue(callable(sample.value))",
    "import sample; sample.value = lambda: 42; self.assertEqual(sample.value(), 42)",
])
def test_import_only_or_replacement_function_cannot_supply_api_evidence(scene, body):
    code = "import unittest\nimport sample\nclass Check(unittest.TestCase):\n"
    code += "    def test_value(self):\n        " + body + "\n"
    _, result = run(scene, code)
    assert result["status"] != "passed" and not result["execution_ready"], result


def test_stateful_public_method_runs_in_the_existing_test_sequence(scene):
    (scene / "sample.py").write_text(
        "class Counter:\n    def __init__(self): self.total=0\n"
        "    def add(self, n):\n        self.total += n\n        return self.total\n",
        encoding="utf-8")
    code = """import unittest
from sample import Counter
class Check(unittest.TestCase):
    def test_value(self):
        counter = Counter()
        self.assertEqual(counter.add(2), 2)
        self.assertEqual(counter.add(3), 5)
"""
    _, result = run(scene, code, [api_check(api_entry="sample:Counter.add")])
    assert result["status"] == "passed", result
    assert result["execution"]["checks"][0]["calls"]["Check.test_value"] == 2


def test_api_and_command_checks_keep_their_own_original_receipts(scene):
    (scene / 'cli.py').write_text('print(42)\n', encoding='utf-8')
    command = {'id': 'cli', 'artifact_type': 'repository_entry', 'consumer': 'command',
               'entry_file': 'cli.py', 'interpreter': 'python', 'argv': []}
    runner, result = run(scene, API_TEST, [api_check(), command])
    assert result['status'] == 'passed', result
    assert (runner.artifact_dir / 'candidate.command-execution.json').is_file()
    assert [r['id'] for r in result['execution']['checks']] == ['library', 'cli']


def producer_spec(scene, value=42):
    (scene / 'producer.py').write_text(
        "import json\nfrom pathlib import Path\n"
        "p=Path('output/result.json')\np.parent.mkdir(exist_ok=True)\n"
        "old=json.loads(p.read_text()) if p.exists() else {'build':0}\n"
        f"p.write_text(json.dumps({{'value':{value},'build':old['build']+1}}))\n", encoding='utf-8')
    checks = [artifact_check(), {'id': 'producer', 'artifact_type': 'repository_entry',
              'consumer': 'command', 'entry_file': 'producer.py', 'interpreter': 'python', 'argv': []}]
    return VerificationSpec(('The output value must be 42; build metadata may change.',),
                            test_code=ARTIFACT_TEST, execution_checks=checks,
                            output_paths=('output/result.json',))


def test_file_consumer_preserves_delivered_bytes_when_command_rebuilds_output(scene):
    spec = producer_spec(scene)
    (scene / 'output').mkdir()
    (scene / 'output/result.json').write_text('{"value":42,"build":0}', encoding='utf-8')
    runner = AcceptanceRun(scene, spec, environment=LocalEnvironment(scene, artifact_root=scene.parent / 'runtime'),
                           artifacts_root=scene.parent / 'acceptance')
    result = runner.run('candidate')
    assert result['status'] == 'passed' and result['execution_ready'], result
    assert json.loads((scene / 'output/result.json').read_text()) == {'value': 42, 'build': 1}
    retained = runner.artifact_dir / 'candidate.consumed-files/output/result.json'
    assert json.loads(retained.read_text()) == {'value': 42, 'build': 0}
    with pytest.raises(ModelRequestNotSent):
        runner.run('candidate')
    assert json.loads((scene / 'output/result.json').read_text())['build'] == 1


def test_saved_producer_receipt_stops_before_any_test_side_effect(scene):
    spec = producer_spec(scene)
    runner = AcceptanceRun(scene, spec, environment=LocalEnvironment(scene, artifact_root=scene.parent / 'runtime'),
                           artifacts_root=scene.parent / 'acceptance')
    saved = runner.artifact_dir / 'candidate.command-execution.json'
    saved.write_text('{"preserved": true}', encoding='utf-8')
    with pytest.raises(ModelRequestNotSent):
        runner.run('candidate')
    assert not (scene / 'output/result.json').exists()
    assert not (runner.artifact_dir / 'candidate.tests.json').exists()
    assert json.loads(saved.read_text()) == {'preserved': True}


def test_following_command_cannot_replace_preserved_artifact_evidence(scene):
    spec = producer_spec(scene)
    (scene / 'output').mkdir()
    (scene / 'output/result.json').write_text('{"value":42,"build":0}', encoding='utf-8')
    runner = AcceptanceRun(scene, spec, environment=LocalEnvironment(scene, artifact_root=scene.parent / 'runtime'),
                           artifacts_root=scene.parent / 'acceptance')
    target = runner.artifact_dir / 'candidate.consumed-files/output/result.json'
    (scene / 'producer.py').write_text(
        'from pathlib import Path\nPath(' + repr(str(target)) + ").write_text('forged')\n", encoding='utf-8')
    with pytest.raises(ModelRequestNotSent):
        runner.run('candidate')
    assert json.loads((scene / 'output/result.json').read_text())['value'] == 42


def test_original_trial_produces_outputs_in_copy_and_reports_real_failure(scene):
    from agent.acceptance.revisions import run_report
    from agent.source.snapshots import SourceSnapshot
    from agent.task.budget import StepBudget
    spec = producer_spec(scene, value=0)
    saved = SourceSnapshot.capture(scene, scene.parent / 'original')
    loop = SimpleNamespace(environment=LocalEnvironment(scene, artifact_root=scene.parent / 'runtime'),
                           budget=StepBudget(timeout_seconds=120))
    result = run_report(loop, saved, spec, scene.parent / 'trial', 'original')
    assert result['status'] == 'failed' and result['tests_ready'], result
    assert result['source_integrity']['valid'] and result['snapshot_integrity']['valid']
    assert saved.verify_integrity()['valid'] and not (saved.root / 'output/result.json').exists()


def test_original_trial_rejects_source_edits_in_its_writable_copy(scene):
    from agent.acceptance.revisions import run_report
    from agent.source.snapshots import SourceSnapshot
    from agent.task.budget import StepBudget
    code = API_TEST + "        from pathlib import Path\n        Path('sample.py').write_text('changed')\n"
    spec = VerificationSpec(('Keep the original source intact during acceptance.',), test_code=code)
    saved = SourceSnapshot.capture(scene, scene.parent / 'original')
    loop = SimpleNamespace(environment=LocalEnvironment(scene, artifact_root=scene.parent / 'runtime'),
                           budget=StepBudget(timeout_seconds=120))
    result = run_report(loop, saved, spec, scene.parent / 'trial', 'original')
    assert result['status'] == 'error' and not result['integrity_ok'], result
    assert saved.verify_integrity()['valid']


def test_file_created_only_in_acceptance_copy_is_not_claimed_as_delivered(scene):
    from agent.acceptance.final import verify_final
    from agent.source.snapshots import SourceSnapshot
    spec = producer_spec(scene)
    saved = SourceSnapshot.capture(scene, scene.parent / 'candidate')
    environment = LocalEnvironment(scene, artifact_root=scene.parent / 'runtime')
    result = verify_final(environment, saved, spec, '', scene.parent / 'final', 120)
    assert result['status'] == 'failed' and result['tests_ready'], result
    assert result['source_integrity']['valid'] and result['snapshot_integrity']['valid']
    assert not (scene / 'output/result.json').exists()


def test_file_artifact_is_read_as_original_bytes_and_validated(scene):
    (scene / "output").mkdir()
    (scene / "output/result.json").write_text('{"value":42}', encoding="utf-8")
    _, result = run(scene, ARTIFACT_TEST, [artifact_check()])
    assert result["status"] == "passed", result
    row = result["execution"]["checks"][0]
    assert row["artifact_path"] == "output/result.json"
    assert row["bytes_read"] == len(b'{"value":42}')
    assert len(row["artifact_sha256"]) == 64


def test_file_change_after_read_is_not_accepted(scene):
    (scene / "output").mkdir()
    (scene / "output/result.json").write_text('{"value":42}', encoding="utf-8")
    code = ARTIFACT_TEST + "        from pathlib import Path\n"
    code += "        Path('output/result.json').write_text('{}')\n"
    _, result = run(scene, code, [artifact_check()])
    assert result['status'] != 'passed' and not result['execution_ready'], result


def test_missing_required_artifact_is_original_failure_not_environment_failure(scene):
    _, result = run(scene, ARTIFACT_TEST, [artifact_check()])
    assert result["status"] == "failed" and result["tests_ready"], result
    assert not (scene / "output/result.json").exists()


def test_replacement_file_cannot_satisfy_confirmed_artifact(scene):
    (scene / "output").mkdir()
    (scene / "output/result.json").write_text('{"value":0}', encoding="utf-8")
    (scene / "other.json").write_text('{"value":42}', encoding="utf-8")
    _, result = run(scene, ARTIFACT_TEST.replace("output/result.json", "other.json"),
                    [artifact_check()])
    assert result["status"] != "passed" and not result["execution_ready"], result


def test_test_created_artifact_does_not_masquerade_as_delivered_output(scene):
    code = ARTIFACT_TEST.replace(
        "data = json.loads", "from pathlib import Path\n"
        "        Path('output').mkdir()\n"
        "        Path('output/result.json').write_text('{\"value\":42}')\n"
        "        data = json.loads")
    _, result = run(scene, code, [artifact_check()])
    assert result["status"] != "passed" and not result["execution_ready"], result


def test_existing_test_receipt_is_preserved_instead_of_replaying_after_gap(scene):
    spec = VerificationSpec(("Original behavior",), test_code=API_TEST,
                            execution_checks=[api_check()])
    env = LocalEnvironment(scene, artifact_root=scene.parent / "runtime")
    runner = AcceptanceRun(scene, spec, environment=env,
                           artifacts_root=scene.parent / "acceptance")
    saved = runner.artifact_dir / "candidate.tests.json"
    saved.write_text('{"preserved": true}', encoding="utf-8")
    with pytest.raises(ModelRequestNotSent):
        runner.run("candidate")
    assert json.loads(saved.read_text()) == {"preserved": True}


@pytest.mark.parametrize("kind,entry,check", [
    ("python_api", "sample:value", api_check()),
    ("file_artifact", "output/result.json", artifact_check()),
])
def test_requirement_binding_cannot_be_replaced(kind, entry, check):
    target = {"id": "delivery", "artifact_type": kind, "target_runtime": "python",
              "entry": entry, "behavior": "Original behavior", "input_scope": "Original inputs",
              "quotes": [{"source_id": "task_request", "text": "Original behavior"}]}
    packet = confirmed("Original behavior", [target],
                       confirmation={"id": "human", "text": "Continue"})
    assert checked(packet) == packet
    owner = SimpleNamespace(state={"requirements": packet})
    assert check_execution(owner, [check]) == packet["digest"]
    changed = {**check, ("api_entry" if kind == "python_api" else "entry_file"): "wrong"}
    with pytest.raises(ValueError):
        check_execution(owner, [changed])


def test_requested_library_scope_does_not_invent_a_fixed_api_or_allow_another_package():
    target = {'id': 'library', 'artifact_type': 'python_api', 'target_runtime': 'python',
              'entry': 'sample:*', 'behavior': 'Repair the sample library',
              'input_scope': 'Complete original behavior',
              'quotes': [{'source_id': 'task_request', 'text': 'Repair the sample library'}]}
    packet = confirmed('Repair the sample library', [target],
                       confirmation={'id': 'human', 'text': 'Continue'})
    loop = SimpleNamespace(state={'requirements': packet})
    for entry in ['sample:value', 'sample.public:Reader.read']:
        assert check_execution(loop, [api_check(api_entry=entry)]) == packet['digest']
    with pytest.raises(ValueError):
        check_execution(loop, [api_check(api_entry='sample_copy:value')])
