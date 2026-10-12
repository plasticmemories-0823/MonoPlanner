"""Startup evidence does not execute code, change acceptance, or invent visible source."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage, message_to_dict
from agent.execution.executor import OperationExecutor
from agent.source.paging import visible_reads
from agent.source.work_materials import work_materials, supplied_locations, MAX_CHARS, MAX_LINES
from agent.source.snapshots import SourceSnapshot
from agent.source.versions import SourceVersions
from agent.execution.local import LocalEnvironment
from agent.task.runner import TaskRunner
from agent.storage.recovery import code_key
from agent.settings import PROJECT_ROOT


class WorkMaterialsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.write('sample/actions.py', 'class Actions:\n    def run(self):\n        return self.allowed()\n\n'
                   '    def allowed(self):\n        return True\n')
        self.write('tests/runtests.py', 'import os\nos.environ.setdefault("APP_SETTINGS_MODULE", "test_local")\n'
                   'raise RuntimeError("DO NOT EXECUTE STARTUP EVIDENCE")\n')
        self.write('tests/test_local.py', 'DATABASE = "native in-memory fixture"\n')
        self.write('tests/run/tests.py', 'from .models import Existing\nclass RunTests:\n    pass\n')
        self.write('tests/run/models.py', 'class Existing:\n    value = "原文\\\\路径和引号"\n')
        self.write('tests/model_fields/tests.py', '# UNRELATED TEST INFRASTRUCTURE\n')

    def write(self, name, value):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8')

    def executor(self):
        environment = LocalEnvironment(self.repo, artifact_root=self.root / 'execution')
        baseline = SourceSnapshot.capture(self.repo, self.root / 'baseline')
        evidence = SourceVersions(environment, self.root / 'records', baseline=baseline)
        return OperationExecutor(environment, evidence, self.root / 'tasks', versioned_reads=True)

    def materials(self, executor=None):
        return work_materials(executor or self.executor(), 'Fix run() for models; see Sample.actions:3-3.')

    def test_exact_reference_native_conditions_and_static_neighbors_arrive_without_execution(self):
        executor = self.executor()
        with patch.object(executor.environment, 'run_process', side_effect=AssertionError('no program execution')):
            data = self.materials(executor)
        self.assertEqual(data['source_locations'][0]['path'], 'sample/actions.py')
        self.assertEqual(data['test_suite_candidates'], ['tests/run'])
        self.assertEqual(data['test_entrypoints'], ['tests/runtests.py'])
        by_name = {p['path']: p for p in data['reads']}
        self.assertIn('tests/test_local.py', by_name)
        self.assertIn('tests/run/models.py', by_name)
        self.assertNotIn('tests/model_fields/tests.py', by_name)
        self.assertIn('def allowed', by_name['sample/actions.py']['text'])
        raw = executor.source_bytes('tests/run/models.py')
        self.assertEqual(by_name['tests/run/models.py']['sha256'], hashlib.sha256(raw).hexdigest())
        self.assertIn(raw.decode().splitlines()[1], by_name['tests/run/models.py']['text'])

    def test_unknown_absolute_traversal_and_case_ambiguous_paths_are_not_guessed(self):
        names = {'sample/actions.py', 'case.py', 'Case.py'}
        for request in ('../sample/actions.py:3', 'C:\\sample\\actions.py:3', '/sample/actions.py:3',
                        'unknown.actions:3', 'CASE.py:2'):
            with self.subTest(request=request):
                self.assertEqual(supplied_locations(request, names), [])

    def test_nearest_package_tests_work_without_a_repository_specific_rule(self):
        self.write('widget/printing/cprint.py', 'def render():\n    return 1\n')
        self.write('widget/printing/tests/conftest.py', 'NATIVE_FIXTURE = True\n')
        data = work_materials(self.executor(), 'Fix widget/printing/cprint.py:2.')
        self.assertEqual(data['test_suite_candidates'], ['widget/printing/tests'])
        self.assertTrue(any(p['path'] == 'widget/printing/tests/conftest.py' for p in data['reads']))

    def test_only_literal_settings_defaults_are_followed(self):
        self.write('tests/runtests.py', 'import unrelated\nsettings = compute_settings()\n')
        self.write('tests/unrelated.py', '# DO NOT GUESS SETTINGS FROM IMPORTS\n')
        data = self.materials()
        self.assertFalse(any('test_local.py' in p['path'] or 'unrelated.py' in p['path'] for p in data['reads']))

    def test_native_loader_middle_is_complete_and_no_project_code_is_executed(self):
        head = '# header\n' * 100
        setup = ('def prepare(labels):\n    settings.INSTALLED_APPS = []\n'
                 '    for label in labels:\n        settings.INSTALLED_APPS.append(label)\n'
                 '    setup()\n    return "FIXTURE_LOADING_COMPLETE"\n')
        self.write('tests/runtests.py', head + setup + '# unrelated runner body\n' * 300)
        executor = self.executor()
        with patch.object(executor.environment, 'run_process', side_effect=AssertionError('no execution')):
            data = self.materials(executor)
        pages = [p for p in data['reads'] if p['path'] == 'tests/runtests.py']
        originals = ['\n'.join(line.split(': ', 1)[1] for line in p['text'].splitlines()) for p in pages]
        self.assertTrue(any(setup.rstrip() in text for text in originals))
        self.assertFalse(any('unrelated runner body\n' * 100 in p['text'] for p in pages))
        self.assertLessEqual(sum(len(p['text']) for p in data['reads']), MAX_CHARS)
        self.assertLessEqual(sum(b-a for p in data['reads'] for a,b in p['visible_ranges']), MAX_LINES)

    def test_large_material_is_bounded_and_not_recursively_expanded(self):
        self.write('tests/run/models.py', 'class Many:\n' + '    value = "' + 'x' * 20000 + '"\n')
        data = self.materials()
        self.assertLessEqual(sum(len(p['text']) for p in data['reads']), MAX_CHARS)
        self.assertLessEqual(sum(b - a for p in data['reads'] for a, b in p['visible_ranges']), MAX_LINES)
        self.assertFalse(any(p['path'] == 'tests/model_fields/tests.py' for p in data['reads']))

    def test_initial_text_is_visible_but_removed_text_and_changed_content_are_not(self):
        executor = self.executor()
        data = self.materials(executor)
        initial = message_to_dict(HumanMessage(content=json.dumps({'initial_evidence': {'work_materials': data}})))
        seen = visible_reads([initial], [])
        sha = executor.inventory()['sample/actions.py']
        self.assertTrue(seen[('sample/actions.py', sha)])
        executor.visible_reads = seen
        result = executor.read_continuation('sample/actions.py', 1, 2)
        self.assertTrue(result['already_visible'])
        self.assertEqual(visible_reads([message_to_dict(HumanMessage(content='summary only'))], []), {})
        executor.visible_reads = {('sample/actions.py', 'stale-sha'): [[0, 100]]}
        reread = executor.read_continuation('sample/actions.py', 1, 2)
        self.assertIn('def run', reread['text'])

    def test_unavailable_native_material_keeps_existing_fallback_available(self):
        for name in ('tests/runtests.py', 'tests/test_local.py', 'tests/run/models.py', 'tests/run/tests.py'):
            (self.repo / name).unlink()
        data = self.materials()
        self.assertEqual(data['test_entrypoints'], [])
        self.assertEqual(data['test_suite_candidates'], [])
        self.assertTrue(any(p['path'] == 'sample/actions.py' for p in data['reads']))

    def test_cli_initial_material_has_same_evidence_without_changing_state(self):
        executor = self.executor()
        runner = object.__new__(TaskRunner)
        runner.executor, runner.environment, runner.evidence = executor, executor.environment, executor.evidence
        runner.state = {'request': 'Fix run(); see Sample.actions:3.', 'verification': {'status': 'failed'}}
        before = json.dumps(runner.state, sort_keys=True)
        data = runner.initial_material()
        self.assertEqual(data['work_materials']['test_suite_candidates'], ['tests/run'])
        self.assertEqual(json.dumps(runner.state, sort_keys=True), before)

    def test_material_selection_is_part_of_recovery_fingerprint(self):
        module = Path('agent/source/work_materials.py')
        alternate = self.root / module
        alternate.parent.mkdir(parents=True)
        alternate.write_bytes((PROJECT_ROOT / module).read_bytes() + b'\n')
        class Overlay:
            def __truediv__(self, name):
                return alternate if str(name) == module.as_posix() else PROJECT_ROOT / name
        original = code_key()
        with patch('agent.storage.recovery.PROJECT_ROOT', Overlay()):
            self.assertNotEqual(code_key(), original)


if __name__ == '__main__':
    unittest.main()
