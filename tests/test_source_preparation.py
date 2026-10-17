"""验证版本匹配的扫描复用和有界准备，不发起模型服务请求。"""

import copy
import json
import os
import re
import shutil
import threading
import time
import unittest
from unittest.mock import patch

from agent.task.budget import StepBudget
from agent.model.payload import wire_input
from agent.source.snapshots import SourceSnapshot
from agent.source.view import SourceView
from agent.source.versions import SourceVersions
from agent.source.index import RepositoryNavigator
from agent.execution.executor import OperationExecutor
from agent.source.preparation import Preparation, build_preparation
from agent.task.runner import TaskRunner
from agent.task.contracts import Operation
from agent.model.tools import submit_plan
import test_task_runner as f


class SourceTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp
    candidate = f.TaskRunnerTests.candidate
    shell = f.TaskRunnerTests.shell

    def evidence(self):
        base = SourceSnapshot.capture(self.repo, self.root / 'baseline')
        return SourceVersions(self.env, self.root / 'evidence', baseline=base)

    def test_reads_and_searches_use_old_bytes_even_when_live_files_change(self):
        (self.repo / 'data.json').write_text('{"value": 0}')
        ev = self.evidence()
        reader = OperationExecutor(self.env, ev, self.root / 'records', versioned_reads=True)
        old = (self.repo / 'sample.py').stat()
        (self.repo / 'sample.py').write_text('def value():\n    return 9\n')
        os.utime(self.repo / 'sample.py', ns=(old.st_atime_ns, old.st_mtime_ns))
        (self.repo / 'data.json').unlink()
        (self.repo / 'new.py').write_text('NEW_SEARCH_HIT = 1')
        self.assertIn('return 0', reader.read('sample.py', 0, 80)['text'])
        self.assertIn('value', reader.read('data.json', 0, 80)['text'])
        self.assertEqual(
            reader.search(Operation(id='s', op='search', query='NEW_SEARCH_HIT'))['hits'], []
        )
        self.assertEqual(
            reader.search(Operation(id='s', op='search', query='return 0'))['hits'][0]['path'],
            'sample.py',
        )

    def test_missing_or_corrupt_record_never_reads_live_source(self):
        ev = self.evidence()
        (ev.baseline.root / 'sample.py').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'Recorded source bytes changed'):
            ev.view().read_bytes('sample.py')
        (ev.baseline.root / 'sample.py').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            SourceView(ev).read_bytes('sample.py')

    def test_manifest_tampering_and_path_escape_are_rejected(self):
        ev = self.evidence()
        with self.assertRaises(ValueError):
            ev.view().read_bytes('../private')
        path = ev.root / 'states' / (ev.current + '.json')
        data = json.loads(path.read_text())
        data['files']['sample.py'] = '0' * 64
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'manifest fingerprint'):
            SourceView(ev)

    def test_export_uses_archived_version_and_preserves_source_integrity(self):
        ev = self.evidence()
        (self.repo / 'sample.py').write_text('def changed(): pass')
        snapshot = ev.view().export(self.root / 'export')
        self.assertIn('return 0', (snapshot.root / 'sample.py').read_text())
        self.assertTrue(snapshot.verify_integrity()['valid'])

    def test_live_change_during_model_request_blocks_planned_write(self):
        repo = self.repo

        class Model(f.Model):
            def invoke(self, messages):
                reply = super().invoke(messages)
                if len(self.calls) == 2:
                    (repo / 'sample.py').write_text('def value():\n    return 9\n')
                return reply

        scan = {
            'kind': 'inspect',
            'note': 'read',
            'steps': [{'id': 'r', 'op': 'read', 'path': 'sample.py'}],
        }
        model = Model([scan, self.candidate()])
        result = TaskRunner(model, self.env, state_dir=self.root / 'state').run(
            [], 'return 4', StepBudget(max_steps=5)
        )
        self.assertEqual(result['status'], 'state_conflict', result)
        self.assertIn('return 9', (repo / 'sample.py').read_text())
        self.assertEqual(result['rounds'], [])

    def test_read_events_do_not_rescan_and_writes_keep_strong_boundaries(self):
        steps = [{'id': 'r' + str(i), 'op': 'read', 'path': 'sample.py'} for i in range(6)]
        model = f.Model([{'kind': 'inspect', 'note': 'read', 'steps': steps}, self.candidate()])
        result = TaskRunner(model, self.env, state_dir=self.root / 'state').run(
            [], 'return 4', StepBudget(max_steps=5)
        )
        self.assertEqual(result['status'], 'verified_success', result)
        events = [
            json.loads(l)
            for l in (self.root / 'state/source-records/events.jsonl').read_text().splitlines()
        ]
        self.assertEqual(sum(e['scan_kind'] == 'reference' for e in events), 12)
        self.assertEqual(sum(e['scan_kind'] == 'full' for e in events), 4)
        self.assertFalse(any(t.name.startswith('source-prepare') for t in threading.enumerate()))

    def test_prepared_map_never_reopens_mutable_workspace(self):
        (self.repo / 'second.py').write_text('def stable(): return 1\n')
        ev = self.evidence()
        prepared = build_preparation(ev.view(), {}, threading.Event())
        nav = RepositoryNavigator.from_prepared(ev.view(), prepared, self.root / 'map')
        (self.repo / 'sample.py').write_text('def replacement(): pass')
        with patch('agent.source.index.os.walk', side_effect=AssertionError('no live walk')):
            nav.refresh()
        self.assertEqual(nav.entries['sample.py']['symbols'][0]['name'], 'value')
        self.assertNotIn('replacement', str(nav.entries))
        self.assertEqual(nav.version, ev.current)
        old_cache = prepared['parse_cache']
        ev.record('new_version')
        next_data = build_preparation(ev.view(), old_cache, threading.Event())
        self.assertEqual(next_data['parsed_files'], 1)
        self.assertEqual(next_data['reused_files'], 1)

    def test_cold_transcript_matches_unoptimized_mode(self):
        scan = {
            'kind': 'inspect',
            'note': 'read',
            'steps': [
                {'id': 'r', 'op': 'read', 'path': 'sample.py'},
                {'id': 's', 'op': 'search', 'path': '.', 'query': 'value'},
            ],
        }
        outputs = []
        for enabled in (False, True):
            model = f.Model([scan, self.candidate()])
            result = TaskRunner(
                model, self.env, state_dir=self.root / 'state', optimize_source=enabled
            ).run([], 'return 4', StepBudget(max_steps=5))
            self.assertEqual(result['status'], 'verified_success', result)
            outputs.append([wire_input(m, [submit_plan]) for m in model.calls])
            shutil.rmtree(self.root / 'state')
            shutil.rmtree(self.env.artifact_root)
            self.env.artifact_root.mkdir()
            (self.repo / 'sample.py').write_text('def value():\n    return 0\n')

        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items() if k != 'seconds'}
            if isinstance(value, list):
                return [clean(v) for v in value]
            if isinstance(value, str) and value.startswith('{'):
                try:
                    return clean(json.loads(value))
                except ValueError:
                    pass
            if isinstance(value, str):
                value = re.sub(r"acceptance-[a-z0-9_]{8}(?=[/\\]|$)", "acceptance-TEMP", value)
                return re.sub(r"(Ran \d+ tests? in )\d+\.\d+s", r"\1TIME", value)
            return value

        self.assertEqual(clean(outputs[0]), clean(outputs[1]))

    def test_incomplete_map_never_hides_text_matches(self):
        (self.repo / 'other.c').write_text('// needle\nint needle=1;')
        ev = self.evidence()
        reader = OperationExecutor(self.env, ev, self.root / 'records', versioned_reads=True)
        reader.prepared = {'version': ev.current, 'lines': {}}
        self.assertEqual(
            len(reader.search(Operation(id='s', op='search', query='needle'))['hits']), 2
        )

    def test_optional_preparation_does_not_block_and_is_cancelled(self):
        ev = self.evidence()
        entered = threading.Event()

        def slow(view, previous, cancel):
            entered.set()
            self.assertTrue(cancel.wait(3), 'cleanup must signal a running worker')
            return build_preparation(view, previous, cancel)

        prep = Preparation(self.root / 'prepared')
        with patch('agent.source.preparation.build_preparation', side_effect=slow):
            prep.start(ev.view())
            self.assertTrue(entered.wait(2))
            self.assertIsNone(prep.collect(ev.current))
            prep.close(ev.current)
        self.assertTrue(prep.closed)
        self.assertFalse(any(t.name.startswith('source-prepare') for t in threading.enumerate()))

    def test_stale_result_is_not_published_to_new_source(self):
        ev = self.evidence()
        prep = Preparation(self.root / 'prepared')
        prep.start(ev.view())
        prep.active['future'].result(timeout=5)
        (self.repo / 'sample.py').write_text('def new(): pass\n')
        ev.record('change')
        self.assertIsNone(prep.collect(ev.current))
        self.assertEqual(prep.jobs[0]['status'], 'stale')
        self.assertFalse((self.root / 'prepared').exists())
        prep.close(ev.current)

    def test_model_and_local_worker_overlap_without_another_model(self):
        entered, proceed = threading.Event(), threading.Event()
        original = build_preparation

        def prepare(view, previous, cancel):
            entered.set()
            self.assertTrue(proceed.wait(3))
            return original(view, previous, cancel)

        outer = self

        class Model(f.Model):
            def invoke(self, messages):
                outer.assertTrue(entered.wait(3))
                proceed.set()
                return super().invoke(messages)

        model = Model([self.candidate()])
        with patch('agent.source.preparation.build_preparation', side_effect=prepare):
            result = TaskRunner(model, self.env, state_dir=self.root / 'state').run(
                [], 'return 4', StepBudget(max_steps=4)
            )
        self.assertEqual(result['status'], 'verified_success', result)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(result['source_performance']['preparation']['worker_model_calls'], 0)
        self.assertEqual(result['source_performance']['preparation_errors'], [])

    def test_derived_artifact_registration_runs_on_caller_thread(self):
        ev = self.evidence()
        owner = threading.get_ident()
        registered = []

        class Archive:
            def register_derived(self, *args):
                registered.append(threading.get_ident())

        prep = Preparation(self.root / 'prepared')
        prep.start(ev.view())
        prep.active['future'].result(timeout=5)
        prep.collect(ev.current, Archive())
        prep.close(ev.current)
        self.assertEqual(registered, [owner])
