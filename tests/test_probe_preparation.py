"""Only immutable source copying overlaps model waits; execution stays lazy."""
from concurrent.futures import ThreadPoolExecutor, CancelledError
from pathlib import Path
from threading import Event
from unittest.mock import patch
import unittest

from agent.source.probe_preparation import ProbePreparation
from agent.source.preparation import Preparation
from agent.source.snapshots import SourceSnapshot
from agent.source.versions import SourceVersions
import test_task_runner as f


class ProbePreparationTests(unittest.TestCase):
    setUp = f.TaskRunnerTests.setUp

    def view(self):
        base = SourceSnapshot.capture(self.repo, self.root/'baseline')
        self.evidence = SourceVersions(self.env, self.root/'evidence', baseline=base)
        return self.evidence.view()

    def worker(self):
        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(pool.shutdown, wait=True, cancel_futures=True)
        return ProbePreparation(pool, self.root/'prepared')

    def test_ready_copy_is_used_once_and_keeps_archived_bytes(self):
        view = self.view()
        worker = self.worker()
        worker.start(view)
        worker.active['future'].result(timeout=10)
        (self.repo/'sample.py').write_text('changed workspace')
        snapshot = worker.take(view.version)
        self.assertIn('return 0', (snapshot.root/'sample.py').read_text())
        self.assertIsNone(worker.take(view.version))
        self.assertEqual(worker.jobs[0]['status'], 'consumed')

    def test_wrong_version_cannot_consume_ready_copy(self):
        view = self.view()
        worker = self.worker()
        worker.start(view)
        worker.active['future'].result(timeout=10)
        self.assertIsNone(worker.take('other-version'))
        self.assertEqual(worker.jobs[0]['status'], 'stale')

    def test_corrupt_copy_is_not_replaced_by_a_fallback(self):
        view = self.view()
        worker = self.worker()
        worker.start(view)
        snapshot = worker.active['future'].result(timeout=10)['snapshot']
        (snapshot.root/'sample.py').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'integrity failed'):
            worker.take(view.version)

    def test_unfinished_copy_is_optional_and_job_queue_is_bounded(self):
        view = self.view()
        worker = self.worker()
        entered, release = Event(), Event()
        def copy(view, directory, cancel):
            entered.set()
            release.wait(5)
            raise CancelledError()
        with patch('agent.source.probe_preparation.prepare_copy', side_effect=copy) as build:
            worker.start(view)
            self.assertTrue(entered.wait(5))
            future = worker.active['future']
            for i in range(20):
                worker.start(view)
                self.assertIsNone(worker.take(view.version))
            self.assertIs(worker.active['future'], future)
            self.assertEqual(build.call_count, 1)
            self.assertTrue(worker.active['cancel'].is_set())
            worker.cancel()
            self.assertTrue(worker.active['cancel'].is_set())
            release.set()
            with self.assertRaises(CancelledError):
                future.result(timeout=5)
            worker.finish()
            self.assertIsNone(worker.active)

    def test_cancelled_copy_can_be_prepared_again_on_a_later_request(self):
        view = self.view()
        worker = self.worker()
        worker.start(view)
        worker.active['future'].result(timeout=10)
        worker.cancel()
        worker.start(view)
        worker.active['future'].result(timeout=10)
        self.assertIsNotNone(worker.take(view.version))
        self.assertEqual(worker.jobs[0]['status'], 'stale')

    def test_copy_and_index_share_one_worker_and_close_cleanly(self):
        view = self.view()
        preparation = Preparation(self.root/'prepared', prepare_probes=True)
        preparation.start(view)
        preparation.probes.active['future'].result(timeout=10)
        preparation.active['future'].result(timeout=10)
        self.assertEqual(preparation.pool._max_workers, 1)
        preparation.close(view.version)
        self.assertTrue(preparation.closed)
        self.assertIsNone(preparation.active)
        self.assertIsNone(preparation.probes.active)
        self.assertEqual(preparation.report()['worker_model_calls'], 0)
        self.assertEqual(preparation.report()['worker_sqlite_writes'], 0)

    def test_cancelled_export_cannot_produce_a_usable_snapshot(self):
        view = self.view()
        cancel = Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            view.export(self.root/'cancelled', cancel=cancel)
        self.assertFalse((self.root/'cancelled/manifest.json').exists())

    def test_short_wait_cancels_queued_copy_without_delaying_the_index(self):
        view = self.view()
        entered, release = Event(), Event()
        preparation = Preparation(self.root/'prepared', prepare_probes=True)
        def index(view, previous, cancel):
            entered.set()
            release.wait(5)
            raise CancelledError()
        with patch('agent.source.preparation.build_preparation', side_effect=index), \
                patch('agent.source.probe_preparation.prepare_copy') as copy:
            preparation.start(view)
            self.assertTrue(entered.wait(5))
            self.assertIsNone(preparation.probes.take(view.version))
            self.assertTrue(preparation.probes.active['future'].cancelled())
            self.assertEqual(copy.call_count, 0)
            release.set()
            preparation.close(view.version)
