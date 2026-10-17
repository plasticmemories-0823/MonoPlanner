import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.storage.live_reasoning import LiveReasoning, read_live, _active


class DisplayResilienceTests(unittest.TestCase):
    def test_open_old_display_file_cannot_stop_new_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / 'display/model.json'
            old.parent.mkdir()
            old.write_text('{}')
            with old.open('rb'):
                if __import__('os').name == 'nt':
                    temporary = old.with_suffix('.tmp')
                    temporary.write_text('{}')
                    with self.assertRaises(PermissionError):
                        temporary.replace(old)
                live = LiveReasoning(root, 1, 'user')
                live.append('实时内容')
                live.finish('completed', '完整内容')
            self.assertEqual(read_live(root)[0][0]['detail'], '完整内容')
            self.assertEqual(old.read_text(), '{}')

    def test_partial_line_does_not_advance_cursor_or_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            live = LiveReasoning(root, 1, 'user')
            _, cursor, version, _, _ = read_live(root)
            row = {
                **live.row,
                'version': live.row['version'] + 1,
                'detail': '新片段',
                'append': True,
            }
            raw = (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')
            with live.path.open('ab') as stream:
                stream.write(raw[:-2])
            result = read_live(root, cursor, version)
            self.assertEqual(result[:2], ([], cursor))
            with live.path.open('ab') as stream:
                stream.write(raw[-2:])
            result = read_live(root, cursor, version)
            self.assertEqual(result[0][0]['detail'], '新片段')
            self.assertEqual(read_live(root, result[1], result[2])[0], [])

    def test_completed_instances_do_not_accumulate_in_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            key = str(root.resolve())
            live = LiveReasoning(root, 1, 'user')
            live.finish('completed')
            del live
            gc.collect()
            self.assertNotIn(key, _active)


def test_critical_work_response_save_failure_stops_without_model_replay(tmp_path):
    from contextlib import closing
    from agent.execution.local import LocalEnvironment
    from agent.storage.history import LocalHistory
    from agent.task.budget import StepBudget
    from agent.task.runner import TaskRunner, save
    from tests.fixtures.models import ScriptedModel, text_reply

    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'source.py').write_text('VALUE = 1\n')
    root = tmp_path / 'task'
    environment = LocalEnvironment(repo, artifact_root=tmp_path / 'execution')
    model = ScriptedModel([text_reply()])
    runner = TaskRunner(model, environment, state_dir=root, interactive=True)

    def fail_response(path, value):
        if path.name == 'response.json':
            raise PermissionError('response archive denied')
        return save(path, value)

    with patch('agent.task.runner.save', side_effect=fail_response):
        stopped = runner.run([], 'Read the source.', StepBudget(max_steps=5))
    assert stopped['status'] == 'runtime_error'
    assert 'response archive denied' in stopped['stop_reason']
    with closing(LocalHistory(root)) as history, history.lease():
        saved = history.load()
    assert saved['request_pending'] is not None
    assert not (root / 'plans').exists()
    resumed = TaskRunner(model, environment, state_dir=root, interactive=True).run(
        [], 'Read the source.', StepBudget(max_steps=5)
    )
    assert resumed['status'] == 'model_request_unknown'
    assert len(model.calls) == 1
