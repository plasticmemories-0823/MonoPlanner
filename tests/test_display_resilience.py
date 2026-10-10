import gc
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.storage.live_reasoning import LiveReasoning, read_live, _active
import test_web_graph as fixtures


class DisplayResilienceTests(unittest.TestCase):
    def test_open_old_display_file_cannot_stop_new_stream(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root/'display/model.json'; old.parent.mkdir(); old.write_text('{}')
            with old.open('rb'):
                if __import__('os').name == 'nt':
                    temporary = old.with_suffix('.tmp'); temporary.write_text('{}')
                    with self.assertRaises(PermissionError):
                        temporary.replace(old)
                live = LiveReasoning(root, 1, 'user')
                live.append('实时内容'); live.finish('completed', '完整内容')
            self.assertEqual(read_live(root)[0][0]['detail'], '完整内容')
            self.assertEqual(old.read_text(), '{}')

    def test_partial_line_does_not_advance_cursor_or_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); live = LiveReasoning(root, 1, 'user')
            _, cursor, version, _, _ = read_live(root)
            row = {**live.row, 'version': live.row['version']+1, 'detail': '新片段', 'append': True}
            raw = (json.dumps(row, ensure_ascii=False)+'\n').encode('utf-8')
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
            root = Path(folder); key = str(root.resolve())
            live = LiveReasoning(root, 1, 'user'); live.finish('completed')
            del live; gc.collect()
            self.assertNotIn(key, _active)


class GraphDisplayFailureTests(unittest.TestCase):
    setUp = fixtures.ConversationTests.setUp
    invoke = fixtures.ConversationTests.invoke

    def test_display_write_failure_keeps_response_usage_and_no_retry(self):
        owner = self
        class Streaming(fixtures.ScriptedModel):
            def invoke_with_progress(self, messages, receive):
                self.calls.append(messages)
                receive('第一段'); receive('第二段')
                from agent.storage.activity import read_activity
                task = next((owner.root/'runs/tasks').iterdir())
                view = read_activity(owner.root/'runs', [{'id': task.name, 'input_id': 'user-one'}], {})
                owner.assertEqual(view['live'][0]['detail'], '第一段')
                return next(self.replies)
        model = Streaming([fixtures.text_reply()])
        original = Path.open
        def opened(path, *args, **kwargs):
            if path.name == 'model.jsonl':
                raise PermissionError('injected display lock')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', opened):
            result = self.invoke(model)
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result['result']['usage']['total_tokens'], 5)
        response = self.root/'runs/tasks'/result['task_id']/'requests/call-0001/response.json'
        self.assertTrue(response.exists())

    def test_critical_response_failure_stops_and_is_not_retried(self):
        from agent.task.runner import save
        model = fixtures.ScriptedModel([fixtures.text_reply()])
        def persist(path, data):
            if Path(path).name == 'response.json':
                raise PermissionError('critical archive failure')
            return save(path, data)
        with patch('agent.task.runner.save', side_effect=persist):
            result = self.invoke(model)
        self.assertEqual(result['status'], 'runtime_error')
        self.assertEqual(len(model.calls), 1)
        from agent.storage.history import LocalHistory
        history = LocalHistory(self.root/'runs/tasks'/result['task_id'])
        try:
            with history.lease():
                self.assertIsNotNone(history.load()['request_pending'])
        finally:
            history.close()
