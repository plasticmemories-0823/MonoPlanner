"""Official graph integration keeps private task records out of public state."""
import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from agent.server.graph import graph
from agent.storage.conversations import ConversationStore
from langgraph.checkpoint.memory import InMemorySaver
from agent.server.graph import builder


class BlockedModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content='', tool_calls=[{'name': 'submit_work', 'id': 'blocked',
            'args': {'kind': 'blocked', 'note': 'private explanation'}}],
            usage_metadata={'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2})


class WebGraphTests(unittest.TestCase):
    def test_existing_runner_publishes_only_safe_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            repo = root/'repo'
            repo.mkdir()
            with patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(root/'runs')}), \
                    patch('agent.server.graph.model_factory', return_value=BlockedModel()):
                result = graph.invoke({'repo': str(repo), 'messages': [HumanMessage(content='do work', id='first')]},
                                      {'configurable': {'thread_id': str(uuid.uuid4())}, 'recursion_limit': 100})
            self.assertEqual(result['status'], 'task_blocked')
            self.assertNotIn('private explanation', repr(result))
            self.assertNotIn('verification', result['result'])
            self.assertTrue((root/'runs'/'tasks'/result['task_id']/'history.sqlite3').exists())


class ScriptedModel:
    def __init__(self, replies, hook=None):
        self.replies = iter(replies)
        self.calls = []
        self.hook = hook

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if self.hook:
            self.hook(len(self.calls))
        return next(self.replies)


def text_reply(text='公开回答'):
    return AIMessage(content=text, additional_kwargs={'reasoning_content': 'PRIVATE_REASONING'},
                     usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5})


def investigation(paths):
    return AIMessage(content='先读取必要的源码。', tool_calls=[{'name': 'submit_work', 'id': 'investigate',
        'args': {'kind': 'inspect', 'note': 'PRIVATE_NOTE',
                 'steps': [{'id': 'read'+str(i), 'op': 'read', 'path': p} for i, p in enumerate(paths)]}}],
        usage_metadata={'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5})


class ConversationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root/'repo'
        self.repo.mkdir()
        for name in ('first.py', 'second.py'):
            (self.repo/name).write_text('VALUE = 1\n', encoding='utf-8')
        self.thread = str(uuid.uuid4())
        self.config = {'configurable': {'thread_id': self.thread}, 'recursion_limit': 100}
        self.compiled = builder.compile(checkpointer=InMemorySaver())
        self.store = ConversationStore(self.root/'runs')
        guard = patch.dict(os.environ, {'MONOPLANER_STATE_DIR': str(self.root/'runs')})
        guard.start()
        self.addCleanup(guard.stop)

    def invoke(self, model, text='你好', identity='user-one'):
        with patch('agent.server.graph.model_factory', return_value=model):
            return self.compiled.invoke({'repo': str(self.repo),
                'messages': [HumanMessage(content=text, id=identity)]}, self.config)

    def test_chat_uses_one_request_and_no_operations_or_private_output(self):
        model = ScriptedModel([text_reply()])
        result = self.invoke(model)
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 1)
        self.assertNotIn('PRIVATE_REASONING', repr(result))
        self.assertEqual(len(list((self.root/'runs'/'tasks').glob('*/plans/*/plan.json'))), 0)

    def test_startup_failure_is_public_and_persistent_without_a_model_call(self):
        model = ScriptedModel([text_reply('新的普通聊天正常回答。')])
        with patch('agent.server.runtime.recovery.code_key', side_effect=FileNotFoundError('PRIVATE_FILE_PATH')):
            failed = self.invoke(model)
        self.assertEqual(failed['status'], 'runtime_error')
        self.assertEqual(len(model.calls), 0)
        self.assertTrue(any('未能启动' in m.content for m in failed['messages'] if m.type == 'ai'))
        self.assertNotIn('PRIVATE_FILE_PATH', repr(failed))
        session = self.store.session(self.thread)
        self.assertEqual(session['status'], 'runtime_error')
        self.assertTrue(session['initialization_failed'])
        self.assertEqual(session['result']['usage']['calls'], 0)
        self.assertTrue((self.root/'runs'/'errors'/f"{failed['task_id']}.json").is_file())
        with patch('agent.server.app.state_root', return_value=self.root/'runs'):
            from agent.server.app import control_value
            import json
            restored = json.loads(control_value(self.thread).body)
        self.assertEqual(restored['status'], 'runtime_error')
        self.assertTrue(any('未能启动' in m['content'] for m in restored['messages'] if m['type'] == 'ai'))
        chatted = self.invoke(model, '你好', 'after-startup-failure')
        self.assertEqual(chatted['status'], 'chat_complete')
        self.assertNotEqual(chatted['task_id'], failed['task_id'])
        self.assertEqual(len(model.calls), 1)

    def test_escaped_failure_does_not_settle_an_existing_task(self):
        from agent.server.runtime import ConversationRunner
        def interrupted(runner, *args, **kwargs):
            runner.root = runner.explicit_state
            runner.root.mkdir(parents=True)
            runner.state = {'running_step': 'write', 'budget': {'unknown_usage_steps': 1}}
            raise OSError('PRIVATE_EXECUTION_ERROR')

        model = ScriptedModel([])
        with patch.object(ConversationRunner, 'run', interrupted):
            failed = self.invoke(model)
        session = self.store.session(self.thread)
        self.assertEqual(failed['status'], 'runtime_error')
        self.assertFalse(session['initialization_failed'])
        from agent.server.graph import finished
        self.assertFalse(finished(session, self.store))
        self.assertNotIn('PRIVATE_EXECUTION_ERROR', repr(failed))

    def test_graph_interrupt_is_not_converted_to_a_runtime_error(self):
        from langgraph.errors import GraphInterrupt
        with patch('agent.server.runtime.ConversationRunner.run', side_effect=GraphInterrupt()):
            result = self.invoke(ScriptedModel([]))
        self.assertNotEqual(result.get('status'), 'runtime_error')
        self.assertFalse(self.store.session(self.thread).get('initialization_failed'))

    def test_code_draft_is_public_chat_and_does_not_initialize_work(self):
        draft = '这是页面草稿，还未保存。\n```html\n<!doctype html>\n<h1>淘宝示例</h1>\n```\n是否按这个方案开始创建文件？'
        model = ScriptedModel([text_reply(draft)])
        result = self.invoke(model, '先不改代码，帮我做一个首页 HTML 可以吗')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(any(m.type == 'ai' and m.content == draft for m in result['messages']))
        self.assertEqual(sorted(p.name for p in self.repo.iterdir()), ['first.py', 'second.py'])
        self.assertFalse(list((self.root/'runs'/'tasks').glob('*/baseline/*')))
        self.assertFalse(list((self.root/'runs'/'tasks').glob('*/plans/*/plan.json')))

    def test_explicit_start_after_discussion_preserves_agreed_material(self):
        draft = '先使用一个独立 HTML 文件，橙色导航、搜索框和商品列表。是否现在开工？'
        model = ScriptedModel([text_reply(draft), investigation(['first.py']), text_reply('已读取授权源码。')])
        first = self.invoke(model, '帮我做一个页面')
        self.assertEqual(first['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 1)
        started = self.invoke(model, '按刚才的方案，你直接动手', 'start-work')
        self.assertIn(draft, repr(model.calls[1]))
        self.assertIn('按刚才的方案，你直接动手', repr(model.calls[1]))
        self.assertEqual(len(model.calls), 3)
        self.assertNotEqual(first['task_id'], started['task_id'])
        self.assertTrue(list((self.root/'runs'/'tasks'/started['task_id']).glob('plans/*/plan.json')))

    def test_new_turn_preserves_public_history_and_duplicate_does_not_recall(self):
        model = ScriptedModel([text_reply('第一次'), text_reply('第二次')])
        first = self.invoke(model)
        repeated = self.invoke(model)
        self.assertEqual(repeated['task_id'], first['task_id'])
        self.assertEqual(len(model.calls), 1)
        self.invoke(model, '再问', 'user-two')
        self.assertEqual(len(model.calls), 2)
        self.assertIn('第一次', repr(model.calls[-1]))
        self.assertNotIn('PRIVATE_REASONING', repr(model.calls[-1]))

    def test_new_message_after_lost_graph_view_keeps_saved_public_history(self):
        model = ScriptedModel([text_reply('已保存的公开回答'), text_reply('继续回答')])
        self.invoke(model)
        self.compiled = builder.compile(checkpointer=InMemorySaver())
        result = self.invoke(model, '新问题', 'user-two')
        self.assertEqual(len(model.calls), 2)
        self.assertIn('已保存的公开回答', repr(model.calls[-1]))
        self.assertTrue(any(m.type == 'ai' and m.content == '已保存的公开回答' for m in result['messages']))

    def test_duplicate_after_lost_graph_view_restores_reply_without_another_request(self):
        model = ScriptedModel([text_reply('已有回答')])
        first = self.invoke(model)
        self.compiled = builder.compile(checkpointer=InMemorySaver())
        repeated = self.invoke(model)
        self.assertEqual(repeated['task_id'], first['task_id'])
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(any(m.type == 'ai' and m.content == '已有回答' for m in repeated['messages']))

    def test_recovered_pending_messages_do_not_replace_the_submitted_input(self):
        from langchain_core.messages import messages_to_dict
        model = ScriptedModel([text_reply('已有回答'), text_reply('接入此前消息')])
        self.invoke(model)
        session = self.store.session(self.thread)
        saved = [HumanMessage(content='第一条排队消息', id='queued-first'),
                 HumanMessage(content='第二条排队消息', id='queued-later')]
        session['public_messages'].extend(messages_to_dict(saved))
        self.store.save_session(self.thread, session)
        for message in saved:
            self.store.enqueue(self.thread, message.id, 'message', message.content)
        self.compiled = builder.compile(checkpointer=InMemorySaver())
        self.invoke(model, '第一条排队消息', 'queued-first')
        self.assertEqual(self.store.session(self.thread)['request'], '第一条排队消息')

    def test_chat_after_runtime_error_starts_a_new_task_without_replaying_failed_work(self):
        from agent.source.snapshots import SnapshotError
        model = ScriptedModel([investigation(['first.py']), text_reply('你好，上一项工作仍保留失败现场。')])
        with patch('agent.server.runtime.ConversationRunner.bootstrap_work', side_effect=SnapshotError('ignored cache link')):
            failed = self.invoke(model, '直接读取源码', 'work-request')
        self.assertEqual(failed['status'], 'runtime_error')
        self.assertTrue(any(m.type == 'ai' and m.content == '先读取必要的源码。' for m in failed['messages']))
        old_root = self.root/'runs'/'tasks'/failed['task_id']
        originals = {str(p.relative_to(old_root)): p.read_bytes() for p in old_root.rglob('*') if p.is_file()}
        chatted = self.invoke(model, '你好', 'after-error')
        self.assertEqual(chatted['status'], 'chat_complete')
        self.assertNotEqual(chatted['task_id'], failed['task_id'])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(originals, {str(p.relative_to(old_root)): p.read_bytes() for p in old_root.rglob('*') if p.is_file()})
        self.assertFalse(list((self.root/'runs'/'tasks').glob('*/plans/*/plan.json')))
        self.assertFalse(self.store.pending(self.thread))

    def test_pause_during_model_saves_response_and_resume_does_not_resend(self):
        model = ScriptedModel([investigation(['first.py']), text_reply()],
            hook=lambda n: self.store.enqueue(self.thread, 'pause', 'pause') if n == 1 else None)
        result = self.invoke(model, '读取文件')
        self.assertEqual(result['status'], 'paused')
        self.assertEqual(len(model.calls), 1)
        # Resume, retaining the paid response, without a new requirement.
        self.store.enqueue(self.thread, 'resume', 'resume')
        with patch('agent.server.graph.model_factory', return_value=model):
            continued = self.compiled.invoke({'repo': str(self.repo), 'messages': []}, self.config)
        self.assertEqual(len(model.calls), 2)
        self.assertNotEqual(continued['status'], 'runtime_error')
        self.assertTrue(list((self.root/'runs'/'tasks').glob('*/plans/*/plan.json')))

    def test_input_during_model_cancels_unexecuted_tool_and_balances_history(self):
        model = ScriptedModel([investigation(['first.py']), text_reply('按新要求回答')],
            hook=lambda n: self.store.enqueue(self.thread, 'new', 'message', '只解释，不读取') if n == 1 else None)
        result = self.invoke(model, '原要求')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 2)
        self.assertIn('superseded_by_user', repr(model.calls[-1]))
        self.assertIn('只解释，不读取', repr(model.calls[-1]))
        self.assertFalse(list((self.root/'runs'/'tasks').glob('*/plans/*/plan.json')))

    def test_input_after_first_operation_cancels_remaining_operations(self):
        from agent.execution.executor import OperationExecutor
        original = OperationExecutor.execute
        executed = []
        def operation(executor, step, *args, **kwargs):
            result = original(executor, step, *args, **kwargs)
            executed.append(step.path)
            self.store.enqueue(self.thread, 'new', 'message', '不要读取第二个文件')
            return result
        # A coding supplement selects the work route. Plain chat now preserves
        # the old plan; it must not serve as an implicit cancellation command.
        stopped = AIMessage(content='停止剩余读取。', tool_calls=[{'name': 'submit_work', 'id': 'stop',
            'args': {'kind': 'blocked', 'note': 'User cancelled the remaining read'}}])
        model = ScriptedModel([investigation(['first.py', 'second.py']), stopped])
        with patch.object(OperationExecutor, 'execute', operation):
            result = self.invoke(model, '读取两个文件')
        self.assertEqual(executed, ['first.py'])
        self.assertEqual(result['status'], 'task_blocked')
        from agent.storage.history import LocalHistory
        archive = LocalHistory(self.root/'runs'/'tasks'/result['task_id'])
        with archive.lease():
            self.assertIn('superseded_by_user', repr(archive.load()['history']))
        archive.close()
        self.assertEqual(len(model.calls), 2)

    def test_unknown_request_is_not_automatically_reissued(self):
        class UnknownModel(ScriptedModel):
            def invoke(self, messages):
                self.calls.append(messages)
                raise TimeoutError('No response received')
        model = UnknownModel([])
        first = self.invoke(model, '工作')
        self.assertEqual(first['status'], 'runtime_error')
        self.invoke(model, '继续', 'user-two')
        self.assertEqual(len(model.calls), 1)

    def test_error_with_unknown_execution_or_compaction_cannot_start_another_task(self):
        from agent.storage.history import LocalHistory
        for marker, value in [('running_step', 'write'), ('assistant_running', {'id': 'run'}),
                              ('verification_running', True), ('compaction_pending', {'stage': 'committing'})]:
            with self.subTest(marker=marker):
                task = uuid.uuid4().hex
                archive = LocalHistory(self.root/'runs'/'tasks'/task)
                try:
                    with archive.lease():
                        archive.record({'status': 'runtime_error', 'history': [],
                                        'budget': {'unknown_usage_steps': 0}, marker: value}, 'task_returned', {})
                    self.assertFalse(self.store.error_is_settled(task))
                finally:
                    archive.close()

    def test_mailbox_rejects_changed_content_and_can_be_reopened(self):
        self.store.enqueue(self.thread, 'x', 'message', '原文')
        self.store.enqueue(self.thread, 'x', 'message', '原文')
        with self.assertRaises(ValueError):
            self.store.enqueue(self.thread, 'x', 'message', '改写')
        self.assertEqual(len(ConversationStore(self.root/'runs').pending(self.thread)), 1)

    def test_new_graph_checkpoint_restores_old_messages_before_new_input(self):
        first = ScriptedModel([text_reply('第一轮回复')])
        self.invoke(first, '第一轮要求', 'first-user')
        self.compiled = builder.compile(checkpointer=InMemorySaver())
        second = ScriptedModel([text_reply('第二轮回复')])
        result = self.invoke(second, '第二轮要求', 'second-user')
        self.assertEqual([m.content for m in result['messages']],
                         ['第一轮要求', '第一轮回复', '第二轮要求', '第二轮回复'])
        saved = self.store.session(self.thread)['public_messages']
        self.assertEqual([m['data']['content'] for m in saved],
                         ['第一轮要求', '第一轮回复', '第二轮要求', '第二轮回复'])
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)

    def test_real_acceptance_failure_returns_to_model_and_completed_write_does_not_replay(self):
        from test_task_runner import CODE, Model
        from support import python_command as shell_command
        from agent.settings import PROJECT_ROOT
        python_command = lambda code: shell_command(code, str(PROJECT_ROOT/"launcher/runtime/python/python.exe"))
        import json
        (self.repo/'sample.py').write_text('def value():\n    return 0\n')
        def candidate(value):
            return {'kind': 'candidate', 'note': 'Implement the required behavior', 'test_code': CODE,
                    'steps': [{'id': 'edit', 'op': 'shell', 'command': python_command(
                        "from pathlib import Path; Path('sample.py').write_text('def value():\\n    return %d\\n')" % value)}]}
        model = Model([candidate(2), candidate(4)])
        result = self.invoke(model, 'return 4')
        self.assertEqual(result['status'], 'verified_success', result)
        root = self.root/'runs'/'tasks'/result['task_id']
        from agent.storage.history import LocalHistory
        archive = LocalHistory(root)
        try:
            with archive.lease():
                state = archive.load()
        finally:
            archive.close()
        self.assertEqual([r['verification']['status'] for r in state['rounds']], ['failed', 'passed'])
        self.assertEqual(len(model.calls), 3)
        before = (root/'events.jsonl').read_bytes()
        again = self.invoke(model, 'return 4')
        self.assertEqual(again['status'], 'verified_success')
        self.assertEqual(before, (root/'events.jsonl').read_bytes())
        self.assertEqual(len(model.calls), 3)

    def test_new_requirement_after_write_keeps_completed_change_and_cancels_next_write(self):
        from agent.execution.executor import OperationExecutor
        from support import python_command as shell_command
        from agent.settings import PROJECT_ROOT
        python_command = lambda code: shell_command(code, str(PROJECT_ROOT/"launcher/runtime/python/python.exe"))
        from test_task_runner import CODE, Model
        original = OperationExecutor.execute
        done = []
        def operation(executor, step, *args, **kwargs):
            result = original(executor, step, *args, **kwargs)
            done.append(step.id)
            self.store.enqueue(self.thread, 'new', 'message', '停止剩余修改')
            return result
        code = "from pathlib import Path; Path('changed.txt').write_text('原样修改')"
        (self.repo/'sample.py').write_text('def value():\n    return 0\n')
        plan = {'kind': 'candidate', 'note': 'Two changes', 'test_code': CODE,
                'steps': [{'id': 'edit', 'op': 'shell', 'command': python_command(code)},
                          {'id': 'later', 'op': 'shell', 'command': python_command(
                              "from pathlib import Path; Path('not-created.txt').write_text('bad')")}]}
        # Explicit preflight confirmation is provided by the existing fixture.
        model = Model([plan, {'kind': 'blocked', 'note': 'User stopped changes'}])
        with patch.object(OperationExecutor, 'execute', operation):
            result = self.invoke(model, 'return 4')
        self.assertEqual(done, ['edit'])
        self.assertEqual((self.repo/'changed.txt').read_text(), '原样修改')
        self.assertFalse((self.repo/'not-created.txt').exists())
        self.assertEqual(result['status'], 'task_blocked')

    def test_saved_input_applies_once_after_acknowledgment_interruption(self):
        model = ScriptedModel([investigation(['first.py']), text_reply()],
            hook=lambda n: self.store.enqueue(self.thread, 'new', 'message', '只解释') if n == 1 else None)
        original = ConversationStore.acknowledge
        def interrupted(store, thread, identities):
            if 'new' in identities:
                raise InterruptedError('crash after task checkpoint, before mailbox ack')
            return original(store, thread, identities)
        with patch.object(ConversationStore, 'acknowledge', interrupted), self.assertRaises(InterruptedError):
            self.invoke(model)
        result = self.invoke(model)
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(sum(m.id == 'new' for m in model.calls[-1] if isinstance(m, HumanMessage)), 1)

    def test_unsupported_attachment_is_explicitly_rejected(self):
        model = ScriptedModel([])
        with patch('agent.server.graph.model_factory', return_value=model), self.assertRaises(ValueError):
            self.compiled.invoke({'repo': str(self.repo), 'messages': [HumanMessage(
                content=[{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,x'}}], id='file')]}, self.config)
        self.assertEqual(len(model.calls), 0)

    def test_early_input_before_session_is_applied_without_an_extra_classification_call(self):
        self.store.enqueue(self.thread, 'early', 'message', '只回答，不操作')
        model = ScriptedModel([text_reply()])
        result = self.invoke(model, '读取文件')
        self.assertEqual(len(model.calls), 1)
        self.assertIn('只回答，不操作', repr(model.calls[0]))
        self.assertEqual(result['status'], 'chat_complete')

    def test_completed_tool_pair_and_reasoning_survive_graph_boundaries_only_in_private_request(self):
        model_first = investigation(['first.py'])
        model_first.additional_kwargs['reasoning_content'] = 'PRIVATE_REASONING'
        model = ScriptedModel([model_first, text_reply()])
        result = self.invoke(model, '读取文件')
        self.assertEqual(len(model.calls), 2)
        from langchain_core.messages import ToolMessage
        prior = model.calls[-1]
        called = next(m for m in prior if isinstance(m, AIMessage) and m.tool_calls)
        self.assertEqual(called.additional_kwargs['reasoning_content'], 'PRIVATE_REASONING')
        self.assertEqual(called.tool_calls[0]['args'], model_first.tool_calls[0]['args'])
        self.assertEqual(next(m for m in prior if isinstance(m, ToolMessage)).tool_call_id,
                         model_first.tool_calls[0]['id'])
        self.assertNotIn('PRIVATE_REASONING', repr(result))

    def test_lost_presentation_checkpoint_restores_public_history_without_replaying_request(self):
        model = ScriptedModel([investigation(['first.py']), text_reply()],
            hook=lambda n: self.store.enqueue(self.thread, 'pause', 'pause') if n == 1 else None)
        first = self.invoke(model, '读取文件')
        self.assertEqual(first['status'], 'paused')
        self.compiled = builder.compile(checkpointer=InMemorySaver())
        self.store.enqueue(self.thread, 'resume', 'resume')
        with patch('agent.server.graph.model_factory', return_value=model):
            recovered = self.compiled.invoke({'messages': []}, self.config)
        self.assertEqual(recovered['task_id'], first['task_id'])
        self.assertEqual(recovered['status'], 'awaiting_user')
        self.assertEqual(len(model.calls), 2)
        self.assertNotIn('PRIVATE_NOTE', repr(recovered))

    def test_product_model_suppresses_private_automatic_streaming_without_changing_api_options(self):
        from agent.model.client import ConfiguredModel
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from agent.server.graph import model_factory
        model = ConfiguredModel(FakeMessagesListChatModel(responses=[text_reply()]))
        before = model.request_options
        with patch('agent.server.graph.create_model', return_value=model), patch('agent.server.graph.load_settings', return_value={}):
            selected = model_factory()
        self.assertEqual(selected.client.tags, ['nostream', 'langsmith:do-not-render'])
        self.assertEqual(selected.request_options, before)
