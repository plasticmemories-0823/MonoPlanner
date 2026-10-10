"""General tools use real disposable files/processes and fake model replies."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from agent.execution.assistant_files import AssistantFiles
from agent.execution.existing import command_for
from agent.model.tools import CHAT_TOOLS, ASSISTANT_SCHEMA
from agent.server.assistant import validate
from agent.storage.activity import read_activity
from agent.storage.history import LocalHistory
import test_web_graph as fixtures
from test_web_graph import ScriptedModel, text_reply, investigation


def reply(name, **args):
    return AIMessage(content='开始查询或运行。', tool_calls=[{'name': name, 'id': 'call-'+name, 'args': args}],
        additional_kwargs={'reasoning_content': 'PRIVATE_ASSISTANT_REASONING'},
        usage_metadata={'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12,
                        'input_token_details': {'cache_read': 6}})


class GeneralAssistantTests(unittest.TestCase):
    setUp = fixtures.ConversationTests.setUp
    invoke = fixtures.ConversationTests.invoke

    def task(self, code, **args):
        (self.repo/'task.py').write_text(code, encoding='utf-8')
        return reply('run_existing', entry='task.py', args=args.get('argv', []), timeout=args.get('timeout', 10),
                     usage_file=args.get('usage_file', ''))

    def test_fixed_tools_and_aggregate_parameter_feedback(self):
        self.assertEqual(len(CHAT_TOOLS), 14)
        for schema in ASSISTANT_SCHEMA.values():
            self.assertFalse(schema['additionalProperties'])
            self.assertEqual(set(schema['required']), set(schema['properties']))
        with self.assertRaises(ValueError) as errors:
            validate('read_file', {'path': 4, 'offset': True, 'unexpected': 3})
        for value in ('path', 'offset', 'limit', 'reread_reason', 'unexpected'):
            self.assertIn(value, str(errors.exception))

    def test_read_does_not_bootstrap_coding_and_retains_tool_pair(self):
        model = ScriptedModel([reply('read_file', path='first.py', offset=0, limit=80, reread_reason=''), text_reply('VALUE 是1')])
        with patch('agent.server.runtime.ConversationRunner.activate', side_effect=AssertionError('not execution')):
            result = self.invoke(model, '查询文件')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 2)
        self.assertIn('VALUE = 1', model.calls[1][-1].content)
        self.assertEqual(model.calls[1][-1].tool_call_id, 'call-read_file')
        self.assertFalse(result['workspace_bound'])
        self.assertNotIn('PRIVATE_ASSISTANT_REASONING', repr(result))

    def test_additional_read_root_works_without_workspace_but_cannot_run(self):
        docs = self.root/'docs'; docs.mkdir()
        (docs/'note.txt').write_text('用户授权的资料', encoding='utf-8')
        model = ScriptedModel([reply('read_file', path=str(docs/'note.txt'), offset=0, limit=80, reread_reason=''), text_reply()])
        with patch('agent.server.graph.model_factory', return_value=model):
            result = self.compiled.invoke({'repo': '', 'read_roots': [str(docs)],
                'messages': [HumanMessage(content='看资料', id='one')]}, self.config)
        self.assertEqual(result['status'], 'chat_complete')
        self.assertIn('用户授权的资料', model.calls[1][-1].content)
        with self.assertRaises(ValueError):
            command_for(str(self.repo), str(docs/'note.txt'), [])

    def test_source_continuation_only_reuses_visible_unchanged_lines(self):
        path = self.repo/'long.txt'
        path.write_text('\n'.join(str(i) for i in range(100)), encoding='utf-8')
        files = AssistantFiles(str(self.repo))
        first = files.read('long.txt', 0, 60, [])
        history = [{'type': 'tool', 'data': {'content': json.dumps(first)}}]
        second = files.read('long.txt', 0, 80, history)
        self.assertEqual(second['visible_ranges'], [[60, 80]])
        self.assertEqual(files.read('long.txt', 0, 80, [])['visible_ranges'], [[0, 80]])
        path.write_text('Changed\n' + path.read_text(), encoding='utf-8')
        self.assertEqual(files.read('long.txt', 0, 80, history)['visible_ranges'], [[0, 80]])

    def test_path_secret_inline_code_and_application_rejection(self):
        files = AssistantFiles(str(self.repo))
        (self.repo/'.env').write_text('SENSITIVE_KEY=secret')
        outside = self.root/'outside.txt'; outside.write_text('outside')
        for path in (str(outside), '../outside.txt', '.env'):
            with self.assertRaises(ValueError):
                files.path(path)
        for entry in ('python -c print(1)', 'module:os', 'QQ.exe', 'task.py; echo something'):
            with self.assertRaises((ValueError, OSError)):
                command_for(str(self.repo), entry, [])

    def test_existing_run_records_real_failure_and_does_not_initialize_acceptance(self):
        model = ScriptedModel([self.task("print('真实失败')\nraise SystemExit(3)\n"), text_reply('失败，要修复吗')])
        with patch('agent.server.runtime.ConversationRunner.bootstrap_work', side_effect=AssertionError('not coding')):
            result = self.invoke(model, '运行已有任务')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(result['result']['existing_runs'][0]['status'], 'failed')
        self.assertEqual(json.loads(model.calls[1][-1].content)['exit_code'], 3)
        self.assertTrue(result['workspace_bound'])
        self.assertNotIn('verification_spec', result['result'])
        self.assertEqual(result['result']['usage']['total_tokens'], 17)
        self.assertEqual(result['result']['usage']['cached_input_tokens'], 6)
        repeated = self.invoke(model, '运行已有任务')
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(repeated['task_id'], result['task_id'])

    def test_unsolicited_repair_after_run_is_rejected_without_source_execution(self):
        model = ScriptedModel([self.task('raise SystemExit(1)'), investigation(['first.py']), text_reply('请确认是否修复')])
        result = self.invoke(model, '只运行')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertIn('不自动转入源码修复', model.calls[-1][-1].content)
        self.assertTrue(result['workspace_bound'])
        self.assertFalse(list((self.root/'runs/tasks'/result['task_id']).glob('plans/*/plan.json')))
        self.assertEqual((self.repo/'first.py').read_text(), 'VALUE = 1\n')

    def test_timeout_and_subprocess_environment_do_not_leak_provider_credentials(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'TEST_PROVIDER_SECRET'}):
            model = ScriptedModel([self.task("import os,time\nprint(os.getenv('DEEPSEEK_API_KEY','absent'),flush=True)\ntime.sleep(5)", timeout=1), text_reply()])
            result = self.invoke(model, '运行')
        output = json.loads(model.calls[1][-1].content)
        self.assertTrue(output['timed_out'])
        self.assertIn('absent', output['stdout'])
        self.assertNotIn('TEST_PROVIDER_SECRET', repr(result) + repr(model.calls))

    def test_child_usage_is_separate_and_unknown_is_not_zero(self):
        model = ScriptedModel([self.task("import json\nfrom pathlib import Path\nPath('usage.json').write_text(json.dumps(dict(input_tokens=100,output_tokens=20,total_tokens=120,cached_input_tokens=90)))", usage_file='usage.json'), text_reply()])
        result = self.invoke(model, '运行')
        child = result['result']['existing_runs'][0]['child_usage']
        self.assertEqual(child['total_tokens'], 120)
        self.assertEqual(child['status'], 'reported_by_task')
        self.assertEqual(result['result']['usage']['total_tokens'], 17)

    def test_unknown_operation_is_not_replayed(self):
        message = self.task('print(1)')
        def unknown(loop, args, folder):
            loop.state['assistant_process'] = {'pid': 42}
            raise OSError('operation outcome unknown')
        model = ScriptedModel([message])
        with patch('agent.execution.existing.execute', side_effect=unknown) as execute:
            result = self.invoke(model, '运行')
            again = self.invoke(model, '继续', 'new')
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result['status'], 'execution_unknown')
        self.assertEqual(again['status'], 'execution_unknown')

    def test_two_reads_pause_resume_and_public_activity(self):
        calls = [reply('read_file', path=p, offset=0, limit=80, reread_reason='').tool_calls[0]
                 for p in ('first.py', 'second.py')]
        calls[1]['id'] = 'second-read'
        model = ScriptedModel([AIMessage(content='读取两个文件', tool_calls=calls), text_reply()])
        original = AssistantFiles.read
        def paused_read(*args, **kwargs):
            self.store.enqueue(self.thread, 'pause', 'pause')
            return original(*args, **kwargs)
        with patch.object(AssistantFiles, 'read', side_effect=paused_read, autospec=True):
            paused = self.invoke(model, '读两份')
        self.assertEqual(paused['status'], 'paused')
        self.store.enqueue(self.thread, 'resume', 'resume')
        with patch('agent.server.graph.model_factory', return_value=model):
            result = self.compiled.invoke({'messages': []}, self.config)
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len([m for m in model.calls[1] if m.type == 'tool']), 2)
        activity = read_activity(self.root/'runs', self.store.session(self.thread)['activity_tasks'], {})
        operations = {r['id']: r for r in activity['items'] if r['kind'] == 'operation'}
        self.assertEqual(len(operations), 2)
        self.assertTrue(all(r['status'] == 'succeeded' for r in operations.values()))

    def test_rejected_arguments_remain_paired_for_author_correction(self):
        model = ScriptedModel([reply('read_file', path='first.py', offset='wrong', limit=80, reread_reason=''),
            reply('read_file', path='first.py', offset=0, limit=80, reread_reason=''), text_reply()])
        result = self.invoke(model, '查询')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertIn('offset', model.calls[1][-1].content)
        self.assertIn('VALUE = 1', model.calls[2][-1].content)

    def test_query_to_coding_uses_saved_response_and_visible_source(self):
        model = ScriptedModel([reply('read_file', path='first.py', offset=0, limit=80, reread_reason=''),
                               investigation(['first.py']), text_reply()])
        result = self.invoke(model, '调查后进入源码工作')
        self.assertEqual(len(model.calls), 3)
        self.assertTrue(result['workspace_bound'])
        results = list((self.root/'runs'/'tasks'/result['task_id']/'plans').glob('*/*/result.json'))
        self.assertEqual(len(results), 1)
        self.assertTrue(json.loads(results[0].read_text())['data']['already_visible'])

    def test_declared_package_script_and_local_module_keep_argv(self):
        package = self.repo/'package.json'
        package.write_text(json.dumps({'scripts': {'test': 'node existing.js'}}))
        command, _, _ = command_for(str(self.repo), 'package.json#test', ['quoted value', '中文\\path'])
        self.assertEqual(command, ['/usr/bin/env', 'node', '--run', 'test', '--', 'quoted value', '中文\\path'])
        command, _, _ = command_for(str(self.repo), 'module:first', ['--one'])
        self.assertEqual(command[-3:], ['-m', 'first', '--one'])
        with self.assertRaises(ValueError):
            command_for(str(self.repo), 'package.json#missing', [])

    def test_unchanged_child_usage_is_unknown_and_never_reused(self):
        (self.repo/'usage.json').write_text(json.dumps(dict(input_tokens=10, output_tokens=2, total_tokens=12, cached_input_tokens=1)))
        model = ScriptedModel([self.task('print(1)', usage_file='usage.json'), text_reply()])
        result = self.invoke(model, '运行')
        child = result['result']['existing_runs'][0]['child_usage']
        self.assertEqual(child['status'], 'unknown')
        self.assertNotIn('total_tokens', child)

    def test_read_search_and_disabled_web_in_one_reply_remain_paired(self):
        first = reply('list_files', path='.', offset=0, limit=20)
        first.tool_calls.extend([reply('search_files', path='.', query='VALUE', offset=0, limit=20).tool_calls[0],
                                 reply('web_fetch', url='https://example.com').tool_calls[0]])
        model = ScriptedModel([first, text_reply()])
        with patch('agent.execution.web.WebProvider.fetch', side_effect=AssertionError('network disabled')):
            result = self.invoke(model, '查询资料')
        self.assertEqual(result['status'], 'chat_complete')
        self.assertEqual([m.name for m in model.calls[1] if m.type == 'tool'], ['list_files', 'search_files', 'web_fetch'])
        self.assertIn('VALUE', repr(model.calls[1]))

    def test_conflicting_run_and_read_do_not_execute_and_are_both_reported(self):
        first = self.task("from pathlib import Path\nPath('marker').write_text('ran')")
        first.tool_calls.append(reply('list_files', path='.', offset=0, limit=20).tool_calls[0])
        model = ScriptedModel([first, text_reply()])
        self.invoke(model, '查询和运行')
        self.assertFalse((self.repo/'marker').exists())
        self.assertEqual(len([m for m in model.calls[1] if m.type == 'tool']), 2)
