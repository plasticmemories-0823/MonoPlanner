"""工作工具在原件保存后整理展示；续读沿用相同的命中去重规则。"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, message_to_dict

from agent.acceptance.revisions import append_result
from agent.execution.executor import OperationExecutor
from agent.execution.web import execute_calls
from agent.model.tool_results import normalize_result
from agent.source.exploration import ExplorationMap
from agent.storage.history import LocalHistory
from agent.task.budget import StepBudget
from agent.task.contracts import Operation
from agent.task.operations import finish_plan


class WorkResultNormalizationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.environment = SimpleNamespace(repo_root=self.repo)
        self.loop = SimpleNamespace(
            root=self.root,
            state={'history': [], 'source': 'a' * 64},
            environment=self.environment,
            persist=lambda *args: None,
        )

    def response(self, name='submit_plan', identity='plan-call'):
        return message_to_dict(AIMessage(content='', tool_calls=[{
            'id': identity, 'name': name, 'args': {},
        }]))

    def hits(self):
        return [
            {'path': 'a.py', 'line': 4, 'text': 'old'},
            {'path': 'b.py', 'line': 5, 'text': 'other'},
            {'path': 'a.py', 'line': 4, 'text': 'new'},
        ]

    def test_plan_result_is_saved_before_normalization_and_keeps_operation_envelopes(self):
        row = {'id': 'find', 'op': 'search', 'status': 'succeeded', 'before': 'b' * 64,
               'data': {'hits': self.hits(), 'truncated': False}}
        rows = [row, {**deepcopy(row), 'id': 'another-find'}]
        self.loop.state['pending'] = {
            'id': 'p1', 'response': self.response('repair_work'), 'rows': rows,
        }
        original = deepcopy(rows)
        saved = self.root / 'plans/p1/result.json'

        def normalize(name, value, **kwargs):
            self.assertEqual(json.loads(saved.read_bytes()), value)
            return normalize_result(name, value, **kwargs)

        with patch('agent.model.tool_results.normalize_result', side_effect=normalize):
            finish_plan(self.loop, 'inspected')
        raw = json.loads(saved.read_bytes())
        self.assertEqual(raw['steps'], original)
        self.assertEqual(self.loop.state['last_result'], raw)
        delivered = self.loop.state['history'][-1]['data']
        self.assertEqual(delivered['tool_call_id'], 'plan-call')
        self.assertEqual(delivered['name'], 'repair_work')
        result = json.loads(delivered['content'])
        self.assertEqual([row['id'] for row in result['steps']], ['find', 'another-find'])
        self.assertEqual(result['steps'][0]['data']['hits'], self.hits()[1:])
        self.assertEqual(result['steps'][1]['data']['hits'], self.hits()[1:])
        self.assertEqual(rows, original)

    def test_acceptance_result_and_duplicate_test_results_remain_unchanged(self):
        result = {
            'outcome': 'acceptance_preflight', 'trial_id': 'trial',
            'baseline': {'status': 'failed', 'test_results': [
                {'name': 'same_name', 'status': 'passed'},
                {'name': 'same_name', 'status': 'failed'},
            ]},
        }
        original = deepcopy(result)
        append_result(self.loop, self.response('revise_acceptance'), result)
        delivered = self.loop.state['history'][-1]['data']
        self.assertEqual(json.loads(delivered['content']), original)
        self.assertEqual(delivered['tool_call_id'], 'plan-call')
        self.assertEqual(result, original)

    def test_real_import_dependencies_survive_saved_result_and_model_tool_message(self):
        source = b'from dep import alpha, beta\n'
        files = {'consumer.py': source, 'dep.py': b'alpha = 1\nbeta = 2\n'}
        digests = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
        reader = SimpleNamespace(
            inventory=lambda: digests,
            evidence=SimpleNamespace(current='a' * 64),
            prepared=None,
            source_bytes=lambda name: files[name],
        )
        mapping = ExplorationMap(reader, {})
        self.loop.executor = SimpleNamespace(exploration=mapping)
        data = {'path': 'consumer.py', 'sha256': digests['consumer.py'],
                'text': source.decode(), 'visible_ranges': [[0, 1]], 'offset': 0}
        data['repository_map'] = mapping.observe(data)
        expected = data['repository_map']['updates'][0]['new_dependencies']
        self.assertEqual([row['import_target'] for row in expected],
                         ['dep.alpha', 'dep.beta'])
        self.assertEqual(normalize_result('retrieve_evidence', data), data)
        row = {'id': 'read-imports', 'op': 'read', 'status': 'succeeded', 'data': data}
        self.loop.state['pending'] = {
            'id': 'p1', 'response': self.response(), 'rows': [row],
        }
        prefix = [self.response('submit_plan', 'earlier-call'), {
            'type': 'tool', 'data': {'content': 'previous result',
                                     'tool_call_id': 'earlier-call'},
        }]
        self.loop.state['history'] = deepcopy(prefix)
        with patch('agent.model.tool_payloads.project', side_effect=OSError('projection failed')):
            with self.assertRaisesRegex(OSError, 'projection failed'):
                finish_plan(self.loop, 'inspected')
        self.assertEqual(mapping.state['consumer.py']['delivered']['ids'], [])
        self.assertEqual(self.loop.state['history'], prefix)
        finish_plan(self.loop, 'inspected')
        self.assertEqual(mapping.state['consumer.py']['delivered']['ids'],
                         ['import:1:0:0', 'import:1:0:1'])
        self.assertEqual(self.loop.state['history'][:-2], prefix)
        message = self.loop.state['history'][-1]['data']
        self.assertEqual(message['tool_call_id'], 'plan-call')
        delivered = json.loads(message['content'])
        saved = json.loads((self.root / 'plans/p1/result.json').read_bytes())
        for result in (delivered, saved):
            self.assertEqual(
                result['steps'][0]['data']['repository_map']['updates'][0]['new_dependencies'],
                expected,
            )

    def test_from_step_and_display_use_same_latest_hits_before_five_hit_limit(self):
        hits = [self.hits()[0]] * 6 + [
            {'path': f'file-{index}.py', 'line': 10, 'text': str(index)}
            for index in range(6)
        ] + [self.hits()[-1]]
        raw = b'fixture content'
        sha = hashlib.sha256(raw).hexdigest()
        hits = [{**hit, 'version': sha, 'source_sha256': sha} for hit in hits]
        prior = {'find': {'id': 'find', 'op': 'search', 'before': 'b' * 64,
                          'data': {'hits': hits, 'truncated': False}}}
        original = deepcopy(prior)
        reader = OperationExecutor(self.environment, None, self.root)
        reader.inventory = lambda: {hit['path']: sha for hit in hits}
        reader.source_bytes = lambda _: raw
        shown = normalize_result(
            'submit_plan', {'steps': [prior['find']]}, source_version='c' * 64,
            project=self.repo,
        )['steps'][0]['data']['hits']

        def read(path, offset, limit, repeat, *, budget):
            return {'path': path, 'offset': offset, 'sha256': sha,
                    'text': path, 'visible_ranges': [], 'truncated': False}

        with patch.object(reader, 'read_continuation', side_effect=read) as read_call:
            result = reader.execute(
                Operation(id='read', op='read', from_step='find', limit=3),
                prior, self.root, 10,
            )
        self.assertEqual([call.args[0] for call in read_call.call_args_list],
                         [hit['path'] for hit in shown[:5]])
        self.assertEqual(read_call.call_count, 5)
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(prior, original)

    def test_from_step_without_source_identity_rejects_before_reading(self):
        prior = {'find': {'data': {'hits': self.hits()}}}
        reader = OperationExecutor(self.environment, None, self.root)
        page = {'path': 'a.py', 'sha256': 'd' * 64, 'text': '',
                'visible_ranges': [], 'truncated': False}
        with patch.object(reader, 'read_continuation', return_value=page) as read_call:
            with self.assertRaisesRegex(ValueError, 'source version'):
                reader.execute(Operation(id='read', op='read', from_step='find'),
                               prior, self.root, 10)
        self.assertEqual(read_call.call_count, 0)

    def test_distinct_old_source_versions_remain_visible_but_cannot_drive_new_reads(self):
        hits = [
            {**self.hits()[0], 'source_version': 'b' * 64},
            {**self.hits()[0], 'source_version': 'c' * 64},
            {**self.hits()[-1], 'source_version': 'b' * 64},
        ]
        prior = {'find': {'before': 'a' * 64, 'data': {'hits': hits}}}
        original = deepcopy(prior)
        shown = normalize_result('retrieve_evidence', prior['find']['data'])
        self.assertEqual(len(shown['hits']), 2)
        reader = OperationExecutor(self.environment, None, self.root)
        page = {'path': 'a.py', 'sha256': 'd' * 64, 'text': '',
                'visible_ranges': [], 'truncated': False}
        with patch.object(reader, 'read_continuation', return_value=page) as read_call:
            with self.assertRaisesRegex(ValueError, 'source version'):
                reader.execute(Operation(id='read', op='read', from_step='find'),
                               prior, self.root, 10)
        self.assertEqual(read_call.call_count, 0)
        self.assertEqual(prior, original)

    def test_web_original_is_saved_before_normalization_and_pairing_is_unchanged(self):
        store = LocalHistory(self.root)
        lease = store.lease()
        lease.__enter__()
        self.addCleanup(store.close)
        self.addCleanup(lease.__exit__, None, None, None)
        self.loop.history_store = store
        self.loop.budget = StepBudget()
        self.loop.runtime_settings = {'online': True}
        self.loop.yield_boundary = False
        response = message_to_dict(AIMessage(content='', tool_calls=[{
            'id': 'web-call', 'name': 'web_search', 'args': {'query': 'offline'},
        }]))
        raw = {'sources': [
            {'url': 'https://example.com/a', 'title': 'old'},
            {'url': 'https://example.com/b', 'title': 'other'},
            {'url': 'https://example.com/a', 'title': 'new'},
        ], 'truncated': False}
        original = deepcopy(raw)
        saved = self.root / 'web/call-0000-00/result.json'

        def normalize(name, value, **kwargs):
            self.assertEqual(json.loads(saved.read_bytes()), {'status': 'succeeded', 'data': raw})
            return normalize_result(name, value, **kwargs)

        with patch('agent.execution.web.WebProvider.search', return_value=raw), patch(
            'agent.execution.web.normalize_result', side_effect=normalize,
        ):
            execute_calls(self.loop, response)
        message = self.loop.state['history'][-1]['data']
        self.assertEqual(message['tool_call_id'], 'web-call')
        self.assertEqual(message['name'], 'web_search')
        self.assertEqual(json.loads(message['content'])['sources'], raw['sources'][1:])
        self.assertEqual(raw, original)
