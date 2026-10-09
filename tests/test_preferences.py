"""Settings persist without exposing credentials or changing active request policy."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage
from starlette.testclient import TestClient

from agent.storage.preferences import DEFAULTS, Preferences, protect
from agent.model.client import create_model
from agent.model.tools import CHAT_TOOLS


class PreferencesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.preferences = Preferences(self.temp.name)
        environment = patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-environment'})
        environment.start()
        self.addCleanup(environment.stop)

    def test_real_windows_encryption_roundtrip_and_blank_preserves_key(self):
        if os.name != 'nt':
            self.skipTest('Windows DPAPI')
        value = 'offline-secret-中文-not-a-real-key'
        public = self.preferences.update({'api_key': value, 'reasoning_effort': 'low', 'online': True})
        self.assertEqual(self.preferences.key(), value)
        self.assertNotIn(value, repr(public))
        self.assertNotIn(value.encode(), self.preferences.secret.read_bytes())
        self.assertNotIn('api_key', self.preferences.path.read_text())
        self.preferences.update({'api_key': ''})
        self.assertEqual(self.preferences.key(), value)
        self.assertEqual(self.preferences.delete_key()['key_source'], 'environment')
        self.assertEqual(self.preferences.key(), 'offline-environment')

    def test_invalid_policy_does_not_replace_existing_defaults_or_secret(self):
        self.preferences.update({'reasoning_effort': 'max'})
        before = self.preferences.path.read_bytes()
        for value in [{'model_name': 'pro'}, {'reasoning_effort': 'medium'}, {'online': 'yes'}, {'unknown': 1}, {'api_key': 'bad\nkey'}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.preferences.update(value)
        self.assertEqual(self.preferences.path.read_bytes(), before)
        self.assertFalse(self.preferences.secret.exists())

    def test_model_settings_use_saved_defaults_and_environment_fallback(self):
        from agent.settings import load_settings
        self.preferences.update({'reasoning_effort': 'none'})
        with patch('agent.storage.preferences.state_root', return_value=Path(self.temp.name)):
            value = load_settings()
            self.assertEqual(value['reasoning_effort'], 'none')
            self.assertEqual(value['api_key'], 'offline-environment')
            self.assertEqual(load_settings(DEFAULTS)['reasoning_effort'], 'high')

    def test_http_configuration_is_redacted_and_rejects_external_origin(self):
        from agent.server.app import app
        with patch('agent.server.app.Preferences', return_value=self.preferences), TestClient(app) as client:
            response = client.post('/settings', json={'reasoning_effort': 'low'}, headers={'origin': 'http://127.0.0.1:3000'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['reasoning_effort'], 'low')
            self.assertNotIn('offline-environment', response.text)
            self.assertEqual(response.headers['cache-control'], 'no-store')
            rejected = client.post('/settings', json={'online': True}, headers={'origin': 'https://external.example'})
            self.assertEqual(rejected.status_code, 403)
            self.assertFalse(self.preferences.defaults()['online'])
            self.assertEqual(client.post('/settings', content='bad').status_code, 415)

    def test_settings_filesystem_setup_never_runs_on_event_loop(self):
        import asyncio
        from agent.server.app import app
        def factory():
            with self.assertRaises(RuntimeError):
                asyncio.get_running_loop()
            return self.preferences
        with patch('agent.server.app.Preferences', side_effect=factory), TestClient(app) as client:
            self.assertEqual(client.get('/settings').status_code, 200)
            self.assertEqual(client.post('/settings', json={'online': True}).status_code, 200)
            self.assertEqual(client.delete('/settings/api-key').status_code, 200)

    def test_four_efforts_reach_real_sdk_wire_and_tools_stay_identical(self):
        bodies = []
        def respond(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={'id': 'offline', 'object': 'chat.completion', 'created': 1,
                'model': 'deepseek-flash', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'offline'}}]})
        for effort in ('none', 'low', 'high', 'max'):
            with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
                    patch('agent.model.network.model_http_clients', return_value={'http_client': client}):
                model = create_model({'model_name': 'deepseek-flash', 'api_key': 'offline', 'reasoning_effort': effort}).bind_tools(CHAT_TOOLS)
                model.invoke([HumanMessage(content='hello')])
        self.assertEqual(bodies[0]['thinking']['type'], 'disabled')
        self.assertNotIn('reasoning_effort', bodies[0])
        for body, effort in zip(bodies[1:], ('low', 'high', 'max')):
            self.assertEqual(body['reasoning_effort'], effort)
            self.assertEqual(body['thinking']['type'], 'enabled')
        self.assertTrue(all(body['tools'] == bodies[0]['tools'] for body in bodies))
        self.assertEqual([t['function']['name'] for t in bodies[0]['tools']],
                             ['submit_work', 'repair_work', 'freeze_acceptance', 'revise_acceptance', 'web_search', 'web_fetch',
                              'list_files', 'read_file', 'search_files', 'run_existing'])
