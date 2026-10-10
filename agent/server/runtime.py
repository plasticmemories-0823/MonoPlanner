"""Lazy conversation routing around the existing single-owner work runner."""
import json
from pathlib import Path
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, message_to_dict, messages_from_dict, messages_to_dict

from agent.execution.executor import OperationExecutor
from agent.execution.native import NativeEnvironment
from agent.execution.web import WEB_NAMES, execute_calls
from agent.model.client import configure_call
from agent.model.context import admit
from agent.model import compaction
from agent.source.snapshots import SourceSnapshot
from agent.source.versions import SourceVersions
from agent.storage.history import LocalHistory
from agent.storage import recovery
from agent.storage.records import digest
from agent.task import dialogue
from agent.task.instructions import TASK_RULES, CONVERSATION_RULES, ARTIFACT_RULES
from agent.acceptance.artifact_checks import required_consumer
from agent.task.runner import TaskRunner
from agent.server import assistant


class ConversationRunner(TaskRunner):
    def __init__(self, model, repo, *, phase='decide', read_roots=None, read_sources=None, access_mode='sandbox', conversation_id=None, **kwargs):
        kwargs.pop('interactive', None)
        super().__init__(model, None, interactive=True, yield_boundary=True, **kwargs)
        self.repo, self.phase = repo, phase
        self.read_roots = list(read_roots or [])
        self.read_sources = list(read_sources if read_sources is not None else
                                 ({'path': p, 'kind': 'directory'} for p in self.read_roots))
        self.access_mode = access_mode
        self.conversation_id = conversation_id
        self.mailbox = self.control_source
        self.control_source = lambda: [r for r in self.mailbox() if r['kind'] != 'message']

    def activate(self):
        if not self.repo or not Path(self.repo).is_dir():
            raise ValueError('工作需要选择一个已有仓库；普通聊天无需仓库')
        self.environment = NativeEnvironment(self.repo, artifact_root=self.root.parent.parent/'execution'/self.root.name,
                                             access_mode=self.access_mode, read_sources=self.read_sources)
        if self.root.is_relative_to(self.environment.repo_root) or self.root.is_relative_to(self.environment.artifact_root):
            raise ValueError('Task records must stay outside source and execution mounts')
        self.env_key = recovery.environment_key(self.environment)

    def run(self, history, text, budget, verify_command=None, acceptance=None):
        self.budget, self.public_history = budget, list(history)
        self.root = self.explicit_state
        self.run_started, self.prior_elapsed = time.monotonic(), 0
        self.calls_before = budget.used_steps
        self.reused_result, self.fresh_verification = False, False
        budget.timeout_seconds = budget.timeout_seconds or self.task_timeout
        self.identity = digest([self.repo, text, messages_to_dict(history)])
        self.code_key = recovery.code_key()
        self.history_store = LocalHistory(self.root)
        try:
            with self.history_store.lease():
                stored = self.history_store.load()
                if stored:
                    if stored['identity'] != self.identity or stored['code_key'] != self.code_key:
                        raise ValueError('Task or engine changed; old checkpoint requires its frozen version')
                    if stored.get('runtime_settings') != self.runtime_settings:
                        raise ValueError('Task model/network settings changed; new defaults apply to new tasks only')
                    self.state, self.rules = stored, stored['rules']
                    self.return_status = stored['status']
                    if stored.get('assistant_running'):
                        stored.update(status='execution_unknown', reason='已有操作结果不明，保留现场，不自动重试')
                        return self.result()
                    if stored.get('baseline'):
                        self.activate()
                        return recovery.restore(self, stored, self.identity)
                    recovery.restore_budget(self, stored)
                    if stored.get('request_pending') or stored.get('web_running') or stored.get('web_request_pending'):
                        stored.update(status='model_request_unknown', reason='Interrupted request; no automatic retry')
                        return self.result()
                    if (stored.get('compaction_pending') or {}).get('stage') == 'committing':
                        stored.update(status='compaction_commit_unknown', reason='Interrupted compaction commit; no automatic replay')
                        return self.result()
                    if stored['status'] in {'paused', 'awaiting_user'} and self.mailbox():
                        stored['status'] = 'needs_plan'
                    return self._drive()
                return self._run_new(history, text, budget)
        finally:
            from agent.source.lifecycle import close
            close(self)
            self.history_store.close()

    def _run_new(self, history, text, budget, verify_command=None, acceptance=None):
        self.rules = CONVERSATION_RULES + '\nThe following rules apply ONLY to coding work tools:\n' + TASK_RULES
        self.rules += '\nSelected workspace: ' + (self.repo or '(none)')
        self.rules += '\nAuthorized read-only materials: ' + json.dumps(self.read_sources, ensure_ascii=False)
        self.rules += '\nCurrent user-selected permission: ' + self.access_mode
        self.rules += ('\nExecution environment: Windows; shell commands use cmd.exe, not Bash. '
                       'Use fixed read/search tools for source lookup. For Python execution use the '
                       'explicit executable below; Unix commands and heredocs are not available. '
                       'Sandbox/ask mode uses native LPAC isolation; full mode uses Windows user permissions.')
        from agent.settings import PROJECT_ROOT
        self.rules += '\nPython executable: ' + str(PROJECT_ROOT/'launcher/runtime/python/python.exe')
        self.rules += '\nTask networking: ' + ('enabled' if self.runtime_settings['online'] else 'disabled')
        if required_consumer(text):
            self.rules += ARTIFACT_RULES
        self.state = {'engine': 'MonoPlaner V0.9', 'identity': self.identity, 'code_key': self.code_key,
            'request': text, 'acceptance': None, 'verify_command': None, 'runtime_settings': self.runtime_settings,
            'rules': self.rules, 'history': messages_to_dict([*history, HumanMessage(content=text)]),
            'source': None,
            'read_roots': self.read_roots,
            'read_sources': self.read_sources,
            'access_mode': self.access_mode,
            'status': 'needs_plan', 'plan_count': 0, 'repair_count': 0, 'plan_error': '', 'rounds': [],
            'seen_plans': [], 'pending': None, 'running_step': None, 'request_pending': None}
        self.return_status = 'needs_plan'
        for incoming in self.mailbox():
            if incoming['kind'] == 'message':
                self.state['history'].append(message_to_dict(HumanMessage(content=incoming['text'], id=incoming['id'])))
                self.state.setdefault('applied_inputs', []).append(incoming['id'])
                self.state.setdefault('public_events', []).append({'id': incoming['id'], 'type': 'human', 'content': incoming['text']})
                self.state.setdefault('user_updates', []).append({'id': incoming['id'], 'text': incoming['text']})
        self.persist('conversation_started')
        return self._drive()

    def request_decision(self):
        turn = self.state.get('conversation_turn')
        selected = configure_call(self.model.bind_tools(self.tools), 'conversation' if turn else 'plan')
        if turn:
            if turn.get('stage') == 'requesting' and self.state.get('response_pending'):
                turn['response'] = self.state['response_pending']
                self.state['response_pending'] = turn.get('work_response')
                turn['stage'] = 'saved'
            if turn.get('response'):
                raw = turn['response']
            else:
                messages = [SystemMessage(content=self.rules), *messages_from_dict(turn['messages'])]
                admission = self.state.setdefault('conversation_admission', {})
                admit(admission, messages, self.tools, getattr(selected, 'context_envelope', {}))
                turn.update(stage='requesting', work_response=self.state.get('response_pending'))
                self.persist('conversation_request_prepared')
                raw = self.request_model(messages, 'conversation')
                turn.update(response=raw, stage='saved')
                self.state['response_pending'] = turn.get('work_response')
        else:
            raw = self.state.get('response_pending')
            if not raw:
                compaction.prepare(self, getattr(selected, 'max_output_tokens', 65536))
                messages = self.planner_messages()
                admit(self.state, messages, self.tools, getattr(selected, 'context_envelope', {}))
                raw = self.request_model(messages, 'plan')
        response = messages_from_dict([raw])[0]
        if response.response_metadata.get('finish_reason') in {'length', 'content_filter', 'aborted', 'insufficient_system_resource'}:
            raise ValueError('Incomplete model response; not delivered or executed')
        calls = response.tool_calls
        names = {c['name'] for c in [*calls, *response.invalid_tool_calls]}
        if not names and isinstance(response.content, str) and response.content.strip():
            mode = 'chat'
        elif names and names <= WEB_NAMES and not response.invalid_tool_calls:
            mode = 'web'
        elif names == {'run_existing'}:
            mode = 'run'
        elif names - {t.name for t in self.tools[:4]}:
            mode = 'read'
        else:
            mode = 'work'
        self.state['dispatch'] = mode
        self.persist('response_routed', {'mode': mode})

    def start_turn(self):
        old = set(self.state.get('applied_inputs', []))
        incoming = next((r for r in self.mailbox() if r['kind'] == 'message' and r['id'] not in old), None)
        if not incoming or self.state.get('conversation_turn'):
            return
        public = [*self.public_history, HumanMessage(content=self.state['request'])]
        public.extend((HumanMessage if r['type'] == 'human' else AIMessage)(
            content=r['content'], id=r['id']) for r in self.state.get('public_events', []))
        brief = {'status': self.return_status, 'source': self.state.get('source'),
                 'last_outcome': self.state.get('last_result', {}).get('outcome'),
                 'external_results': self.state.get('desktop_results', [])}
        public.append(HumanMessage(content='Current work state (not an execution instruction): ' + json.dumps(brief)))
        prefix = len(public)
        public.append(HumanMessage(content=incoming['text'], id=incoming['id']))
        self.state['conversation_turn'] = {'id': incoming['id'], 'text': incoming['text'], 'prefix': prefix,
            'messages': messages_to_dict(public), 'return_status': self.return_status, 'work_response': self.state.get('response_pending')}
        self.state.pop('conversation_admission', None)
        self.persist('conversation_input_saved', {'id': incoming['id']})

    def complete_turn(self):
        turn = self.state['conversation_turn']
        self.state.setdefault('applied_inputs', []).append(turn['id'])
        self.state.setdefault('public_events', []).append({'id': turn['id'], 'type': 'human', 'content': turn['text']})
        return turn

    def on_model_interrupted(self, kind):
        turn = self.state.get('conversation_turn')
        if turn:
            turn = self.complete_turn()
            self.state['history'].extend(turn['messages'][turn['prefix']:])
            self.state.update(response_pending=turn.get('work_response'), conversation_turn=None)
        super().on_model_interrupted(kind)
        if kind == 'message' and not self.state.get('baseline') and not self.state.get('pending'):
            # The first reply was cancelled before work started. Answer the
            # saved new message without silently restarting the old request.
            self.state['status'] = 'awaiting_user'
            self.persist('interrupted_conversation_waiting_for_input')

    def bootstrap_work(self):
        if self.state.get('baseline'):
            return
        self.activate()
        self.env_key = recovery.environment_key(self.environment)
        self.baseline = SourceSnapshot.capture(self.environment.repo_root, self.root/'baseline')
        self.evidence = SourceVersions(self.environment, self.root/'source-records', baseline=self.baseline)
        self.executor = OperationExecutor(self.environment, self.evidence, self.root, versioned_reads=self.optimize_source)
        self.executor.archive = self.history_store
        self.state.update(baseline=str(self.baseline.manifest_path), source=self.evidence.current, environment_key=self.env_key)
        self.persist('work_initialized')

    def advance(self):
        self.budget.check_request(0)
        approvals = self.state.get('desktop_approvals', {})
        if approvals:
            from agent.execution.desktop import DesktopStore
            from agent.storage.preferences import state_root
            for record in DesktopStore(state_root()).list(self.conversation_id):
                if record['id'] not in approvals or record['status'] == approvals[record['id']]:
                    continue
                if record['status'] in {'unknown', 'launching', 'executing'}:
                    self.state.update(status='execution_unknown', reason='外部操作结果不明，不自动重试')
                    self.persist('desktop_result_unknown')
                    return
                result = {k: record.get(k) for k in ('id', 'kind', 'status', 'path', 'destination', 'error', 'backup')}
                self.state.setdefault('desktop_results', []).append(result)
                self.state['history'].append(message_to_dict(HumanMessage(content=
                    'Trusted backend external-operation result: ' + json.dumps(result, ensure_ascii=False))))
                approvals[record['id']] = record['status']
                self.persist('desktop_result_received', result)
        if self.state.get('access_mode') != self.access_mode:
            self.state['access_mode'] = self.access_mode
            self.state['history'].append(message_to_dict(HumanMessage(content=
                'Current user-selected permission is now: ' + self.access_mode)))
            self.persist('permission_updated')
        if self.state.get('read_sources', []) != self.read_sources:
            self.state['read_roots'] = self.read_roots
            self.state['read_sources'] = self.read_sources
            self.state['history'].append(message_to_dict(HumanMessage(content=
                'User-selected read-only materials updated: ' + json.dumps(self.read_sources, ensure_ascii=False))))
            self.persist('read_access_updated')
        # Explicit pause/resume is handled without a model call. Messages are
        # classified by the ordinary first reply, before altering pending work.
        if dialogue.apply_controls(self):
            return
        self.start_turn()
        turn = self.state.get('conversation_turn')
        if turn and self.phase == 'work' and not self.state.get('dispatch'):
            self.state.update(status='needs_plan')
            return
        if turn and self.phase in {'assistant', 'web', 'chat'} and not turn.get('response'):
            self.state.pop('dispatch', None)
            self.state.update(status='needs_plan')
            return
        if self.phase == 'decide':
            if self.state.get('pending') and not turn:
                return
            self.request_decision()
            return
        self.state.pop('dispatch', None)
        raw = turn.get('response') if turn else self.state.get('response_pending')
        if self.phase == 'chat':
            if turn:
                turn = self.complete_turn()
                dialogue.publish(self, raw)
                self.state['response_pending'] = turn.get('work_response')
                status = turn['return_status']
                if status == 'awaiting_user' and not self.state.get('baseline'):
                    status = 'chat_complete'
                self.state.update(status=status, conversation_turn=None, turn_status='chat_complete')
                self.persist('conversation_reply_saved')
            else:
                dialogue.accept_text(self, raw)
                self.state['turn_status'] = 'chat_complete'
            return
        if self.phase in {'web', 'assistant'}:
            target = turn['messages'] if turn else None
            if self.phase == 'assistant':
                raw = (self.state.get('assistant_pending') or {}).get('response') or raw
                dialogue.publish(self, raw)
                assistant.execute_calls(self, raw, target)
            else:
                execute_calls(self, raw, target)
            if turn and not self.state.get('web_pending') and not self.state.get('assistant_pending'):
                turn.pop('response', None)
                turn.pop('stage', None)
                self.state['response_pending'] = turn.get('work_response')
            return
        turn_id = turn['id'] if turn else 'initial'
        if self.state.get('assistant_run_turn') == turn_id and raw:
            target = turn['messages'] if turn else self.state['history']
            target.append(raw)
            for call in raw['data'].get('tool_calls', []):
                target.append(message_to_dict(ToolMessage(content=json.dumps({'error':
                    '已有任务运行不自动转入源码修复。请报告结果；用户明确要求修复后再开始编码。'}, ensure_ascii=False),
                    name=call['name'], tool_call_id=call['id'])))
            if turn:
                turn.pop('response', None)
                turn.pop('stage', None)
            self.state.update(response_pending=turn.get('work_response') if turn else None, status='needs_plan')
            self.persist('unsolicited_repair_rejected')
            return
        if turn:
            turn = self.complete_turn()
            if self.state.get('pending'):
                self.finish_plan('superseded_by_user')
            dialogue.discard_unexecuted_response(self)
            self.state['history'].extend(turn['messages'][turn['prefix']:])
            self.state.setdefault('user_updates', []).append({'id': turn['id'], 'text': turn['text']})
            self.state.update(response_pending=raw, conversation_turn=None, status='needs_plan', plan_error='')
            self.state.pop('acceptance_release', None)
            self.state.pop('repair_draft', None)
            self.persist('user_input_applied', {'id': turn['id']})
        self.state.pop('turn_status', None)
        response = messages_from_dict([raw])[0] if raw else None
        if raw:
            # Public commentary must survive preparation errors as well as
            # successful tool execution; publish deduplicates the call ID.
            dialogue.publish(self, raw)
        blocked = response and len(response.tool_calls) == 1 and response.tool_calls[0]['name'] == 'submit_work' and response.tool_calls[0]['args'].get('kind') == 'blocked'
        if not self.state.get('baseline') and not blocked:
            if not self.repo or not Path(self.repo).is_dir():
                self.state['history'].append(raw)
                for call in [*response.tool_calls, *response.invalid_tool_calls]:
                    self.state['history'].append(message_to_dict(ToolMessage(
                        content=json.dumps({'error': 'workspace_required', 'detail': '请先选择已有仓库；操作未执行。普通聊天不需要仓库。'}),
                        name=call['name'], tool_call_id=call['id'])))
                self.state.update(response_pending=None, status='needs_plan')
                self.persist('workspace_required')
                return
            self.bootstrap_work()
        super().advance()

    def result(self):
        if self.state.get('assistant_pending'):
            self.state['dispatch'] = 'read'
        if self.state.get('web_pending'):
            self.state['dispatch'] = 'web'
        result = super().result()
        result['dispatch'] = self.state.get('dispatch')
        result['turn_status'] = self.state.get('turn_status')
        return result
