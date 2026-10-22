"""后台调度与公开进度；浏览器断开不会创建、重复或取消工作请求。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
import logging
import sqlite3
import uuid

from agent.application.chat import Chatbot
from agent.application.store import ACTIVE, TERMINAL, RESUMABLE, SUPPLEMENTABLE, Conflict
from agent.application.lease import ServiceLease
from agent.application.store import identity, encode


class ServiceClosing(Conflict):
    """退出已锁住派发；重新打开后也不会自动重发被取消的请求。"""


class StorageUnavailable(OSError):
    """关键记录失败后只能读取与停止，不能继续派发。"""


KEY_EVENTS = {
    'environment_preparation', 'plan_accepted', 'plan_rejected', 'plan_finished',
    'acceptance_trial_finished', 'final_acceptance_finished', 'user_input_applied',
    'paused', 'task_returned', 'work_state', 'environment',
    'acceptance_limit_reached', 'acceptance_limit_continued',
}


class Scheduler:
    def __init__(self, store, *, work_factory=None, pause_callback=None, model_factory=None,
                 tool_invoker=None, title_factory=None,
                 supplement_callback=None):
        self.store = store
        self.work_factory = work_factory or self._run_work
        self.pause_callback = pause_callback or self._pause_work
        self.supplement_callback = supplement_callback or self._supplement_work
        self.title_factory = title_factory
        self.chatbot = Chatbot(store, self.dispatch_work, factory=model_factory,
                              invoke_tool=tool_invoker)
        self.lock = Lock()
        self.chats = set()
        self.works = set()
        self.titles = set()
        self.controls_inflight = {}
        self.active_jobs = {}
        self.storage_fault = None
        self.emergency_controls = {}
        self.report_task = None
        self.reporting = False
        self.stop = Event()
        self.wake = Event()
        self.chat_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='application-chat')
        self.work_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='application-work')
        self.report_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='application-progress')
        self.title_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='application-title')
        self.control_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='application-control')
        self.worker = None
        self.starting = False
        self.lease = None
        from agent.application.shutdown import Shutdown

        self.shutdown = Shutdown(self)

    def storage_error(self):
        with self.lock:
            return dict(self.storage_fault) if self.storage_fault else None

    def storage_failed(self, operation, error):
        """健康闭锁不代替任务记录；只阻止下一步，并让界面知道记录无法更新。"""
        with self.lock:
            if self.storage_fault is None:
                self.storage_fault = {
                    'code': 'storage_unavailable', 'operation': operation,
                    'message': '本地运行记录写入失败，已停止新任务和模型请求；请释放空间后核对现场。',
                    'dispatch_suspended': True, 'id': 'storage-fault:' + uuid.uuid4().hex,
                }
            fault = dict(self.storage_fault)
            owners = [*self.active_jobs, *('chat:' + value for value in self.chats),
                      *('title:' + value for value in self.titles)]
            if self.report_task:
                owners.append('progress:' + self.report_task)
        from agent.task.control import interrupt

        for owner in owners:
            try:
                interrupt(self.store.root, owner, 'pause', fault['id'])
            except Exception:
                logging.getLogger(__name__).exception('Storage-fault cancellation was not confirmed')
        self.wake.set()
        return fault

    def ensure_dispatch_available(self):
        if self.stop.is_set():
            raise ServiceClosing('MonoPlanner 正在退出，已停止接收新操作。')
        if self.storage_error():
            raise StorageUnavailable(self.storage_error()['message'])

    def _pause_identity(self, conversation, command):
        if set(command) != {'id', 'kind'} or command['kind'] != 'pause':
            raise ValueError('Pause accepts only its command identity')
        conversation, command_id = identity(conversation), identity(command['id'])
        try:
            value = self.store.get(conversation)
            if value.get('read_only') or value.get('visibility') != 'active':
                raise Conflict('只读、已归档或删除中的会话不允许控制；请先核对会话。')
            with self.store.connection() as db:
                old = db.execute('SELECT body,state FROM app_commands WHERE conversation=? AND id=?',
                                 (conversation, command_id)).fetchone()
                if old and old['body'] != encode({'id': command_id, 'kind': 'pause'}):
                    raise Conflict('Command identity was reused for different content')
                dispatched = old and db.execute("SELECT 1 FROM app_events WHERE conversation=? "
                    "AND kind='control_dispatched' AND json_extract(payload,'$.id')=?",
                    (conversation, command_id)).fetchone()
            task = self.store.work(value['work_id'])
            if task and task['conversation_id'] != conversation:
                raise Conflict('Task does not belong to this conversation')
            receipt = ({'received': True, 'id': command_id, 'applied': old['state'] == 'applied'}
                       if old and (old['state'] == 'applied' or dispatched) else None)
            return conversation, command_id, task['task_id'] if task else None, receipt
        except (OSError, sqlite3.Error) as exc:
            self.storage_failed('pause_identity', exc)
            with self.lock:
                task = next((key for key, thread in self.active_jobs.items()
                             if thread == conversation), None)
                if task or conversation in self.chats:
                    return conversation, command_id, task, None
            raise StorageUnavailable('无法核对这条会话的执行归属，未操作任何其他任务。') from exc

    def _emergency_pause(self, target):
        conversation, command_id, task = target[:3]
        with self.lock:
            old = self.emergency_controls.get(command_id)
            if old is not None and old != (conversation, task):
                raise Conflict('Control identity was reused for another conversation')
            self.emergency_controls[command_id] = (conversation, task)
            if old is None and task:
                self.controls_inflight[command_id] = task
        if old is None and task:
            self.control_pool.submit(self._control, conversation, task,
                                     {'id': command_id, 'kind': 'pause'}, self.pause_callback, False)
        return {'received': False, 'id': command_id, 'applied': False, 'recorded': False,
                'control_requested': True, 'termination_confirmed': False,
                'message': '已请求终止当前操作，但记录保存失败；停止结果仍需核对。',
                'storage_error': self.storage_error(), 'status': 'control_unconfirmed', 'event_cursor': 0}

    @staticmethod
    def _run_work(job, emit, controls):
        from agent.runtime.bridge import run_work

        return run_work(job, emit, controls)

    def _pause_work(self, task, command):
        from agent.runtime.bridge import pause_work

        return pause_work(task, command, root=self.store.root)

    def _supplement_work(self, task, command):
        from agent.runtime.bridge import supplement_work

        return supplement_work(task, command, root=self.store.root)

    def start(self):
        with self.lock:
            if self.worker is not None or self.starting or self.stop.is_set():
                return
            self.starting = True
        lease = ServiceLease(self.store.root)
        try:
            lease.acquire()
            with self.lock:
                self.lease = lease
            self.store.recover_interrupted()
            from agent.runtime.bridge import pause_work

            with self.store.connection() as db:
                interrupted_tasks = [row[0] for row in db.execute('SELECT id FROM app_work_state')]
            for task_id in interrupted_tasks:
                task = self.store.work(task_id)
                if task and task['status'] in {'interrupted', 'execution_unknown', 'pausing'}:
                    # 启动只收拢遗留执行，不续跑任务。已有资源协议核验归属后才停止。
                    self._control(task['conversation_id'], task['task_id'],
                                  {'id': 'startup-stop-' + uuid.uuid4().hex, 'kind': 'pause'},
                                  lambda task_id, command_id: pause_work(
                                      task_id, command_id, root=self.store.root))
            with self.lock:
                if not self.stop.is_set():
                    self.worker = Thread(target=self._loop, name='application-dispatch', daemon=True)
                    self.worker.start()
        except Exception:
            lease.close()
            with self.lock:
                if self.lease is lease:
                    self.lease = None
            raise
        finally:
            with self.lock:
                self.starting = False
            self._release_if_idle()

    def close(self):
        self.shutdown.start()
        self.shutdown.finished.wait(timeout=2)
        self._release_if_idle()

    def _release_if_idle(self):
        with self.lock:
            if (self.shutdown.finished.is_set() and self.shutdown.controls_idle()) and not (
                self.starting or self.chats or self.works or self.titles or self.reporting
                or self.controls_inflight
            ):
                if self.lease:
                    self.lease.close()
                    self.lease = None

    def submit(self, conversation, command):
        if self.stop.is_set():
            raise ServiceClosing('MonoPlanner 正在退出，已停止接收新操作。')
        target = None
        if isinstance(command, dict) and command.get('kind') == 'pause':
            target = self._pause_identity(conversation, command)
            if target[3]:
                return target[3]
            from agent.task.control import interrupt

            interrupt(self.store.root, 'chat:' + target[0], 'pause', target[1])
            if target[2]:
                interrupt(self.store.root, target[2], 'pause', target[1])
        if self.storage_error():
            if target:
                return self._emergency_pause(target)
            self.ensure_dispatch_available()
        try:
            return self._submit_saved(conversation, command)
        except (OSError, sqlite3.Error) as exc:
            self.storage_failed('command_save', exc)
            if target:
                return self._emergency_pause(target)
            raise StorageUnavailable(self.storage_error()['message']) from exc

    def _submit_saved(self, conversation, command):
        if self.stop.is_set():
            raise ServiceClosing('MonoPlanner 正在退出，已停止接收新操作。')
        result = self.store.submit(conversation, command)
        if command['kind'] == 'message' and not result['applied']:
            try:
                self._reserve_title(conversation, command)
            except (OSError, sqlite3.Error):
                raise
            except Exception:
                logging.getLogger(__name__).exception('Title unavailable; saved message continues')
        if command['kind'] in {'pause', 'supplement'} and not result['applied']:
            task = self.store.get(conversation)['work_id']
            if task and self.store.reserve_control(conversation, task, command):
                callback = self.pause_callback if command['kind'] == 'pause' else self.supplement_callback
                with self.lock:
                    self.controls_inflight[command['id']] = task
                self.control_pool.submit(self._control, conversation, task, command, callback)
            if command['kind'] == 'pause':
                from agent.task.control import interrupt

                interrupt(self.store.root, 'chat:' + conversation, 'pause', command['id'])
        self.wake.set()
        return result

    def _control(self, conversation, task, command, callback, persist=True):
        try:
            result = callback(task, command['id'])
            if not persist:
                return
            termination = result.get('termination', []) if isinstance(result, dict) else []
            confirmed = all(row.get('stopped') is True for row in termination)
            if not confirmed and command['kind'] == 'pause':
                self.store.control_unconfirmed(task, command['id'])
            self.store.record_event(conversation,
                                    'control_delivered' if confirmed else 'control_unconfirmed',
                                    {'id': command['id'], 'kind': command['kind'], 'task_id': task})
        except Exception as error:
            logging.getLogger(__name__).exception('Immediate control could not be confirmed')
            if isinstance(error, (OSError, sqlite3.Error)):
                self.storage_failed('control_record', error)
            if persist:
                try:
                    if command['kind'] == 'pause':
                        self.store.control_unconfirmed(task, command['id'])
                    self.store.record_event(conversation, 'control_unconfirmed',
                                            {'id': command['id'], 'kind': command['kind'], 'task_id': task})
                except (OSError, sqlite3.Error) as exc:
                    self.storage_failed('control_result', exc)
        finally:
            with self.lock:
                self.controls_inflight.pop(command['id'], None)
            self._release_if_idle()
            self.wake.set()

    def dispatch_work(self, conversation, command, *, kind='coding', arguments=None,
                      work_options=None, project_path='.', create=False):
        self.ensure_dispatch_available()
        value = self.store.get(conversation)
        previous = self.store.work(value['work_id'])
        if previous and previous.get('input_id') == command['id']:
            return self.store.create_work(conversation, command['id'], kind=kind,
                                          arguments=arguments, work_options=work_options,
                                          project_path=project_path, create=create)
        if previous and previous['status'] not in TERMINAL:
            if kind != 'coding':
                raise Conflict('当前工作尚未结束，不能同时运行另一任务')
            from agent.application.project_location import resolve_location

            selected = resolve_location(value['repo'], project_path, create)
            if create or selected['repo'] != previous['repo']:
                raise Conflict('已有工作仍绑定原项目；补充不能创建或切换项目。')
            control = {'id': str(uuid.uuid5(uuid.UUID(command['id']), 'supplement')),
                       'kind': 'supplement', 'text': command['text']}
            if (work_options or {}).get('confirmed_requirements'):
                packet = work_options['confirmed_requirements']
                from agent.application.requirements import latest

                old_sources = {r['id'] for r in (latest(self.store, conversation) or {}).get('sources', [])}
                control['text'] = '\n\n'.join(r['text'] for r in packet['sources'] if r['id'] not in old_sources)
                control['confirmed_requirements'] = packet
            self.submit(conversation, control)
            return previous
        work = self.store.create_work(conversation, command['id'], kind=kind, arguments=arguments,
                                      work_options=work_options,
                                      project_path=project_path, create=create)
        self.wake.set()
        return work

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.tick()
            except ServiceClosing:
                return
            except Exception:
                logging.getLogger(__name__).exception('Application dispatcher failed')
            self.wake.wait(0.5)
            self.wake.clear()

    def tick(self):
        """每轮只派发未持有执行者的工作；实际调用在独立线程内。"""
        if self.stop.is_set() or self.storage_error():
            return
        try:
            self._tick()
        except (OSError, sqlite3.Error) as exc:
            self.storage_failed('dispatch_record', exc)
            raise StorageUnavailable(self.storage_error()['message']) from exc

    def _tick(self):
        commands = self.store.commands(states=('pending',))
        # 原生线程可能先消费补充并进入等待；这不等于聊天端已生成新的确认方案。
        for row in self.store.commands(states=('applied',)):
            if row['kind'] != 'supplement' or row['value'].get('confirmed_requirements'):
                continue
            work = self.store.work(row['task_id'])
            if work and work.get('confirmed_requirements') and work['status'] in SUPPLEMENTABLE:
                commands.append(row)
        for row in commands:
            if self.stop.is_set():
                return
            conversation, command = row['conversation'], row['value']
            if row['kind'] == 'supplement' and not command.get('confirmed_requirements'):
                target = self.store.work(row['task_id'])
                if not target or target['status'] not in SUPPLEMENTABLE:
                    self.store.command_state(conversation, command['id'], 'failed',
                                             error='WorkNoLongerAcceptsControl')
                    continue
                if row['state'] == 'pending' and target['status'] in RESUMABLE:
                    self._schedule_work(target)
            if row['kind'] == 'message' or (row['kind'] == 'supplement'
                                           and not command.get('confirmed_requirements')):
                if row['kind'] == 'supplement':
                    with self.store.connection() as db:
                        discussed = db.execute("SELECT 1 FROM app_events WHERE conversation=? "
                            "AND kind='requirement_discussion_started' AND json_extract(payload,'$.id')=?",
                            (conversation, command['id'])).fetchone()
                    if discussed:
                        continue
                with self.lock:
                    if self.stop.is_set():
                        return
                    if conversation in self.chats or len(self.chats) >= 2:
                        continue
                    self.chats.add(conversation)
                try:
                    if row['kind'] == 'supplement':
                        self.store.record_event(conversation, 'requirement_discussion_started',
                                                {'id': command['id']})
                    else:
                        self.store.command_state(conversation, command['id'], 'running')
                    self.chat_pool.submit(self._chat, conversation, command)
                except BaseException:
                    with self.lock:
                        self.chats.discard(conversation)
                    raise
            elif row['kind'] == 'confirm_work':
                from agent.application.requirements import accept, work_for_proposal

                if work_for_proposal(self.store, conversation, command['proposal_id']):
                    self.store.command_state(conversation, command['id'], 'applied')
                    continue
                self.store.command_state(conversation, command['id'], 'running')
                try:
                    proposal_id = command['proposal_id']
                    from agent.application.requirements import load_proposal

                    proposal = load_proposal(self.store, conversation, proposal_id)
                    packet, original = accept(
                        self.store, conversation, command, proposal_id,
                        proposal['project_path'], proposal['create'],
                    )
                    self.dispatch_work(
                        conversation, command, project_path=proposal['project_path'], create=proposal['create'],
                        work_options={'confirmed_requirements': packet, 'original_request': original},
                    )
                    self.store.command_state(conversation, command['id'], 'applied')
                    self.store.chat_state(conversation, 'chat_complete')
                except (ValueError, KeyError) as error:
                    self.store.command_state(conversation, command['id'], 'failed', error=str(error))
                    self.store.publish(conversation, command['id'] + ':error', str(error), source='system')
            elif row['kind'] == 'workspace':
                # 选择的是授权工作区；实际项目在开工交接中确定，不能提前安装父目录。
                self.store.command_state(conversation, command['id'], 'applied')
            elif row['kind'] == 'pause':
                task = self.store.work(row['task_id'])
                if not task or task['status'] not in ACTIVE:
                    self.store.command_state(conversation, command['id'], 'applied')
            elif row['kind'] in {'resume', 'supplement'}:
                task = self.store.work(row['task_id'])
                if (row['kind'] == 'supplement' and task and task['status'] in RESUMABLE
                        and any(command['kind'] == 'resume'
                                for command in self.store.commands(task=task['task_id']))):
                    # 本轮明确继续会携带原补充；不能因旧的中断状态先把它判失败。
                    continue
                allowed = RESUMABLE if row['kind'] == 'resume' else SUPPLEMENTABLE
                if task and task['status'] in RESUMABLE and task['status'] in allowed:
                    self._schedule_work(task)
                elif not task or task['status'] not in ACTIVE:
                    self.store.command_state(conversation, command['id'], 'failed',
                                             error='WorkNoLongerAcceptsControl')
        for conversation in self.store.list():
            if self.stop.is_set():
                return
            if conversation.get('read_only'):
                continue
            task = self.store.work(conversation['work_id'])
            if task and task['status'] == 'queued':
                self._schedule_work(task)
        with self.lock:
            if self.stop.is_set() or self.reporting:
                return
            pending = self.store.reserve_progress()
            if pending:
                self.reporting = True
                self.report_task = pending['task_id']
        if pending:
            self.report_pool.submit(self._report, pending)

    def _chat(self, conversation, command):
        try:
            self.ensure_dispatch_available()
            self.store.chat_state(conversation, 'running')
            from agent.task.control import active, current

            with active(self.store.root, 'chat:' + conversation):
                if self.stop.is_set():
                    current().interrupt('pause', self.shutdown.identity)
                if self.store.pause_after(conversation, command['id']):
                    current().interrupt('pause')
                result = self.chatbot.run(conversation, command)
            if command['kind'] != 'supplement':
                self.store.command_state(conversation, command['id'], 'applied')
            self.store.chat_state(conversation, 'awaiting_user' if result.get('waiting_approval')
                                  else 'chat_complete')
        except Exception as exc:
            if isinstance(exc, (OSError, sqlite3.Error)):
                self.storage_failed('chat_record', exc)
                return
            from agent.model.errors import ModelInterrupted, ModelRequestNotSent
            from agent.application.chat import ChatStopped

            status = (
                'paused' if isinstance(exc, (ModelInterrupted, ServiceClosing))
                else exc.status if isinstance(exc, (ModelRequestNotSent, ChatStopped))
                else 'model_request_unknown'
            )
            try:
                if command['kind'] != 'supplement':
                    self.store.command_state(conversation, command['id'],
                                             'applied' if status == 'paused'
                                             else 'unknown' if status == 'model_request_unknown' else 'failed',
                                             error=type(exc).__name__)
                self.store.chat_state(conversation, status)
                self.store.publish(conversation, command['id'] + ':error',
                               '回复已暂停。' if status == 'paused'
                               else '请先配置 DeepSeek API Key，再发送消息。' if status == 'settings_required'
                               else '回复达到输出上限，部分内容未执行；记录已保留。'
                               if status == 'output_limit_reached'
                               else '工具处理发生内部错误；已保存模型响应，工具结果需核对，未自动重试。'
                               if status == 'tool_internal_error'
                               else '模型响应已保存，但本地处理失败；记录已保留，未重新请求。'
                               if status == 'response_processing_error'
                               else '工具调用与返回记录不完整；没有发送新请求，请核对保留现场。'
                               if status == 'invalid_message_history'
                               else '聊天上下文未通过完整性或容量检查；记录已保留，没有发送新请求。'
                               if status in {'context_prefix_changed', 'context_capacity_exhausted',
                                             'compaction_failed', 'compaction_commit_unknown'}
                                   else '本次回复未完整完成，记录已保留；没有自动重发请求。', 'system')
            except (OSError, sqlite3.Error) as error:
                self.storage_failed('chat_failure_record', error)
            logging.getLogger(__name__).exception('Chatbot request stopped')
        finally:
            with self.lock:
                self.chats.discard(conversation)
            self._release_if_idle()
            self.wake.set()

    def _schedule_work(self, job):
        self.ensure_dispatch_available()
        with self.lock:
            if self.stop.is_set():
                return
            if self.works or job['task_id'] in self.controls_inflight.values():
                return
            self.works.add(job['task_id'])
            self.active_jobs[job['task_id']] = job['conversation_id']
        try:
            self.store.work_state(job['task_id'], 'running')
            self.work_pool.submit(self._work, job)
        except BaseException as exc:
            with self.lock:
                self.works.discard(job['task_id'])
                self.active_jobs.pop(job['task_id'], None)
            if isinstance(exc, (OSError, sqlite3.Error)):
                self.storage_failed('work_dispatch_record', exc)
                raise StorageUnavailable(self.storage_error()['message']) from exc
            raise

    def controls(self, task):
        if self.stop.is_set():
            return [{'id': self.shutdown.identity, 'kind': 'pause', 'text': ''}]
        fault = self.storage_error()
        if fault:
            return [{'id': fault['id'], 'kind': 'pause', 'text': ''}]
        return [{'id': row['id'], 'kind': 'message' if row['kind'] == 'supplement' else row['kind'],
                 'text': row['value'].get('text', ''),
                 **({'confirmed_requirements': row['value']['confirmed_requirements']}
                    if row['value'].get('confirmed_requirements') else {}),
                 **({'acceptance_pause_id': row['acceptance_pause_id']}
                    if row.get('acceptance_pause_id') else {})}
                for row in self.store.commands(task=task, states=('pending', 'running'))
                if row['kind'] in {'pause', 'resume', 'supplement'}]

    def emit(self, task, kind, payload):
        try:
            return self._emit(task, kind, payload)
        except (OSError, sqlite3.Error) as exc:
            self.storage_failed('work_projection', exc)
            raise StorageUnavailable(self.storage_error()['message']) from exc

    def _emit(self, task, kind, payload):
        work = self.store.work(task)
        if kind == 'work_state':
            allowed = {key: payload[key] for key in (
                'status', 'reason', 'usage', 'applied_inputs', 'acceptance_limit',
            ) if key in payload}
            seq = self.store.work_state(task, **allowed, event=payload.get('event'),
                                        details=payload.get('details', {}))
        else:
            # 公开事实不复制工作回复、思考、源码、冻结测试及完整检查点。
            safe = {key: payload[key] for key in
                    ('status', 'reason', 'event', 'phase', 'label', 'exit_code', 'outcome',
                     'seconds', 'passed', 'failed', 'count', 'message') if key in payload}
            safe = {key: (item[:4000] if isinstance(item, str) else item)
                    for key, item in safe.items()}
            seq = self.store.record_event(work['conversation_id'], 'work_fact',
                                          {'task_id': task, 'event': kind, **safe})
        relevant = payload.get('event', kind)
        if payload.get('status') in {'preparation_failed', 'runtime_error',
                                     'acceptance_limit_reached'}:
            self.store.finish_progress(task, seq)
            return
        if relevant in KEY_EVENTS:
            self.store.progress_pending(task, seq)
            self.wake.set()

    def _work(self, job):
        task = job['task_id']
        attempted = set()
        try:
            attempted = {row['id'] for row in self.store.commands(task=task)
                         if row['kind'] in {'resume', 'supplement'}}
            self.ensure_dispatch_available()
            result = self.work_factory(job, lambda kind, data: self.emit(task, kind, data),
                                       lambda: self.controls(task))
            self.emit(task, 'work_state', {
                'status': result['status'], 'reason': result.get('stop_reason', result.get('reason', '')),
                'usage': result.get('usage', {}), 'applied_inputs': result.get('applied_inputs', []),
                **({'acceptance_limit': result['acceptance_limit']}
                   if 'acceptance_limit' in result else {}),
            })
            if result['status'] in {'preparation_failed', 'runtime_error'}:
                self.store.publish(job['conversation_id'], 'work:' + task + ':failure',
                                   result.get('stop_reason', result.get('reason', '工作已停止，运行记录已保留。')),
                                   'system')
        except ServiceClosing:
            self.store.work_state(task, 'paused', reason='应用退出，工作尚未派发；可明确继续。')
        except Exception as error:
            logging.getLogger(__name__).exception('Work bridge stopped')
            from agent.runtime.bridge import WorkerUnavailable

            if isinstance(error, (OSError, sqlite3.Error)):
                self.storage_failed('work_record', error)
            try:
                unavailable = isinstance(error, WorkerUnavailable)
                self.emit(task, 'work_state', {
                    'status': 'paused' if unavailable else 'execution_unknown',
                    'reason': ('工作尚未派发，内核或运行设置不可用：' + str(error))
                    if unavailable else '执行结果未确认，已保留任务记录，不自动重试',
                })
            except (OSError, sqlite3.Error):
                pass  # 已闭锁；不能再把写入失败伪装成已保存的终态。
        finally:
            # 入口失败或拒绝的旧控制必须结束，不能在同一暂停状态反复派发。
            try:
                for row in self.store.commands(task=task):
                    if row['id'] in attempted:
                        self.store.command_state(job['conversation_id'], row['id'], 'failed',
                                                 error='WorkControlNotApplied')
            except (OSError, sqlite3.Error) as error:
                self.storage_failed('work_control_completion', error)
            with self.lock:
                self.works.discard(task)
                self.active_jobs.pop(task, None)
            self._release_if_idle()
            self.wake.set()

    def _report(self, pending):
        task, conversation = pending['task_id'], pending['conversation']
        through = pending['reported_seq']
        try:
            self.ensure_dispatch_available()
            if self.store.get(conversation)['work_id'] != task:
                # 新工作已接管展示时，取消旧工作的待汇报，不再为过期事实请求模型。
                through = pending['pending_seq']
                return
            before = self.store.work(task)
            if before and before['status'] == 'acceptance_limit_reached':
                through = pending['pending_seq']
                return
            facts = self.chatbot.work_facts(conversation, pending['reported_seq'], 50,
                                            latest=False, through=pending['pending_seq'])
            through = facts['cursor']
            from agent.task.control import active

            with active(self.store.root, 'progress:' + task):
                self.ensure_dispatch_available()
                result = self.chatbot.progress(conversation, facts)
            current = self.store.work(task)
            # 终态变化后不发布旧的“正在执行”；下次事件汇报会读取新事实。
            if (current and before and current['status'] == before['status']
                    and self.store.get(conversation)['work_id'] == task
                    and result.get('output', '').strip()):
                self.store.publish(conversation, 'progress:' + task + ':' + str(through),
                                   result['output'], 'progress')
        except Exception as error:
            logging.getLogger(__name__).exception('Progress display unavailable; work continues')
            if isinstance(error, (OSError, sqlite3.Error)):
                self.storage_failed('progress_record', error)
            try:
                self.store.record_event(conversation, 'summary_failed',
                                        {'task_id': task, 'through': through})
            except (OSError, sqlite3.Error) as exc:
                self.storage_failed('progress_failure_record', exc)
        finally:
            try:
                self.store.finish_progress(task, through)
            except (OSError, sqlite3.Error) as exc:
                self.storage_failed('progress_completion_record', exc)
            with self.lock:
                self.reporting = False
                self.report_task = None
            self._release_if_idle()
            self.wake.set()

    def status(self):
        queued, pending = 0, 0
        if not self.storage_error():
            try:
                queued = self.store.queued_work_count()
                pending = len(self.store.commands(states=('pending',))) + queued
            except (OSError, sqlite3.Error) as exc:
                self.storage_failed('status_read', exc)
        try:
            from agent.execution.preparation import active_count

            preparing = active_count(root=self.store.root)
        except (OSError, sqlite3.Error) as exc:
            self.storage_failed('preparation_status_read', exc)
            preparing = 0
        except ValueError:
            preparing = 1
        fault = self.storage_error()
        with self.lock:
            return {'running_work': len(self.works), 'running_chat': len(self.chats),
                    'running_progress': int(self.reporting), 'running_title': len(self.titles),
                    'running_control': len(self.controls_inflight),
                    'pending': pending, 'queued_work': queued, 'environment_preparations': preparing,
                    'storage_error': fault, 'shutting_down': self.stop.is_set()}

    def _reserve_title(self, conversation, command):
        from agent.application.titles import reserve

        if not reserve(self.store, conversation, command):
            return
        with self.lock:
            if self.stop.is_set():
                return
            self.titles.add(conversation)
        self.title_pool.submit(self._title, conversation)

    def _title(self, conversation):
        from agent.application.titles import generate
        from agent.task.control import active

        try:
            self.ensure_dispatch_available()
            with active(self.store.root, 'title:' + conversation):
                self.ensure_dispatch_available()
                generate(self.store, conversation, self.title_factory)
        except (OSError, sqlite3.Error) as exc:
            self.storage_failed('title_record', exc)
        finally:
            with self.lock:
                self.titles.discard(conversation)
            self._release_if_idle()

    def purge(self, conversation, retain_environment=False):
        from agent.application.purge import cleanup

        return self.store.purge(conversation, lambda thread, tasks:
                                cleanup(self.store, thread, tasks, retain_environment))
