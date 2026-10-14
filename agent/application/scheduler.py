"""后台调度与公开进度；浏览器断开不会创建、重复或取消工作请求。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
import logging
import time
import uuid

from agent.application.chat import Chatbot
from agent.application.store import ACTIVE, TERMINAL, RESUMABLE, SUPPLEMENTABLE, Conflict
from agent.application.lease import ServiceLease


KEY_EVENTS = {
    'environment_preparation', 'plan_accepted', 'plan_rejected', 'plan_finished',
    'acceptance_trial_finished', 'final_acceptance_finished', 'user_input_applied',
    'paused', 'task_returned', 'work_state', 'environment',
}


class Scheduler:
    def __init__(self, store, *, work_factory=None, pause_callback=None, model_factory=None,
                 tool_invoker=None, prepare_callback=None, title_factory=None,
                 supplement_callback=None):
        self.store = store
        self.work_factory = work_factory or self._run_work
        self.pause_callback = pause_callback or self._pause_work
        self.supplement_callback = supplement_callback or self._supplement_work
        self.prepare_callback = prepare_callback or self._prepare
        self.title_factory = title_factory
        self.chatbot = Chatbot(store, self.dispatch_work, factory=model_factory,
                              invoke_tool=tool_invoker)
        self.lock = Lock()
        self.chats = set()
        self.works = set()
        self.titles = set()
        self.controls_inflight = {}
        self.reporting = False
        self.stop = Event()
        self.wake = Event()
        self.chat_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='application-chat')
        self.work_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='application-work')
        self.report_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='application-progress')
        self.title_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='application-title')
        self.control_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='application-control')
        self.worker = None
        self.lease = None

    @staticmethod
    def _run_work(job, emit, controls):
        from agent.application.work_bridge import run_work

        return run_work(job, emit, controls)

    def _pause_work(self, task, command):
        from agent.application.work_bridge import pause_work

        return pause_work(task, command, root=self.store.root)

    def _supplement_work(self, task, command):
        from agent.application.work_bridge import supplement_work

        return supplement_work(task, command, root=self.store.root)

    def _prepare(self, repo):
        if repo:
            from agent.execution.preparation import jobs

            return jobs(root=self.store.root).start(repo)

    def start(self):
        if self.worker is not None:
            return
        self.lease = ServiceLease(self.store.root)
        self.lease.acquire()
        try:
            self.store.recover_interrupted()
        except Exception:
            self.lease.close()
            self.lease = None
            raise
        self.worker = Thread(target=self._loop, name='application-dispatch', daemon=True)
        self.worker.start()

    def close(self):
        self.stop.set()
        self.wake.set()
        if self.worker:
            self.worker.join(timeout=2)
        for pool in (self.chat_pool, self.work_pool, self.report_pool, self.title_pool,
                     self.control_pool):
            pool.shutdown(wait=False, cancel_futures=True)
        self._release_if_idle()

    def _release_if_idle(self):
        with self.lock:
            if self.stop.is_set() and not (
                self.chats or self.works or self.titles or self.reporting or self.controls_inflight
            ):
                if self.lease:
                    self.lease.close()
                    self.lease = None

    def submit(self, conversation, command):
        result = self.store.submit(conversation, command)
        if command['kind'] == 'message' and not result['applied']:
            try:
                self._reserve_title(conversation, command)
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

    def _control(self, conversation, task, command, callback):
        try:
            result = callback(task, command['id'])
            termination = result.get('termination', []) if isinstance(result, dict) else []
            confirmed = all(row.get('stopped') is True for row in termination)
            if not confirmed and command['kind'] == 'pause':
                self.store.control_unconfirmed(task, command['id'])
            self.store.record_event(conversation,
                                    'control_delivered' if confirmed else 'control_unconfirmed',
                                    {'id': command['id'], 'kind': command['kind'], 'task_id': task})
        except Exception:
            logging.getLogger(__name__).exception('Immediate control could not be confirmed')
            if command['kind'] == 'pause':
                self.store.control_unconfirmed(task, command['id'])
            self.store.record_event(conversation, 'control_unconfirmed',
                                    {'id': command['id'], 'kind': command['kind'], 'task_id': task})
        finally:
            with self.lock:
                self.controls_inflight.pop(command['id'], None)
            self._release_if_idle()
            self.wake.set()

    def dispatch_work(self, conversation, command, *, kind='coding', arguments=None,
                      work_options=None):
        value = self.store.get(conversation)
        previous = self.store.work(value['work_id'])
        if previous and previous['status'] not in TERMINAL:
            if kind != 'coding':
                raise Conflict('当前工作尚未结束，不能同时运行另一任务')
            control = {'id': str(uuid.uuid5(uuid.UUID(command['id']), 'supplement')),
                       'kind': 'supplement', 'text': command['text']}
            self.submit(conversation, control)
            return previous
        work = self.store.create_work(conversation, command['id'], kind=kind, arguments=arguments,
                                      work_options=work_options)
        self.wake.set()
        return work

    def _loop(self):
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception:
                logging.getLogger(__name__).exception('Application dispatcher failed')
            self.wake.wait(0.5)
            self.wake.clear()

    def tick(self):
        """每轮只派发未持有执行者的工作；实际调用在独立线程内。"""
        for row in self.store.commands(states=('pending',)):
            conversation, command = row['conversation'], row['value']
            if row['kind'] == 'message':
                with self.lock:
                    if conversation in self.chats or len(self.chats) >= 2:
                        continue
                    self.chats.add(conversation)
                self.store.command_state(conversation, command['id'], 'running')
                self.chat_pool.submit(self._chat, conversation, command)
            elif row['kind'] == 'workspace':
                self.store.command_state(conversation, command['id'], 'running')
                try:
                    self.prepare_callback(self.store.get(conversation)['repo'])
                    self.store.command_state(conversation, command['id'], 'applied')
                except Exception as exc:
                    self.store.command_state(conversation, command['id'], 'failed',
                                             error=type(exc).__name__)
            elif row['kind'] == 'pause':
                task = self.store.work(row['task_id'])
                if not task or task['status'] not in ACTIVE:
                    self.store.command_state(conversation, command['id'], 'applied')
            elif row['kind'] in {'resume', 'supplement'}:
                task = self.store.work(row['task_id'])
                allowed = RESUMABLE if row['kind'] == 'resume' else SUPPLEMENTABLE
                if task and task['status'] in RESUMABLE and task['status'] in allowed:
                    self._schedule_work(task)
                elif not task or task['status'] not in ACTIVE:
                    self.store.command_state(conversation, command['id'], 'failed',
                                             error='WorkNoLongerAcceptsControl')
        for conversation in self.store.list():
            if conversation.get('read_only'):
                continue
            task = self.store.work(conversation['work_id'])
            if task and task['status'] == 'queued':
                self._schedule_work(task)
        with self.lock:
            if self.reporting:
                return
            pending = self.store.reserve_progress()
            if pending:
                self.reporting = True
        if pending:
            self.report_pool.submit(self._report, pending)

    def _chat(self, conversation, command):
        try:
            self.store.chat_state(conversation, 'running')
            from agent.task.control import active, current

            with active(self.store.root, 'chat:' + conversation):
                if self.store.pause_after(conversation, command['id']):
                    current().interrupt('pause')
                result = self.chatbot.run(conversation, command)
            self.store.command_state(conversation, command['id'], 'applied')
            self.store.chat_state(conversation, 'awaiting_user' if result.get('waiting_approval')
                                  else 'chat_complete')
        except Exception as exc:
            from agent.model.errors import ModelInterrupted, ModelRequestNotSent
            from agent.application.chat import ChatStopped

            status = (
                'paused' if isinstance(exc, ModelInterrupted)
                else exc.status if isinstance(exc, (ModelRequestNotSent, ChatStopped))
                else 'model_request_unknown'
            )
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
                               else '聊天上下文未通过完整性或容量检查；记录已保留，没有发送新请求。'
                               if status in {'context_prefix_changed', 'context_capacity_exhausted',
                                             'compaction_failed', 'compaction_commit_unknown',
                                             'invalid_message_history'}
                               else '本次回复未完整完成，记录已保留；没有自动重发请求。', 'system')
            logging.getLogger(__name__).exception('Chatbot request stopped')
        finally:
            with self.lock:
                self.chats.discard(conversation)
            self._release_if_idle()
            self.wake.set()

    def _schedule_work(self, job):
        with self.lock:
            if self.works or job['task_id'] in self.controls_inflight.values():
                return
            self.works.add(job['task_id'])
        self.store.work_state(job['task_id'], 'running')
        self.work_pool.submit(self._work, job)

    def controls(self, task):
        return [{'id': row['id'], 'kind': 'message' if row['kind'] == 'supplement' else row['kind'],
                 'text': row['value'].get('text', '')}
                for row in self.store.commands(task=task)
                if row['kind'] in {'pause', 'resume', 'supplement'}]

    def emit(self, task, kind, payload):
        work = self.store.work(task)
        if kind == 'work_state':
            allowed = {key: payload[key] for key in ('status', 'reason', 'usage', 'applied_inputs')
                       if key in payload}
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
        if relevant in KEY_EVENTS:
            self.store.progress_pending(task, seq)
            self.wake.set()

    def _work(self, job):
        task = job['task_id']
        try:
            result = self.work_factory(job, lambda kind, data: self.emit(task, kind, data),
                                       lambda: self.controls(task))
            self.emit(task, 'work_state', {
                'status': result['status'], 'reason': result.get('stop_reason', result.get('reason', '')),
                'usage': result.get('usage', {}), 'applied_inputs': result.get('applied_inputs', []),
            })
        except Exception:
            logging.getLogger(__name__).exception('Work bridge stopped')
            self.emit(task, 'work_state', {'status': 'execution_unknown',
                                          'reason': '执行结果未确认，已保留任务记录，不自动重试'})
        finally:
            with self.lock:
                self.works.discard(task)
            self._release_if_idle()
            self.wake.set()

    def _report(self, pending):
        task, conversation = pending['task_id'], pending['conversation']
        through = pending['reported_seq']
        try:
            if self.store.get(conversation)['work_id'] != task:
                # 新工作已接管展示时，取消旧工作的待汇报，不再为过期事实请求模型。
                through = pending['pending_seq']
                return
            before = self.store.work(task)
            facts = self.chatbot.work_facts(conversation, pending['reported_seq'], 50,
                                            latest=False, through=pending['pending_seq'])
            through = facts['cursor']
            result = self.chatbot.progress(conversation, facts)
            current = self.store.work(task)
            # 终态变化后不发布旧的“正在执行”；下次事件汇报会读取新事实。
            if (current and before and current['status'] == before['status']
                    and self.store.get(conversation)['work_id'] == task
                    and result.get('output', '').strip()):
                self.store.publish(conversation, 'progress:' + task + ':' + str(through),
                                   result['output'], 'progress')
        except Exception:
            logging.getLogger(__name__).exception('Progress display unavailable; work continues')
            self.store.record_event(conversation, 'summary_failed',
                                    {'task_id': task, 'through': through})
        finally:
            self.store.finish_progress(task, through)
            with self.lock:
                self.reporting = False
            self._release_if_idle()
            self.wake.set()

    def status(self):
        queued = self.store.queued_work_count()
        pending = len(self.store.commands(states=('pending',))) + queued
        try:
            from agent.execution.preparation import active_count

            preparing = active_count(root=self.store.root)
        except (OSError, ValueError):
            preparing = 1
        with self.lock:
            return {'running_work': len(self.works), 'running_chat': len(self.chats),
                    'running_progress': int(self.reporting), 'running_title': len(self.titles),
                    'running_control': len(self.controls_inflight),
                    'pending': pending, 'queued_work': queued, 'environment_preparations': preparing}

    def _reserve_title(self, conversation, command):
        from agent.application.titles import reserve

        if not reserve(self.store, conversation, command):
            return
        with self.lock:
            self.titles.add(conversation)
        self.title_pool.submit(self._title, conversation)

    def _title(self, conversation):
        from agent.application.titles import generate

        try:
            generate(self.store, conversation, self.title_factory)
        finally:
            with self.lock:
                self.titles.discard(conversation)
            self._release_if_idle()

    def purge(self, conversation, retain_environment=False):
        from agent.application.purge import cleanup

        return self.store.purge(conversation, lambda thread, tasks:
                                cleanup(self.store, thread, tasks, retain_environment))
