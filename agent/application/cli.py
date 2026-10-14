"""控制台复用同一应用服务，避免保留第二套聊天与工作调度。"""

import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from agent.application.scheduler import Scheduler
from agent.application.store import ACTIVE, ApplicationStore
from agent.storage.preferences import state_root, Preferences


class ConsoleConversation:
    def __init__(self, model, repo, *, store=None, scheduler=None):
        self.repo = repo
        self.store = store or ApplicationStore(state_root() / 'cli')
        self.conversation = self.store.create(
            repo=repo, runtime_settings=Preferences().defaults())['thread_id']
        if scheduler is None:
            from agent.application.work_bridge import run_work
            from agent.application.chat import close_model
            from agent.model.client import ConfiguredModel, create_model
            from agent.settings import load_settings

            def factory(settings):
                return (create_model(load_settings(settings), model.policy)
                        if isinstance(model, ConfiguredModel) else model)

            def work(job, emit, controls):
                selected = factory(job['settings'])
                try:
                    return run_work(job, emit, controls, model=selected)
                finally:
                    close_model(selected)

            scheduler = Scheduler(
                self.store, model_factory=factory, work_factory=work,
            )
        self.scheduler = scheduler
        self.scheduler.start()

    def close(self):
        self.scheduler.close()

    def run(self, history, text, budget, verify_command=None, acceptance=None):
        command = {'id': str(uuid.uuid4()), 'kind': 'message', 'text': text}
        # 固定验收设置属于开工输入，由工作接口原样承接，不能由聊天摘要替代。
        original = self.scheduler.dispatch_work

        def dispatch(conversation, incoming, **kwargs):
            return original(conversation, incoming, work_options={
                'verify_command': verify_command, 'acceptance': acceptance,
                'budget': {'max_steps': budget.max_steps},
            }, **kwargs)

        self.scheduler.chatbot.dispatch_work = dispatch
        self.scheduler.submit(self.conversation, command)
        try:
            while True:
                snapshot = self.store.snapshot(self.conversation)
                work = snapshot['work']
                if snapshot['chat_status'] not in ACTIVE and (not work or work['status'] not in ACTIVE):
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.scheduler.submit(self.conversation, {'id': str(uuid.uuid4()), 'kind': 'pause'})
            raise
        finally:
            self.scheduler.chatbot.dispatch_work = original
        totals = snapshot['usage']
        budget.used_steps = sum(value.get('calls', 0) for value in totals.values())
        budget.reported_tokens = sum(value.get('total_tokens', 0) for value in totals.values())
        budget.unknown_usage_steps = sum(value.get('unknown_calls', 0) for value in totals.values())
        return {
            'status': snapshot['status'], 'stop_reason': (work or {}).get('reason', ''),
            'messages': [(HumanMessage if row['type'] == 'human' else AIMessage)(
                content=row['content'], id=row['id']) for row in snapshot['messages']],
            'artifact_dir': str(self.store.root), 'usage': totals,
        }
