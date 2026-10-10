"""One existing planner loop; persistent lookup and resume do not need a model."""
import argparse
import json
import uuid
from pathlib import Path
from langchain_core.messages import messages_from_dict
from agent.task.budget import StepBudget
from agent.storage.history import LocalHistory
from agent.server.runtime import ConversationRunner
from agent.execution.projects import workspace_binding


class LazyModel:
    def __init__(self):
        self.model = None

    def bind_tools(self, tools):
        if self.model is None:
            from agent.settings import load_settings
            from agent.model.client import create_model
            self.model = create_model(load_settings())
        return self.model.bind_tools(tools)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-dir',required=True)
    parser.add_argument('--repo')
    parser.add_argument('--task')
    parser.add_argument('--verify-command')
    parser.add_argument('--acceptance')
    parser.add_argument('--max-steps',type=int)
    parser.add_argument('--pause-after-steps',type=int)
    parser.add_argument('--find',help='Search existing history metadata; no model or shell')
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args(argv)
    folder=Path(args.state_dir).resolve()
    old=None
    if (folder/'history.sqlite3').exists():
        store=LocalHistory(folder)
        try:
            with store.lease():
                old=store.load()
                if args.find is not None:
                    print(json.dumps(store.find(args.find),ensure_ascii=False,indent=2))
                    return 0
                if args.status:
                    print(json.dumps({k:old.get(k) for k in ('status','reason','source','plan_count')},ensure_ascii=False,indent=2))
                    return 0
        finally:
            store.close()
    elif args.find is not None or args.status:
        parser.error('No existing indexed task at this path')
    old=old or {}
    repo=args.repo or old.get('repo') or old.get('environment_key',{}).get('repo')
    task=args.task or old.get('request')
    if not repo or not task:
        parser.error('A new task requires --repo and --task')
    if old and not old.get('execution_binding'):
        parser.error('旧任务须使用冻结源码和原执行后端恢复，不迁移检查点')
    saved=old.get('budget',{})
    policy=('max_steps','max_tokens','timeout_seconds','report_reserve','report_token_reserve',
            'report_time_reserve','transport_retry_limit')
    values={k:saved[k] for k in policy if k in saved}
    if args.max_steps is not None:
        values['max_steps']=args.max_steps
    budget=StepBudget(**values)
    history=[]
    if old:
        history=messages_from_dict(old.get('public_conversation',[]))
    controls=[{'id': uuid.uuid4().hex, 'kind': 'resume'}] if old.get('status') in {'paused', 'awaiting_user'} else []
    loop=ConversationRunner(LazyModel(),repo,state_dir=folder,pause_after_steps=args.pause_after_steps,
                            control_source=lambda: controls,
                            yield_boundary=False, execution_binding=old.get('execution_binding') or workspace_binding(repo))
    result=loop.run(history,task,budget,
        verify_command=args.verify_command if args.verify_command is not None else old.get('verify_command'),
        acceptance=args.acceptance if args.acceptance is not None else old.get('acceptance'))
    print(json.dumps({k:v for k,v in result.items() if k not in {'messages','rounds','verification'}},ensure_ascii=False,indent=2))
    return 0 if result['status'] in {'verified_success','paused','chat_complete'} else 1


if __name__=='__main__':
    raise SystemExit(main())
