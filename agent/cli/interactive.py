"""MonoPlanner 交互入口：记住工作区，每条需求使用同一个规划流程。"""

import argparse
import sys

from langchain_core.messages import HumanMessage

from agent.task.budget import StepBudget
from agent.settings import load_settings
from agent.execution.native import NativeEnvironment
from agent.cli.workspace import load_saved_repo, save_repo


SESSION_HELP = (
    "/repo [路径] 查看或切换仓库；/clear 清空对话；/help 帮助；/exit 退出\n"
    "/verify-command [命令]、/acceptance [要求]：查看或设置下一条需求的验收；用 - 清除设置"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="在工作区使用 MonoPlanner 连续对话。"
    )
    parser.add_argument("--repo", help="目标工作区目录；省略时使用上次保存的仓库")
    parser.add_argument(
        "--check-workspace",
        action="store_true",
        help="只检查工作区与计划 Schema，不读取 API 配置或调用模型",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="每条需求最多调用模型的次数",
    )
    parser.add_argument("--verify-command", help="验收命令，在目标仓库运行；交互模式仅用于下一条需求")
    parser.add_argument("--acceptance", help="补充验收要求；交互模式仅用于下一条需求")
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps 必须大于 0")
    for option in ("verify_command", "acceptance"):
        value = getattr(args, option)
        if value is not None and not value.strip():
            parser.error(f"--{option.replace('_', '-')} 不能为空")
    return args


def open_workspace(path: str) -> NativeEnvironment:
    """交互输入可以粘贴带引号、带空格的目录路径。"""
    path = path.strip()
    if len(path) >= 2 and path[0] == path[-1] and path[0] in ("'", '"'):
        path = path[1:-1]
    if not path.strip():
        raise ValueError("仓库路径不能为空")
    return NativeEnvironment(path)


def select_workspace(args) -> NativeEnvironment:
    if args.repo is not None:
        return open_workspace(args.repo)

    interactive = not args.check_workspace
    try:
        saved_repo = load_saved_repo()
        if saved_repo:
            return open_workspace(saved_repo)
    except (OSError, ValueError) as exc:
        if not interactive:
            raise
        print(f"无法使用上次的仓库：{exc}")

    if not interactive:
        raise ValueError("没有可用的默认仓库，请通过 --repo 指定目录，或先启动交互模式选择仓库")
    print("请选择工作仓库。选好后会记住路径，下次启动自动使用。输入 /exit 退出。")
    while True:
        path = input("仓库路径> ").strip()
        if path == "/exit":
            raise EOFError
        try:
            return open_workspace(path)
        except (OSError, ValueError) as exc:
            print(f"路径无效：{exc}")


def load_model():
    """输入第一条自然语言需求时才读取密钥、创建模型。"""
    settings = load_settings()
    from agent.model.client import create_model
    return create_model(settings)


def create_workspace_agent(model, environment):
    from agent.task.runner import TaskRunner
    return TaskRunner(model, environment, interactive=True)


def run_one_turn(agent, history, text, max_steps, verify_command=None, acceptance=None):
    budget = StepBudget(max_steps=max_steps)
    print("正在准备并执行任务……", flush=True)
    result = agent.run(history, text, budget, verify_command=verify_command, acceptance=acceptance)
    if result["status"] == "chat_complete":
        pass
    elif result["status"] == "verified_success":
        print("[验收结果] 固定程序验收通过。")
    else:
        notice = f"[程序状态] {result.get('stop_reason', result['status'])}；任务尚未确认完成。"
        print(notice)
        result["messages"] = [*result["messages"], HumanMessage(content=notice)]
    if result.get("final_report"):
        if result['status'] != 'chat_complete':
            print(f"\n程序：{result['final_report']}")
    from langchain_core.messages import AIMessage
    for message in result.get('messages', []):
        if isinstance(message, AIMessage) and message not in history:
            print(message.text)
    for key, label in (("patch_path", "候选补丁"), ("plan_path", "当前计划"), ("history_dir", "任务记录")):
        if result.get(key):
            print(f"[{label}] {result[key]}")
    print(f"[预算] 模型调用 {budget.used_steps}/{budget.max_steps}；"
          f"已报告 Token {budget.reported_tokens}；未报告用量 {budget.unknown_usage_steps} 次")
    return result


def run_session(
    environment,
    max_steps: int = 1000,
    verify_command=None,
    acceptance=None,
) -> int:
    model = None
    agent = None
    messages = []

    print(f"工作区：{environment.repo_root}")
    print("直接输入需求即可连续对话。重启程序会记住仓库，但不恢复聊天记录。")
    print(SESSION_HELP)

    while True:
        text = input("\n你> ").strip()
        if not text:
            continue
        if text == "/exit":
            return 0
        if text == "/help":
            print(SESSION_HELP)
            continue
        if text == "/clear":
            messages = []
            agent = None
            verify_command, acceptance = None, None
            print("已清空对话及待用验收设置，工作仓库保持不变。")
            continue
        command, *arguments = text.split(maxsplit=1)
        if command in {"/verify-command", "/acceptance"}:
            if arguments:
                value = None if arguments[0] == "-" else arguments[0]
                if command == "/verify-command":
                    verify_command = value
                else:
                    acceptance = value
            current = verify_command if command == "/verify-command" else acceptance
            print(f"{command}（仅下一条需求）：{current or '未设置'}")
            continue
        if command == "/repo":
            if not arguments:
                print(f"工作区：{environment.repo_root}")
                continue
            try:
                new_environment = open_workspace(arguments[0])
                if new_environment.repo_root == environment.repo_root:
                    print(f"已在当前工作区：{environment.repo_root}")
                    continue
                # 全部准备成功再切换，路径错误不会丢掉当前会话。
                save_repo(new_environment.repo_root)
            except (OSError, ValueError) as exc:
                print(f"切换失败，仍在 {environment.repo_root}：{exc}")
                continue
            environment = new_environment
            agent, messages = None, []
            verify_command, acceptance = None, None
            print(f"已切换工作区：{environment.repo_root}，并开始新会话。")
            continue

        if text.startswith("/"):
            print(f"未知命令。{SESSION_HELP}")
            continue

        # 一旦提交需求就消费本次设置，避免失败后泄漏到不相关的新需求。
        current_command, current_acceptance = verify_command, acceptance
        verify_command, acceptance = None, None
        try:
            if model is None:
                model = load_model()
            if agent is None:
                agent = create_workspace_agent(model, environment)
            result = run_one_turn(
                agent, messages, text, max_steps,
                verify_command=current_command, acceptance=current_acceptance,
            )
            messages = result["messages"]

        except Exception as exc:
            # 失败步骤可能已执行工具，不能声称回滚或保留了完整执行轨迹。
            agent, messages = None, []
            print(f"本轮失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            print("已清空对话；已执行的文件修改不会回滚，下一条需求请先检查当前文件状态。")


def check_workspace(environment):
    from agent.model.tools import WORK_SCHEMA
    if WORK_SCHEMA.get("type") != "object":
        raise ValueError("计划 Schema 必须是 JSON 对象")
    print(f"工作区：{environment.repo_root}")
    print("检查通过：工作区有效，p0401 计划 Schema 已加载。")
    print("本次未读取 API 配置，也未创建或调用模型。")


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        # 选择工作区，模型仍在第一条需求提交时才创建。
        environment = select_workspace(args)

        # 离线检查：不创建模型。
        if args.check_workspace:
            check_workspace(environment)
            return 0

        save_repo(environment.repo_root)
        return run_session(environment, args.max_steps, args.verify_command, args.acceptance)

    except EOFError:
        print("\n已退出。")
        return 0

    except KeyboardInterrupt:
        print("\n任务已被用户中断。", file=sys.stderr)
        return 130

    except Exception as exc:
        print(
            f"运行失败：{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
